from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from model_core.loader import load_classifier
from model_core.preprocess import preprocess_crops
from pipeline import constants, utils


logger = utils.get_logger("stage_15_crop_duplicate_detection")

REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_AUTO_RESOLVED = "auto_resolved"
REVIEW_STATUS_CONFIRMED = "confirmed"
REVIEW_STATUS_EXCLUDED = "excluded"
MANUAL_REVIEW_STATUSES = {REVIEW_STATUS_CONFIRMED, REVIEW_STATUS_EXCLUDED}
IGNORED_MANUAL_REVIEW_LABELS = {
    constants.NOISE_REVIEW_LABEL_BAD_CROP,
    constants.NOISE_REVIEW_LABEL_OUT_OF_LABEL_SPACE,
}


def bbox_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Return IoU for two xyxy crop rows."""
    intersection_x1 = max(float(left["box_x1"]), float(right["box_x1"]))
    intersection_y1 = max(float(left["box_y1"]), float(right["box_y1"]))
    intersection_x2 = min(float(left["box_x2"]), float(right["box_x2"]))
    intersection_y2 = min(float(left["box_y2"]), float(right["box_y2"]))
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height

    left_area = max(
        0.0,
        (float(left["box_x2"]) - float(left["box_x1"]))
        * (float(left["box_y2"]) - float(left["box_y1"])),
    )
    right_area = max(
        0.0,
        (float(right["box_x2"]) - float(right["box_x1"]))
        * (float(right["box_y2"]) - float(right["box_y1"])),
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def cluster_duplicate_rows(
    rows: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[list[dict[str, Any]]]:
    """Greedily group near-identical boxes using the oldest crop as anchor."""
    clusters: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: int(item["crop_id"])):
        best_cluster = None
        best_iou = -1.0
        for cluster in clusters:
            current_iou = bbox_iou(row, cluster[0])
            if current_iou >= iou_threshold and current_iou > best_iou:
                best_cluster = cluster
                best_iou = current_iou
        if best_cluster is None:
            clusters.append([row])
        else:
            best_cluster.append(row)
    return clusters


def duplicate_group_key(
    *,
    source_sha1: str,
    detector_model: str,
    nms_iou_threshold: float,
    anchor_crop_id: int,
) -> str:
    payload = (
        f"{source_sha1}|{detector_model}|{float(nms_iou_threshold):.8f}|"
        f"{int(anchor_crop_id)}"
    )
    return "cropdup_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def choose_representative(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer reviewed/corrected and high-score rows while retaining stable provenance."""

    def sort_key(row: dict[str, Any]) -> tuple[int, int, float, int]:
        review_label = str(row.get("noise_review_label") or "").strip()
        corrected_label = str(row.get("manual_corrected_label") or "").strip()
        return (
            int(bool(corrected_label)),
            int(review_label == constants.NOISE_REVIEW_LABEL_OK),
            float(row.get("detector_score") or 0.0),
            -int(row["crop_id"]),
        )

    return max(members, key=sort_key)


def build_duplicate_groups(
    rows: pd.DataFrame,
    *,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    required = {
        "crop_id",
        "source_sha1",
        "detector_model",
        "nms_iou_threshold",
        "effective_label",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"重复crop探测输入缺少列: {sorted(missing)}")
    if not 0 < float(iou_threshold) <= 1:
        raise ValueError("crop_duplicate_detection.bbox_iou_threshold必须在(0, 1]范围内")

    groups: list[dict[str, Any]] = []
    grouped = rows.groupby(
        ["source_sha1", "detector_model", "nms_iou_threshold"],
        sort=True,
        dropna=False,
    )
    for (source_sha1, detector_model, nms_iou_threshold), frame in grouped:
        records = frame.to_dict(orient="records")
        for members in cluster_duplicate_rows(records, iou_threshold=float(iou_threshold)):
            if len(members) < 2:
                continue
            members = sorted(members, key=lambda row: int(row["crop_id"]))
            anchor = members[0]
            representative = choose_representative(members)
            candidate_labels = sorted(
                {
                    str(row["effective_label"]).strip()
                    for row in members
                    if str(row["effective_label"]).strip()
                }
            )
            if not candidate_labels:
                raise ValueError(
                    f"重复crop组没有可用label: crop_ids={[row['crop_id'] for row in members]}"
                )
            auto_resolved = len(candidate_labels) == 1
            groups.append(
                {
                    "group_key": duplicate_group_key(
                        source_sha1=str(source_sha1),
                        detector_model=str(detector_model),
                        nms_iou_threshold=float(nms_iou_threshold),
                        anchor_crop_id=int(anchor["crop_id"]),
                    ),
                    "source_sha1": str(source_sha1),
                    "detector_model": str(detector_model),
                    "nms_iou_threshold": float(nms_iou_threshold),
                    "representative_crop_id": int(representative["crop_id"]),
                    "member_crop_ids": [int(row["crop_id"]) for row in members],
                    "candidate_labels": candidate_labels,
                    "member_count": len(members),
                    "candidate_label_count": len(candidate_labels),
                    "box_x1": float(anchor["box_x1"]),
                    "box_y1": float(anchor["box_y1"]),
                    "box_x2": float(anchor["box_x2"]),
                    "box_y2": float(anchor["box_y2"]),
                    "review_status": (
                        REVIEW_STATUS_AUTO_RESOLVED
                        if auto_resolved
                        else REVIEW_STATUS_PENDING
                    ),
                    "resolved_label": candidate_labels[0] if auto_resolved else None,
                    "exclusion_reason": None,
                    "review_note": None,
                    "reviewed_at": None,
                }
            )
    return groups


def resolve_torch_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def proposal_model_name(model_dir: Path) -> str:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        return model_dir.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest.get("artifact_version") or model_dir.name)


@torch.inference_mode()
def add_linear_head_proposals(
    groups: list[dict[str, Any]],
    *,
    crop_rows: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    if not groups:
        return
    duplicate_config = config["crop_duplicate_detection"]
    model_dir = utils.join_data_root(
        duplicate_config["prediction_model_dir"],
        config=config,
    )
    device = resolve_torch_device(str(duplicate_config["prediction_device"]))
    loaded = load_classifier(model_dir, device=device, local_files_only=True)
    label_to_id = {
        str(row["label"]): int(row["label_id"])
        for row in loaded.labels
    }
    batch_size = int(duplicate_config["prediction_batch_size"])
    top_k = int(duplicate_config["prediction_top_k"])
    if batch_size < 1:
        raise ValueError("crop_duplicate_detection.prediction_batch_size必须是正整数")
    if top_k < 1:
        raise ValueError("crop_duplicate_detection.prediction_top_k必须是正整数")

    row_by_crop_id = {
        int(row["crop_id"]): row
        for row in crop_rows.to_dict(orient="records")
    }
    model_name = proposal_model_name(model_dir)
    for start in tqdm(
        range(0, len(groups), batch_size),
        desc="重复crop线性头提议",
        unit="batch",
    ):
        batch_groups = groups[start : start + batch_size]
        images = [
            utils.load_crop(
                row_by_crop_id[int(group["representative_crop_id"])],
                config=config,
                pad_frac=float(config["crops_storage"]["crop_pad_frac"]),
            )
            for group in batch_groups
        ]
        pixel_values = preprocess_crops(
            images=images,
            processor=loaded.processor,
            image_size=int(loaded.model_config["image_size"]),
        ).to(device)
        logits = loaded.model(pixel_values)
        probabilities = torch.softmax(logits, dim=1)
        global_k = min(top_k, probabilities.shape[1])
        global_probs, global_ids = probabilities.topk(global_k, dim=1)

        for (
            group,
            sample_logits,
            sample_probabilities,
            sample_global_probs,
            sample_global_ids,
        ) in zip(
            batch_groups,
            logits,
            probabilities,
            global_probs,
            global_ids,
            strict=True,
        ):
            global_top_k = [
                {
                    "label": loaded.id_to_label[int(label_id)],
                    "probability": float(probability),
                }
                for probability, label_id in zip(
                    sample_global_probs.cpu().tolist(),
                    sample_global_ids.cpu().tolist(),
                    strict=True,
                )
            ]
            group["global_top1_label"] = global_top_k[0]["label"]
            group["global_top1_prob"] = global_top_k[0]["probability"]
            group["global_top_k"] = global_top_k
            group["proposal_model"] = model_name

            available_candidates = [
                label
                for label in group["candidate_labels"]
                if label in label_to_id
            ]
            if not available_candidates:
                group["proposed_label"] = None
                group["proposed_candidate_prob"] = None
                group["proposed_candidate_margin"] = None
                group["candidate_scores"] = []
                continue

            candidate_ids = torch.tensor(
                [label_to_id[label] for label in available_candidates],
                device=sample_logits.device,
                dtype=torch.long,
            )
            candidate_probabilities = torch.softmax(
                sample_logits.index_select(0, candidate_ids),
                dim=0,
            )
            order = candidate_probabilities.argsort(descending=True)
            candidate_scores = [
                {
                    "label": available_candidates[int(index)],
                    "probability": float(candidate_probabilities[int(index)].cpu()),
                    "global_probability": float(
                        sample_probabilities[
                            label_to_id[available_candidates[int(index)]]
                        ].cpu()
                    ),
                }
                for index in order.cpu().tolist()
            ]
            group["candidate_scores"] = candidate_scores
            group["proposed_label"] = candidate_scores[0]["label"]
            group["proposed_candidate_prob"] = candidate_scores[0]["probability"]
            group["proposed_candidate_margin"] = (
                candidate_scores[0]["probability"] - candidate_scores[1]["probability"]
                if len(candidate_scores) > 1
                else 1.0
            )


def load_crop_rows(conn: sqlite3.Connection, *, label_column: str) -> pd.DataFrame:
    if label_column not in {
        row[1] for row in conn.execute("PRAGMA table_info(images)").fetchall()
    }:
        raise ValueError(f"images表不存在配置的label列: {label_column!r}")
    return pd.read_sql_query(
        f"""
        SELECT
            c.id AS crop_id,
            c.image_id,
            c.detector_model,
            c.nms_iou_threshold,
            c.detector_score,
            c.box_x1,
            c.box_y1,
            c.box_x2,
            c.box_y2,
            c.noise_review_label,
            c.manual_corrected_label,
            i.sha1 AS source_sha1,
            i.downloaded_path,
            COALESCE(
                NULLIF(TRIM(c.manual_corrected_label), ''),
                NULLIF(TRIM(i.{label_column}), ''),
                NULLIF(TRIM(i.series), '')
            ) AS effective_label
        FROM crops c
        JOIN images i ON i.id = c.image_id
        WHERE i.sha1 IS NOT NULL
          AND TRIM(i.sha1) != ''
          AND i.downloaded_path IS NOT NULL
          AND TRIM(i.downloaded_path) != ''
          AND COALESCE(TRIM(c.noise_review_label), '') NOT IN (?, ?)
        ORDER BY i.sha1, c.detector_model, c.nms_iou_threshold, c.id
        """,
        conn,
        params=sorted(IGNORED_MANUAL_REVIEW_LABELS),
    )


def existing_manual_reviews(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT group_key, member_crop_ids_json, candidate_labels_json,
               review_status, resolved_label, exclusion_reason, review_note, reviewed_at
        FROM crop_duplicate_groups
        WHERE review_status IN ('confirmed', 'excluded')
        """
    ).fetchall()
    return {
        str(row[0]): {
            "member_crop_ids_json": str(row[1]),
            "candidate_labels_json": str(row[2]),
            "review_status": str(row[3]),
            "resolved_label": row[4],
            "exclusion_reason": row[5],
            "review_note": row[6],
            "reviewed_at": row[7],
        }
        for row in rows
    }


def preserve_matching_manual_reviews(
    groups: list[dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
) -> None:
    for group in groups:
        existing = reviews.get(group["group_key"])
        if existing is None:
            continue
        member_json = json.dumps(group["member_crop_ids"], ensure_ascii=False)
        candidate_json = json.dumps(group["candidate_labels"], ensure_ascii=False)
        if (
            existing["member_crop_ids_json"] != member_json
            or existing["candidate_labels_json"] != candidate_json
        ):
            continue
        group["review_status"] = existing["review_status"]
        group["resolved_label"] = existing["resolved_label"]
        group["exclusion_reason"] = existing["exclusion_reason"]
        group["review_note"] = existing["review_note"]
        group["reviewed_at"] = existing["reviewed_at"]


INSERT_GROUP_SQL = """
INSERT INTO crop_duplicate_groups (
    group_key, source_sha1, detector_model, nms_iou_threshold,
    representative_crop_id, member_crop_ids_json, candidate_labels_json,
    member_count, candidate_label_count, box_x1, box_y1, box_x2, box_y2,
    global_top1_label, global_top1_prob, global_top_k_json,
    proposed_label, proposed_candidate_prob, proposed_candidate_margin,
    candidate_scores_json, proposal_model, review_status, resolved_label,
    exclusion_reason, review_note, reviewed_at, detected_at, updated_at
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
)
"""


def replace_duplicate_groups(
    conn: sqlite3.Connection,
    groups: list[dict[str, Any]],
) -> None:
    conn.execute("DELETE FROM crop_duplicate_groups")
    rows = []
    for group in groups:
        rows.append(
            (
                group["group_key"],
                group["source_sha1"],
                group["detector_model"],
                group["nms_iou_threshold"],
                group["representative_crop_id"],
                json.dumps(group["member_crop_ids"], ensure_ascii=False),
                json.dumps(group["candidate_labels"], ensure_ascii=False),
                group["member_count"],
                group["candidate_label_count"],
                group["box_x1"],
                group["box_y1"],
                group["box_x2"],
                group["box_y2"],
                group.get("global_top1_label"),
                group.get("global_top1_prob"),
                json.dumps(group.get("global_top_k", []), ensure_ascii=False),
                group.get("proposed_label"),
                group.get("proposed_candidate_prob"),
                group.get("proposed_candidate_margin"),
                json.dumps(group.get("candidate_scores", []), ensure_ascii=False),
                group.get("proposal_model"),
                group["review_status"],
                group.get("resolved_label"),
                group.get("exclusion_reason"),
                group.get("review_note"),
                group.get("reviewed_at"),
            )
        )
    conn.executemany(INSERT_GROUP_SQL, rows)
    conn.commit()


def main(config: dict[str, Any] | None = None) -> int:
    config = config or utils.load_pipeline_config()
    utils.init_db(config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    duplicate_config = config["crop_duplicate_detection"]
    label_column = config["crops_storage"]["label_column"]

    with sqlite3.connect(db_path) as conn:
        crop_rows = load_crop_rows(conn, label_column=label_column)
        reviews = existing_manual_reviews(conn)

    groups = build_duplicate_groups(
        crop_rows,
        iou_threshold=float(duplicate_config["bbox_iou_threshold"]),
    )
    add_linear_head_proposals(groups, crop_rows=crop_rows, config=config)
    preserve_matching_manual_reviews(groups, reviews)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        replace_duplicate_groups(conn, groups)

    status_counts = pd.Series(
        [group["review_status"] for group in groups],
        dtype="string",
    ).value_counts().to_dict()
    duplicate_crops = sum(int(group["member_count"]) for group in groups)
    logger.info(
        "重复crop探测完成：groups=%d，member crops=%d，status=%s。"
        "跨标签pending组请运行 python tools/crop_duplicate_review_gradio.py 复核。",
        len(groups),
        duplicate_crops,
        status_counts,
    )
    return constants.STAGE_COMPLETED


if __name__ == "__main__":
    main()
