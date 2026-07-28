from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from pipeline import constants, utils


ARTIFACT_SCHEMA_VERSION = 1
PROBE_BUCKET_LIKELY_FALSE_KILL = "likely_false_kill"
PROBE_BUCKET_UNCERTAIN = "uncertain"
PROBE_BUCKET_LIKELY_TRUE_NOISE = "likely_true_noise"
PROBE_BUCKET_LABEL_NOT_IN_MODEL = "label_not_in_model"
PROBE_BUCKET_MISSING_FEATURE = "missing_feature"
PROBE_BUCKETS = [
    PROBE_BUCKET_LIKELY_FALSE_KILL,
    PROBE_BUCKET_UNCERTAIN,
    PROBE_BUCKET_LIKELY_TRUE_NOISE,
    PROBE_BUCKET_LABEL_NOT_IN_MODEL,
    PROBE_BUCKET_MISSING_FEATURE,
]


def label_expr_for_granularity(label_granularity: str) -> str:
    if label_granularity == "submodel":
        base_label_expr = "COALESCE(i.submodel, i.fine_grained_series, c.series)"
    elif label_granularity == "fine_grained_series":
        base_label_expr = "COALESCE(i.fine_grained_series, c.series)"
    elif label_granularity == "series":
        base_label_expr = "c.series"
    else:
        raise ValueError(
            "noise_detection.label_granularity 必须是以下值之一: "
            "series, fine_grained_series, submodel"
        )
    return f"COALESCE(NULLIF(c.manual_corrected_label, ''), {base_label_expr})"


def relative_to_data_root(path: Path, config: dict) -> str:
    data_root = utils.get_data_root(config).resolve()
    return str(path.resolve().relative_to(data_root))


def write_linear_head_artifact(
    *,
    config: dict,
    loss_round_dir: Path,
    checkpoint_path: Path,
    feature_cache_path: Path,
    label_map_path: Path,
    input_dim: int,
    num_classes: int,
) -> Path:
    loss_cfg = config["loss_noise_tracking"]
    artifact_path = loss_round_dir / loss_cfg["linear_head_artifact_file_name"]
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "loss_round": loss_round_dir.name,
        "checkpoint_path": relative_to_data_root(checkpoint_path, config),
        "feature_cache_path": relative_to_data_root(feature_cache_path, config),
        "label_map_path": label_map_path.name,
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "label_granularity": config["noise_detection"]["label_granularity"],
        "image_size": int(config["noise_detection"]["image_size"]),
    }
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact_path


def load_linear_head_artifact(
    *,
    config: dict,
    loss_round_dir: Path,
    checkpoint_path_override: str | Path | None = None,
) -> dict[str, Any]:
    artifact_name = config["loss_noise_tracking"]["linear_head_artifact_file_name"]
    artifact_path = loss_round_dir / artifact_name
    if artifact_path.is_file():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if int(artifact["schema_version"]) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported linear head artifact schema: {artifact['schema_version']}"
            )
    elif checkpoint_path_override is None:
        raise FileNotFoundError(
            f"Linear head artifact not found: {artifact_path}. "
            "请重跑 loss_tracking，或为 review 工具传入 --checkpoint-path。"
        )
    else:
        noise_cfg = config["noise_detection"]
        feature_cache_dir = utils.join_data_root(
            noise_cfg["feature_cache_dir"],
            config=config,
        )
        latest_pointer = feature_cache_dir / noise_cfg["latest_feature_cache_file"]
        feature_cache_file = (
            latest_pointer.read_text(encoding="utf-8").strip()
            if noise_cfg["active_feature_cache_file"] == "latest"
            else noise_cfg["active_feature_cache_file"]
        )
        artifact = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "loss_round": loss_round_dir.name,
            "checkpoint_path": str(checkpoint_path_override),
            "feature_cache_path": str(feature_cache_dir / feature_cache_file),
            "label_map_path": config["old_noise_recovery"]["label_map_file_name"],
            "input_dim": int(config["loss_noise_tracking"]["embedding_feature_dim"]),
            "num_classes": None,
            "label_granularity": noise_cfg["label_granularity"],
            "image_size": int(noise_cfg["image_size"]),
        }

    if checkpoint_path_override is not None:
        artifact["checkpoint_path"] = str(checkpoint_path_override)
    return artifact


def resolve_artifact_path(path_value: str | Path, *, config: dict, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    data_path = utils.join_data_root(path, config=config)
    if data_path.exists():
        return data_path
    project_path = utils.join_project_root(path)
    if project_path.exists():
        return project_path
    return base_dir / path


def resolve_active_lr_model(config: dict) -> str:
    model_dir = utils.join_data_root(config["path"]["model_dir"], config=config)
    pointer_path = (
        model_dir / config["logistic_regression_filter"]["model_pointer_path"]
    )
    if not pointer_path.is_file():
        raise FileNotFoundError(f"Latest LR model pointer not found: {pointer_path}")
    model_name = pointer_path.read_text(encoding="utf-8").strip()
    if not model_name:
        raise ValueError(f"Latest LR model pointer is empty: {pointer_path}")
    return model_name


def load_old_noise_candidates(
    *,
    db_path: Path,
    config: dict,
    active_lr_model: str,
) -> pd.DataFrame:
    recovery_cfg = config["old_noise_recovery"]
    labels = list(recovery_cfg["historical_noise_labels"])
    if not labels:
        raise ValueError("old_noise_recovery.historical_noise_labels 不能为空")
    placeholders = ", ".join("?" for _ in labels)
    label_expr = label_expr_for_granularity(
        config["noise_detection"]["label_granularity"]
    )
    sql = f"""
        SELECT
            c.id AS crop_id,
            c.image_id,
            c.series,
            c.power_type,
            c.crop_index,
            c.detector_label,
            c.detector_score,
            c.box_x1,
            c.box_y1,
            c.box_x2,
            c.box_y2,
            c.box_area,
            c.noise_review_label AS review_label,
            c.noise_review_note AS review_note,
            c.noise_reviewed_at AS reviewed_at,
            c.manual_corrected_label AS corrected_label,
            c.noise_predicted_label AS historical_noise_label,
            c.noise_predicted_prob AS historical_noise_prob,
            c.noise_prediction_model AS historical_noise_model,
            {label_expr} AS assigned_label,
            i.file_title,
            i.downloaded_path,
            i.category
        FROM crops c
        JOIN images i ON i.id = c.image_id
        WHERE c.noise_predicted_label IN ({placeholders})
          AND c.noise_predicted_prob >= ?
          AND COALESCE(TRIM(c.noise_prediction_model), '') != ?
          AND i.downloaded_path IS NOT NULL
        ORDER BY c.id
    """
    params = [
        *labels,
        float(recovery_cfg["historical_noise_min_prob"]),
        active_lr_model,
    ]
    with sqlite3.connect(db_path) as conn:
        candidates = pd.read_sql_query(sql, conn, params=params)
    candidates["crop_id"] = candidates["crop_id"].astype(int)
    return candidates


def load_label_map(label_map_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    label_to_id = {
        str(label): int(label_id)
        for label, label_id in payload["label_to_id"].items()
    }
    id_to_label = {
        int(label_id): str(label)
        for label_id, label in payload["id_to_label"].items()
    }
    return label_to_id, id_to_label


def _load_linear_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Linear head checkpoint is not a state dict: {checkpoint_path}")
    required = {"linear.weight", "linear.bias"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"Linear head checkpoint missing keys: {sorted(missing)}")
    return state


def score_old_noise_candidates(
    candidates: pd.DataFrame,
    *,
    feature_cache: dict,
    checkpoint_path: Path,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    batch_size: int,
    disagreement_min_confidence: float,
) -> pd.DataFrame:
    output = candidates.copy()
    if output.empty:
        return output.assign(
            probe_pred_id=pd.Series(dtype="Int64"),
            probe_pred_label=pd.Series(dtype="string"),
            probe_top1_prob=pd.Series(dtype=float),
            probe_assigned_prob=pd.Series(dtype=float),
            probe_best_other_prob=pd.Series(dtype=float),
            probe_assigned_margin=pd.Series(dtype=float),
            probe_bucket=pd.Series(dtype="string"),
        )

    features = feature_cache["features"].float()
    crop_ids = feature_cache["crop_ids"].long()
    feature_index = {
        int(crop_id): index for index, crop_id in enumerate(crop_ids.tolist())
    }
    candidate_indices = output["crop_id"].map(feature_index)
    output["probe_bucket"] = PROBE_BUCKET_MISSING_FEATURE
    output["probe_pred_id"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["probe_pred_label"] = pd.Series(pd.NA, index=output.index, dtype="string")
    for column in [
        "probe_top1_prob",
        "probe_assigned_prob",
        "probe_best_other_prob",
        "probe_assigned_margin",
    ]:
        output[column] = float("nan")

    available = candidate_indices.notna()
    if not available.any():
        return output

    state = _load_linear_state(checkpoint_path)
    weight = state["linear.weight"].float()
    bias = state["linear.bias"].float()
    if features.shape[1] != weight.shape[1]:
        raise ValueError(
            "Feature dimension does not match linear head: "
            f"features={features.shape[1]}, checkpoint={weight.shape[1]}"
        )
    if weight.shape[0] != len(id_to_label):
        raise ValueError(
            "Label map size does not match linear head: "
            f"labels={len(id_to_label)}, checkpoint={weight.shape[0]}"
        )

    available_rows = output.index[available].tolist()
    available_feature_indices = candidate_indices.loc[available].astype(int).tolist()
    all_probs = []
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("old_noise_recovery.batch_size 必须大于0")
    with torch.inference_mode():
        for start in range(0, len(available_feature_indices), batch_size):
            indices = available_feature_indices[start : start + batch_size]
            logits = features[indices] @ weight.T + bias
            all_probs.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(all_probs, dim=0)
    top1_prob, pred_ids = probs.max(dim=1)

    for probe_index, row_index in enumerate(available_rows):
        pred_id = int(pred_ids[probe_index])
        assigned_label = str(output.at[row_index, "assigned_label"])
        assigned_id = label_to_id.get(assigned_label)
        output.at[row_index, "probe_pred_id"] = pred_id
        output.at[row_index, "probe_pred_label"] = id_to_label[pred_id]
        output.at[row_index, "probe_top1_prob"] = float(top1_prob[probe_index])
        if assigned_id is None:
            output.at[row_index, "probe_bucket"] = PROBE_BUCKET_LABEL_NOT_IN_MODEL
            continue

        assigned_prob = float(probs[probe_index, assigned_id])
        other_probs = probs[probe_index].clone()
        other_probs[assigned_id] = -1.0
        best_other_prob = float(other_probs.max())
        assigned_margin = assigned_prob - best_other_prob
        output.at[row_index, "probe_assigned_prob"] = assigned_prob
        output.at[row_index, "probe_best_other_prob"] = best_other_prob
        output.at[row_index, "probe_assigned_margin"] = assigned_margin
        if (
            pred_id == assigned_id
            and float(top1_prob[probe_index]) >= float(disagreement_min_confidence)
        ):
            bucket = PROBE_BUCKET_LIKELY_FALSE_KILL
        elif float(top1_prob[probe_index]) >= float(disagreement_min_confidence):
            bucket = PROBE_BUCKET_LIKELY_TRUE_NOISE
        else:
            bucket = PROBE_BUCKET_UNCERTAIN
        output.at[row_index, "probe_bucket"] = bucket

    bucket_order = {bucket: index for index, bucket in enumerate(PROBE_BUCKETS)}
    output["_bucket_order"] = output["probe_bucket"].map(bucket_order)
    output = output.sort_values(
        ["_bucket_order", "probe_assigned_margin", "historical_noise_prob"],
        ascending=[True, False, False],
    ).drop(columns="_bucket_order")
    return output.reset_index(drop=True)


def build_label_rescue_stats(
    probe_rows: pd.DataFrame,
    dataset_metadata: pd.DataFrame,
) -> pd.DataFrame:
    required_probe_columns = {
        "assigned_label",
        "probe_bucket",
        "review_label",
        "corrected_label",
    }
    missing_probe_columns = required_probe_columns - set(probe_rows.columns)
    if missing_probe_columns:
        raise ValueError(
            f"Probe rows missing label rescue columns: {sorted(missing_probe_columns)}"
        )
    if "label" not in dataset_metadata.columns:
        raise ValueError("Dataset metadata missing label column")

    dataset_counts = (
        dataset_metadata["label"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values != ""]
        .value_counts()
        .rename("dataset_count")
    )
    probe = probe_rows.copy()
    probe["assigned_label"] = probe["assigned_label"].astype("string").str.strip()
    probe = probe[
        probe["assigned_label"].notna() & probe["assigned_label"].ne("")
    ].copy()
    review_label = probe["review_label"].fillna("").astype(str).str.strip()
    corrected = (
        probe["corrected_label"].notna()
        & probe["corrected_label"].astype(str).str.strip().ne("")
    )
    probe["unreviewed"] = review_label.eq("")
    probe["reviewed_recovered"] = review_label.eq(
        constants.NOISE_REVIEW_LABEL_OK
    ) | corrected

    grouped = probe.groupby("assigned_label", dropna=False)
    stats = grouped.agg(
        old_noise_candidates=("assigned_label", "size"),
        unreviewed_candidates=("unreviewed", "sum"),
        reviewed_recovered=("reviewed_recovered", "sum"),
    )
    for bucket in PROBE_BUCKETS:
        bucket_counts = (
            probe.loc[probe["probe_bucket"].eq(bucket)]
            .groupby("assigned_label")
            .size()
        )
        stats[bucket] = bucket_counts
        unreviewed_bucket_counts = (
            probe.loc[
                probe["probe_bucket"].eq(bucket) & probe["unreviewed"]
            ]
            .groupby("assigned_label")
            .size()
        )
        stats[f"{bucket}_unreviewed"] = unreviewed_bucket_counts

    all_labels = dataset_counts.index.union(stats.index)
    stats = stats.reindex(all_labels)
    stats.insert(0, "dataset_count", dataset_counts.reindex(all_labels))
    numeric_columns = list(stats.columns)
    stats[numeric_columns] = stats[numeric_columns].fillna(0).astype(int)
    stats["rescue_priority"] = (
        stats[f"{PROBE_BUCKET_LIKELY_FALSE_KILL}_unreviewed"] * 2
        + stats[f"{PROBE_BUCKET_UNCERTAIN}_unreviewed"]
    )
    stats["has_recovery_candidates"] = stats["old_noise_candidates"].gt(0)
    return (
        stats.reset_index(names="label")
        .sort_values(
            [
                "has_recovery_candidates",
                "dataset_count",
                "rescue_priority",
                "old_noise_candidates",
                "label",
            ],
            ascending=[False, True, False, False, True],
        )
        .reset_index(drop=True)
    )


def generate_recovery_probe(
    *,
    config: dict,
    checkpoint_path_override: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    recovery_cfg = config["old_noise_recovery"]
    loss_round_dir = utils.get_loss_round_dir(
        config=config,
        active_round=recovery_cfg["loss_round"],
    )
    artifact = load_linear_head_artifact(
        config=config,
        loss_round_dir=loss_round_dir,
        checkpoint_path_override=checkpoint_path_override,
    )
    checkpoint_path = resolve_artifact_path(
        artifact["checkpoint_path"],
        config=config,
        base_dir=loss_round_dir,
    )
    feature_cache_path = resolve_artifact_path(
        artifact["feature_cache_path"],
        config=config,
        base_dir=loss_round_dir,
    )
    label_map_path = resolve_artifact_path(
        artifact["label_map_path"],
        config=config,
        base_dir=loss_round_dir,
    )
    for name, path in [
        ("checkpoint", checkpoint_path),
        ("feature cache", feature_cache_path),
        ("label map", label_map_path),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Old-noise recovery {name} not found: {path}")

    active_lr_model = resolve_active_lr_model(config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    candidates = load_old_noise_candidates(
        db_path=db_path,
        config=config,
        active_lr_model=active_lr_model,
    )
    feature_cache = torch.load(feature_cache_path, map_location="cpu")
    label_to_id, id_to_label = load_label_map(label_map_path)
    scored = score_old_noise_candidates(
        candidates,
        feature_cache=feature_cache,
        checkpoint_path=checkpoint_path,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        batch_size=int(recovery_cfg["batch_size"]),
        disagreement_min_confidence=float(
            recovery_cfg["disagreement_min_confidence"]
        ),
    )
    scored.insert(0, "probe_round", loss_round_dir.name)
    scored.insert(1, "probe_checkpoint", str(checkpoint_path))
    scored.insert(2, "active_lr_model", active_lr_model)
    review_path = utils.join_data_root(
        recovery_cfg["review_file_path"],
        config=config,
    )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(review_path, index=False, encoding="utf-8")
    return scored, review_path


def save_manual_review(
    *,
    db_path: Path,
    crop_id: int,
    review_label: str,
    review_note: str | None,
    corrected_label: str | None,
    score_source: str,
) -> None:
    if review_label not in constants.NOISE_REVIEW_LABELS:
        raise ValueError(f"Unsupported noise review label: {review_label}")
    corrected_label = str(corrected_label or "").strip() or None
    if (
        review_label == constants.NOISE_REVIEW_LABEL_WRONG_LABEL
        and corrected_label is None
    ):
        raise ValueError("wrong_label review requires manual_corrected_label")
    if review_label != constants.NOISE_REVIEW_LABEL_WRONG_LABEL:
        corrected_label = None

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE crops
            SET noise_review_label = ?,
                noise_review_note = ?,
                noise_reviewed_at = CURRENT_TIMESTAMP,
                noise_review_score_col = ?,
                manual_corrected_label = ?,
                manual_corrected_at =
                    CASE WHEN ? IS NULL THEN NULL ELSE CURRENT_TIMESTAMP END
            WHERE id = ?
            """,
            (
                review_label,
                review_note or None,
                score_source,
                corrected_label,
                corrected_label,
                int(crop_id),
            ),
        )
        conn.commit()
