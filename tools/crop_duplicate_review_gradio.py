from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Project root not found from {start}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import constants, utils  # noqa: E402
from pipeline.stage_15_crop_duplicate_detection import (  # noqa: E402
    IGNORED_MANUAL_REVIEW_LABELS,
    REVIEW_STATUS_AUTO_RESOLVED,
    REVIEW_STATUS_CONFIRMED,
    REVIEW_STATUS_EXCLUDED,
    REVIEW_STATUS_PENDING,
    choose_representative,
)


CONFIG: dict[str, Any] = {}
DB_PATH: Path
REVIEW_SOURCE = "crop_duplicate_review"
GROUP_EXCLUSION_REASONS = {
    constants.NOISE_REVIEW_LABEL_BAD_CROP,
    constants.NOISE_REVIEW_LABEL_OUT_OF_LABEL_SPACE,
}


def placeholder_image(text: str) -> Image.Image:
    image = Image.new("RGB", (960, 640), "white")
    ImageDraw.Draw(image).text((40, 40), text, fill="black")
    return image


def known_label_choices() -> list[str]:
    label_column = CONFIG["crops_storage"]["label_column"]
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT label
            FROM (
                SELECT NULLIF(TRIM({label_column}), '') AS label
                FROM images
                UNION
                SELECT NULLIF(TRIM(manual_corrected_label), '') AS label
                FROM crops
            )
            WHERE label IS NOT NULL
            ORDER BY label
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def load_member_rows_by_id(crop_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not crop_ids:
        return {}
    label_column = CONFIG["crops_storage"]["label_column"]
    member_rows: dict[int, dict[str, Any]] = {}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for start in range(0, len(crop_ids), 900):
            batch = crop_ids[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"""
                SELECT
                    c.id AS crop_id,
                    c.image_id,
                    c.detector_score,
                    c.box_x1,
                    c.box_y1,
                    c.box_x2,
                    c.box_y2,
                    c.noise_review_label,
                    c.manual_corrected_label,
                    i.series,
                    i.{label_column} AS assigned_label,
                    i.category,
                    i.file_title,
                    i.downloaded_path,
                    COALESCE(
                        NULLIF(TRIM(c.manual_corrected_label), ''),
                        NULLIF(TRIM(i.{label_column}), ''),
                        NULLIF(TRIM(i.series), '')
                    ) AS effective_label
                FROM crops c
                JOIN images i ON i.id = c.image_id
                WHERE c.id IN ({placeholders})
                ORDER BY c.id
                """,
                batch,
            ).fetchall()
            member_rows.update(
                (int(row["crop_id"]), dict(row))
                for row in rows
            )
    return member_rows


def load_groups(status_filter: str) -> list[dict[str, Any]]:
    where = ""
    params: list[Any] = []
    if status_filter != "all":
        where = "WHERE review_status = ?"
        params.append(status_filter)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        groups = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT *
                FROM crop_duplicate_groups
                {where}
                ORDER BY
                    CASE review_status WHEN 'pending' THEN 0 ELSE 1 END,
                    proposed_candidate_margin ASC,
                    member_count DESC,
                    id
                """,
                params,
            ).fetchall()
        ]
    all_member_ids: list[int] = []
    for group in groups:
        group["member_crop_ids"] = [
            int(value) for value in json.loads(group["member_crop_ids_json"])
        ]
        all_member_ids.extend(group["member_crop_ids"])
        group["candidate_labels"] = [
            str(value) for value in json.loads(group["candidate_labels_json"])
        ]
        group["global_top_k"] = json.loads(group["global_top_k_json"])
        group["candidate_scores"] = json.loads(group["candidate_scores_json"])
    member_rows = load_member_rows_by_id(sorted(set(all_member_ids)))
    for group in groups:
        group["members"] = [
            member_rows[crop_id]
            for crop_id in group["member_crop_ids"]
            if crop_id in member_rows
        ]
    return groups


def group_summary() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT review_status, COUNT(*), SUM(member_count)
            FROM crop_duplicate_groups
            GROUP BY review_status
            ORDER BY review_status
            """
        ).fetchall()
    if not rows:
        return "尚无重复组；请先运行 `python pipeline_entry.py --only crop_duplicate_detection`。"
    lines = ["### 重复组状态"]
    for status, groups, members in rows:
        lines.append(f"- `{status}`: {groups} groups / {members} member crops")
    return "\n".join(lines)


def load_representative_image(group: dict[str, Any]) -> Image.Image:
    representative_id = int(group["representative_crop_id"])
    member = next(
        (
            row
            for row in group["members"]
            if int(row["crop_id"]) == representative_id
        ),
        group["members"][0] if group["members"] else None,
    )
    if member is None:
        return placeholder_image("Duplicate group has no surviving crop members.")
    return utils.load_crop(
        member,
        config=CONFIG,
        pad_frac=float(CONFIG["crops_storage"]["crop_pad_frac"]),
    )


def format_probability(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def group_markdown(group: dict[str, Any], index: int, total: int) -> str:
    candidate_scores = ", ".join(
        f"{row['label']}={float(row['probability']):.3f}"
        for row in group["candidate_scores"]
    )
    global_top_k = ", ".join(
        f"{row['label']}={float(row['probability']):.3f}"
        for row in group["global_top_k"]
    )
    return "\n".join(
        [
            f"### 重复组 {index + 1}/{total}",
            f"- **group_id**: {group['id']}",
            f"- **SHA1**: `{group['source_sha1']}`",
            f"- **members**: {group['member_count']}",
            f"- **candidate labels**: {', '.join(group['candidate_labels'])}",
            f"- **proposal**: {group.get('proposed_label') or ''}",
            (
                "- **candidate probability / margin**: "
                f"{format_probability(group.get('proposed_candidate_prob'))} / "
                f"{format_probability(group.get('proposed_candidate_margin'))}"
            ),
            f"- **candidate ranking**: {candidate_scores}",
            f"- **global top-k**: {global_top_k}",
            f"- **proposal model**: {group.get('proposal_model') or ''}",
            f"- **status**: {group['review_status']}",
            f"- **exclusion reason**: {group.get('exclusion_reason') or ''}",
        ]
    )


MEMBER_COLUMNS = [
    "crop_id",
    "image_id",
    "effective_label",
    "assigned_label",
    "series",
    "category",
    "detector_score",
    "noise_review_label",
    "manual_corrected_label",
]


def member_table(group: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(group["members"])
    if frame.empty:
        return pd.DataFrame(columns=MEMBER_COLUMNS)
    return frame[[column for column in MEMBER_COLUMNS if column in frame.columns]]


def display_group(records: list[dict[str, Any]], index: int):
    total = len(records or [])
    if total == 0:
        return (
            placeholder_image("No duplicate groups loaded."),
            "没有符合过滤条件的重复组。",
            pd.DataFrame(columns=MEMBER_COLUMNS),
            None,
            "",
            gr.update(choices=[], value=None),
            "0/0",
        )
    index = max(0, min(int(index), total - 1))
    group = records[index]
    selected_label = group.get("resolved_label") or group.get("proposed_label")
    member_ids = [int(row["crop_id"]) for row in group["members"]]
    representative_id = int(group["representative_crop_id"])
    selected_member_id = (
        representative_id
        if representative_id in member_ids
        else (member_ids[0] if member_ids else None)
    )
    return (
        load_representative_image(group),
        group_markdown(group, index, total),
        member_table(group),
        selected_label,
        group.get("review_note") or "",
        gr.update(choices=member_ids, value=selected_member_id),
        f"{index + 1}/{total}",
    )


def load_review_batch(status_filter: str):
    records = load_groups(status_filter)
    display = display_group(records, 0)
    return records, 0, group_summary(), *display


def save_resolution(
    *,
    group_id: int,
    status: str,
    resolved_label: str | None,
    note: str,
    exclusion_reason: str | None = None,
) -> None:
    resolved_label = str(resolved_label or "").strip() or None
    exclusion_reason = str(exclusion_reason or "").strip() or None
    if status == REVIEW_STATUS_CONFIRMED:
        if resolved_label is None:
            raise gr.Error("确认重复组时必须选择最终 label。")
        if resolved_label not in set(known_label_choices()):
            raise gr.Error(f"最终 label 不在当前标签空间中: {resolved_label}")
        exclusion_reason = None
    elif status == REVIEW_STATUS_EXCLUDED:
        resolved_label = None
        if exclusion_reason not in GROUP_EXCLUSION_REASONS:
            raise gr.Error(
                "整组排除必须选择 bad_crop 或 out_of_label_space。"
            )
    else:
        raise ValueError(f"不支持的人工状态: {status}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        group_row = conn.execute(
            """
            SELECT member_crop_ids_json
            FROM crop_duplicate_groups
            WHERE id = ?
            """,
            (int(group_id),),
        ).fetchone()
        if group_row is None:
            raise gr.Error(f"找不到duplicate group id={group_id}")
        member_ids = [int(value) for value in json.loads(str(group_row[0]))]
        updated = conn.execute(
            """
            UPDATE crop_duplicate_groups
            SET review_status = ?,
                resolved_label = ?,
                exclusion_reason = ?,
                review_note = ?,
                reviewed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                status,
                resolved_label,
                exclusion_reason,
                note or None,
                int(group_id),
            ),
        ).rowcount
        if updated != 1:
            raise gr.Error(f"找不到duplicate group id={group_id}")
        if status == REVIEW_STATUS_EXCLUDED:
            mark_crops_excluded(
                conn,
                crop_ids=member_ids,
                exclusion_reason=exclusion_reason,
                note=note,
            )
        conn.commit()


def mark_crops_excluded(
    conn: sqlite3.Connection,
    *,
    crop_ids: list[int],
    exclusion_reason: str,
    note: str,
) -> None:
    if exclusion_reason not in GROUP_EXCLUSION_REASONS:
        raise ValueError(f"不支持的crop排除原因: {exclusion_reason!r}")
    if not crop_ids:
        return
    placeholders = ",".join("?" for _ in crop_ids)
    conn.execute(
        f"""
        UPDATE crops
        SET noise_review_label = ?,
            noise_review_note = ?,
            noise_reviewed_at = CURRENT_TIMESTAMP,
            noise_review_score_col = ?,
            manual_corrected_label = NULL,
            manual_corrected_at = NULL
        WHERE id IN ({placeholders})
        """,
        (
            exclusion_reason,
            note or None,
            REVIEW_SOURCE,
            *[int(crop_id) for crop_id in crop_ids],
        ),
    )


def reconcile_group_after_member_exclusion(
    conn: sqlite3.Connection,
    *,
    group_id: int,
    label_column: str,
) -> None:
    image_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(images)").fetchall()
    }
    if label_column not in image_columns:
        raise ValueError(f"images表不存在配置的label列: {label_column!r}")
    group_row = conn.execute(
        """
        SELECT member_crop_ids_json
        FROM crop_duplicate_groups
        WHERE id = ?
        """,
        (int(group_id),),
    ).fetchone()
    if group_row is None:
        raise gr.Error(f"找不到duplicate group id={group_id}")
    member_ids = [int(value) for value in json.loads(str(group_row[0]))]
    placeholders = ",".join("?" for _ in member_ids)
    rows = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT
                c.id AS crop_id,
                c.detector_score,
                c.noise_review_label,
                c.manual_corrected_label,
                c.box_x1,
                c.box_y1,
                c.box_x2,
                c.box_y2,
                COALESCE(
                    NULLIF(TRIM(c.manual_corrected_label), ''),
                    NULLIF(TRIM(i.{label_column}), ''),
                    NULLIF(TRIM(i.series), '')
                ) AS effective_label
            FROM crops c
            JOIN images i ON i.id = c.image_id
            WHERE c.id IN ({placeholders})
              AND COALESCE(TRIM(c.noise_review_label), '') NOT IN (?, ?)
            ORDER BY c.id
            """,
            (
                *member_ids,
                *sorted(IGNORED_MANUAL_REVIEW_LABELS),
            ),
        ).fetchall()
    ]
    if len(rows) < 2:
        conn.execute(
            "DELETE FROM crop_duplicate_groups WHERE id = ?",
            (int(group_id),),
        )
        return

    candidate_labels = sorted(
        {
            str(row["effective_label"]).strip()
            for row in rows
            if str(row["effective_label"] or "").strip()
        }
    )
    if not candidate_labels:
        raise ValueError(
            f"重复crop组没有可用label: crop_ids={[row['crop_id'] for row in rows]}"
        )
    representative = choose_representative(rows)
    anchor = rows[0]
    auto_resolved = len(candidate_labels) == 1
    conn.execute(
        """
        UPDATE crop_duplicate_groups
        SET representative_crop_id = ?,
            member_crop_ids_json = ?,
            candidate_labels_json = ?,
            member_count = ?,
            candidate_label_count = ?,
            box_x1 = ?,
            box_y1 = ?,
            box_x2 = ?,
            box_y2 = ?,
            proposed_label = NULL,
            proposed_candidate_prob = NULL,
            proposed_candidate_margin = NULL,
            candidate_scores_json = '[]',
            review_status = ?,
            resolved_label = ?,
            exclusion_reason = NULL,
            review_note = NULL,
            reviewed_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            int(representative["crop_id"]),
            json.dumps([int(row["crop_id"]) for row in rows], ensure_ascii=False),
            json.dumps(candidate_labels, ensure_ascii=False),
            len(rows),
            len(candidate_labels),
            float(anchor["box_x1"]),
            float(anchor["box_y1"]),
            float(anchor["box_x2"]),
            float(anchor["box_y2"]),
            REVIEW_STATUS_AUTO_RESOLVED if auto_resolved else REVIEW_STATUS_PENDING,
            candidate_labels[0] if auto_resolved else None,
            int(group_id),
        ),
    )


def exclude_member(
    *,
    group_id: int,
    crop_id: int,
    exclusion_reason: str,
    note: str,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        group_row = conn.execute(
            """
            SELECT member_crop_ids_json
            FROM crop_duplicate_groups
            WHERE id = ?
            """,
            (int(group_id),),
        ).fetchone()
        if group_row is None:
            raise gr.Error(f"找不到duplicate group id={group_id}")
        member_ids = {
            int(value) for value in json.loads(str(group_row["member_crop_ids_json"]))
        }
        if int(crop_id) not in member_ids:
            raise gr.Error(f"crop_id={crop_id} 不属于 duplicate group id={group_id}")
        mark_crops_excluded(
            conn,
            crop_ids=[int(crop_id)],
            exclusion_reason=exclusion_reason,
            note=note,
        )
        reconcile_group_after_member_exclusion(
            conn,
            group_id=int(group_id),
            label_column=CONFIG["crops_storage"]["label_column"],
        )
        conn.commit()


def save_and_advance(
    records: list[dict[str, Any]],
    index: int,
    selected_label: str | None,
    note: str,
    action: str,
):
    if not records:
        raise gr.Error("请先加载重复组。")
    index = max(0, min(int(index), len(records) - 1))
    group = records[index]
    if action == "proposal":
        selected_label = group.get("proposed_label")
        if not selected_label:
            raise gr.Error("当前组没有线性头建议，请手工选择 label。")
        status = REVIEW_STATUS_CONFIRMED
    elif action == "selected":
        status = REVIEW_STATUS_CONFIRMED
    else:
        raise ValueError(f"未知操作: {action}")

    save_resolution(
        group_id=int(group["id"]),
        status=status,
        resolved_label=selected_label,
        note=note,
    )
    group["review_status"] = status
    group["resolved_label"] = selected_label if status == REVIEW_STATUS_CONFIRMED else None
    group["exclusion_reason"] = None
    group["review_note"] = note or None
    next_index = min(index + 1, len(records) - 1)
    return records, next_index, group_summary(), *display_group(records, next_index)


def exclude_and_reload(
    records: list[dict[str, Any]],
    index: int,
    status_filter: str,
    member_crop_id: int | None,
    note: str,
    exclusion_reason: str,
    *,
    whole_group: bool,
):
    if not records:
        raise gr.Error("请先加载重复组。")
    index = max(0, min(int(index), len(records) - 1))
    group = records[index]
    if whole_group:
        save_resolution(
            group_id=int(group["id"]),
            status=REVIEW_STATUS_EXCLUDED,
            resolved_label=None,
            exclusion_reason=exclusion_reason,
            note=note,
        )
    else:
        if member_crop_id is None:
            raise gr.Error("请先选择要排除的 member crop。")
        exclude_member(
            group_id=int(group["id"]),
            crop_id=int(member_crop_id),
            exclusion_reason=exclusion_reason,
            note=note,
        )

    refreshed = load_groups(status_filter)
    next_index = min(index, max(0, len(refreshed) - 1))
    return refreshed, next_index, group_summary(), *display_group(
        refreshed,
        next_index,
    )


def move(records: list[dict[str, Any]], index: int, delta: int):
    if not records:
        return index, *display_group(records, index)
    next_index = max(0, min(int(index) + int(delta), len(records) - 1))
    return next_index, *display_group(records, next_index)


def build_app() -> gr.Blocks:
    label_choices = known_label_choices()
    with gr.Blocks(title="Crop Duplicate Review") as app:
        gr.Markdown(
            "# Crop Duplicate Review\n\n"
            "同一 SHA1 内高 IoU crop 的线性头建议仅用于排序；"
            "人工确认后 Stage 16 才会保留一个代表 crop。"
        )
        records_state = gr.State([])
        index_state = gr.State(0)

        with gr.Row():
            status_filter = gr.Dropdown(
                choices=["pending", "confirmed", "excluded", "auto_resolved", "all"],
                value="pending",
                label="Review status",
            )
            load_btn = gr.Button("加载 / 刷新", variant="primary")
        summary = gr.Markdown(group_summary())

        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(type="pil", label="Canonical crop")
                members = gr.Dataframe(
                    headers=MEMBER_COLUMNS,
                    interactive=False,
                    label="Duplicate members and label provenance",
                )
            with gr.Column(scale=2):
                metadata = gr.Markdown()
                selected_label = gr.Dropdown(
                    choices=label_choices,
                    allow_custom_value=False,
                    label="Resolved label",
                )
                note = gr.Textbox(label="Review note", lines=3)
                member_crop_id = gr.Dropdown(
                    choices=[],
                    label="Member crop_id to exclude",
                )
                progress = gr.Markdown("0/0")
                with gr.Row():
                    accept_btn = gr.Button("接受线性头建议并下一条", variant="primary")
                    selected_btn = gr.Button("保存所选标签并下一条")
                gr.Markdown(
                    "下面的排除操作会直接写入 `crops.noise_review_label`；"
                    "单个成员排除后会立即重算当前重复组。"
                )
                with gr.Row():
                    member_bad_crop_btn = gr.Button("当前 crop → Bad crop")
                    member_out_of_scope_btn = gr.Button(
                        "当前 crop → Out of label space"
                    )
                with gr.Row():
                    group_bad_crop_btn = gr.Button(
                        "整组 → Bad crop",
                        variant="stop",
                    )
                    group_out_of_scope_btn = gr.Button(
                        "整组 → Out of label space",
                        variant="stop",
                    )
                with gr.Row():
                    previous_btn = gr.Button("Previous")
                    next_btn = gr.Button("Skip / Next")

        load_btn.click(
            load_review_batch,
            inputs=[status_filter],
            outputs=[
                records_state,
                index_state,
                summary,
                image,
                metadata,
                members,
                selected_label,
                note,
                member_crop_id,
                progress,
            ],
        )
        accept_btn.click(
            lambda records, index, label, review_note: save_and_advance(
                records, index, label, review_note, "proposal"
            ),
            inputs=[records_state, index_state, selected_label, note],
            outputs=[
                records_state,
                index_state,
                summary,
                image,
                metadata,
                members,
                selected_label,
                note,
                member_crop_id,
                progress,
            ],
        )
        selected_btn.click(
            lambda records, index, label, review_note: save_and_advance(
                records, index, label, review_note, "selected"
            ),
            inputs=[records_state, index_state, selected_label, note],
            outputs=[
                records_state,
                index_state,
                summary,
                image,
                metadata,
                members,
                selected_label,
                note,
                member_crop_id,
                progress,
            ],
        )
        member_bad_crop_btn.click(
            lambda records, index, current_filter, crop_id, review_note: (
                exclude_and_reload(
                    records,
                    index,
                    current_filter,
                    crop_id,
                    review_note,
                    constants.NOISE_REVIEW_LABEL_BAD_CROP,
                    whole_group=False,
                )
            ),
            inputs=[
                records_state,
                index_state,
                status_filter,
                member_crop_id,
                note,
            ],
            outputs=[
                records_state,
                index_state,
                summary,
                image,
                metadata,
                members,
                selected_label,
                note,
                member_crop_id,
                progress,
            ],
        )
        member_out_of_scope_btn.click(
            lambda records, index, current_filter, crop_id, review_note: (
                exclude_and_reload(
                    records,
                    index,
                    current_filter,
                    crop_id,
                    review_note,
                    constants.NOISE_REVIEW_LABEL_OUT_OF_LABEL_SPACE,
                    whole_group=False,
                )
            ),
            inputs=[
                records_state,
                index_state,
                status_filter,
                member_crop_id,
                note,
            ],
            outputs=[
                records_state,
                index_state,
                summary,
                image,
                metadata,
                members,
                selected_label,
                note,
                member_crop_id,
                progress,
            ],
        )
        group_bad_crop_btn.click(
            lambda records, index, current_filter, crop_id, review_note: (
                exclude_and_reload(
                    records,
                    index,
                    current_filter,
                    crop_id,
                    review_note,
                    constants.NOISE_REVIEW_LABEL_BAD_CROP,
                    whole_group=True,
                )
            ),
            inputs=[
                records_state,
                index_state,
                status_filter,
                member_crop_id,
                note,
            ],
            outputs=[
                records_state,
                index_state,
                summary,
                image,
                metadata,
                members,
                selected_label,
                note,
                member_crop_id,
                progress,
            ],
        )
        group_out_of_scope_btn.click(
            lambda records, index, current_filter, crop_id, review_note: (
                exclude_and_reload(
                    records,
                    index,
                    current_filter,
                    crop_id,
                    review_note,
                    constants.NOISE_REVIEW_LABEL_OUT_OF_LABEL_SPACE,
                    whole_group=True,
                )
            ),
            inputs=[
                records_state,
                index_state,
                status_filter,
                member_crop_id,
                note,
            ],
            outputs=[
                records_state,
                index_state,
                summary,
                image,
                metadata,
                members,
                selected_label,
                note,
                member_crop_id,
                progress,
            ],
        )
        previous_btn.click(
            lambda records, index: move(records, index, -1),
            inputs=[records_state, index_state],
            outputs=[
                index_state,
                image,
                metadata,
                members,
                selected_label,
                note,
                member_crop_id,
                progress,
            ],
        )
        next_btn.click(
            lambda records, index: move(records, index, 1),
            inputs=[records_state, index_state],
            outputs=[
                index_state,
                image,
                metadata,
                members,
                selected_label,
                note,
                member_crop_id,
                progress,
            ],
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动重复 crop 人工解析 Gradio UI。")
    parser.add_argument("--config", type=str, default=None, help="pipeline_config.yaml 路径。")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    global CONFIG, DB_PATH

    args = parse_args()
    CONFIG = utils.load_pipeline_config(args.config)
    utils.init_db(config=CONFIG)
    DB_PATH = utils.join_data_root(CONFIG["path"]["db_path"], config=CONFIG)
    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
