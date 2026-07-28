from __future__ import annotations

import argparse
import copy
import math
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

from pipeline import constants, old_noise_recovery, utils  # noqa: E402


CONFIG: dict[str, Any] = {}
DB_PATH: Path
REVIEW_PATH: Path
CLI_DEFAULTS: dict[str, Any] = {}


def known_label_choices() -> list[str]:
    label_expr = old_noise_recovery.label_expr_for_granularity(
        CONFIG["noise_detection"]["label_granularity"]
    )
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT {label_expr} AS label
            FROM crops c
            JOIN images i ON i.id = c.image_id
            WHERE {label_expr} IS NOT NULL
              AND TRIM({label_expr}) != ''
            ORDER BY label
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def current_review_overlay() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT
                id AS crop_id,
                noise_review_label AS current_review_label,
                noise_review_note AS current_review_note,
                noise_reviewed_at AS current_reviewed_at,
                manual_corrected_label AS current_corrected_label
            FROM crops
            """,
            conn,
        )


def load_probe_rows() -> pd.DataFrame:
    if not REVIEW_PATH.is_file():
        raise FileNotFoundError(
            f"Probe CSV not found: {REVIEW_PATH}. 请先点击 Generate / refresh probe。"
        )
    probe = pd.read_csv(REVIEW_PATH)
    required = {
        "crop_id",
        "probe_bucket",
        "assigned_label",
        "probe_pred_label",
        "probe_assigned_margin",
    }
    missing = required - set(probe.columns)
    if missing:
        raise ValueError(f"Probe CSV missing columns: {sorted(missing)}")
    probe["crop_id"] = probe["crop_id"].astype(int)
    stale_review_columns = [
        "review_label",
        "review_note",
        "reviewed_at",
        "corrected_label",
    ]
    probe = probe.drop(
        columns=[column for column in stale_review_columns if column in probe.columns]
    )
    overlay = current_review_overlay()
    rows = probe.merge(overlay, on="crop_id", how="left")
    return rows.rename(
        columns={
            "current_review_label": "review_label",
            "current_review_note": "review_note",
            "current_reviewed_at": "reviewed_at",
            "current_corrected_label": "corrected_label",
        }
    )


def available_loss_rounds() -> list[str]:
    root = utils.join_data_root(
        CONFIG["path"]["loss_analysis_data_dir"],
        config=CONFIG,
    )
    label_map_name = CONFIG["old_noise_recovery"]["label_map_file_name"]
    rounds = sorted(
        (
            path.name
            for path in root.iterdir()
            if path.is_dir() and (path / label_map_name).is_file()
        ),
        reverse=True,
    )
    return ["latest", *rounds]


def available_checkpoints() -> list[str]:
    model_dir = utils.join_data_root(CONFIG["path"]["model_dir"], config=CONFIG)
    prefix = CONFIG["loss_noise_tracking"]["model_checkpoint_prefix"]
    data_root = utils.get_data_root(CONFIG)
    checkpoints = sorted(
        model_dir.glob(f"{prefix}_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return ["auto", *(str(path.relative_to(data_root)) for path in checkpoints)]


def dataset_metadata() -> pd.DataFrame:
    path = (
        utils.join_data_root(CONFIG["path"]["dataset_dir"], config=CONFIG)
        / CONFIG["crops_storage"]["metadata_file_name"]
    )
    if not path.is_file():
        return pd.DataFrame({"label": pd.Series(dtype="string")})
    return pd.read_csv(path, usecols=["label"])


def label_rescue_stats(rows: pd.DataFrame) -> pd.DataFrame:
    return old_noise_recovery.build_label_rescue_stats(rows, dataset_metadata())


def label_filter_choices(stats: pd.DataFrame, max_dataset_count: int) -> list:
    rare = stats[
        stats["dataset_count"].le(int(max_dataset_count))
        & stats["old_noise_candidates"].gt(0)
    ]
    choices: list = [("All labels", "all")]
    for row in rare.itertuples(index=False):
        display = (
            f"{row.label} | dataset={row.dataset_count} | "
            f"false-kill={row.likely_false_kill_unreviewed} | "
            f"uncertain={row.uncertain_unreviewed}"
        )
        choices.append((display, str(row.label)))
    return choices


def label_stats_preview(stats: pd.DataFrame, max_dataset_count: int) -> pd.DataFrame:
    columns = [
        "label",
        "dataset_count",
        "old_noise_candidates",
        "unreviewed_candidates",
        "likely_false_kill_unreviewed",
        "uncertain_unreviewed",
        "likely_true_noise_unreviewed",
        "reviewed_recovered",
        "rescue_priority",
    ]
    return stats[
        stats["dataset_count"].le(int(max_dataset_count))
        & stats["old_noise_candidates"].gt(0)
    ][columns].reset_index(drop=True)


def generate_probe(
    *,
    loss_round: str,
    checkpoint_path: str | None,
    historical_noise_min_prob: float,
    disagreement_min_confidence: float,
    batch_size: int,
) -> tuple[pd.DataFrame, str]:
    probe_config = copy.deepcopy(CONFIG)
    probe_config["old_noise_recovery"]["loss_round"] = str(loss_round).strip()
    probe_config["old_noise_recovery"]["historical_noise_min_prob"] = float(
        historical_noise_min_prob
    )
    probe_config["old_noise_recovery"]["disagreement_min_confidence"] = float(
        disagreement_min_confidence
    )
    probe_config["old_noise_recovery"]["batch_size"] = int(batch_size)
    checkpoint_override = str(checkpoint_path or "").strip()
    if checkpoint_override == "auto":
        checkpoint_override = ""
    rows, path = old_noise_recovery.generate_recovery_probe(
        config=probe_config,
        checkpoint_path_override=checkpoint_override or None,
    )
    counts = rows["probe_bucket"].value_counts(dropna=False).to_dict()
    return (
        rows,
        f"Generated {len(rows)} candidates from round `{loss_round}` at `{path}`"
        f"\n\nBucket counts: `{counts}`",
    )


def sample_rows(
    rows: pd.DataFrame,
    *,
    bucket: str,
    historical_model: str,
    assigned_label: str,
    skip_reviewed: bool,
    sampling_mode: str,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    filtered = rows.copy()
    if bucket != "all":
        filtered = filtered[filtered["probe_bucket"].eq(bucket)]
    if historical_model != "all":
        filtered = filtered[filtered["historical_noise_model"].eq(historical_model)]
    if assigned_label != "all":
        filtered = filtered[filtered["assigned_label"].eq(assigned_label)]
    if skip_reviewed:
        review = filtered["review_label"].fillna("").astype(str).str.strip()
        filtered = filtered[review.eq("")]

    sample_size = min(max(1, int(sample_size)), len(filtered))
    if filtered.empty:
        return filtered
    if sampling_mode == "random":
        sampled = filtered.sample(n=sample_size, random_state=int(seed))
    elif sampling_mode == "label_balanced":
        labels = max(1, filtered["assigned_label"].nunique(dropna=False))
        per_label = max(1, math.ceil(sample_size / labels))
        parts = []
        for _, group in filtered.groupby("assigned_label", dropna=False, sort=False):
            parts.append(
                group.sample(
                    n=min(per_label, len(group)),
                    random_state=int(seed),
                )
            )
        sampled = pd.concat(parts, ignore_index=True).head(sample_size)
    elif sampling_mode == "priority":
        sampled = filtered.sort_values(
            ["probe_assigned_margin", "probe_top1_prob"],
            ascending=[False, False],
            na_position="last",
        ).head(sample_size)
    else:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
    return sampled.reset_index(drop=True)


def placeholder_image(message: str, size: tuple[int, int] = (640, 480)) -> Image.Image:
    image = Image.new("RGB", size, "#f2f2f2")
    ImageDraw.Draw(image).multiline_text((20, 20), message, fill="#333333")
    return image


def load_crop_image(row: dict[str, Any]) -> Image.Image:
    try:
        return utils.load_crop(
            row,
            config=CONFIG,
            pad_frac=float(CONFIG["noise_detection"]["crop_pad_frac"]),
        )
    except Exception as exc:
        return placeholder_image(
            f"Failed to load crop_id={row.get('crop_id')}\n{exc}"
        )


def _format_float(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return f"{float(value):.4f}"


def row_markdown(row: dict[str, Any], index: int, total: int) -> str:
    fields = [
        ("progress", f"{index + 1}/{total}"),
        ("bucket", row.get("probe_bucket")),
        ("crop_id", row.get("crop_id")),
        ("assigned label", row.get("assigned_label")),
        ("probe prediction", row.get("probe_pred_label")),
        ("assigned probability", _format_float(row.get("probe_assigned_prob"))),
        ("top-1 probability", _format_float(row.get("probe_top1_prob"))),
        ("assigned margin", _format_float(row.get("probe_assigned_margin"))),
        ("historical LR probability", _format_float(row.get("historical_noise_prob"))),
        ("historical LR model", row.get("historical_noise_model")),
        ("series", row.get("series")),
        ("power type", row.get("power_type")),
        ("detector score", _format_float(row.get("detector_score"))),
        ("file", row.get("file_title")),
        ("category", row.get("category")),
        ("existing review", row.get("review_label")),
        ("corrected label", row.get("corrected_label")),
    ]
    lines = ["### Old-noise recovery review"]
    for key, value in fields:
        if value is not None and not pd.isna(value):
            lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def display_record(records: list[dict[str, Any]], index: int):
    total = len(records or [])
    if total == 0:
        return (
            placeholder_image("No candidates loaded."),
            "No candidates loaded.",
            None,
            "",
            None,
            "0/0",
        )
    index = max(0, int(index))
    if index >= total:
        message = f"Review batch complete: {total}/{total}"
        return (
            placeholder_image(message),
            f"### Review complete\n\n{message}",
            None,
            "",
            None,
            message,
        )
    row = records[index]
    return (
        load_crop_image(row),
        row_markdown(row, index, total),
        row.get("review_label") or None,
        row.get("review_note") or "",
        row.get("corrected_label") or None,
        f"{index + 1}/{total}",
    )


def validate_corrected_label(
    review_label: str,
    corrected_label: str | None,
) -> str | None:
    corrected = str(corrected_label or "").strip() or None
    if (
        review_label == constants.NOISE_REVIEW_LABEL_WRONG_LABEL
        and corrected is None
    ):
        raise gr.Error("标记为 wrong_label 时必须选择 Correct label。")
    if review_label != constants.NOISE_REVIEW_LABEL_WRONG_LABEL:
        return None
    if corrected not in set(known_label_choices()):
        raise gr.Error(f"Correct label 不在当前标签空间中: {corrected}")
    return corrected


def build_review_app() -> gr.Blocks:
    initial_rows = load_probe_rows() if REVIEW_PATH.is_file() else pd.DataFrame()
    bucket_choices = ["all", *old_noise_recovery.PROBE_BUCKETS]
    model_choices = ["all"]
    initial_stats = (
        label_rescue_stats(initial_rows)
        if not initial_rows.empty
        else pd.DataFrame()
    )
    rare_label_max_default = int(CLI_DEFAULTS["rare_label_max_samples"])
    assigned_label_choices = (
        label_filter_choices(initial_stats, rare_label_max_default)
        if not initial_stats.empty
        else [("All labels", "all")]
    )
    initial_stats_preview = (
        label_stats_preview(initial_stats, rare_label_max_default)
        if not initial_stats.empty
        else pd.DataFrame()
    )
    if not initial_rows.empty:
        model_choices.extend(
            sorted(initial_rows["historical_noise_model"].dropna().astype(str).unique())
        )
    corrected_label_choices = known_label_choices()
    loss_round_choices = available_loss_rounds()
    checkpoint_choices = available_checkpoints()
    loss_round_default = str(CLI_DEFAULTS["loss_round"])
    checkpoint_default = str(CLI_DEFAULTS["checkpoint_path"] or "auto")

    with gr.Blocks(title="Old Noise Recovery Review") as app:
        gr.Markdown(
            "# Old Noise Recovery Review\n\n"
            "最新干净线性头只用于排序历史 LR 噪声；只有人工 review 会写数据库。"
        )
        records_state = gr.State([])
        index_state = gr.State(0)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Probe parameters", open=True):
                    loss_round = gr.Dropdown(
                        choices=loss_round_choices,
                        value=loss_round_default,
                        label="Loss round",
                        allow_custom_value=True,
                    )
                    checkpoint_path = gr.Dropdown(
                        choices=checkpoint_choices,
                        value=checkpoint_default,
                        label="Linear-head checkpoint",
                        allow_custom_value=True,
                        info=(
                            "auto reads linear_head_artifact.json; "
                            "legacy rounds require an explicit checkpoint."
                        ),
                    )
                    historical_noise_min_prob = gr.Slider(
                        0.0,
                        1.0,
                        value=float(CLI_DEFAULTS["historical_noise_min_prob"]),
                        step=0.01,
                        label="Historical LR minimum probability",
                    )
                    disagreement_min_confidence = gr.Slider(
                        0.0,
                        1.0,
                        value=float(CLI_DEFAULTS["disagreement_min_confidence"]),
                        step=0.01,
                        label="Probe confidence threshold",
                    )
                    probe_batch_size = gr.Number(
                        value=int(CLI_DEFAULTS["batch_size"]),
                        precision=0,
                        label="Probe batch size",
                    )
                    generate_btn = gr.Button(
                        "Generate / refresh probe",
                        variant="primary",
                    )
                generation_status = gr.Markdown(
                    f"Probe CSV: `{REVIEW_PATH}`"
                    if REVIEW_PATH.is_file()
                    else "Probe CSV has not been generated."
                )
                bucket = gr.Dropdown(
                    choices=bucket_choices,
                    value="likely_false_kill",
                    label="Probe bucket",
                )
                historical_model = gr.Dropdown(
                    choices=model_choices,
                    value="all",
                    label="Historical LR model",
                )
                assigned_label = gr.Dropdown(
                    choices=assigned_label_choices,
                    value="all",
                    label="Target label rescue",
                )
                rare_label_max_samples = gr.Slider(
                    0,
                    1000,
                    value=rare_label_max_default,
                    step=1,
                    label="Max current dataset samples",
                )
                refresh_label_stats_btn = gr.Button("Refresh label rescue list")
                skip_reviewed = gr.Checkbox(value=True, label="Skip reviewed")
                sampling_mode = gr.Radio(
                    choices=["priority", "label_balanced", "random"],
                    value="priority",
                    label="Sampling",
                )
                sample_size = gr.Slider(1, 1000, value=100, step=1, label="Batch size")
                seed = gr.Number(value=42, precision=0, label="Random seed")
                load_btn = gr.Button("Load review batch", variant="primary")
                progress = gr.Textbox(label="Progress", interactive=False)

            with gr.Column(scale=2):
                image = gr.Image(type="pil", height=560, label="Crop")
                metadata = gr.Markdown()

            with gr.Column(scale=1):
                review_label = gr.Radio(
                    choices=constants.NOISE_REVIEW_LABELS,
                    label="Manual review label",
                )
                with gr.Row():
                    ok_btn = gr.Button("OK & next", variant="primary")
                    wrong_pred_btn = gr.Button("Wrong = probe pred")
                with gr.Row():
                    bad_crop_btn = gr.Button("Bad crop", variant="stop")
                    out_space_btn = gr.Button("Out of label space")
                ambiguous_btn = gr.Button("Ambiguous")
                corrected_label = gr.Dropdown(
                    choices=corrected_label_choices,
                    label="Correct label",
                )
                note = gr.Textbox(label="Note", lines=4)
                save_btn = gr.Button("Save & next", variant="primary")
                with gr.Row():
                    previous_btn = gr.Button("Previous")
                    skip_btn = gr.Button("Skip")

        label_stats_table = gr.Dataframe(
            value=initial_stats_preview,
            label="Rare-label rescue overview",
            interactive=False,
            wrap=True,
        )
        preview = gr.Dataframe(label="Current review batch", interactive=False, wrap=True)

        def on_generate(
            loss_round_value,
            checkpoint_path_value,
            historical_noise_min_prob_value,
            disagreement_min_confidence_value,
            probe_batch_size_value,
            rare_label_max_samples_value,
        ):
            _, message = generate_probe(
                loss_round=str(loss_round_value).strip() or "latest",
                checkpoint_path=checkpoint_path_value,
                historical_noise_min_prob=float(historical_noise_min_prob_value),
                disagreement_min_confidence=float(
                    disagreement_min_confidence_value
                ),
                batch_size=int(probe_batch_size_value),
            )
            rows = load_probe_rows()
            models = ["all", *sorted(rows["historical_noise_model"].dropna().astype(str).unique())]
            stats = label_rescue_stats(rows)
            labels = label_filter_choices(stats, int(rare_label_max_samples_value))
            return (
                message,
                gr.update(choices=models, value="all"),
                gr.update(choices=labels, value="all"),
                label_stats_preview(stats, int(rare_label_max_samples_value)),
            )

        def on_refresh_label_stats(max_dataset_samples):
            rows = load_probe_rows()
            stats = label_rescue_stats(rows)
            choices = label_filter_choices(stats, int(max_dataset_samples))
            return (
                gr.update(choices=choices, value="all"),
                label_stats_preview(stats, int(max_dataset_samples)),
            )

        def on_load(
            bucket_value,
            model_value,
            label_value,
            skip_reviewed_value,
            sampling_mode_value,
            sample_size_value,
            seed_value,
        ):
            rows = load_probe_rows()
            sampled = sample_rows(
                rows,
                bucket=bucket_value,
                historical_model=model_value,
                assigned_label=label_value,
                skip_reviewed=bool(skip_reviewed_value),
                sampling_mode=sampling_mode_value,
                sample_size=int(sample_size_value),
                seed=int(seed_value),
            )
            records = sampled.to_dict(orient="records")
            shown = display_record(records, 0)
            preview_columns = [
                column
                for column in [
                    "crop_id",
                    "probe_bucket",
                    "assigned_label",
                    "probe_pred_label",
                    "probe_assigned_prob",
                    "probe_top1_prob",
                    "probe_assigned_margin",
                    "historical_noise_prob",
                    "historical_noise_model",
                    "review_label",
                ]
                if column in sampled.columns
            ]
            return records, 0, *shown, sampled[preview_columns]

        def save_and_advance(
            records,
            index,
            selected_review_label,
            note_value,
            corrected_label_value,
        ):
            records = records or []
            index = int(index)
            if not records or index >= len(records):
                raise gr.Error("当前没有可保存的 candidate。")
            if not selected_review_label:
                raise gr.Error("请选择 review label。")
            corrected = validate_corrected_label(
                selected_review_label,
                corrected_label_value,
            )
            row = records[index]
            old_noise_recovery.save_manual_review(
                db_path=DB_PATH,
                crop_id=int(row["crop_id"]),
                review_label=selected_review_label,
                review_note=note_value,
                corrected_label=corrected,
                score_source=(
                    f"old_noise_recovery:{row['probe_round']}:"
                    f"{row['probe_bucket']}"
                ),
            )
            row["review_label"] = selected_review_label
            row["review_note"] = note_value
            row["corrected_label"] = corrected
            index += 1
            return records, index, *display_record(records, index)

        def quick_save(
            records,
            index,
            note_value,
            corrected_label_value,
            selected_review_label,
        ):
            return save_and_advance(
                records,
                index,
                selected_review_label,
                note_value,
                corrected_label_value,
            )

        def wrong_as_probe(records, index, note_value):
            records = records or []
            index = int(index)
            if not records or index >= len(records):
                raise gr.Error("当前没有 candidate。")
            probe_label = records[index].get("probe_pred_label")
            if not probe_label or pd.isna(probe_label):
                raise gr.Error("当前 candidate 没有 probe prediction。")
            return save_and_advance(
                records,
                index,
                constants.NOISE_REVIEW_LABEL_WRONG_LABEL,
                note_value,
                str(probe_label),
            )

        def navigate(records, index, delta):
            records = records or []
            index = max(0, min(int(index) + int(delta), len(records)))
            return index, *display_record(records, index)

        generate_btn.click(
            on_generate,
            inputs=[
                loss_round,
                checkpoint_path,
                historical_noise_min_prob,
                disagreement_min_confidence,
                probe_batch_size,
                rare_label_max_samples,
            ],
            outputs=[
                generation_status,
                historical_model,
                assigned_label,
                label_stats_table,
            ],
        )
        refresh_label_stats_btn.click(
            on_refresh_label_stats,
            inputs=[rare_label_max_samples],
            outputs=[assigned_label, label_stats_table],
        )
        load_btn.click(
            on_load,
            inputs=[
                bucket,
                historical_model,
                assigned_label,
                skip_reviewed,
                sampling_mode,
                sample_size,
                seed,
            ],
            outputs=[
                records_state,
                index_state,
                image,
                metadata,
                review_label,
                note,
                corrected_label,
                progress,
                preview,
            ],
        )
        save_btn.click(
            save_and_advance,
            inputs=[records_state, index_state, review_label, note, corrected_label],
            outputs=[
                records_state,
                index_state,
                image,
                metadata,
                review_label,
                note,
                corrected_label,
                progress,
            ],
        )
        ok_btn.click(
            lambda records, index, note_value, corrected: quick_save(
                records,
                index,
                note_value,
                corrected,
                constants.NOISE_REVIEW_LABEL_OK,
            ),
            inputs=[records_state, index_state, note, corrected_label],
            outputs=[
                records_state,
                index_state,
                image,
                metadata,
                review_label,
                note,
                corrected_label,
                progress,
            ],
        )
        wrong_pred_btn.click(
            wrong_as_probe,
            inputs=[records_state, index_state, note],
            outputs=[
                records_state,
                index_state,
                image,
                metadata,
                review_label,
                note,
                corrected_label,
                progress,
            ],
        )
        bad_crop_btn.click(
            lambda records, index, note_value, corrected: quick_save(
                records,
                index,
                note_value,
                corrected,
                constants.NOISE_REVIEW_LABEL_BAD_CROP,
            ),
            inputs=[records_state, index_state, note, corrected_label],
            outputs=[
                records_state,
                index_state,
                image,
                metadata,
                review_label,
                note,
                corrected_label,
                progress,
            ],
        )
        out_space_btn.click(
            lambda records, index, note_value, corrected: quick_save(
                records,
                index,
                note_value,
                corrected,
                constants.NOISE_REVIEW_LABEL_OUT_OF_LABEL_SPACE,
            ),
            inputs=[records_state, index_state, note, corrected_label],
            outputs=[
                records_state,
                index_state,
                image,
                metadata,
                review_label,
                note,
                corrected_label,
                progress,
            ],
        )
        ambiguous_btn.click(
            lambda records, index, note_value, corrected: quick_save(
                records,
                index,
                note_value,
                corrected,
                constants.NOISE_REVIEW_LABEL_AMBIGUOUS,
            ),
            inputs=[records_state, index_state, note, corrected_label],
            outputs=[
                records_state,
                index_state,
                image,
                metadata,
                review_label,
                note,
                corrected_label,
                progress,
            ],
        )
        previous_btn.click(
            lambda records, index: navigate(records, index, -1),
            inputs=[records_state, index_state],
            outputs=[
                index_state,
                image,
                metadata,
                review_label,
                note,
                corrected_label,
                progress,
            ],
        )
        skip_btn.click(
            lambda records, index: navigate(records, index, 1),
            inputs=[records_state, index_state],
            outputs=[
                index_state,
                image,
                metadata,
                review_label,
                note,
                corrected_label,
                progress,
            ],
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review historical LR noise with the latest clean linear head."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "Initial GUI checkpoint value. GUI selection overrides it."
        ),
    )
    parser.add_argument(
        "--loss-round",
        type=str,
        default=None,
        help="Initial GUI loss round value. GUI selection overrides it.",
    )
    parser.add_argument(
        "--historical-noise-min-prob",
        type=float,
        default=None,
        help="Initial historical LR probability threshold.",
    )
    parser.add_argument(
        "--probe-confidence",
        type=float,
        default=None,
        help="Initial clean-head confidence threshold.",
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=None,
        help="Initial probe inference batch size.",
    )
    parser.add_argument(
        "--rare-label-max-samples",
        type=int,
        default=50,
        help="Initial maximum dataset count shown in label rescue.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the probe CSV before launching.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    global CONFIG, DB_PATH, REVIEW_PATH, CLI_DEFAULTS

    args = parse_args()
    CONFIG = utils.load_pipeline_config(args.config)
    utils.init_db(config=CONFIG)
    DB_PATH = utils.join_data_root(CONFIG["path"]["db_path"], config=CONFIG)
    REVIEW_PATH = utils.join_data_root(
        CONFIG["old_noise_recovery"]["review_file_path"],
        config=CONFIG,
    )
    recovery_cfg = CONFIG["old_noise_recovery"]
    CLI_DEFAULTS = {
        "loss_round": args.loss_round or recovery_cfg["loss_round"],
        "checkpoint_path": args.checkpoint_path,
        "historical_noise_min_prob": (
            args.historical_noise_min_prob
            if args.historical_noise_min_prob is not None
            else recovery_cfg["historical_noise_min_prob"]
        ),
        "disagreement_min_confidence": (
            args.probe_confidence
            if args.probe_confidence is not None
            else recovery_cfg["disagreement_min_confidence"]
        ),
        "batch_size": (
            args.probe_batch_size
            if args.probe_batch_size is not None
            else recovery_cfg["batch_size"]
        ),
        "rare_label_max_samples": args.rare_label_max_samples,
    }
    if args.regenerate:
        generate_probe(
            loss_round=CLI_DEFAULTS["loss_round"],
            checkpoint_path=CLI_DEFAULTS["checkpoint_path"],
            historical_noise_min_prob=CLI_DEFAULTS[
                "historical_noise_min_prob"
            ],
            disagreement_min_confidence=CLI_DEFAULTS[
                "disagreement_min_confidence"
            ],
            batch_size=CLI_DEFAULTS["batch_size"],
        )
    app = build_review_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
