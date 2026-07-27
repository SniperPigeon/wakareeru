import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import constants
import utils

logger = utils.get_logger("stage_14_store_crops")

CROP_SELECTION_MODES = {"filtered", "all"}
NOISE_PREDICTION_SCOPES = {"active_model", "all_stored"}
DUPLICATE_RESOLVED_STATUSES = {"auto_resolved", "confirmed"}
DUPLICATE_REVIEW_STATUSES = {
    "pending",
    *DUPLICATE_RESOLVED_STATUSES,
    "excluded",
}


def resolve_crop_selection_mode(crops_storage_config: dict) -> str:
    """Validate and normalize the configured dataset crop selection mode."""
    selection_mode = str(crops_storage_config["selection_mode"]).strip().lower()
    if selection_mode not in CROP_SELECTION_MODES:
        raise ValueError(
            "crops_storage.selection_mode 必须是以下值之一: "
            f"{sorted(CROP_SELECTION_MODES)}"
        )
    return selection_mode


def resolve_noise_prediction_scope(crops_storage_config: dict) -> str:
    scope = str(crops_storage_config["noise_prediction_scope"]).strip().lower()
    if scope not in NOISE_PREDICTION_SCOPES:
        raise ValueError(
            "crops_storage.noise_prediction_scope 必须是以下值之一: "
            f"{sorted(NOISE_PREDICTION_SCOPES)}"
        )
    return scope


def load_crop_duplicate_groups(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT
                id,
                group_key,
                representative_crop_id,
                member_crop_ids_json,
                candidate_labels_json,
                member_count,
                candidate_label_count,
                review_status,
                resolved_label,
                proposal_model,
                detected_at,
                reviewed_at
            FROM crop_duplicate_groups
            ORDER BY id
            """,
            conn,
        )


def summarize_crop_duplicate_review(groups: pd.DataFrame) -> dict:
    if groups.empty:
        return {
            "group_count": 0,
            "member_crop_count": 0,
            "pending_count": 0,
            "auto_resolved_count": 0,
            "confirmed_count": 0,
            "excluded_count": 0,
            "review_complete": True,
            "tool_command": "python tools/crop_duplicate_review_gradio.py",
        }
    statuses = groups["review_status"].fillna("").astype(str)
    unknown = sorted(set(statuses) - DUPLICATE_REVIEW_STATUSES)
    if unknown:
        raise ValueError(f"crop_duplicate_groups包含未知review_status: {unknown}")
    summary = {
        "group_count": int(len(groups)),
        "member_crop_count": int(groups["member_count"].sum()),
        "pending_count": int(statuses.eq("pending").sum()),
        "auto_resolved_count": int(statuses.eq("auto_resolved").sum()),
        "confirmed_count": int(statuses.eq("confirmed").sum()),
        "excluded_count": int(statuses.eq("excluded").sum()),
        "tool_command": "python tools/crop_duplicate_review_gradio.py",
    }
    summary["review_complete"] = summary["pending_count"] == 0
    return summary


def apply_crop_duplicate_resolutions(
    metadata: pd.DataFrame,
    *,
    duplicate_groups: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Collapse reviewed duplicate groups after normal crop filters are applied."""
    if "crop_id" not in metadata.columns:
        raise ValueError("重复crop去重要求metadata包含crop_id")
    if label_column not in metadata.columns:
        raise ValueError(f"重复crop去重要求metadata包含label列: {label_column!r}")

    metadata = metadata.copy()
    if duplicate_groups.empty or metadata.empty:
        return (
            metadata.reset_index(drop=True),
            pd.Series(False, index=range(len(metadata)), dtype=bool),
            {
                "removed_count": 0,
                "excluded_count": 0,
                "label_override_count": 0,
                "resolved_group_count": 0,
            },
        )

    crop_ids = metadata["crop_id"].astype(int)
    crop_id_set = set(crop_ids)
    seen_member_ids: set[int] = set()
    remove_ids: set[int] = set()
    label_overrides: dict[int, str] = {}
    excluded_count = 0
    resolved_group_count = 0

    for group in duplicate_groups.to_dict(orient="records"):
        member_ids = {
            int(value)
            for value in json.loads(str(group["member_crop_ids_json"]))
        }
        overlap = seen_member_ids & member_ids
        if overlap:
            raise ValueError(
                "crop_duplicate_groups成员重复出现在多个组: "
                f"{sorted(overlap)[:10]}"
            )
        seen_member_ids.update(member_ids)
        surviving_ids = sorted(member_ids & crop_id_set)
        if not surviving_ids:
            continue

        status = str(group["review_status"])
        if status == "pending":
            continue
        if status == "excluded":
            remove_ids.update(surviving_ids)
            excluded_count += len(surviving_ids)
            resolved_group_count += 1
            continue
        if status not in DUPLICATE_RESOLVED_STATUSES:
            raise ValueError(f"未知重复crop review_status: {status!r}")

        resolved_label = str(group.get("resolved_label") or "").strip()
        if not resolved_label:
            raise ValueError(
                f"已解析重复组缺少resolved_label: group_id={group['id']}"
            )
        group_rows = metadata[crop_ids.isin(surviving_ids)]
        matching_ids = sorted(
            group_rows.loc[
                group_rows[label_column].astype(str).str.strip().eq(resolved_label),
                "crop_id",
            ]
            .astype(int)
            .tolist()
        )
        preferred_id = int(group["representative_crop_id"])
        if preferred_id in matching_ids:
            kept_id = preferred_id
        elif matching_ids:
            kept_id = matching_ids[0]
        elif preferred_id in surviving_ids:
            kept_id = preferred_id
        else:
            kept_id = surviving_ids[0]

        remove_ids.update(set(surviving_ids) - {kept_id})
        current_label = str(
            metadata.loc[crop_ids.eq(kept_id), label_column].iloc[0]
        ).strip()
        if current_label != resolved_label:
            label_overrides[kept_id] = resolved_label
        resolved_group_count += 1

    before_count = len(metadata)
    metadata = metadata.loc[~crop_ids.isin(remove_ids)].copy()
    changed_mask = pd.Series(False, index=metadata.index, dtype=bool)
    for crop_id, resolved_label in label_overrides.items():
        target = metadata["crop_id"].astype(int).eq(crop_id)
        metadata.loc[target, label_column] = resolved_label
        changed_mask.loc[target] = True

    result = metadata.reset_index(drop=True)
    changed_mask = changed_mask.reset_index(drop=True)
    return (
        result,
        changed_mask,
        {
            "removed_count": before_count - len(result),
            "excluded_count": excluded_count,
            "label_override_count": len(label_overrides),
            "resolved_group_count": resolved_group_count,
        },
    )


def label_to_ascii(label: str, fallback: str = "label") -> str:
    """Convert a Japanese train label into a deterministic ASCII slug.

    This is intentionally rule-based rather than LLM-based so saved dataset
    paths remain stable across runs.
    """
    text = unicodedata.normalize("NFKC", str(label)).strip().lower()
    for src, dst in constants.LABEL_ASCII_REPLACEMENTS:
        text = text.replace(src.lower(), dst)
    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    text = re.sub(r"\b(kiha|kumoha|kuha|moha|saha|deha|roha|saro|moro|kani)_(\d)", r"\1\2", text)
    return text or fallback


def save_crop_image(
    *,
    source_image_path: str | Path,
    output_path: str | Path,
    box_x1: float,
    box_y1: float,
    box_x2: float,
    box_y2: float,
    pad_frac: float = 0.04,
    image_format: str | None = None,
    jpeg_quality: int = 95,
) -> Path:
    """Crop one explicit bbox from an image and save it to disk."""
    source_image_path = Path(source_image_path)
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".jpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = utils.load_img_with_orientation(source_image_path)
    box_width = float(box_x2) - float(box_x1)
    box_height = float(box_y2) - float(box_y1)
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid crop box: ({box_x1}, {box_y1}, {box_x2}, {box_y2})")

    pad = max(box_width, box_height) * float(pad_frac)
    left = max(0, int(float(box_x1) - pad))
    top = max(0, int(float(box_y1) - pad))
    right = min(img.width, int(float(box_x2) + pad))
    bottom = min(img.height, int(float(box_y2) + pad))
    if right <= left or bottom <= top:
        raise ValueError(
            "Padded crop box is outside image bounds: "
            f"({left}, {top}, {right}, {bottom}) for {source_image_path}"
        )

    crop = img.crop((left, top, right, bottom))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    suffix = output_path.suffix.lower()
    save_format = image_format
    if save_format is None:
        save_format = "JPEG" if suffix in {".jpg", ".jpeg"} else suffix.lstrip(".").upper()

    save_kwargs = {}
    if save_format.upper() in {"JPEG", "JPG"}:
        save_format = "JPEG"
        save_kwargs["quality"] = int(jpeg_quality)
    crop.save(output_path, format=save_format, **save_kwargs)
    return output_path


def validate_config_column_names(columns: list[str]) -> None:
    unsafe = [col for col in columns if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", col)]
    if unsafe:
        raise ValueError(f"Unsafe configured column names: {unsafe}")


def build_en_label(
    metadata: pd.DataFrame,
    *,
    label_column: str,
) -> pd.DataFrame:
    if label_column not in metadata.columns:
        raise ValueError(f"image metadata中不存在label列: {label_column!r}")

    metadata = metadata.copy()
    metadata["label"] = metadata[label_column]
    metadata["label_en"] = metadata["label"].map(lambda label: label_to_ascii(label, fallback="unknown"))
    return metadata


def invalidate_metadata_for_manual_corrections(
    metadata: pd.DataFrame,
    *,
    corrected_mask: pd.Series,
    columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    validate_config_column_names(columns)
    if label_column in columns:
        raise ValueError(
            "crops_storage.manual_correction_invalidate_metadata_columns "
            f"不能包含当前label列: {label_column!r}"
        )
    missing_columns = [col for col in columns if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"人工纠正后要清空的metadata列不存在: {missing_columns}")

    metadata = metadata.copy()
    if columns and corrected_mask.any():
        metadata.loc[corrected_mask, columns] = pd.NA
    return metadata


def _clean_metadata_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def refill_unique_metadata_for_manual_corrections(
    metadata: pd.DataFrame,
    *,
    corrected_mask: pd.Series,
    label_column: str,
    operator_columns: list[str],
    submodel_bandai_columns: list[str],
) -> pd.DataFrame:
    validate_config_column_names(operator_columns)
    validate_config_column_names(submodel_bandai_columns)
    refill_columns = [*operator_columns, *submodel_bandai_columns]
    if label_column in refill_columns:
        raise ValueError("人工纠正metadata补齐列不能包含当前label列。")
    missing_columns = [col for col in [label_column, *refill_columns] if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"人工纠正metadata补齐所需列不存在: {missing_columns}")
    if len(submodel_bandai_columns) != 2:
        raise ValueError("crops_storage.manual_correction_refill_submodel_bandai_columns 必须包含两个列名。")

    metadata = metadata.copy()
    reference = metadata.loc[~corrected_mask].copy()
    if reference.empty or not corrected_mask.any():
        return metadata

    labels = metadata.loc[corrected_mask, label_column].dropna().astype(str).str.strip().unique()
    for label in labels:
        if not label:
            continue
        same_label = reference[reference[label_column].astype(str).str.strip() == label]
        if same_label.empty:
            continue
        target = corrected_mask & metadata[label_column].astype(str).str.strip().eq(label)

        for col in operator_columns:
            values = sorted({_clean_metadata_value(value) for value in same_label[col] if _clean_metadata_value(value)})
            if len(values) == 1:
                metadata.loc[target, col] = values[0]

        pair_cols = submodel_bandai_columns
        pairs = {
            tuple(_clean_metadata_value(row[col]) for col in pair_cols)
            for _, row in same_label[pair_cols].iterrows()
        }
        pairs = {pair for pair in pairs if any(pair)}
        if len(pairs) == 1:
            pair = next(iter(pairs))
            for col, value in zip(pair_cols, pair, strict=True):
                metadata.loc[target, col] = value or pd.NA

    return metadata


def build_label_tables(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = (
        metadata[["label", "label_en"]]
        .dropna(subset=["label"])
        .drop_duplicates()
        .sort_values(["label_en", "label"], kind="stable")
        .reset_index(drop=True)
    )
    labels.insert(0, "label_id", range(len(labels)))

    counts = metadata["label"].value_counts(dropna=False).rename("count").reset_index()
    counts.columns = ["label", "count"]
    labels = labels.merge(counts, on="label", how="left")

    metadata = metadata.merge(labels[["label", "label_id"]], on="label", how="left")
    metadata["label_id"] = metadata["label_id"].astype("Int64")
    return metadata, labels


def _parse_l10n_json_array(value: object, *, label_ja: str, column: str) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"label_metadata {label_ja!r} 的 {column} 不是合法JSON") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"label_metadata {label_ja!r} 的 {column} 必须是字符串数组")
    if any(not item.strip() for item in parsed):
        raise ValueError(f"label_metadata {label_ja!r} 的 {column} 包含空字符串")
    return [item.strip() for item in parsed]


def _validate_canonical_l10n_text(value: str, *, label_ja: str, field: str) -> None:
    if "http://" in value or "https://" in value or re.search(r"\]\([^)]*\)", value):
        raise ValueError(f"label_metadata {label_ja!r} 的 {field} 包含链接污染")


def build_l10n_metadata_from_db(labels: pd.DataFrame, db_path: Path) -> list[dict]:
    required_label_columns = {"label_id", "label"}
    missing_label_columns = required_label_columns - set(labels.columns)
    if missing_label_columns:
        raise ValueError(f"生成l10n metadata缺少labels列: {sorted(missing_label_columns)}")

    with sqlite3.connect(db_path) as conn:
        canonical = pd.read_sql_query(
            """
            SELECT label_ja, label_en, label_zh,
                   operator_ja_json, operator_en_json, operator_zh_json,
                   wiki_title_ja
            FROM label_metadata
            """,
            conn,
        )
    canonical_by_label = canonical.set_index("label_ja", drop=False).to_dict(orient="index")
    missing_labels = sorted(set(labels["label"]) - set(canonical_by_label))
    if missing_labels:
        raise ValueError(
            "label_metadata缺少当前数据集标签，拒绝从images旧字段或既有JSON回填: "
            f"{missing_labels}"
        )

    result = []
    for row in labels.itertuples(index=False):
        source = canonical_by_label[row.label]
        operators = {
            language: _parse_l10n_json_array(
                source[f"operator_{language}_json"],
                label_ja=row.label,
                column=f"operator_{language}_json",
            )
            for language in ("ja", "en", "zh")
        }
        if len({len(values) for values in operators.values()}) != 1:
            raise ValueError(f"label_metadata {row.label!r} 的三语operator数组长度不一致")
        if len(operators["ja"]) != len(set(operators["ja"])):
            raise ValueError(f"label_metadata {row.label!r} 包含重复operator_ja")

        scalar_fields = {
            "label_en": str(source["label_en"]).strip(),
            "label_zh": str(source["label_zh"]).strip(),
            "wiki_title_ja": str(source["wiki_title_ja"]).strip(),
        }
        if not scalar_fields["label_en"] or not scalar_fields["label_zh"]:
            raise ValueError(f"label_metadata {row.label!r} 的英中label翻译不能为空")
        for field, value in scalar_fields.items():
            _validate_canonical_l10n_text(value, label_ja=row.label, field=field)
        for language, values in operators.items():
            for value in values:
                _validate_canonical_l10n_text(
                    value,
                    label_ja=row.label,
                    field=f"operator_{language}_json",
                )
        if re.search(r"[ぁ-んァ-ヶ一-龠々]", scalar_fields["label_en"]):
            raise ValueError(f"label_metadata {row.label!r} 的label_en包含日文污染")
        if re.search(r"[ぁ-んァ-ヶ]", scalar_fields["label_zh"]):
            raise ValueError(f"label_metadata {row.label!r} 的label_zh包含日文假名污染")
        if any("/" in value for value in operators["ja"]):
            raise ValueError(f"label_metadata {row.label!r} 的operator_ja包含双语言分隔符")
        if any(re.search(r"[ぁ-んァ-ヶ一-龠々]", value) for value in operators["en"]):
            raise ValueError(f"label_metadata {row.label!r} 的operator_en包含日文污染")
        if any(re.search(r"[ぁ-んァ-ヶ]", value) for value in operators["zh"]):
            raise ValueError(f"label_metadata {row.label!r} 的operator_zh包含日文假名污染")

        result.append(
            {
                "id": int(row.label_id),
                "label": {
                    "ja": row.label,
                    "en": scalar_fields["label_en"],
                    "zh": scalar_fields["label_zh"],
                },
                "operator": operators,
                "wiki_title_ja": scalar_fields["wiki_title_ja"],
            }
        )
    return result


def write_l10n_metadata(labels: pd.DataFrame, db_path: Path, output_path: Path) -> int:
    l10n_metadata = build_l10n_metadata_from_db(labels, db_path)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(l10n_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return len(l10n_metadata)


def flush_crop_save_updates(db_path: Path, updates: list[tuple[str, int]]) -> None:
    if not updates:
        return
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "UPDATE crops SET saved = 1, crop_path = ? WHERE id = ?",
            updates,
        )
        conn.commit()


def summarize_old_noise_recovery_review(
    *,
    config: dict,
    db_path: Path,
) -> dict:
    review_path = utils.join_data_root(
        config["old_noise_recovery"]["review_file_path"],
        config=config,
    )
    summary = {
        "workflow": "manual_auxiliary_tool",
        "pipeline_stage": False,
        "tool_command": "python tools/old_noise_recovery_review_gradio.py",
        "review_file": str(
            config["old_noise_recovery"]["review_file_path"]
        ),
        "resolved_review_file": str(review_path),
    }
    if not review_path.is_file():
        return {
            **summary,
            "status": "missing",
            "candidate_count": 0,
            "reviewed_count": 0,
            "unreviewed_count": 0,
        }

    probe = pd.read_csv(review_path)
    if "crop_id" not in probe.columns:
        raise ValueError(f"旧噪声恢复review CSV缺少crop_id列: {review_path}")
    probe["crop_id"] = probe["crop_id"].astype(int)
    probe = probe.drop_duplicates("crop_id", keep="last")
    with sqlite3.connect(db_path) as conn:
        reviews = pd.read_sql_query(
            """
            SELECT
                id AS crop_id,
                noise_review_label,
                manual_corrected_label
            FROM crops
            """,
            conn,
        )
    probe = probe.merge(reviews, on="crop_id", how="left")
    reviewed = (
        probe["noise_review_label"].fillna("").astype(str).str.strip().ne("")
        | probe["manual_corrected_label"].fillna("").astype(str).str.strip().ne("")
    )
    manual_ok = (
        probe["noise_review_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(constants.NOISE_REVIEW_LABEL_OK)
    )
    corrected = (
        probe["manual_corrected_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    probe_rounds = (
        sorted(probe["probe_round"].dropna().astype(str).unique())
        if "probe_round" in probe.columns
        else []
    )
    active_lr_models = (
        sorted(probe["active_lr_model"].dropna().astype(str).unique())
        if "active_lr_model" in probe.columns
        else []
    )
    bucket_counts = (
        {
            str(bucket): int(count)
            for bucket, count in probe["probe_bucket"].value_counts().items()
        }
        if "probe_bucket" in probe.columns
        else {}
    )
    return {
        **summary,
        "status": "available",
        "candidate_count": int(len(probe)),
        "reviewed_count": int(reviewed.sum()),
        "unreviewed_count": int((~reviewed).sum()),
        "manual_ok_count": int(manual_ok.sum()),
        "manual_corrected_count": int(corrected.sum()),
        "probe_rounds": probe_rounds,
        "active_lr_models": active_lr_models,
        "bucket_counts": bucket_counts,
    }


def log_old_noise_recovery_preflight(summary: dict) -> None:
    if summary["status"] == "missing":
        logger.warning(
            "旧噪声恢复review是导出前人工辅助流程，不是pipeline stage；"
            "当前未找到probe CSV：%s。需要复核时先运行：%s",
            summary["resolved_review_file"],
            summary["tool_command"],
        )
        return
    log = logger.warning if summary["unreviewed_count"] else logger.info
    log(
        "旧噪声恢复review（非pipeline stage）：候选%d，已复核%d，未复核%d，"
        "人工ok=%d，人工纠正=%d。工具：%s",
        summary["candidate_count"],
        summary["reviewed_count"],
        summary["unreviewed_count"],
        summary["manual_ok_count"],
        summary["manual_corrected_count"],
        summary["tool_command"],
    )


def write_dataset_manifest(
    *,
    manifest_path: Path,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    crops_storage_config: dict,
    resolved_noise_prediction_model: str | None,
    old_noise_recovery_review: dict,
    crop_duplicate_review: dict,
) -> None:
    selection_mode = resolve_crop_selection_mode(crops_storage_config)
    prediction_scope = resolve_noise_prediction_scope(crops_storage_config)
    filters_enabled = selection_mode == "filtered"
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "metadata_file": crops_storage_config["metadata_file_name"],
        "labels_file": crops_storage_config["labels_file_name"],
        "l10n_metadata_file": crops_storage_config["l10n_metadata_file_name"],
        "num_samples": int(len(metadata)),
        "num_labels": int(len(labels)),
        "label_column": crops_storage_config["label_column"],
        "image_extension": crops_storage_config["image_extension"],
        "crop_pad_frac": crops_storage_config["crop_pad_frac"],
        "selection_mode": selection_mode,
        "manual_reviewed_count": int(metadata["manual_reviewed"].sum()),
        "manual_corrected_count": int(
            metadata["manual_corrected_label"].notna().sum()
            if "manual_corrected_label" in metadata.columns
            else 0
        ),
        "noise_filtering": {
            "save_only_manual_reviewed": (
                filters_enabled and crops_storage_config["save_only_manual_reviewed"]
            ),
            "exclude_manual_noise": (
                filters_enabled and crops_storage_config["exclude_manual_noise"]
            ),
            "manual_noise_labels": crops_storage_config["manual_noise_labels"],
            "exclude_predicted_noise": (
                filters_enabled and crops_storage_config["exclude_predicted_noise"]
            ),
            "prediction_source": "crops.noise_predicted_label/noise_predicted_prob",
            "prediction_scope": prediction_scope,
            "configured_prediction_model": (
                crops_storage_config["noise_prediction_model"]
                if prediction_scope == "active_model"
                else None
            ),
            "resolved_prediction_model": resolved_noise_prediction_model,
            "human_review_overrides_prediction": True,
            "predicted_noise_labels": crops_storage_config["predicted_noise_labels"],
            "predicted_noise_min_prob": crops_storage_config["predicted_noise_min_prob"],
            "manual_correction_invalidate_metadata_columns": crops_storage_config[
                "manual_correction_invalidate_metadata_columns"
            ],
            "manual_correction_refill_operator_columns": crops_storage_config[
                "manual_correction_refill_operator_columns"
            ],
            "manual_correction_refill_submodel_bandai_columns": crops_storage_config[
                "manual_correction_refill_submodel_bandai_columns"
            ],
        },
        "old_noise_recovery_review": old_noise_recovery_review,
        "crop_duplicate_review": crop_duplicate_review,
        "notes": "Generated by stage_14_store_crops.py",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_noise_prediction_model(
    config: dict,
    crops_storage_config: dict,
) -> str:
    configured_model = str(crops_storage_config["noise_prediction_model"]).strip()
    if not configured_model:
        raise ValueError("crops_storage.noise_prediction_model 不能为空")
    if configured_model != "latest":
        return configured_model

    model_dir = utils.join_data_root(config["path"]["model_dir"], config=config)
    pointer_path = (
        model_dir
        / config["logistic_regression_filter"]["model_pointer_path"]
    )
    if not pointer_path.is_file():
        raise FileNotFoundError(f"Latest LR model pointer not found: {pointer_path}")
    resolved_model = pointer_path.read_text(encoding="utf-8").strip()
    if not resolved_model:
        raise ValueError(f"Latest LR model pointer is empty: {pointer_path}")
    return resolved_model


def build_db_predicted_noise_mask(
    metadata: pd.DataFrame,
    *,
    corrected_mask: pd.Series,
    crops_storage_config: dict,
    prediction_scope: str,
    active_prediction_model: str | None,
) -> pd.Series:
    prediction_scope = str(prediction_scope).strip().lower()
    if prediction_scope not in NOISE_PREDICTION_SCOPES:
        raise ValueError(
            "prediction_scope 必须是以下值之一: "
            f"{sorted(NOISE_PREDICTION_SCOPES)}"
        )
    required_columns = {
        "noise_review_label",
        "noise_predicted_prob",
        "noise_predicted_label",
    }
    if prediction_scope == "active_model":
        required_columns.add("noise_prediction_model")
        if not active_prediction_model:
            raise ValueError("active_model范围要求提供active_prediction_model")
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"crop数据库预测字段缺失: {sorted(missing_columns)}")

    review_label = (
        metadata["noise_review_label"].fillna("").astype(str).str.strip()
    )
    predicted_prob = pd.to_numeric(
        metadata["noise_predicted_prob"],
        errors="coerce",
    ).fillna(0.0)
    human_override = (
        review_label.eq(constants.NOISE_REVIEW_LABEL_OK)
        | corrected_mask.astype(bool)
    )
    predicted_noise = (
        metadata["noise_predicted_label"].isin(
            set(crops_storage_config["predicted_noise_labels"])
        )
        & (
            predicted_prob
            >= float(crops_storage_config["predicted_noise_min_prob"])
        )
        & ~human_override
    )
    if prediction_scope == "all_stored":
        return predicted_noise
    return (
        metadata["noise_prediction_model"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(active_prediction_model)
        & predicted_noise
    )


def main(config: dict | None = None) -> None:
    if config is None:
        config = utils.load_pipeline_config()

    utils.init_db(config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    crops_storage_config = config["crops_storage"]
    selection_mode = resolve_crop_selection_mode(crops_storage_config)
    prediction_scope = resolve_noise_prediction_scope(crops_storage_config)
    export_all_crops = selection_mode == "all"
    resolved_noise_prediction_model = None
    old_noise_recovery_review = summarize_old_noise_recovery_review(
        config=config,
        db_path=db_path,
    )
    log_old_noise_recovery_preflight(old_noise_recovery_review)
    duplicate_groups = load_crop_duplicate_groups(db_path)
    crop_duplicate_review = summarize_crop_duplicate_review(duplicate_groups)
    if (
        config["crop_duplicate_detection"]["require_review_complete_before_store"]
        and not crop_duplicate_review["review_complete"]
    ):
        logger.warning(
            "重复crop仍有%d个跨标签组未复核；Stage 14中断。请运行：%s",
            crop_duplicate_review["pending_count"],
            crop_duplicate_review["tool_command"],
        )
        return constants.STAGE_INTERRUPT
    if (
        not export_all_crops
        and crops_storage_config["exclude_predicted_noise"]
        and not config["lr_prediction"]["sync_to_db"]
    ):
        raise ValueError(
            "store_crops使用crops表中的LR预测字段过滤，"
            "要求lr_prediction.sync_to_db=true"
        )
    if (
        not export_all_crops
        and crops_storage_config["exclude_predicted_noise"]
        and prediction_scope == "active_model"
    ):
        resolved_noise_prediction_model = resolve_noise_prediction_model(
            config,
            crops_storage_config,
        )
        logger.info(
            "仅使用LR模型%s写入数据库的预测结果。",
            resolved_noise_prediction_model,
        )
    elif not export_all_crops and crops_storage_config["exclude_predicted_noise"]:
        logger.info(
            "使用数据库中所有已存LR模型轮次的预测结果；人工ok/纠正标签优先保留。"
        )

    if crops_storage_config["reprocess"]:
        logger.info("crops_storage配置为reprocess=true，将重新裁剪所有入选crop图像")
    else:
        logger.info("crops_storage配置为reprocess=false，将复用已存在的crop图像并只补齐缺失文件")
    if crops_storage_config["format"] == "flatten":
        with sqlite3.connect(db_path) as conn:
            metadata_columns = list(crops_storage_config["image_metadata_columns"])
            validate_config_column_names(metadata_columns)
            image_select_cols = [f"i.{col}" for col in metadata_columns]
            crop_sql = f"""
                SELECT
                    c.id AS crop_id,
                    c.image_id,
                    c.saved,
                    c.crop_path,
                    c.noise_review_label,
                    c.manual_corrected_label,
                    c.noise_predicted_prob,
                    c.noise_predicted_label,
                    c.noise_prediction_model,
                    CASE
                        WHEN c.noise_review_label = '{constants.NOISE_REVIEW_LABEL_OK}' THEN 1
                        ELSE 0
                    END AS manual_reviewed,
                    c.box_x1,
                    c.box_y1,
                    c.box_x2,
                    c.box_y2,
                    {', '.join(image_select_cols)}
                FROM crops c
                JOIN images i ON i.id = c.image_id
                ORDER BY c.id
            """
            metadata = pd.read_sql_query(crop_sql, conn)
        logger.info("已加载%d条crop及图片元数据，开始应用筛选策略。", len(metadata))

        if export_all_crops:
            logger.info(
                "crops_storage.selection_mode=all：不按人工复核或LR预测排除任何crop；"
                "人工纠正标签仍会应用。"
            )
        elif crops_storage_config["save_only_manual_reviewed"]:
            logger.info("将仅保存人工审核为ok的crop。")
            metadata = metadata[
                metadata["noise_review_label"].eq(constants.NOISE_REVIEW_LABEL_OK)
            ].copy()
        else:
            logger.info("将保存人工审核过、自动审核过和未审核的入选crop。")

        corrected_mask = (
            metadata["manual_corrected_label"].notna()
            & (metadata["manual_corrected_label"].astype(str).str.strip() != "")
        )
        
        if not export_all_crops and crops_storage_config["exclude_manual_noise"]:
            before_count = len(metadata)
            review_label = metadata["noise_review_label"].fillna("").astype(str).str.strip()
            #为选定噪声label的数据
            manual_noise = review_label.isin(set(crops_storage_config["manual_noise_labels"]))
            corrected_wrong_label = (
                review_label.eq(constants.NOISE_REVIEW_LABEL_WRONG_LABEL)
                & corrected_mask
            ) #被人工纠正为非clean的旧轮次数据
            
            #去掉manual_noise,但是用并集保留被人工纠正为非clean的旧轮次数据（不管新旧轮次）
            metadata = metadata.loc[~manual_noise | corrected_wrong_label].reset_index(drop=True)
            
            logger.info(
                "按人工复核过滤crop：过滤%d条，保留%d条。",
                before_count - len(metadata),
                len(metadata),
            )

        
        label_column = crops_storage_config["label_column"]
        if label_column not in metadata.columns:
            raise ValueError(f"image metadata中不存在label列: {label_column!r}")
        corrected_mask = (
            metadata["manual_corrected_label"].notna()
            & (metadata["manual_corrected_label"].astype(str).str.strip() != "")
        )
        if corrected_mask.any():
            # 将人工纠正的标签应用到label_column中，覆盖原有标签
            metadata.loc[corrected_mask, label_column] = metadata.loc[
                corrected_mask,
                "manual_corrected_label",
            ].astype(str)
            logger.info("已应用%d条人工纠正标签。", int(corrected_mask.sum()))
            invalidate_columns = list(crops_storage_config["manual_correction_invalidate_metadata_columns"])
            metadata = invalidate_metadata_for_manual_corrections(
                metadata,
                corrected_mask=corrected_mask,
                columns=invalidate_columns,
                label_column=label_column,
            )
            if invalidate_columns:
                logger.info(
                    "已清空%d条人工纠正样本的metadata列: %s",
                    int(corrected_mask.sum()),
                    invalidate_columns,
                )
            metadata = refill_unique_metadata_for_manual_corrections(
                metadata,
                corrected_mask=corrected_mask,
                label_column=label_column,
                operator_columns=list(crops_storage_config["manual_correction_refill_operator_columns"]),
                submodel_bandai_columns=list(
                    crops_storage_config["manual_correction_refill_submodel_bandai_columns"]
                ),
            )
            logger.info(
                "已按唯一反查规则尝试补齐%d条人工纠正样本的operator与submodel/bandai。",
                int(corrected_mask.sum()),
            )
        # 应用人工结论后，再按数据库中的LR预测结果过滤未复核样本。
        if not export_all_crops and crops_storage_config["exclude_predicted_noise"]:
            before_count = len(metadata)
            predicted_noise_mask = build_db_predicted_noise_mask(
                metadata,
                corrected_mask=corrected_mask,
                crops_storage_config=crops_storage_config,
                prediction_scope=prediction_scope,
                active_prediction_model=resolved_noise_prediction_model,
            )
            metadata = metadata.loc[~predicted_noise_mask].reset_index(drop=True)
            logger.info(
                "按数据库LR预测过滤crop（scope=%s，人工ok/纠正优先）："
                "过滤%d条，保留%d条。",
                prediction_scope,
                before_count - len(metadata),
                len(metadata),
            )
        metadata, duplicate_changed_mask, duplicate_apply_summary = (
            apply_crop_duplicate_resolutions(
                metadata,
                duplicate_groups=duplicate_groups,
                label_column=label_column,
            )
        )
        crop_duplicate_review["stage_14_apply"] = duplicate_apply_summary
        if duplicate_changed_mask.any():
            invalidate_columns = list(
                crops_storage_config["manual_correction_invalidate_metadata_columns"]
            )
            metadata = invalidate_metadata_for_manual_corrections(
                metadata,
                corrected_mask=duplicate_changed_mask,
                columns=invalidate_columns,
                label_column=label_column,
            )
            metadata = refill_unique_metadata_for_manual_corrections(
                metadata,
                corrected_mask=duplicate_changed_mask,
                label_column=label_column,
                operator_columns=list(
                    crops_storage_config["manual_correction_refill_operator_columns"]
                ),
                submodel_bandai_columns=list(
                    crops_storage_config[
                        "manual_correction_refill_submodel_bandai_columns"
                    ]
                ),
            )
        logger.info(
            "应用重复crop解析：处理groups=%d，移除=%d，整组排除crop=%d，"
            "label覆盖=%d，保留=%d。",
            duplicate_apply_summary["resolved_group_count"],
            duplicate_apply_summary["removed_count"],
            duplicate_apply_summary["excluded_count"],
            duplicate_apply_summary["label_override_count"],
            len(metadata),
        )
        #按格式变换ASCII label
        metadata = build_en_label(
            metadata,
            label_column=label_column,
        )
        
        logger.info("已加载%d条待裁剪crop及图片元数据。", len(metadata))
        logger.info("metadata列: %s", list(metadata.columns))
        
        dataset_root = utils.join_data_root(config['path']["dataset_dir"], config=config)
        dataset_img_subdir = config['path']["dataset_img_subdir"]
        image_extension = crops_storage_config["image_extension"].lower().lstrip(".")
        output_filenames = (
            metadata["image_id"].map(lambda value: f"{int(value):08d}")
            + "_"
            + metadata["crop_id"].map(lambda value: f"{int(value):08d}")
            + f".{image_extension}"
        )
        metadata["image_path"] = dataset_img_subdir + "/" + output_filenames
        metadata["output_path"] = metadata["image_path"].map(lambda path: dataset_root / path) # type: ignore
        metadata["source_path"] = metadata["downloaded_path"].map(
            lambda path: utils.join_data_root(str(path), config=config)
        )

        if crops_storage_config["reprocess"]:
            reusable_mask = pd.Series(False, index=metadata.index)
        else:
            reusable_mask = (
                metadata["crop_path"].notna()
                & (metadata["crop_path"].astype(str).str.strip() == metadata["image_path"].astype(str))
                & metadata["output_path"].map(lambda path: Path(path).exists())
            )
        reusable_count = int(reusable_mask.sum())
        to_save = metadata.loc[~reusable_mask].copy()
        reusable_rows = metadata.loc[reusable_mask].copy()
        logger.info(
            "crop图像复用%d条，需要裁剪保存%d条。",
            reusable_count,
            len(to_save),
        )

        saved_rows = reusable_rows.to_dict(orient="records")
        db_updates = []
        db_update_batch_size = 100
        for _, row in tqdm(to_save.iterrows(), total=len(to_save), desc="存盘裁剪图像"):
            try:
                save_crop_image(
                    source_image_path=row["source_path"],
                    output_path=row["output_path"],
                    box_x1=row["box_x1"],
                    box_y1=row["box_y1"],
                    box_x2=row["box_x2"],
                    box_y2=row["box_y2"],
                    pad_frac=crops_storage_config["crop_pad_frac"],
                    image_format=image_extension,
                    jpeg_quality=crops_storage_config['jpeg_quality']
                )
                saved_row = row.to_dict()
                saved_rows.append(saved_row)
                db_updates.append((row["image_path"], int(row["crop_id"])))
                if len(db_updates) >= db_update_batch_size:
                    flush_crop_save_updates(db_path, db_updates)
                    db_updates.clear()
            except Exception as e:
                logger.error(f"保存crop_id={row['crop_id']}失败: {e}")
                continue

        flush_crop_save_updates(db_path, db_updates)
        
        metadata = pd.DataFrame(saved_rows)
        if metadata.empty:
            logger.warning("没有成功保存的crop，跳过metadata和labels写入。")
            return constants.STAGE_PASS  # type: ignore
        
        metadata, labels = build_label_tables(metadata)
        output_metadata_columns = list(crops_storage_config["metadata_columns"])
        missing_metadata_columns = [col for col in output_metadata_columns if col not in metadata.columns]
        if missing_metadata_columns:
            raise ValueError(f"metadata输出列不存在: {missing_metadata_columns}")
        metadata = metadata[output_metadata_columns]

        metadata_path = dataset_root / crops_storage_config["metadata_file_name"]
        labels_path = dataset_root / crops_storage_config["labels_file_name"]
        l10n_metadata_path = dataset_root / crops_storage_config["l10n_metadata_file_name"]
        manifest_path = dataset_root / "manifest.json"
        dataset_root.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_path, index=False, encoding="utf-8")
        labels.to_csv(labels_path, index=False, encoding="utf-8")
        write_dataset_manifest(
            manifest_path=manifest_path,
            metadata=metadata,
            labels=labels,
            crops_storage_config=crops_storage_config,
            resolved_noise_prediction_model=resolved_noise_prediction_model,
            old_noise_recovery_review=old_noise_recovery_review,
            crop_duplicate_review=crop_duplicate_review,
        )

        logger.info("crop图像保存完成，已成功保存%d条crop数据。", len(metadata))
        logger.info("metadata已保存至%s，labels已保存至%s，manifest已保存至%s。", metadata_path, labels_path, manifest_path)
        exported_l10n_count = write_l10n_metadata(
            labels,
            db_path,
            l10n_metadata_path,
        )
        logger.info(
            "已从label_metadata规范表导出%d条多语言metadata至%s。",
            exported_l10n_count,
            l10n_metadata_path,
        )

        return constants.STAGE_COMPLETED #type:ignore flatten格式处理完成，退出程序


if __name__ == "__main__":
    main()
