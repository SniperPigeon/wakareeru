import sqlite3
import sys
import json
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_16_store_crops as stage_16  # noqa: E402


def _crops_storage_config() -> dict:
    return {
        "noise_prediction_scope": "active_model",
        "noise_prediction_model": "latest",
        "predicted_noise_labels": ["wrong_label"],
        "predicted_noise_min_prob": 0.8,
    }


def test_crop_selection_mode_supports_filtered_and_all() -> None:
    assert stage_16.resolve_crop_selection_mode({"selection_mode": "filtered"}) == "filtered"
    assert stage_16.resolve_crop_selection_mode({"selection_mode": " ALL "}) == "all"


def test_crop_selection_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="selection_mode"):
        stage_16.resolve_crop_selection_mode({"selection_mode": "latest"})


def test_noise_prediction_scope_supports_active_and_all_stored() -> None:
    assert (
        stage_16.resolve_noise_prediction_scope(
            {"noise_prediction_scope": "active_model"}
        )
        == "active_model"
    )
    assert (
        stage_16.resolve_noise_prediction_scope(
            {"noise_prediction_scope": " ALL_STORED "}
        )
        == "all_stored"
    )


def test_noise_prediction_scope_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="noise_prediction_scope"):
        stage_16.resolve_noise_prediction_scope(
            {"noise_prediction_scope": "historical"}
        )


def test_apply_crop_duplicate_resolutions_prefers_member_with_resolved_label() -> None:
    metadata = pd.DataFrame(
        [
            {"crop_id": 1, "fine_grained_series": "E257", "submodel": "wrong"},
            {"crop_id": 2, "fine_grained_series": "E259", "submodel": "correct"},
            {"crop_id": 3, "fine_grained_series": "E261", "submodel": "wrong"},
            {"crop_id": 4, "fine_grained_series": "other", "submodel": "other"},
        ]
    )
    groups = pd.DataFrame(
        [
            {
                "id": 1,
                "representative_crop_id": 1,
                "member_crop_ids_json": json.dumps([1, 2, 3]),
                "review_status": "confirmed",
                "resolved_label": "E259",
            }
        ]
    )

    result, changed_mask, summary = stage_16.apply_crop_duplicate_resolutions(
        metadata,
        duplicate_groups=groups,
        label_column="fine_grained_series",
    )

    assert result["crop_id"].tolist() == [2, 4]
    assert result.loc[result["crop_id"].eq(2), "fine_grained_series"].item() == "E259"
    assert not changed_mask.any()
    assert summary["removed_count"] == 2


def test_apply_crop_duplicate_resolutions_overrides_label_when_needed() -> None:
    metadata = pd.DataFrame(
        [{"crop_id": 10, "fine_grained_series": "old"}]
    )
    groups = pd.DataFrame(
        [
            {
                "id": 2,
                "representative_crop_id": 10,
                "member_crop_ids_json": json.dumps([10, 11]),
                "review_status": "confirmed",
                "resolved_label": "corrected",
            }
        ]
    )

    result, changed_mask, summary = stage_16.apply_crop_duplicate_resolutions(
        metadata,
        duplicate_groups=groups,
        label_column="fine_grained_series",
    )

    assert result["fine_grained_series"].tolist() == ["corrected"]
    assert changed_mask.tolist() == [True]
    assert summary["label_override_count"] == 1


def test_apply_crop_duplicate_resolutions_excludes_whole_group() -> None:
    metadata = pd.DataFrame(
        [
            {"crop_id": 20, "fine_grained_series": "a"},
            {"crop_id": 21, "fine_grained_series": "b"},
            {"crop_id": 22, "fine_grained_series": "keep"},
        ]
    )
    groups = pd.DataFrame(
        [
            {
                "id": 3,
                "representative_crop_id": 20,
                "member_crop_ids_json": json.dumps([20, 21]),
                "review_status": "excluded",
                "resolved_label": None,
            }
        ]
    )

    result, _, summary = stage_16.apply_crop_duplicate_resolutions(
        metadata,
        duplicate_groups=groups,
        label_column="fine_grained_series",
    )

    assert result["crop_id"].tolist() == [22]
    assert summary["excluded_count"] == 2


def test_db_prediction_filter_respects_human_review_and_correction() -> None:
    active_model = "LR_pipeline_current.joblib"
    metadata = pd.DataFrame(
        [
            {
                "noise_review_label": None,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.90,
                "noise_prediction_model": active_model,
            },
            {
                "noise_review_label": "ok",
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.99,
                "noise_prediction_model": active_model,
            },
            {
                "noise_review_label": "wrong_label",
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.99,
                "noise_prediction_model": active_model,
            },
            {
                "noise_review_label": None,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.79,
                "noise_prediction_model": active_model,
            },
            {
                "noise_review_label": None,
                "noise_predicted_label": "ok",
                "noise_predicted_prob": 0.99,
                "noise_prediction_model": active_model,
            },
            {
                "noise_review_label": None,
                "noise_predicted_label": None,
                "noise_predicted_prob": None,
                "noise_prediction_model": None,
            },
            {
                "noise_review_label": None,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.99,
                "noise_prediction_model": "LR_pipeline_old.joblib",
            },
        ]
    )
    corrected_mask = pd.Series(
        [False, False, True, False, False, False, False]
    )

    mask = stage_16.build_db_predicted_noise_mask(
        metadata,
        corrected_mask=corrected_mask,
        crops_storage_config=_crops_storage_config(),
        prediction_scope="active_model",
        active_prediction_model=active_model,
    )

    assert mask.tolist() == [True, False, False, False, False, False, False]


def test_db_prediction_filter_all_stored_includes_old_models() -> None:
    metadata = pd.DataFrame(
        [
            {
                "noise_review_label": None,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.99,
            },
            {
                "noise_review_label": "ok",
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.99,
            },
            {
                "noise_review_label": "wrong_label",
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.99,
            },
            {
                "noise_review_label": None,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.79,
            },
        ]
    )
    corrected_mask = pd.Series([False, False, True, False])

    mask = stage_16.build_db_predicted_noise_mask(
        metadata,
        corrected_mask=corrected_mask,
        crops_storage_config=_crops_storage_config(),
        prediction_scope="all_stored",
        active_prediction_model=None,
    )

    assert mask.tolist() == [True, False, False, False]


def test_db_prediction_filter_requires_database_columns() -> None:
    metadata = pd.DataFrame(
        {
            "noise_review_label": [None],
            "noise_predicted_label": ["wrong_label"],
        }
    )

    with pytest.raises(ValueError, match="noise_predicted_prob"):
        stage_16.build_db_predicted_noise_mask(
            metadata,
            corrected_mask=pd.Series([False]),
            crops_storage_config=_crops_storage_config(),
            prediction_scope="active_model",
            active_prediction_model="LR_pipeline_current.joblib",
        )


def test_resolve_noise_prediction_model_from_latest_pointer(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "latest_lr_model.txt").write_text(
        "LR_pipeline_current.joblib\n",
        encoding="utf-8",
    )
    config = {
        "path": {
            "in_project_root": False,
            "data_root": str(tmp_path),
            "model_dir": "model",
        },
        "logistic_regression_filter": {
            "model_pointer_path": "latest_lr_model.txt",
        },
    }

    resolved_model = stage_16.resolve_noise_prediction_model(
        config,
        _crops_storage_config(),
    )

    assert resolved_model == "LR_pipeline_current.joblib"


def test_resolve_noise_prediction_model_accepts_explicit_model() -> None:
    crops_storage_config = _crops_storage_config()
    crops_storage_config["noise_prediction_model"] = "LR_pipeline_fixed.joblib"

    resolved_model = stage_16.resolve_noise_prediction_model(
        {},
        crops_storage_config,
    )

    assert resolved_model == "LR_pipeline_fixed.joblib"


def test_old_noise_recovery_summary_uses_current_manual_review_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "dataset.sqlite"
    review_path = tmp_path / "review" / "old_noise_recovery.csv"
    review_path.parent.mkdir()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE crops (
                id INTEGER PRIMARY KEY,
                noise_review_label TEXT,
                manual_corrected_label TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO crops VALUES (?, ?, ?)",
            [
                (1, "ok", None),
                (2, "wrong_label", "corrected"),
                (3, None, None),
            ],
        )
    pd.DataFrame(
        [
            {
                "crop_id": 1,
                "probe_round": "round-a",
                "active_lr_model": "current.joblib",
                "probe_bucket": "likely_false_kill",
            },
            {
                "crop_id": 2,
                "probe_round": "round-a",
                "active_lr_model": "current.joblib",
                "probe_bucket": "likely_true_noise",
            },
            {
                "crop_id": 3,
                "probe_round": "round-a",
                "active_lr_model": "current.joblib",
                "probe_bucket": "uncertain",
            },
        ]
    ).to_csv(review_path, index=False)
    config = {
        "path": {
            "in_project_root": False,
            "data_root": str(tmp_path),
        },
        "old_noise_recovery": {
            "review_file_path": "review/old_noise_recovery.csv",
        },
    }

    summary = stage_16.summarize_old_noise_recovery_review(
        config=config,
        db_path=db_path,
    )

    assert summary["pipeline_stage"] is False
    assert summary["candidate_count"] == 3
    assert summary["reviewed_count"] == 2
    assert summary["unreviewed_count"] == 1
    assert summary["manual_ok_count"] == 1
    assert summary["manual_corrected_count"] == 1
    assert summary["probe_rounds"] == ["round-a"]
