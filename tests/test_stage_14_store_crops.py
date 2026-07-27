import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_14_store_crops as stage_14  # noqa: E402


def _crops_storage_config() -> dict:
    return {
        "noise_prediction_model": "latest",
        "predicted_noise_labels": ["wrong_label"],
        "predicted_noise_min_prob": 0.8,
    }


def test_crop_selection_mode_supports_filtered_and_all() -> None:
    assert stage_14.resolve_crop_selection_mode({"selection_mode": "filtered"}) == "filtered"
    assert stage_14.resolve_crop_selection_mode({"selection_mode": " ALL "}) == "all"


def test_crop_selection_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="selection_mode"):
        stage_14.resolve_crop_selection_mode({"selection_mode": "latest"})


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

    mask = stage_14.build_db_predicted_noise_mask(
        metadata,
        corrected_mask=corrected_mask,
        crops_storage_config=_crops_storage_config(),
        active_prediction_model=active_model,
    )

    assert mask.tolist() == [True, False, False, False, False, False, False]


def test_db_prediction_filter_requires_database_columns() -> None:
    metadata = pd.DataFrame(
        {
            "noise_review_label": [None],
            "noise_predicted_label": ["wrong_label"],
        }
    )

    with pytest.raises(ValueError, match="noise_predicted_prob"):
        stage_14.build_db_predicted_noise_mask(
            metadata,
            corrected_mask=pd.Series([False]),
            crops_storage_config=_crops_storage_config(),
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

    resolved_model = stage_14.resolve_noise_prediction_model(
        config,
        _crops_storage_config(),
    )

    assert resolved_model == "LR_pipeline_current.joblib"


def test_resolve_noise_prediction_model_accepts_explicit_model() -> None:
    crops_storage_config = _crops_storage_config()
    crops_storage_config["noise_prediction_model"] = "LR_pipeline_fixed.joblib"

    resolved_model = stage_14.resolve_noise_prediction_model(
        {},
        crops_storage_config,
    )

    assert resolved_model == "LR_pipeline_fixed.joblib"
