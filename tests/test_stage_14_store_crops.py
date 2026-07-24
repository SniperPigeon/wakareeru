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
    metadata = pd.DataFrame(
        [
            {
                "noise_review_label": None,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.90,
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
            {
                "noise_review_label": None,
                "noise_predicted_label": "ok",
                "noise_predicted_prob": 0.99,
            },
            {
                "noise_review_label": None,
                "noise_predicted_label": None,
                "noise_predicted_prob": None,
            },
        ]
    )
    corrected_mask = pd.Series([False, False, True, False, False, False])

    mask = stage_14.build_db_predicted_noise_mask(
        metadata,
        corrected_mask=corrected_mask,
        crops_storage_config=_crops_storage_config(),
    )

    assert mask.tolist() == [True, False, False, False, False, False]


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
        )
