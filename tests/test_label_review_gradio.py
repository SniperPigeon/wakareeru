from __future__ import annotations

import pandas as pd
import pytest

from tools import label_review_gradio


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "crop_id": 1,
                "label": "A",
                "pred_label": "B",
                "pred_label_rate": 0.99,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.82,
                "noise_review_label": "",
                "manual_corrected_label": "",
            },
            {
                "crop_id": 2,
                "label": "A",
                "pred_label": "B",
                "pred_label_rate": 0.98,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.95,
                "noise_review_label": "",
                "manual_corrected_label": "",
            },
            {
                "crop_id": 3,
                "label": "A",
                "pred_label": "C",
                "pred_label_rate": 0.98,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.90,
                "noise_review_label": "",
                "manual_corrected_label": "",
            },
            {
                "crop_id": 4,
                "label": "A",
                "pred_label": "A",
                "pred_label_rate": 1.00,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.99,
                "noise_review_label": "",
                "manual_corrected_label": "",
            },
            {
                "crop_id": 5,
                "label": "A",
                "pred_label": "B",
                "pred_label_rate": 1.00,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.99,
                "noise_review_label": "ok",
                "manual_corrected_label": "",
            },
            {
                "crop_id": 6,
                "label": "A",
                "pred_label": "B",
                "pred_label_rate": 1.00,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.99,
                "noise_review_label": "",
                "manual_corrected_label": "B",
            },
            {
                "crop_id": 7,
                "label": "A",
                "pred_label": "B",
                "pred_label_rate": 1.00,
                "noise_predicted_label": "wrong_label",
                "noise_predicted_prob": 0.79,
                "noise_review_label": "",
                "manual_corrected_label": "",
            },
            {
                "crop_id": 8,
                "label": "A",
                "pred_label": "B",
                "pred_label_rate": 1.00,
                "noise_predicted_label": "ok",
                "noise_predicted_prob": 0.99,
                "noise_review_label": "",
                "manual_corrected_label": "",
            },
        ]
    )


@pytest.fixture(autouse=True)
def configure_label_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        label_review_gradio,
        "CONFIG",
        {
            "crops_storage": {
                "predicted_noise_labels": ["wrong_label"],
                "predicted_noise_min_prob": 0.8,
            }
        },
    )


def test_lr_candidates_filter_by_pred_label_rate_and_mismatch() -> None:
    sample = label_review_gradio.sample_rows(
        rows=_rows(),
        stats=pd.DataFrame(),
        sample_mode="lr_auto_filtered",
        lr_min_pred_label_rate=0.95,
        lr_only_prediction_mismatch=True,
        samples_per_label=3,
        sample_size=20,
        seed=42,
    )

    assert sample["crop_id"].tolist() == [1, 2, 3]
    assert sample["pred_label_rate"].tolist() == [0.99, 0.98, 0.98]
    assert sample["noise_predicted_prob"].tolist() == [0.82, 0.95, 0.90]


def test_lr_candidates_can_include_matching_predictions() -> None:
    sample = label_review_gradio.sample_rows(
        rows=_rows(),
        stats=pd.DataFrame(),
        sample_mode="lr_auto_filtered",
        lr_min_pred_label_rate=0.98,
        lr_only_prediction_mismatch=False,
        samples_per_label=3,
        sample_size=20,
        seed=42,
    )

    assert sample["crop_id"].tolist() == [4, 1, 2, 3]


def test_lr_candidates_require_pred_label_rate() -> None:
    with pytest.raises(label_review_gradio.gr.Error, match="pred_label_rate"):
        label_review_gradio.sample_rows(
            rows=_rows().drop(columns=["pred_label_rate"]),
            stats=pd.DataFrame(),
            sample_mode="lr_auto_filtered",
            lr_min_pred_label_rate=0.95,
            lr_only_prediction_mismatch=True,
            samples_per_label=3,
            sample_size=20,
            seed=42,
        )
