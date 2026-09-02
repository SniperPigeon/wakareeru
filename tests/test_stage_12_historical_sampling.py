import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_12_logistic_regression_filter as stage_12  # noqa: E402


def _sampling_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "score_column": "noise_score_v1",
        "bins": 4,
        "sample_to_current_ratio": 2.0,
        "random_seed": 42,
    }
    config.update(overrides)
    return config


def _review_rows() -> pd.DataFrame:
    rows = [
        {
            "crop_id": 1,
            "noise_review_score_col": "loss_round:round-current:noise_score_v1",
            "effective_noise_review_label": "ok",
            "noise_score_v1": 0.8,
        },
        {
            "crop_id": 2,
            "noise_review_score_col": "loss_round:round-current:noise_score_v1",
            "effective_noise_review_label": "ok",
            "noise_score_v1": 1.0,
        },
        {
            "crop_id": 3,
            "noise_review_score_col": "loss_round:round-current:noise_score_v1",
            "effective_noise_review_label": "wrong_label",
            "noise_score_v1": 1.9,
        },
    ]
    rows.extend(
        {
            "crop_id": 10 + index,
            "noise_review_score_col": "loss_round:old:noise_score_v1",
            "effective_noise_review_label": "ok",
            "noise_score_v1": float(index) / 10,
        }
        for index in range(12)
    )
    rows.extend(
        {
            "crop_id": 30 + index,
            "noise_review_score_col": "loss_round:old:noise_score_v1",
            "effective_noise_review_label": "wrong_label",
            "noise_score_v1": 1.0 + float(index) / 10,
        }
        for index in range(6)
    )
    return pd.DataFrame(rows)


def test_historical_sampling_keeps_current_and_balances_each_label() -> None:
    reviewed = _review_rows()

    sampled = stage_12.sample_historical_review_rows(
        reviewed,
        current_round_id="round-current",
        sampling_config=_sampling_config(),
    )

    current_ids = {1, 2, 3}
    assert current_ids.issubset(set(sampled["crop_id"]))
    historical = sampled[~sampled["crop_id"].isin(current_ids)]
    assert historical["effective_noise_review_label"].value_counts().to_dict() == {
        "ok": 4,
        "wrong_label": 2,
    }
    historical_clean = reviewed[
        reviewed["noise_review_score_col"].eq("loss_round:old:noise_score_v1")
        & reviewed["effective_noise_review_label"].eq("ok")
    ]
    clean_bins = pd.qcut(
        historical_clean["noise_score_v1"].rank(method="first"),
        q=4,
        labels=False,
    )
    crop_to_bin = dict(zip(historical_clean["crop_id"], clean_bins, strict=True))
    sampled_clean_ids = historical.loc[
        historical["effective_noise_review_label"].eq("ok"), "crop_id"
    ]
    assert {crop_to_bin[crop_id] for crop_id in sampled_clean_ids} == {0, 1, 2, 3}
    repeated = stage_12.sample_historical_review_rows(
        reviewed,
        current_round_id="round-current",
        sampling_config=_sampling_config(),
    )
    assert sampled["crop_id"].tolist() == repeated["crop_id"].tolist()


def test_historical_sampling_drops_labels_absent_from_current_round() -> None:
    reviewed = _review_rows()
    reviewed = reviewed[reviewed["crop_id"].ne(3)].copy()

    sampled = stage_12.sample_historical_review_rows(
        reviewed,
        current_round_id="round-current",
        sampling_config=_sampling_config(),
    )

    assert set(sampled["effective_noise_review_label"]) == {"ok"}


def test_historical_sampling_can_be_disabled() -> None:
    reviewed = _review_rows()

    sampled = stage_12.sample_historical_review_rows(
        reviewed,
        current_round_id="round-current",
        sampling_config=_sampling_config(enabled=False),
    )

    pd.testing.assert_frame_equal(sampled, reviewed)


def test_historical_sampling_requires_numeric_current_score() -> None:
    reviewed = _review_rows()
    reviewed.loc[reviewed["crop_id"].eq(10), "noise_score_v1"] = None

    with pytest.raises(ValueError, match="缺少有效noise_score_v1"):
        stage_12.sample_historical_review_rows(
            reviewed,
            current_round_id="round-current",
            sampling_config=_sampling_config(),
        )
