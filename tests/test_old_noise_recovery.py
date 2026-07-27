import json
from pathlib import Path

import pandas as pd
import torch

from pipeline import old_noise_recovery


def test_score_old_noise_candidates_assigns_review_buckets(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "head.pt"
    torch.save(
        {
            "linear.weight": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            "linear.bias": torch.tensor([0.0, 0.0]),
        },
        checkpoint_path,
    )
    candidates = pd.DataFrame(
        [
            {"crop_id": 1, "assigned_label": "A", "historical_noise_prob": 0.9},
            {"crop_id": 2, "assigned_label": "A", "historical_noise_prob": 0.9},
            {"crop_id": 3, "assigned_label": "A", "historical_noise_prob": 0.9},
            {"crop_id": 4, "assigned_label": "C", "historical_noise_prob": 0.9},
            {"crop_id": 5, "assigned_label": "A", "historical_noise_prob": 0.9},
        ]
    )
    feature_cache = {
        "crop_ids": torch.tensor([1, 2, 3, 4]),
        "features": torch.tensor(
            [
                [10.0, 0.0],
                [0.0, 10.0],
                [0.0, 0.0],
                [10.0, 0.0],
            ]
        ),
    }

    scored = old_noise_recovery.score_old_noise_candidates(
        candidates,
        feature_cache=feature_cache,
        checkpoint_path=checkpoint_path,
        label_to_id={"A": 0, "B": 1},
        id_to_label={0: "A", 1: "B"},
        batch_size=2,
        disagreement_min_confidence=0.8,
    ).set_index("crop_id")

    assert scored.at[1, "probe_bucket"] == "likely_false_kill"
    assert scored.at[2, "probe_bucket"] == "likely_true_noise"
    assert scored.at[3, "probe_bucket"] == "uncertain"
    assert scored.at[4, "probe_bucket"] == "label_not_in_model"
    assert scored.at[5, "probe_bucket"] == "missing_feature"
    assert scored.at[1, "probe_assigned_margin"] > 0.99
    assert scored.at[2, "probe_assigned_margin"] < -0.99


def test_write_and_load_linear_head_artifact(tmp_path: Path) -> None:
    loss_round_dir = tmp_path / "loss_analysis" / "round"
    model_dir = tmp_path / "model"
    feature_dir = tmp_path / "feature_cache"
    loss_round_dir.mkdir(parents=True)
    model_dir.mkdir()
    feature_dir.mkdir()
    checkpoint_path = model_dir / "head.pt"
    feature_cache_path = feature_dir / "features.pt"
    label_map_path = loss_round_dir / "label_map.json"
    checkpoint_path.touch()
    feature_cache_path.touch()
    label_map_path.write_text(
        json.dumps({"label_to_id": {}, "id_to_label": {}}),
        encoding="utf-8",
    )
    config = {
        "path": {
            "in_project_root": False,
            "data_root": str(tmp_path),
        },
        "noise_detection": {
            "label_granularity": "fine_grained_series",
            "image_size": 384,
        },
        "loss_noise_tracking": {
            "linear_head_artifact_file_name": "linear_head_artifact.json",
        },
    }

    artifact_path = old_noise_recovery.write_linear_head_artifact(
        config=config,
        loss_round_dir=loss_round_dir,
        checkpoint_path=checkpoint_path,
        feature_cache_path=feature_cache_path,
        label_map_path=label_map_path,
        input_dim=384,
        num_classes=10,
    )
    artifact = old_noise_recovery.load_linear_head_artifact(
        config=config,
        loss_round_dir=loss_round_dir,
    )

    assert artifact_path == loss_round_dir / "linear_head_artifact.json"
    assert artifact["checkpoint_path"] == "model/head.pt"
    assert artifact["feature_cache_path"] == "feature_cache/features.pt"
    assert artifact["label_map_path"] == "label_map.json"
    assert artifact["num_classes"] == 10


def test_build_label_rescue_stats_prioritizes_rare_labels() -> None:
    probe_rows = pd.DataFrame(
        [
            {
                "assigned_label": "rare",
                "probe_bucket": "likely_false_kill",
                "review_label": None,
                "corrected_label": None,
            },
            {
                "assigned_label": "rare",
                "probe_bucket": "uncertain",
                "review_label": None,
                "corrected_label": None,
            },
            {
                "assigned_label": "common",
                "probe_bucket": "likely_false_kill",
                "review_label": "ok",
                "corrected_label": None,
            },
            {
                "assigned_label": "common",
                "probe_bucket": "likely_true_noise",
                "review_label": "wrong_label",
                "corrected_label": "other",
            },
        ]
    )
    metadata = pd.DataFrame(
        {
            "label": ["rare"] * 3 + ["common"] * 20 + ["without_candidate"],
        }
    )

    stats = old_noise_recovery.build_label_rescue_stats(
        probe_rows,
        metadata,
    ).set_index("label")

    assert stats.index[0] == "rare"
    assert stats.at["rare", "dataset_count"] == 3
    assert stats.at["rare", "likely_false_kill_unreviewed"] == 1
    assert stats.at["rare", "uncertain_unreviewed"] == 1
    assert stats.at["rare", "rescue_priority"] == 3
    assert stats.at["common", "reviewed_recovered"] == 2
    assert stats.at["without_candidate", "old_noise_candidates"] == 0
