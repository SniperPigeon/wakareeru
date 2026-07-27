from pathlib import Path
from typing import Any

import pandas as pd
import torch

from trainer import train as trainer_train


class DummyFeatureModel:
    feature_pooling = "test_pooling"
    feature_dim = 2

    def train_linear_head_only(self) -> None:
        pass

    def to(self, _device: torch.device) -> "DummyFeatureModel":
        return self


def test_feature_cache_reuses_paths_across_dataset_changes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    extracted_path_batches: list[list[str]] = []
    feature_by_path = {
        "images/a.jpg": torch.tensor([1.0, 10.0]),
        "images/b.jpg": torch.tensor([2.0, 20.0]),
        "images/c.jpg": torch.tensor([3.0, 30.0]),
        "images/d.jpg": torch.tensor([4.0, 40.0]),
    }

    def fake_make_dataloader(*, metadata: pd.DataFrame, **_kwargs: Any) -> pd.DataFrame:
        return metadata

    def fake_extract_feature_table(
        *,
        dataloader: pd.DataFrame,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        image_paths = dataloader["image_path"].tolist()
        extracted_path_batches.append(image_paths)
        return {
            "features": torch.stack([feature_by_path[path] for path in image_paths]),
            "labels": torch.tensor(dataloader["label_id"].tolist(), dtype=torch.long),
            "sample_indices": torch.arange(len(dataloader), dtype=torch.long),
            "image_paths": image_paths,
        }

    monkeypatch.setattr(trainer_train, "make_dataloader", fake_make_dataloader)
    monkeypatch.setattr(
        trainer_train,
        "extract_feature_table",
        fake_extract_feature_table,
    )

    config = {
        "path": {
            "in_project_root": False,
            "data_root": str(tmp_path),
        }
    }
    trainer_cfg = {
        "feature_cache_dir": "trainer_feature_cache",
        "backbone_model_name": "dummy-backbone",
        "image_size": 384,
        "image_path_column": "image_path",
        "label_id_column": "label_id",
    }
    phase_cfg = {
        "feature_cache_file_name": "features.pt",
        "feature_cache_rebuild": False,
        "feature_cache_amp_enabled": False,
    }
    common_kwargs = {
        "config": config,
        "trainer_cfg": trainer_cfg,
        "phase_cfg": phase_cfg,
        "model": DummyFeatureModel(),
        "dataset_root": tmp_path / "dataset",
        "processor": None,
        "device": torch.device("cpu"),
        "amp_enabled": False,
        "amp_dtype": torch.float32,
    }

    first_cache = trainer_train.load_or_create_feature_cache(
        train_df=pd.DataFrame(
            {
                "image_path": ["images/a.jpg", "images/b.jpg"],
                "label_id": [0, 1],
            }
        ),
        val_df=pd.DataFrame(
            {
                "image_path": ["images/c.jpg"],
                "label_id": [2],
            }
        ),
        **common_kwargs,
    )

    assert extracted_path_batches == [
        ["images/a.jpg", "images/b.jpg", "images/c.jpg"]
    ]
    assert first_cache["train"]["image_paths"] == [
        "images/a.jpg",
        "images/b.jpg",
    ]

    second_cache = trainer_train.load_or_create_feature_cache(
        train_df=pd.DataFrame(
            {
                "image_path": ["images/b.jpg", "images/d.jpg"],
                "label_id": [7, 3],
            }
        ),
        val_df=pd.DataFrame(
            {
                "image_path": ["images/c.jpg"],
                "label_id": [8],
            }
        ),
        **common_kwargs,
    )

    assert extracted_path_batches == [
        ["images/a.jpg", "images/b.jpg", "images/c.jpg"],
        ["images/d.jpg"],
    ]
    assert second_cache["train"]["image_paths"] == [
        "images/b.jpg",
        "images/d.jpg",
    ]
    assert second_cache["train"]["labels"].tolist() == [7, 3]
    assert torch.equal(
        second_cache["train"]["features"],
        torch.stack((feature_by_path["images/b.jpg"], feature_by_path["images/d.jpg"])),
    )

    saved_cache = torch.load(
        tmp_path / "trainer_feature_cache" / "features.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved_cache["cache_format_version"] == 2
    assert saved_cache["feature_bank"]["image_paths"] == [
        "images/a.jpg",
        "images/b.jpg",
        "images/c.jpg",
        "images/d.jpg",
    ]


def test_legacy_split_feature_cache_migrates_to_feature_bank() -> None:
    legacy_cache = {
        "train": {
            "features": torch.tensor([[1.0, 2.0]]),
            "image_paths": ["images/a.jpg"],
        },
        "val": {
            "features": torch.tensor([[3.0, 4.0]]),
            "image_paths": ["images/b.jpg"],
        },
    }

    feature_bank, migrated = trainer_train.load_feature_bank(
        cache=legacy_cache,
        feature_dim=2,
    )

    assert migrated is True
    assert feature_bank["image_paths"] == ["images/a.jpg", "images/b.jpg"]
    assert torch.equal(
        feature_bank["features"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
