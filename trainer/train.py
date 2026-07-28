import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

from model_core.model import BackboneLinearClassifier
from pipeline import utils
from trainer.checkpoint import save_checkpoint, write_json
from trainer.dataset import CropCollator, CropDataset
from trainer.eval import (
    build_eval_report,
    build_top_k_accuracy,
    evaluate,
    make_top_k_correct_counts,
    update_top_k_correct_counts,
    validate_top_k_values,
)

logger = utils.get_logger("trainer")

FEATURE_CACHE_FORMAT_VERSION = 2


def validate_image_size(image_size: int) -> int:
    image_size = int(image_size)
    if image_size < 1:
        raise ValueError("trainer.image_size必须是正整数")
    if image_size % 16 != 0:
        logger.warning("trainer.image_size=%d不是16的倍数，ViT patch输入可能产生额外插值或截断。", image_size)
    return image_size


def get_torch_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tables(config: dict[str, Any]) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    trainer_cfg = config["trainer"]
    dataset_root = utils.join_data_root(config["path"]["dataset_dir"], config=config)
    metadata_path = dataset_root / trainer_cfg["metadata_file_name"]
    labels_path = dataset_root / trainer_cfg["labels_file_name"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata文件不存在: {metadata_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels文件不存在: {labels_path}")

    metadata = pd.read_csv(metadata_path)
    labels = pd.read_csv(labels_path)
    validate_tables(metadata=metadata, labels=labels, trainer_cfg=trainer_cfg)
    return dataset_root, metadata, labels


def validate_tables(
    *,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    trainer_cfg: dict[str, Any],
) -> None:
    image_path_column = trainer_cfg["image_path_column"]
    label_id_column = trainer_cfg["label_id_column"]
    missing_metadata_columns = {image_path_column, label_id_column} - set(metadata.columns)
    if missing_metadata_columns:
        raise ValueError(f"metadata缺少必要列: {sorted(missing_metadata_columns)}")
    if {"label_id", "label"} - set(labels.columns):
        raise ValueError("labels.csv必须包含label_id和label列")
    label_ids = sorted(labels["label_id"].astype(int).tolist())
    if label_ids != list(range(len(label_ids))):
        raise ValueError("labels.csv中的label_id必须从0开始连续编号")
    metadata_label_ids = set(metadata[label_id_column].dropna().astype(int).tolist())
    missing_label_ids = metadata_label_ids - set(label_ids)
    if missing_label_ids:
        raise ValueError(f"metadata中存在labels.csv没有定义的label_id: {sorted(missing_label_ids)}")


def split_metadata(
    *,
    metadata: pd.DataFrame,
    trainer_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_id_column = trainer_cfg["label_id_column"]
    val_ratio = float(trainer_cfg["val_ratio"])
    if not 0 < val_ratio < 1:
        raise ValueError("trainer.val_ratio必须在0和1之间")

    stratify = None
    if bool(trainer_cfg["stratify_split"]):
        counts = metadata[label_id_column].value_counts()
        if int(counts.min()) >= 2:
            stratify = metadata[label_id_column]
        else:
            logger.warning("部分标签样本数小于2，改用随机切分。")

    train_df, val_df = train_test_split(
        metadata,
        test_size=val_ratio,
        random_state=int(trainer_cfg["seed"]),
        shuffle=True,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_dataloader(
    *,
    metadata: pd.DataFrame,
    dataset_root: Path,
    processor: Any,
    trainer_cfg: dict[str, Any],
    train: bool,
) -> DataLoader:
    num_workers = int(trainer_cfg["num_workers"])
    image_size = validate_image_size(int(trainer_cfg["image_size"]))
    dataloader_kwargs = {
        "batch_size": int(trainer_cfg["batch_size"]),
        "shuffle": train,
        "num_workers": num_workers,
        "collate_fn": CropCollator(processor, image_size=image_size),
        "pin_memory": bool(trainer_cfg["pin_memory"]),
        "drop_last": train and bool(trainer_cfg["drop_last"]),
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(trainer_cfg["persistent_workers"])
        dataloader_kwargs["prefetch_factor"] = int(trainer_cfg["prefetch_factor"])

    dataset = CropDataset(
        metadata=metadata,
        dataset_root=dataset_root,
        image_path_column=trainer_cfg["image_path_column"],
        label_id_column=trainer_cfg["label_id_column"],
    )
    return DataLoader(dataset, **dataloader_kwargs)


def make_feature_cache_path(
    *,
    config: dict[str, Any],
    trainer_cfg: dict[str, Any],
    feature_cache_file_name: str,
) -> Path:
    feature_cache_dir = utils.join_data_root(trainer_cfg["feature_cache_dir"], config=config)
    return feature_cache_dir / feature_cache_file_name


def update_latest_trainer_run_pointer(*, run_dir: Path, trainer_cfg: dict[str, Any]) -> None:
    pointer_path = run_dir.parent / trainer_cfg["latest_run_pointer"]
    pointer_path.write_text(run_dir.name + "\n", encoding="utf-8")
    logger.info("已更新最新trainer run指针: %s -> %s", pointer_path, run_dir.name)


def normalize_image_paths(paths: list[Any]) -> list[str]:
    return [str(path).replace("\\", "/") for path in paths]


def validate_feature_cache_compatibility(
    *,
    cache: dict[str, Any],
    trainer_cfg: dict[str, Any],
    model: BackboneLinearClassifier,
) -> None:
    compatibility_fields = (
        ("backbone_model_name", trainer_cfg["backbone_model_name"]),
        ("feature_pooling", model.feature_pooling),
        ("feature_dim", model.feature_dim),
        ("image_size", int(trainer_cfg["image_size"])),
    )
    for field, current_value in compatibility_fields:
        if field not in cache:
            raise ValueError(
                f"linear head特征缓存缺少{field}元数据，"
                "请设置feature_cache_rebuild=true后重建。"
            )
        cached_value = cache[field]
        if field in {"feature_dim", "image_size"}:
            cached_value = int(cached_value)
        if cached_value != current_value:
            raise ValueError(
                f"linear head特征缓存的{field}与当前配置不一致，"
                "请设置feature_cache_rebuild=true后重建。"
            )


def validate_feature_bank(
    *,
    feature_bank: dict[str, Any],
    feature_dim: int,
) -> None:
    features = feature_bank["features"]
    image_paths = normalize_image_paths(feature_bank["image_paths"])
    if features.ndim != 2 or features.shape[1] != feature_dim:
        raise ValueError(
            "linear head特征缓存维度错误: "
            f"shape={tuple(features.shape)}, feature_dim={feature_dim}。"
            "请删除缓存或设置feature_cache_rebuild=true后重建。"
        )
    if features.shape[0] != len(image_paths):
        raise ValueError(
            "linear head特征缓存的feature数量与image_path数量不一致，"
            "请删除缓存或设置feature_cache_rebuild=true后重建。"
        )
    if len(image_paths) != len(set(image_paths)):
        raise ValueError(
            "linear head特征缓存包含重复image_path，"
            "请删除缓存或设置feature_cache_rebuild=true后重建。"
        )
    if not torch.isfinite(features).all():
        nan_count = int(torch.isnan(features).sum().item())
        inf_count = int(torch.isinf(features).sum().item())
        raise ValueError(
            "linear head特征缓存包含非有限feature: "
            f"nan={nan_count}, inf={inf_count}。"
            "请删除缓存或设置feature_cache_rebuild=true后重建。"
        )
    feature_bank["image_paths"] = image_paths


def load_feature_bank(
    *,
    cache: dict[str, Any],
    feature_dim: int,
) -> tuple[dict[str, Any], bool]:
    cache_format_version = cache.get("cache_format_version")
    if cache_format_version == FEATURE_CACHE_FORMAT_VERSION:
        feature_bank = cache["feature_bank"]
        validate_feature_bank(feature_bank=feature_bank, feature_dim=feature_dim)
        return feature_bank, False

    if cache_format_version is not None:
        raise ValueError(
            f"不支持的linear head特征缓存格式版本: {cache_format_version!r}，"
            "请设置feature_cache_rebuild=true后重建。"
        )

    if not {"train", "val"}.issubset(cache):
        raise ValueError(
            "无法识别linear head特征缓存结构，"
            "请设置feature_cache_rebuild=true后重建。"
        )
    feature_bank = {
        "features": torch.cat(
            (cache["train"]["features"], cache["val"]["features"]),
            dim=0,
        ),
        "image_paths": (
            list(cache["train"]["image_paths"])
            + list(cache["val"]["image_paths"])
        ),
    }
    validate_feature_bank(feature_bank=feature_bank, feature_dim=feature_dim)
    logger.info("将旧版train/val整表特征缓存迁移为按image_path索引的增量缓存。")
    return feature_bank, True


def build_feature_table(
    *,
    feature_bank: dict[str, Any],
    metadata: pd.DataFrame,
    trainer_cfg: dict[str, Any],
) -> dict[str, Any]:
    image_paths = normalize_image_paths(
        metadata[trainer_cfg["image_path_column"]].tolist()
    )
    feature_indices_by_path = {
        image_path: index
        for index, image_path in enumerate(feature_bank["image_paths"])
    }
    missing_paths = [
        image_path
        for image_path in image_paths
        if image_path not in feature_indices_by_path
    ]
    if missing_paths:
        raise ValueError(
            "linear head特征缓存缺少当前metadata样本: "
            f"{missing_paths[:5]}"
        )
    feature_indices = torch.tensor(
        [feature_indices_by_path[image_path] for image_path in image_paths],
        dtype=torch.long,
    )
    return {
        "features": feature_bank["features"].index_select(0, feature_indices),
        "labels": torch.tensor(
            metadata[trainer_cfg["label_id_column"]].astype(int).to_numpy(),
            dtype=torch.long,
        ),
        "sample_indices": torch.arange(len(metadata), dtype=torch.long),
        "image_paths": image_paths,
    }


def save_feature_bank_cache(
    *,
    feature_cache_path: Path,
    feature_bank: dict[str, Any],
    trainer_cfg: dict[str, Any],
    model: BackboneLinearClassifier,
    created_at: str,
) -> None:
    cache = {
        "cache_format_version": FEATURE_CACHE_FORMAT_VERSION,
        "created_at": created_at,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "backbone_model_name": trainer_cfg["backbone_model_name"],
        "feature_pooling": model.feature_pooling,
        "feature_dim": model.feature_dim,
        "image_size": int(trainer_cfg["image_size"]),
        "feature_bank": feature_bank,
    }
    temporary_path = feature_cache_path.with_name(feature_cache_path.name + ".tmp")
    torch.save(cache, temporary_path)
    temporary_path.replace(feature_cache_path)


@torch.inference_mode()
def extract_feature_table(
    *,
    model: BackboneLinearClassifier,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    features = []
    labels = []
    sample_indices = []
    image_paths = []
    for batch in tqdm(dataloader, desc="extract features", unit="batch"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            batch_features = model.extract_features(pixel_values)
        features.append(batch_features.float().cpu())
        labels.append(batch["labels"].cpu())
        sample_indices.append(batch["sample_index"].cpu())
        image_paths.extend(batch["image_path"])
    return {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "sample_indices": torch.cat(sample_indices, dim=0),
        "image_paths": image_paths,
    }


def load_or_create_feature_cache(
    *,
    config: dict[str, Any],
    trainer_cfg: dict[str, Any],
    phase_cfg: dict[str, Any],
    model: BackboneLinearClassifier,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_root: Path,
    processor: Any,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    feature_cache_path = make_feature_cache_path(
        config=config,
        trainer_cfg=trainer_cfg,
        feature_cache_file_name=phase_cfg["feature_cache_file_name"],
    )
    feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_changed = False
    cache_created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    if feature_cache_path.exists() and not bool(phase_cfg["feature_cache_rebuild"]):
        logger.info("加载linear head特征缓存: %s", feature_cache_path)
        cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
        validate_feature_cache_compatibility(
            cache=cache,
            trainer_cfg=trainer_cfg,
            model=model,
        )
        cache_created_at = str(cache.get("created_at", cache_created_at))
        feature_bank, cache_changed = load_feature_bank(
            cache=cache,
            feature_dim=model.feature_dim,
        )
    else:
        if feature_cache_path.exists():
            logger.info("按配置重建linear head特征缓存: %s", feature_cache_path)
        else:
            logger.info("开始生成linear head特征缓存: %s", feature_cache_path)
        feature_bank = {
            "features": torch.empty((0, model.feature_dim), dtype=torch.float32),
            "image_paths": [],
        }
        cache_changed = True

    image_path_column = trainer_cfg["image_path_column"]
    current_metadata = pd.concat((train_df, val_df), ignore_index=True)
    current_metadata = current_metadata.copy()
    current_metadata[image_path_column] = normalize_image_paths(
        current_metadata[image_path_column].tolist()
    )
    current_metadata = current_metadata.drop_duplicates(
        subset=[image_path_column],
        keep="first",
    ).reset_index(drop=True)
    cached_paths = set(feature_bank["image_paths"])
    missing_metadata = current_metadata.loc[
        ~current_metadata[image_path_column].isin(cached_paths)
    ].reset_index(drop=True)

    if not missing_metadata.empty:
        logger.info(
            "linear head特征缓存命中%d/%d个当前唯一样本，仅提取%d个新增样本。",
            len(current_metadata) - len(missing_metadata),
            len(current_metadata),
            len(missing_metadata),
        )
        missing_feature_loader = make_dataloader(
            metadata=missing_metadata,
            dataset_root=dataset_root,
            processor=processor,
            trainer_cfg=trainer_cfg,
            train=False,
        )
        model.train_linear_head_only()
        model.to(device)
        extracted = extract_feature_table(
            model=model,
            dataloader=missing_feature_loader,
            device=device,
            amp_enabled=bool(phase_cfg["feature_cache_amp_enabled"]) and amp_enabled,
            amp_dtype=amp_dtype,
        )
        expected_paths = missing_metadata[image_path_column].tolist()
        if extracted["image_paths"] != expected_paths:
            raise ValueError("linear head新增特征的image_path顺序与metadata不一致")
        feature_bank["features"] = torch.cat(
            (feature_bank["features"], extracted["features"]),
            dim=0,
        )
        feature_bank["image_paths"].extend(extracted["image_paths"])
        cache_changed = True
    else:
        logger.info(
            "linear head特征缓存命中全部%d个当前唯一样本，无需运行backbone。",
            len(current_metadata),
        )

    validate_feature_bank(feature_bank=feature_bank, feature_dim=model.feature_dim)
    if cache_changed:
        save_feature_bank_cache(
            feature_cache_path=feature_cache_path,
            feature_bank=feature_bank,
            trainer_cfg=trainer_cfg,
            model=model,
            created_at=cache_created_at,
        )
        logger.info(
            "linear head增量特征缓存已保存: %s（累计%d个样本）",
            feature_cache_path,
            len(feature_bank["image_paths"]),
        )

    runtime_cache = {
        "feature_dim": model.feature_dim,
        "train": build_feature_table(
            feature_bank=feature_bank,
            metadata=train_df,
            trainer_cfg=trainer_cfg,
        ),
        "val": build_feature_table(
            feature_bank=feature_bank,
            metadata=val_df,
            trainer_cfg=trainer_cfg,
        ),
    }
    validate_feature_cache(runtime_cache)
    return runtime_cache


def validate_feature_cache(cache: dict[str, Any]) -> None:
    feature_dim = int(cache["feature_dim"])
    for split in ("train", "val"):
        features = cache[split]["features"]
        labels = cache[split]["labels"]
        sample_indices = cache[split]["sample_indices"]
        image_paths = cache[split]["image_paths"]
        if features.ndim != 2 or features.shape[1] != feature_dim:
            raise ValueError(
                f"linear head特征缓存维度错误: split={split}, "
                f"shape={tuple(features.shape)}, feature_dim={feature_dim}。"
                "请删除缓存或设置feature_cache_rebuild=true后重建。"
            )
        if not torch.isfinite(features).all():
            nan_count = int(torch.isnan(features).sum().item())
            inf_count = int(torch.isinf(features).sum().item())
            raise ValueError(
                f"linear head特征缓存包含非有限feature: split={split}, "
                f"nan={nan_count}, inf={inf_count}。请删除缓存或设置feature_cache_rebuild=true后重建。"
            )
        if not (
            features.shape[0]
            == labels.numel()
            == sample_indices.numel()
            == len(image_paths)
        ):
            raise ValueError(
                f"linear head特征缓存字段长度不一致: split={split}。"
                "请删除缓存或设置feature_cache_rebuild=true后重建。"
            )
        if labels.numel() == 0:
            raise ValueError(f"linear head特征缓存为空: split={split}")


def make_feature_dataloader(
    *,
    feature_table: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        feature_table["features"],
        feature_table["labels"].long(),
        feature_table["sample_indices"].long(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    top_k_values: list[int],
) -> dict[str, float | int]:
    model.train()
    loss_chunks = []
    correct_chunks = []
    top_k_correct_counts = make_top_k_correct_counts(top_k_values)
    sample_count = 0
    for batch in tqdm(dataloader, desc="train", unit="batch"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(pixel_values)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练中出现非有限loss，请检查输入、AMP和学习率。")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_size = int(labels.numel())
            update_top_k_correct_counts(
                counts=top_k_correct_counts,
                logits=logits,
                labels=labels,
            )
            loss_chunks.append(loss.detach() * batch_size)
            correct_chunks.append(preds.eq(labels).sum().detach())
            sample_count += batch_size

    loss_sum = torch.stack(loss_chunks).sum().item() if loss_chunks else 0.0
    correct_count = torch.stack(correct_chunks).sum().item() if correct_chunks else 0
    return {
        "loss": loss_sum / max(1, sample_count),
        "accuracy": correct_count / max(1, sample_count),
        **build_top_k_accuracy(counts=top_k_correct_counts, sample_count=sample_count),
        "n": sample_count,
    }


def train_feature_head_one_epoch(
    *,
    model: BackboneLinearClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    top_k_values: list[int],
) -> dict[str, float | int]:
    model.classifier.train()
    loss_chunks = []
    correct_chunks = []
    top_k_correct_counts = make_top_k_correct_counts(top_k_values)
    sample_count = 0
    for features_cpu, labels_cpu, _sample_indices_cpu in tqdm(dataloader, desc="train features", unit="batch"):
        features = features_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model.classifier(features)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("feature linear head训练中出现非有限loss，请重建特征缓存或关闭AMP。")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_size = int(labels.numel())
            update_top_k_correct_counts(
                counts=top_k_correct_counts,
                logits=logits,
                labels=labels,
            )
            loss_chunks.append(loss.detach() * batch_size)
            correct_chunks.append(preds.eq(labels).sum().detach())
            sample_count += batch_size

    loss_sum = torch.stack(loss_chunks).sum().item() if loss_chunks else 0.0
    correct_count = torch.stack(correct_chunks).sum().item() if correct_chunks else 0
    return {
        "loss": loss_sum / max(1, sample_count),
        "accuracy": correct_count / max(1, sample_count),
        **build_top_k_accuracy(counts=top_k_correct_counts, sample_count=sample_count),
        "n": sample_count,
    }


@torch.inference_mode()
def evaluate_feature_head(
    *,
    model: BackboneLinearClassifier,
    dataloader: DataLoader,
    feature_table: dict[str, Any],
    labels: pd.DataFrame,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    top_k_values: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.classifier.eval()
    records = []
    image_paths = feature_table["image_paths"]
    top_k_correct_counts = make_top_k_correct_counts(top_k_values)
    for features_cpu, labels_cpu, sample_indices_cpu in tqdm(dataloader, desc="eval features", unit="batch"):
        features = features_cpu.to(device, non_blocking=True)
        y_true = labels_cpu.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model.classifier(features)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("feature linear head验证中出现非有限logits，请重建特征缓存或降低学习率。")
        update_top_k_correct_counts(
            counts=top_k_correct_counts,
            logits=logits,
            labels=y_true,
        )
        probs = torch.softmax(logits, dim=1)
        confidence, y_pred = probs.max(dim=1)
        for i, sample_index in enumerate(sample_indices_cpu.tolist()):
            records.append(
                {
                    "sample_index": int(sample_index),
                    "image_path": image_paths[int(sample_index)],
                    "label_id": int(y_true[i].item()),
                    "pred_id": int(y_pred[i].item()),
                    "pred_confidence": float(confidence[i].item()),
                    "correct": bool(y_pred[i].eq(y_true[i]).item()),
                }
            )
    predictions = pd.DataFrame(records)
    return (
        build_eval_report(
            predictions=predictions,
            labels=labels,
            top_k_accuracy=build_top_k_accuracy(
                counts=top_k_correct_counts,
                sample_count=len(predictions),
            ),
        ),
        predictions,
    )


def is_metric_improved(
    *,
    current_value: float,
    best_value: float | None,
    mode: str,
    min_delta: float,
) -> bool:
    if best_value is None:
        return True
    if mode == "max":
        return current_value > best_value + min_delta
    if mode == "min":
        return current_value < best_value - min_delta
    raise ValueError("trainer.early_stopping_mode必须是'max'或'min'")


def get_amp_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError("trainer.amp_dtype必须是float16或bfloat16")


def prepare_phase(model: BackboneLinearClassifier, phase_cfg: dict[str, Any]) -> None:
    train_mode = phase_cfg["train_mode"]
    if train_mode == "linear_head":
        model.train_linear_head_only()
        return
    raise ValueError("trainer.phases[].train_mode必须是linear_head")


def make_phase_optimizer(
    *,
    model: BackboneLinearClassifier,
    phase_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    train_mode = phase_cfg["train_mode"]
    weight_decay = float(phase_cfg["weight_decay"])
    if train_mode == "linear_head":
        return torch.optim.AdamW(
            model.classifier.parameters(),
            lr=float(phase_cfg["learning_rate"]),
            weight_decay=weight_decay,
        )
    raise ValueError(f"未知训练模式: {train_mode!r}")


def run_phase(
    *,
    phase_cfg: dict[str, Any],
    model: BackboneLinearClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_root: Path,
    processor: Any,
    labels: pd.DataFrame,
    criterion: nn.Module,
    device: torch.device,
    trainer_cfg: dict[str, Any],
    config: dict[str, Any],
    run_dir: Path,
    labels_payload: list[dict[str, Any]],
    epoch_rows: list[dict[str, Any]],
    global_epoch_start: int,
    top_k_values: list[int],
) -> tuple[int, dict[str, Any]]:
    prepare_phase(model, phase_cfg)
    model.to(device)
    optimizer = make_phase_optimizer(
        model=model,
        phase_cfg=phase_cfg,
    )

    phase_name = str(phase_cfg["name"])
    amp_enabled = bool(trainer_cfg["amp_enabled"]) and device.type == "cuda"
    amp_dtype = get_amp_dtype(str(trainer_cfg["amp_dtype"]))
    use_feature_cache = bool(phase_cfg["use_feature_cache"])
    early_stopping_enabled = bool(phase_cfg["early_stopping_enabled"])
    early_stopping_monitor = str(phase_cfg["early_stopping_monitor"])
    early_stopping_mode = str(phase_cfg["early_stopping_mode"])
    early_stopping_patience = int(phase_cfg["early_stopping_patience"])
    early_stopping_min_delta = float(phase_cfg["early_stopping_min_delta"])
    if early_stopping_patience < 1:
        raise ValueError("trainer.phases[].early_stopping_patience必须大于等于1")

    best_score = None
    best_checkpoint_path = None
    best_epoch = None
    epochs_without_improvement = 0
    stopped_early = False
    completed_epochs = 0
    feature_cache = None
    feature_train_loader = None
    feature_val_loader = None
    if use_feature_cache:
        if phase_cfg["train_mode"] != "linear_head":
            raise ValueError("feature cache目前只支持linear_head phase")
        feature_cache = load_or_create_feature_cache(
            config=config,
            trainer_cfg=trainer_cfg,
            phase_cfg=phase_cfg,
            model=model,
            train_df=train_df,
            val_df=val_df,
            dataset_root=dataset_root,
            processor=processor,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        feature_train_loader = make_feature_dataloader(
            feature_table=feature_cache["train"],
            batch_size=int(trainer_cfg["batch_size"]),
            shuffle=True,
            drop_last=bool(trainer_cfg["drop_last"]),
            pin_memory=bool(trainer_cfg["pin_memory"]),
        )
        feature_val_loader = make_feature_dataloader(
            feature_table=feature_cache["val"],
            batch_size=int(trainer_cfg["batch_size"]),
            shuffle=False,
            drop_last=False,
            pin_memory=bool(trainer_cfg["pin_memory"]),
        )

    logger.info("开始phase=%s, mode=%s, epochs=%d", phase_name, phase_cfg["train_mode"], int(phase_cfg["epochs"]))
    for phase_epoch in range(int(phase_cfg["epochs"])):
        global_epoch = global_epoch_start + phase_epoch
        if use_feature_cache:
            train_metrics = train_feature_head_one_epoch(
                model=model,
                dataloader=feature_train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                amp_enabled=False,
                amp_dtype=amp_dtype,
                top_k_values=top_k_values,
            )
            eval_report, predictions = evaluate_feature_head(
                model=model,
                dataloader=feature_val_loader,
                feature_table=feature_cache["val"],
                labels=labels,
                device=device,
                amp_enabled=False,
                amp_dtype=amp_dtype,
                top_k_values=top_k_values,
            )
        else:
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                top_k_values=top_k_values,
            )
            eval_report, predictions = evaluate(
                model=model,
                dataloader=val_loader,
                labels=labels,
                device=device,
                top_k_values=top_k_values,
            )
        epoch_row = {
            "phase": phase_name,
            "phase_epoch": phase_epoch,
            "epoch": global_epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": eval_report["accuracy"],
            "val_macro_f1": eval_report["macro_f1"],
            "val_weighted_f1": eval_report["weighted_f1"],
            "train_n": train_metrics["n"],
            "val_n": eval_report["num_samples"],
        }
        for top_k in top_k_values:
            metric_name = f"top_{top_k}_accuracy"
            epoch_row[f"train_{metric_name}"] = train_metrics[metric_name]
            epoch_row[f"val_{metric_name}"] = eval_report["top_k_accuracy"][metric_name]
        if early_stopping_monitor not in epoch_row:
            raise ValueError(f"early stopping监控指标不存在: {early_stopping_monitor!r}")
        epoch_rows.append(epoch_row)
        pd.DataFrame(epoch_rows).to_csv(run_dir / trainer_cfg["epoch_report_file_name"], index=False)
        predictions.to_csv(run_dir / trainer_cfg["prediction_file_name"], index=False)
        write_json(run_dir / trainer_cfg["eval_report_file_name"], eval_report)

        checkpoint_path = run_dir / (
            f"{trainer_cfg['checkpoint_prefix']}_{phase_name}_epoch{phase_epoch:03d}.pt"
        )
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=global_epoch,
            config=trainer_cfg,
            metrics=epoch_row,
            labels=labels_payload,
        )
        monitor_value = float(epoch_row[early_stopping_monitor])
        if is_metric_improved(
            current_value=monitor_value,
            best_value=best_score,
            mode=early_stopping_mode,
            min_delta=early_stopping_min_delta,
        ):
            best_score = monitor_value
            best_epoch = global_epoch
            epochs_without_improvement = 0
            best_checkpoint_path = run_dir / f"{trainer_cfg['checkpoint_prefix']}_{phase_name}_best.pt"
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=global_epoch,
                config=trainer_cfg,
                metrics=epoch_row,
                labels=labels_payload,
            )
        else:
            epochs_without_improvement += 1
        completed_epochs += 1
        logger.info(
            "phase=%s epoch=%d train_loss=%.4f train_acc=%.4f val_acc=%.4f val_top_k=%s val_macro_f1=%.4f best_%s=%.4f stale_epochs=%d",
            phase_name,
            phase_epoch,
            float(train_metrics["loss"]),
            float(train_metrics["accuracy"]),
            float(eval_report["accuracy"]),
            ", ".join(
                f"top_{top_k}={float(eval_report['top_k_accuracy'][f'top_{top_k}_accuracy']):.4f}"
                for top_k in top_k_values
            ),
            float(eval_report["macro_f1"]),
            early_stopping_monitor,
            float(best_score) if best_score is not None else float("nan"),
            epochs_without_improvement,
        )
        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            logger.info(
                "phase=%s early stopping触发: monitor=%s, mode=%s, patience=%d, best_epoch=%s, best_score=%.4f",
                phase_name,
                early_stopping_monitor,
                early_stopping_mode,
                early_stopping_patience,
                best_epoch,
                float(best_score) if best_score is not None else float("nan"),
            )
            break

    return completed_epochs, {
        "phase": phase_name,
        "train_mode": phase_cfg["train_mode"],
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "stopped_early": stopped_early,
        "completed_epochs": completed_epochs,
        "use_feature_cache": use_feature_cache,
    }


def main(config: dict[str, Any] | None = None) -> None:
    if config is None:
        config = utils.load_pipeline_config()

    trainer_cfg = config["trainer"]
    set_seed(int(trainer_cfg["seed"]))
    dataset_root, metadata, labels = load_tables(config)
    train_df, val_df = split_metadata(metadata=metadata, trainer_cfg=trainer_cfg)
    device = get_torch_device(trainer_cfg["device"])
    image_size = validate_image_size(int(trainer_cfg["image_size"]))

    processor = AutoImageProcessor.from_pretrained(trainer_cfg["backbone_model_name"])
    train_loader = make_dataloader(
        metadata=train_df,
        dataset_root=dataset_root,
        processor=processor,
        trainer_cfg=trainer_cfg,
        train=True,
    )
    val_loader = make_dataloader(
        metadata=val_df,
        dataset_root=dataset_root,
        processor=processor,
        trainer_cfg=trainer_cfg,
        train=False,
    )

    model = BackboneLinearClassifier(
        backbone_model_name=trainer_cfg["backbone_model_name"],
        num_classes=int(labels["label_id"].astype(int).max()) + 1,
        freeze_backbone=bool(trainer_cfg["freeze_backbone"]),
    ).to(device)
    top_k_values = validate_top_k_values(
        top_k_values=list(trainer_cfg["top_k"]),
        num_classes=int(labels["label_id"].astype(int).max()) + 1,
    )

    criterion = nn.CrossEntropyLoss()
    run_dir = utils.join_data_root(trainer_cfg["output_dir"], config=config) / time.strftime(
        "%Y%m%d_%H%M%S",
        time.localtime(),
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info(
        "开始训练: train=%d, val=%d, labels=%d, device=%s, run_dir=%s",
        len(train_df),
        len(val_df),
        len(labels),
        device,
        run_dir,
    )

    labels_payload = labels.to_dict(orient="records")
    epoch_rows = []
    phase_summaries = []
    global_epoch = 0
    for phase_cfg in trainer_cfg["phases"]:
        completed_epochs, phase_summary = run_phase(
            phase_cfg=phase_cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_df=train_df,
            val_df=val_df,
            dataset_root=dataset_root,
            processor=processor,
            labels=labels,
            criterion=criterion,
            device=device,
            trainer_cfg=trainer_cfg,
            config=config,
            run_dir=run_dir,
            labels_payload=labels_payload,
            epoch_rows=epoch_rows,
            global_epoch_start=global_epoch,
            top_k_values=top_k_values,
        )
        global_epoch += completed_epochs
        phase_summaries.append(phase_summary)

    run_summary = {
        "run_dir": str(run_dir),
        "phase_summaries": phase_summaries,
        "total_completed_epochs": global_epoch,
        "num_train_samples": int(len(train_df)),
        "num_val_samples": int(len(val_df)),
        "num_classes": int(len(labels)),
        "top_k": top_k_values,
        "image_size": image_size,
    }
    write_json(run_dir / "run_summary.json", run_summary)
    update_latest_trainer_run_pointer(run_dir=run_dir, trainer_cfg=trainer_cfg)


if __name__ == "__main__":
    main()
