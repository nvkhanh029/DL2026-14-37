"""Load the prepared UCI HAR datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DatasetVersion = Literal["acc", "acc_gyro"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
CHANNELS: dict[DatasetVersion, tuple[str, ...]] = {
    "acc": ("total_acc_x", "total_acc_y", "total_acc_z"),
    "acc_gyro": (
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
    ),
}


def load_processed(
    version: DatasetVersion,
    batch_size: int = 64,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load processed train, validation, and test batches from an NPZ file."""
    if version not in CHANNELS:
        raise ValueError(f"version must be one of {tuple(CHANNELS)}, got {version!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    processed_path = Path(data_dir) / "processed" / f"{version}.npz"
    if not processed_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_path}. "
            "Run `python src/prepare_data.py` from the project root first."
        )

    with np.load(processed_path, allow_pickle=False) as stored:
        datasets = []
        for split in ("train", "val", "test"):
            features_key, labels_key = f"X_{split}", f"y_{split}"
            if features_key not in stored or labels_key not in stored:
                raise ValueError(f"{processed_path} is missing {features_key} or {labels_key}")
            features = np.array(stored[features_key], dtype=np.float32, copy=True)
            labels = np.array(stored[labels_key], dtype=np.int64, copy=True)
            if features.ndim != 3 or features.shape[1:] != (128, len(CHANNELS[version])):
                raise ValueError(f"Unexpected {features_key} shape: {features.shape}")
            if labels.shape != (features.shape[0],):
                raise ValueError(f"Unexpected {labels_key} shape: {labels.shape}")
            datasets.append(TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)))

    train, val, test = (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, dataset in zip(("train", "val", "test"), datasets)
    )
    return train, val, test, len(CHANNELS[version])
