"""Create the subject-wise UCI HAR train/validation/test datasets."""

from __future__ import annotations

import argparse
import itertools
import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

if __package__:
    from .data import CHANNELS, DEFAULT_DATA_DIR
else:
    from data import CHANNELS, DEFAULT_DATA_DIR

WINDOW_SIZE = 128
ACTIVITY_COUNT = 6
EXPECTED_OFFICIAL_COUNTS = {"train": 7352, "test": 2947}
RAW_DATASET_DIRNAME = "UCI HAR Dataset"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)


def prepare_datasets(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    val_seed: int = 42,
) -> dict[str, object]:
    """Prepare Acc and Acc+Gyro NPZ files and return their summary."""
    data_dir = Path(data_dir)
    raw_dir = _ensure_raw_dataset(data_dir)
    official = {
        split: _read_official_split(raw_dir, split)
        for split in ("train", "test")
    }

    expected_labels = set(range(ACTIVITY_COUNT))
    for split, (features, labels, subjects) in official.items():
        if features.shape[0] != EXPECTED_OFFICIAL_COUNTS[split]:
            raise ValueError(
                f"Expected {EXPECTED_OFFICIAL_COUNTS[split]} official {split} "
                f"windows; found {features.shape[0]}"
            )
        if features.shape[1] != WINDOW_SIZE:
            raise ValueError(f"Expected windows of {WINDOW_SIZE} samples, got {features.shape}")
        if set(np.unique(labels)) != expected_labels:
            raise ValueError(f"The official {split} split does not contain all six activities")
        if not np.isfinite(features).all():
            raise ValueError(f"The official {split} split contains non-finite sensor values")
        if len(subjects) != len(labels):
            raise ValueError(f"Subject and label counts differ in the official {split} split")

    raw_train = official["train"]
    # Hold out complete subjects so overlapping windows cannot cross split boundaries.
    val_subjects = _select_validation_subjects(
        raw_train[1], raw_train[2], seed=val_seed
    )
    val_mask = np.isin(raw_train[2], val_subjects)
    train_mask = ~val_mask
    if not train_mask.any() or not val_mask.any():
        raise ValueError("The subject-wise train/validation split produced an empty split")

    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "dataset": "UCI Human Activity Recognition Using Smartphones",
        "window_size": WINDOW_SIZE,
        "sampling_rate_hz": 50,
        "label_names": [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING",
        ],
        "val_seed": val_seed,
        "versions": {},
    }

    for version, channels in CHANNELS.items():
        selected = _select_channels(raw_dir, official, channels)
        split_arrays = {
            "train": (selected["train"][0][train_mask], selected["train"][1][train_mask],
                      selected["train"][2][train_mask]),
            "val": (selected["train"][0][val_mask], selected["train"][1][val_mask],
                    selected["train"][2][val_mask]),
            "test": selected["test"],
        }
        # Fit normalization only on training windows to avoid validation/test leakage.
        mean, std = _training_statistics(split_arrays["train"][0])
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std == 0).any():
            raise ValueError(f"Invalid training normalization statistics for {version}")

        payload: dict[str, np.ndarray] = {
            "mean": mean,
            "std": std,
            "channels": np.asarray(channels, dtype=np.str_),
            "val_seed": np.asarray(val_seed, dtype=np.int64),
            "val_subjects": np.asarray(val_subjects, dtype=np.int16),
        }
        split_summary: dict[str, object] = {}
        for split, (features, labels, subjects) in split_arrays.items():
            normalized = ((features - mean) / std).astype(np.float32, copy=False)
            payload[f"X_{split}"] = normalized
            payload[f"y_{split}"] = labels.astype(np.int16, copy=False)
            payload[f"subj_{split}"] = subjects.astype(np.int16, copy=False)
            _validate_prepared_split(split, normalized, labels, subjects)
            split_summary[split] = {
                "windows": int(len(labels)),
                "subjects": sorted(int(subject) for subject in np.unique(subjects)),
                "class_counts": [
                    int(np.count_nonzero(labels == label))
                    for label in range(ACTIVITY_COUNT)
                ],
            }

        _validate_no_subject_leakage(split_arrays)
        train_mean = payload["X_train"].mean(axis=(0, 1), dtype=np.float64)
        train_std = payload["X_train"].std(axis=(0, 1), dtype=np.float64)
        if not np.allclose(train_mean, 0.0, atol=1e-5) or not np.allclose(
            train_std, 1.0, atol=1e-5
        ):
            raise ValueError(f"Training normalization check failed for {version}")

        output_path = output_dir / f"{version}.npz"
        np.savez_compressed(output_path, **payload)
        summary["versions"][version] = {
            "file": output_path.name,
            "channels": list(channels),
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
            "splits": split_summary,
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

# Reads 1 official split's data from the raw UCI HAR dataset and returns the features, labels, and subjects.
def _read_official_split(
    raw_dir: Path, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_dir = raw_dir / split
    labels = _read_vector(split_dir / f"y_{split}.txt", np.int64) - 1
    subjects = _read_vector(split_dir / f"subject_{split}.txt", np.int64)
    total_acc = _read_signals(split_dir, split, CHANNELS["acc"])
    if len(labels) != len(subjects) or len(labels) != len(total_acc):
        raise ValueError(f"Signals, labels, and subjects have different counts in {split}")
    return total_acc, labels, subjects

# Ensures the raw UCI HAR dataset is present in the given directory, extracting it from a local archive if necessary, and returns the path to the raw dataset directory.
def _ensure_raw_dataset(data_dir: Path) -> Path:
    dataset_dir = data_dir / RAW_DATASET_DIRNAME
    if _has_raw_dataset(dataset_dir):
        return dataset_dir

    archive_path = data_dir / "uci_har.zip"
    if not archive_path.is_file():
        _download_archive(archive_path)

    _extract_archive(archive_path, data_dir)
    nested_archive = data_dir / f"{RAW_DATASET_DIRNAME}.zip"
    if not _has_raw_dataset(dataset_dir) and nested_archive.is_file():
        _extract_archive(nested_archive, data_dir)
    if not _has_raw_dataset(dataset_dir):
        raise RuntimeError(
            f"Archive did not contain the expected UCI HAR dataset at {dataset_dir}"
        )
    return dataset_dir


def _download_archive(archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = archive_path.with_name(f"{archive_path.name}.download")
    try:
        urllib.request.urlretrieve(DATASET_URL, temporary_path)
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)


# Validates that all paths in the UCI HAR archive are safe and do not contain any unsafe characters or patterns.
def _validate_archive_paths(archive: zipfile.ZipFile) -> None:
    for member in archive.infolist():
        name = member.filename
        path = PurePosixPath(name)
        if "\\" in name or path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe path in local UCI HAR archive: {name!r}")

# Extracts the UCI HAR archive to the specified destination directory, validating paths and handling errors.
def _extract_archive(archive_path: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _validate_archive_paths(archive)
            archive.extractall(destination)
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"Unable to extract local UCI HAR archive: {error}") from error

# Checks if the raw UCI HAR dataset is present in the specified directory by verifying the existence of required files.
def _has_raw_dataset(dataset_dir: Path) -> bool:
    required = (
        dataset_dir / "train" / "y_train.txt",
        dataset_dir / "train" / "subject_train.txt",
        dataset_dir / "test" / "y_test.txt",
    )
    return all(path.is_file() for path in required)

# Selects the specified channels from the official UCI HAR dataset and returns a dictionary containing the selected features, labels, and subjects for each split.
def _select_channels(
    raw_dir: Path,
    official: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    channels: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    selected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in ("train", "test"):
        signals = _read_signals(raw_dir / split, split, channels)
        _, labels, subjects = official[split]
        if len(signals) != len(labels):
            raise ValueError(f"Signal and label counts differ in {split}")
        selected[split] = (signals, labels, subjects)
    return selected

# Reads the specified channels' signals from the UCI HAR dataset for a given split and returns them as a stacked NumPy array.
def _read_signals(split_dir: Path, split: str, channels: tuple[str, ...]) -> np.ndarray:
    axes = []
    expected_windows: int | None = None
    for channel in channels:
        path = split_dir / "Inertial Signals" / f"{channel}_{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Required UCI HAR signal file is missing: {path}")
        values = np.loadtxt(path, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != WINDOW_SIZE:
            raise ValueError(f"Expected {path} to have shape (N, {WINDOW_SIZE}), got {values.shape}")
        if expected_windows is not None and len(values) != expected_windows:
            raise ValueError(f"Signal files have different window counts in {split}")
        expected_windows = len(values)
        axes.append(values)
    return np.stack(axes, axis=-1)

# Reads a vector from a UCI HAR dataset file and returns it as a NumPy array of the specified dtype.
def _read_vector(path: Path, dtype: type[np.generic]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required UCI HAR metadata file is missing: {path}")
    return np.asarray(np.loadtxt(path, dtype=dtype)).reshape(-1)

# Selects a subject-wise validation split from the official training set, ensuring that all six activities are present in both the training and validation splits.
def _select_validation_subjects(
    labels: np.ndarray, subjects: np.ndarray, *, seed: int
) -> list[int]:
    subject_ids = np.unique(subjects)
    validation_count = max(1, round(len(subject_ids) * 0.2))
    rng = np.random.default_rng(seed)
    shuffled_subjects = rng.permutation(subject_ids).tolist()
    all_classes = set(range(ACTIVITY_COUNT))
    for candidate in itertools.combinations(shuffled_subjects, validation_count):
        validation_mask = np.isin(subjects, candidate)
        validation_classes = set(np.unique(labels[validation_mask]))
        training_classes = set(np.unique(labels[~validation_mask]))
        if validation_classes == all_classes and training_classes == all_classes:
            return sorted(int(subject) for subject in candidate)
    raise ValueError(
        "Could not find a subject-wise validation split with all six activities "
        f"for seed {seed}"
    )

# Computes the mean and standard deviation of the training features across all windows and time steps, returning them as NumPy arrays.
def _training_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = features.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    return mean, std

# Validates the prepared split's features, labels, and subjects to ensure they meet the expected criteria for shape, counts, and values.
def _validate_prepared_split(
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> None:
    if features.ndim != 3 or features.shape[1] != WINDOW_SIZE:
        raise ValueError(f"Unexpected {split} feature shape: {features.shape}")
    if len(features) != len(labels) or len(labels) != len(subjects):
        raise ValueError(f"Window, label, and subject counts differ in {split}")
    if not np.isfinite(features).all():
        raise ValueError(f"Normalized {split} features contain non-finite values")
    if set(np.unique(labels)) != set(range(ACTIVITY_COUNT)):
        raise ValueError(f"The {split} split does not contain all six activities")

# Validates that no subject appears in more than one data split (train, validation, test) to prevent data leakage.
def _validate_no_subject_leakage(
    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> None:
    train_subjects = set(np.unique(splits["train"][2]))
    val_subjects = set(np.unique(splits["val"][2]))
    test_subjects = set(np.unique(splits["test"][2]))
    if train_subjects & val_subjects or train_subjects & test_subjects or val_subjects & test_subjects:
        raise ValueError("A subject appears in more than one data split")

# Main entry point for the script, handling command-line arguments and initiating the dataset preparation process.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download UCI HAR if needed and prepare Acc and Acc+Gyro NPZ files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the raw dataset and receiving processed files.",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=42,
        help="Seed used to choose validation subjects from the official training set.",
    )
    args = parser.parse_args()
    summary = prepare_datasets(args.data_dir, val_seed=args.val_seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
