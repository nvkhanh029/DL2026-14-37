"""
Random Forest baseline for the UCI-HAR experiment.

This script assumes the project's shared data loader already handles:
    - official UCI-HAR train/test split
    - subject-wise validation split
    - 128-sample windows
    - accelerometer / gyroscope channel selection
    - per-channel z-score normalization fitted on the training set

Expected data_loader.py interface:

    load_data(sensors, config_path="shared.yaml")

returning:

    {
        "X_train": np.ndarray,  # (N, 128, C)
        "y_train": np.ndarray,  # (N,)
        "X_val":   np.ndarray,  # (N, 128, C)
        "y_val":   np.ndarray,  # (N,)
        "X_test":  np.ndarray,  # (N, 128, C)
        "y_test":  np.ndarray,  # (N,)
    }

The RF itself performs only handcrafted feature extraction and classification.
It does NOT perform another split, windowing step, or normalization step.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import torch
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

EXPECTED_CHANNELS = {
    "acc": 3,
    "acc_gyro": 6,
}

CLASS_NAMES = [
    "walking",
    "upstairs",
    "downstairs",
    "sitting",
    "standing",
    "laying",
]

FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "energy",
]


# ---------------------------------------------------------------------------
# Configuration / reproducibility
# ---------------------------------------------------------------------------

def load_config(config_path: str = "shared.yaml") -> dict:
    """Load the shared YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration in {config_path}")

    return config


def seed_everything(seed: int) -> None:
    """Set the project's Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------------------------------------------------------------------
# Handcrafted feature extraction
# ---------------------------------------------------------------------------

def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Convert sensor windows into handcrafted features.

    Input:
        X.shape = (N, 128, C)

    Output:
        features.shape = (N, C * 5)

    Five features are calculated independently for every channel:
        mean
        standard deviation
        minimum
        maximum
        energy

    Energy is defined as mean(x^2).
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to have shape (N, 128, C), got {X.shape}"
        )

    if X.shape[1] != 128:
        raise ValueError(
            f"Expected 128 time steps per window, got {X.shape[1]}"
        )

    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    minimum = np.min(X, axis=1)
    maximum = np.max(X, axis=1)
    energy = np.mean(X ** 2, axis=1)

    # Each item has shape (N, C).
    # Stack along the last axis -> (N, C, 5).
    features = np.stack(
        [
            mean,
            std,
            minimum,
            maximum,
            energy,
        ],
        axis=2,
    )

    # Flatten C x 5 into one feature vector per window.
    # Feature order for each channel is:
    # mean, std, min, max, energy
    return features.reshape(features.shape[0], -1)


# ---------------------------------------------------------------------------
# Data-loader validation
# ---------------------------------------------------------------------------

def validate_loaded_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensors: str,
) -> None:
    """Validate data returned by the shared data loader."""

    expected_channels = EXPECTED_CHANNELS[sensors]

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for split, (X, y) in splits.items():
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(
                f"X_{split} must have shape (N, 128, C), got {X.shape}"
            )

        if X.shape[1] != 128:
            raise ValueError(
                f"X_{split} must contain 128 time steps, got {X.shape[1]}"
            )

        if X.shape[2] != expected_channels:
            raise ValueError(
                f"{sensors} expects {expected_channels} channels, "
                f"but X_{split} contains {X.shape[2]}"
            )

        if len(X) != len(y):
            raise ValueError(
                f"X_{split} and y_{split} have different numbers of samples: "
                f"{len(X)} vs {len(y)}"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                f"X_{split} contains NaN or infinite values"
            )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                f"y_{split} contains NaN or infinite values"
            )


# ---------------------------------------------------------------------------
# RF training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensors: str,
    seed: int,
    n_estimators: int = 300,
) -> dict:
    """
    Extract features, train the RF on the training set, and evaluate it.

    The validation set is reported for inspection.
    The required result JSON contains the test metrics.
    """
    validate_loaded_data(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    sensors,
    )

    seed_everything(seed)

    # Convert raw windows to handcrafted feature vectors.
    X_train = extract_features(X_train)
    X_val = extract_features(X_val)
    X_test = extract_features(X_test)

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )

    # Training time means RF fitting time, not data-loading time.
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - start

    # Validation evaluation.
    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )

    # Final test evaluation.
    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )

    print()
    print(f"Random Forest: {sensors}")
    print(f"Training windows:   {len(y_train)}")
    print(f"Validation windows: {len(y_val)}")
    print(f"Test windows:       {len(y_test)}")
    print(f"Features/window:    {X_train.shape[1]}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro-F1: {val_macro_f1:.4f}")
    print(f"Test accuracy:       {test_accuracy:.4f}")
    print(f"Test macro-F1:       {test_macro_f1:.4f}")
    print(f"Training time:       {train_time_sec:.4f} sec")

    # Exactly the fields required by shared.yaml.
    return {
        "model": "rf",
        "sensors": sensors,
        "seed": int(seed),
        "accuracy": float(test_accuracy),
        "macro_f1": float(test_macro_f1),
        "epochs": 0,
        "train_time_sec": float(train_time_sec),
    }


# ---------------------------------------------------------------------------
# Result output
# ---------------------------------------------------------------------------

def save_result(result: dict, results_dir: str = "results") -> Path:
    """Write the required RF result JSON."""
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"rf_{result['sensors']}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="UCI-HAR handcrafted-feature Random Forest baseline"
    )

    parser.add_argument(
        "--sensors",
        choices=["acc", "acc_gyro"],
        required=True,
        help="Sensor configuration for this RF run.",
    )

    parser.add_argument(
        "--config",
        default="configs/shared.yaml",
        help="Path to the shared configuration file.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in the Random Forest.",
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for result JSON files.",
    )

    args = parser.parse_args()

    if args.n_estimators <= 0:
        raise ValueError("--n-estimators must be greater than zero")

    config = load_config(args.config)
    if "seed" not in config:
        raise KeyError("Missing 'seed' in shared configuration")

    seed = int(config["seed"])

    # The shared data loader is deliberately imported only when the script
    # is run. This allows this RF module to be developed before DATA.md and
    # the loader are finished.
    try:
        from ..data import load_data
    except ImportError as exc:
        raise ImportError(
            "Could not import load_data from data_loader.py.\n"
            "Expected interface:\n\n"
            "    load_data(sensors)\n\n"
            "The shared data-loader implementation is not part of this "
            "prototype yet."
        ) from exc

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
    sensors=args.sensors
    )

    result = train_and_evaluate(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        sensors=args.sensors,
        seed=seed,
        n_estimators=args.n_estimators,
    )

    output_path = save_result(result, args.results_dir)

    print(f"Saved result: {output_path}")


if __name__ == "__main__":
    main()