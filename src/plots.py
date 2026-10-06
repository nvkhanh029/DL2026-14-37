"""Visualization utilities for the Human Activity Recognition project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_data

CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

FIGURES_DIR = Path("figures")

def class_distribution() -> None:
    """Plot the class distribution across train, validation, and test splits."""

    _, y_train, _, y_val, _, y_test = load_data("acc_gyro")

    splits = {
        "Train": y_train,
        "Validation": y_val,
        "Test": y_test,
    }

    counts = {
        split: np.bincount(labels, minlength=len(CLASS_NAMES))
        for split, labels in splits.items()
    }

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, (split, values) in enumerate(counts.items()):
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=split,
        )
        ax.bar_label(bars, padding=3)

    ax.set_title("Class Distribution Across Dataset Splits")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.legend()

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    class_distribution()