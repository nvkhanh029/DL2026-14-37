"""Result figures for the report.

Reads every number from results/*.json (nothing is typed by hand), so the
figures always match the report tables.
Run from the repo root:  python src/plot_results.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MODEL_NAMES = {"rf": "Random Forest", "cnn": "1D-CNN", "lstm": "LSTM", "cnn_lstm": "CNN-LSTM"}
SENSOR_NAMES = {"acc": "Acc", "acc_gyro": "Acc+Gyro"}
MODEL_COLORS = {"rf": "#7f7f7f", "cnn": "#1f77b4", "lstm": "#2ca02c", "cnn_lstm": "#d62728"}
SENSOR_COLORS = {"acc": "#9ecae1", "acc_gyro": "#3182bd"}


def load_result(model, sensors):
    with open(RESULTS / f"{model}_{sensors}.json", encoding="utf-8") as f:
        return json.load(f)


def sensor_effect():
    """RF and CNN-LSTM, Acc vs Acc+Gyro (macro F1, one seed-42 run each)."""
    models = ["rf", "cnn_lstm"]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, s in enumerate(["acc", "acc_gyro"]):
        vals = [load_result(m, s)["accuracy"] for m in models]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=SENSOR_NAMES[s], color=SENSOR_COLORS[s])
        ax.bar_label(bars, fmt="%.4f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models])
    ax.set_ylabel("Accuracy (test, 0-1)")
    ax.set_ylim(0, 1)
    ax.set_title("Sensor effect: Acc vs Acc+Gyro (seed 42, single runs)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "sensor_effect.png", dpi=200)
    plt.close(fig)


def model_effect():
    """All four models on Acc+Gyro (macro F1, one seed-42 run each)."""
    models = ["rf", "cnn", "lstm", "cnn_lstm"]
    vals = [load_result(m, "acc_gyro")["accuracy"] for m in models]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([MODEL_NAMES[m] for m in models], vals,
                  color=[MODEL_COLORS[m] for m in models])
    ax.bar_label(bars, fmt="%.4f", fontsize=9)
    ax.set_ylabel("Accuracy (test, 0-1)")
    ax.set_ylim(0, 1)
    ax.set_title("Model effect on Acc+Gyro (seed 42, single runs)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "model_effect.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    sensor_effect()
    model_effect()
    print("saved figures/sensor_effect.png and figures/model_effect.png")