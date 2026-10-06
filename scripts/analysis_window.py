"""Ablation 1: window size for the CNN-LSTM (Acc+Gyro, seed 42).

The dataset is supplied as fixed 128-sample windows, so shorter windows are made
by cutting each window into non-overlapping pieces (64 -> 2 pieces, 32 -> 4
pieces, 16 -> 8 pieces). The model is trained on the pieces. At test time the
softmax outputs of the pieces of one window are averaged, so every size is scored
on the same 2,947 test windows and the numbers are directly comparable.
Windows longer than 128 are not possible without re-segmenting the raw signal.

A single seed turned out to be too noisy for this comparison (one init can move
the accuracy by 2-3 points), so every size is trained with several seeds and the
mean and standard deviation are reported. The seed-42 run of size 128 is NOT
identical to the reported main result, because the main script builds the model
twice (once for describe()) before training, which shifts the random stream.

Usage (from the repository root):
    python scripts/analysis_window.py
Output: results/analysis/window_size.json (every run plus a mean/SD summary per size)
"""

import argparse

import numpy as np

from analysis_common import fit, evaluate, save_json, set_seed, split_windows

from src.data import load_data
from src.models.cnn_lstm import CNNLSTM

SIZES = (16, 32, 64, 128)


def summarise(rows: list[dict]) -> list[dict]:
    """Mean and sample SD over seeds for each window size."""
    out = []
    for size in sorted({r["window_size"] for r in rows}):
        sel = [r for r in rows if r["window_size"] == size]
        acc, f1 = np.array([r["accuracy"] for r in sel]), np.array([r["macro_f1"] for r in sel])
        out.append({"window_size": size, "n_seeds": len(sel),
                    "accuracy_mean": round(float(acc.mean()), 4),
                    "accuracy_std": round(float(acc.std(ddof=1)), 4) if len(sel) > 1 else 0.0,
                    "macro_f1_mean": round(float(f1.mean()), 4),
                    "macro_f1_std": round(float(f1.std(ddof=1)), 4) if len(sel) > 1 else 0.0,
                    "epochs_mean": round(float(np.mean([r["epochs"] for r in sel])), 1),
                    "train_time_sec_mean": round(float(np.mean([r["train_time_sec"] for r in sel])), 1)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data("acc_gyro")
    rows = []
    for size in args.sizes:
        k = 128 // size  # pieces per original window
        for seed in args.seeds:
            print(f"\n=== window size {size} ({k} piece(s)) | seed {seed} ===", flush=True)
            set_seed(seed)
            model = CNNLSTM(X_tr.shape[2], window=size)
            # Labels are repeated for every piece of a training window.
            epochs, seconds = fit(model, split_windows(X_tr, size), y_tr.repeat(k),
                                  split_windows(X_va, size), y_va, pieces=k,
                                  epochs=args.epochs, verbose=False)
            metrics = evaluate(model, split_windows(X_te, size), y_te, pieces=k)
            rows.append({"model": "cnn_lstm", "sensors": "acc_gyro", "seed": seed,
                         "window_size": size, **metrics, "epochs": epochs,
                         "train_time_sec": round(seconds, 1)})
            print(f"  TEST {metrics} | {epochs} epochs | {seconds:.0f}s", flush=True)
            save_json("window_size.json", {"runs": rows, "summary": summarise(rows)})


if __name__ == "__main__":
    main()
