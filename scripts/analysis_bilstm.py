"""Ablation 2: bidirectional LSTM in the CNN-LSTM (Acc+Gyro, seed 42).

The only change from the main model is `bidirectional=True` in the LSTM. The
classifier reads the last hidden state of the forward pass and of the backward
pass (h_n), so it uses both directions over the whole window. (Taking the last
output step would give the backward direction only one sample of context.)
The src/models/cnn_lstm.py file is not modified; the variant lives here.

Both the unidirectional CNN-LSTM and the bidirectional variant are trained here
with the same seeds (42, 0, 1, 2), because a single seed is too noisy to compare
two models that differ by a few tenths of a point.

Usage (from the repository root):
    python scripts/analysis_bilstm.py
Output: results/analysis/bilstm.json (every run plus a mean/SD summary)
"""

import argparse

import numpy as np

import torch
import torch.nn as nn
from analysis_common import fit, evaluate, save_json, set_seed

from src.data import load_data
from src.models.cnn_lstm import CNNLSTM, CONV_CHANNELS, DROPOUT, HIDDEN, LSTM_LAYERS, N_CLASSES


class BiCNNLSTM(CNNLSTM):
    """Same CNN front end as the main model, bidirectional LSTM back end."""

    def __init__(self, c: int):
        super().__init__(c)
        self.rnn = nn.LSTM(CONV_CHANNELS[1], HIDDEN, LSTM_LAYERS, batch_first=True,
                           dropout=DROPOUT, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(2 * HIDDEN, N_CLASSES))

    def forward(self, x):
        z = self.features(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.rnn(z)  # h: (layers * 2, batch, hidden)
        # Last layer: forward state h[-2] and backward state h[-1].
        return self.head(torch.cat([h[-2], h[-1]], dim=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data("acc_gyro")
    rows = []
    for name, build in (("cnn_lstm", CNNLSTM), ("cnn_bilstm", BiCNNLSTM)):
        for seed in args.seeds:
            set_seed(seed)
            model = build(X_tr.shape[2])
            print(f"\n=== {name} | seed {seed} ===", flush=True)
            epochs, seconds = fit(model, X_tr, y_tr, X_va, y_va, epochs=args.epochs, verbose=False)
            metrics = evaluate(model, X_te, y_te)
            rows.append({"model": name, "sensors": "acc_gyro", "seed": seed, **metrics,
                         "epochs": epochs, "train_time_sec": round(seconds, 1),
                         "params": sum(p.numel() for p in model.parameters())})
            print(f"  TEST {metrics} | {epochs} epochs | {seconds:.0f}s", flush=True)
            save_json("bilstm.json", {"runs": rows, "summary": summarise(rows)})


def summarise(rows: list[dict]) -> list[dict]:
    """Mean and sample SD over seeds for each variant."""
    out = []
    for name in dict.fromkeys(r["model"] for r in rows):
        sel = [r for r in rows if r["model"] == name]
        acc, f1 = np.array([r["accuracy"] for r in sel]), np.array([r["macro_f1"] for r in sel])
        out.append({"model": name, "n_seeds": len(sel), "params": sel[0]["params"],
                    "accuracy_mean": round(float(acc.mean()), 4),
                    "accuracy_std": round(float(acc.std(ddof=1)), 4) if len(sel) > 1 else 0.0,
                    "macro_f1_mean": round(float(f1.mean()), 4),
                    "macro_f1_std": round(float(f1.std(ddof=1)), 4) if len(sel) > 1 else 0.0,
                    "epochs_mean": round(float(np.mean([r["epochs"] for r in sel])), 1),
                    "train_time_sec_mean": round(float(np.mean([r["train_time_sec"] for r in sel])), 1)})
    return out


if __name__ == "__main__":
    main()
