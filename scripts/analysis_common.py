"""Shared helpers for the three CNN-LSTM analysis scripts (owner: analysis person).

Every analysis trains the CNN-LSTM with the same protocol as the main run
(configs/shared.yaml, scripts/train_main_model.py): Adam 1e-3, batch 64, at most
30 epochs, early stopping with patience 6 on validation macro-F1, best
validation checkpoint restored, seed 42. The test split is never used for
selection.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets `python scripts/analysis_*.py` import from src/

CONFIG = yaml.safe_load((ROOT / "configs" / "shared.yaml").read_text(encoding="utf-8"))
SEED = int(CONFIG["seed"])
OUT_DIR = ROOT / CONFIG["paths"]["analysis_results"]
EPOCHS, PATIENCE, LR, BATCH = 30, 6, 1e-3, 64  # same as the main run
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG that affects a run (random, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_windows(X: np.ndarray, size: int) -> np.ndarray:
    """Cut every 128-sample window into 128/size non-overlapping pieces.

    (N, 128, C) -> (N * k, size, C). The k pieces of one window stay next to
    each other, so predictions can be averaged back to one per window.
    """
    n, t, c = X.shape
    assert t % size == 0, "window size must divide 128"
    return X.reshape(n, t // size, size, c).reshape(-1, size, c)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, pieces: int = 1) -> np.ndarray:
    """Softmax probabilities per original window (mean over its `pieces` sub-windows)."""
    model.eval()
    probs = []
    for i in range(0, len(X), 512):
        batch = torch.from_numpy(X[i:i + 512]).to(DEVICE)
        probs.append(torch.softmax(model(batch), dim=1).cpu())
    probs = torch.cat(probs).numpy()
    return probs.reshape(-1, pieces, probs.shape[1]).mean(axis=1)


def fit(model, X_train, y_train, X_val, y_val, *, pieces: int = 1,
        epochs: int = EPOCHS, patience: int = PATIENCE, verbose: bool = True):
    """Train with early stopping on validation macro-F1; restore the best weights.

    `y_val` has one label per original window; `X_val` holds `pieces` sub-windows
    per window. Returns (epochs_run, train_time_sec).
    """
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    data = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                          torch.from_numpy(y_train.astype(np.int64)))
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH, shuffle=True)

    best_f1, best_state, stale, epochs_run = -1.0, None, 0, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        for features, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(features.to(DEVICE)), targets.to(DEVICE))
            loss.backward()
            optimizer.step()

        # Model selection uses validation only, never the test split.
        pred = predict_proba(model, X_val, pieces).argmax(axis=1)
        val_f1 = f1_score(y_val, pred, average="macro")
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs} | val acc {accuracy_score(y_val, pred):.4f}"
                  f" | val macro-F1 {val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return epochs_run, time.perf_counter() - started


def evaluate(model, X_test, y_test, pieces: int = 1) -> dict:
    """Test accuracy and macro-F1 (call once, after fit)."""
    pred = predict_proba(model, X_test, pieces).argmax(axis=1)
    return {"accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4)}


def save_json(name: str, payload) -> Path:
    """Write results/analysis/<name>.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
