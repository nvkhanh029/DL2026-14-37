"""Train the 1D-CNN and LSTM on UCI-HAR and save test results.

Temporary training harness for the model/cnn-and-lstm branch. The shared
trainer (src/train.py, owned by the leader) may replace this later; the
loop below is kept short so any team member can explain it.

Usage (from the repository root):
    python scripts/train_cnn_lstm.py                # train cnn + lstm on acc_gyro
    python scripts/train_cnn_lstm.py --models cnn   # train only the CNN

Rules honoured (AGENTS.md / configs/shared.yaml):
- seed 42 for random / numpy / torch,
- model selection and early stopping use ONLY the validation split,
- the test split is evaluated exactly once, after training,
- each run writes results/<model>_<sensors>.json with the agreed fields.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn

# Allow `python scripts/train_cnn_lstm.py` to import from src/ (project root).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_processed  # noqa: E402
from src.models.cnn import CNN1D  # noqa: E402
from src.models.lstm import LSTMClassifier  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "shared.yaml"


def load_config() -> dict:
    """Read the shared fixed rules instead of hard-coding them."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    """Make runs as reproducible as possible on this machine."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, n_channels: int, n_classes: int) -> nn.Module:
    if name == "cnn":
        return CNN1D(n_channels=n_channels, n_classes=n_classes)
    if name == "lstm":
        return LSTMClassifier(n_channels=n_channels, n_classes=n_classes)
    raise ValueError(f"Unknown model {name!r} (expected 'cnn' or 'lstm')")


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device):
    """Return (y_pred, y_true) numpy arrays for a whole data loader."""
    model.eval()  # disables dropout; batches are not shuffled for val/test
    all_preds, all_trues = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_preds.append(logits.argmax(dim=1).cpu())
        all_trues.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_trues).numpy()


def train_one(
    name: str,
    sensors: str,
    *,
    epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    config: dict,
) -> dict:
    """Train one model with early stopping on validation macro-F1."""
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {name} on {sensors} | device: {device} ===")

    train_loader, val_loader, test_loader, n_channels = load_processed(
        sensors, batch_size=batch_size
    )
    n_classes = len(config["dataset"]["classes"])
    model = build_model(name, n_channels, n_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state: dict | None = None
    epochs_since_improve = 0
    epochs_run = 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        running_loss = 0.0
        for x, y in train_loader:  # train loader is shuffled by the data pipeline
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)

        # Model selection uses validation only, never the test split.
        val_pred, val_true = predict(model, val_loader, device)
        val_f1 = f1_score(val_true, val_pred, average="macro")
        val_acc = accuracy_score(val_true, val_pred)
        print(
            f"epoch {epoch:2d}/{epochs} | train loss {running_loss / len(train_loader.dataset):.4f}"
            f" | val acc {val_acc:.4f} | val macro-F1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"early stopping at epoch {epoch} (no val macro-F1 gain for {patience} epochs)")
                break

    train_time = time.perf_counter() - start

    # Restore the best validation checkpoint and evaluate the test split once.
    model.load_state_dict(best_state)
    test_pred, test_true = predict(model, test_loader, device)
    test_acc = accuracy_score(test_true, test_pred)
    test_f1 = f1_score(test_true, test_pred, average="macro")

    label_names = [c.upper() for c in config["dataset"]["classes"]]
    print(classification_report(test_true, test_pred, target_names=label_names, digits=4))
    print(f"TEST {name}_{sensors}: accuracy={test_acc:.4f} macro_f1={test_f1:.4f}")

    # Save the agreed result format and the test predictions (for the
    # visualization person's confusion matrices; predictions are ~12 KB).
    results_dir = ROOT / config["paths"]["results"]
    results_dir.mkdir(exist_ok=True)
    result = {
        "model": name,
        "sensors": sensors,
        "seed": config["seed"],
        "accuracy": round(float(test_acc), 4),
        "macro_f1": round(float(test_f1), 4),
        "epochs": int(epochs_run),
        "train_time_sec": round(float(train_time), 1),
    }
    out_path = results_dir / f"{name}_{sensors}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        results_dir / f"{name}_{sensors}_preds.npz", y_true=test_true, y_pred=test_pred
    )
    print(f"saved {out_path.relative_to(ROOT)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN and/or LSTM on UCI-HAR.")
    parser.add_argument(
        "--models", nargs="+", default=["cnn", "lstm"], choices=["cnn", "lstm"]
    )
    parser.add_argument("--sensors", default="acc_gyro", choices=["acc", "acc_gyro"])
    parser.add_argument("--epochs", type=int, default=30, help="max epochs (early stopping may stop sooner)")
    parser.add_argument("--patience", type=int, default=6, help="early-stopping patience on val macro-F1")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = load_config()
    for name in args.models:
        train_one(
            name,
            args.sensors,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size,
            config=config,
        )


if __name__ == "__main__":
    main()
