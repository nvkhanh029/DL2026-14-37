"""Train the CNN-LSTM main model on UCI-HAR and write the shared result JSON.

Usage (from the repository root):
    python scripts/train_main_model.py                 # acc + acc_gyro, seed 42
    python scripts/train_main_model.py --sensors acc   # one input version only
    python scripts/train_main_model.py --save-model    # also save checkpoints/cnn_lstm_<sensors>.pt
    python scripts/train_main_model.py --robustness    # extra 3-seed sweep (see below)
    python scripts/train_main_model.py --epochs 3      # smoke test -> results/smoke/ (never reported)

--save-model re-trains the model, so it also overwrites results/cnn_lstm_*.json
and results/preds/*.npz. Only machine D (the machine of the reported numbers)
should commit those files. On any other machine, run it for the checkpoint and
then `git restore results/` before committing.

Any run that changes the protocol (--epochs, --patience, --lr, --batch-size or a
--seed other than the config seed) is treated as a smoke test and written to
results/smoke/ instead, so it can never overwrite the reported results.

Protocol (shared with the CNN/LSTM baselines and fixed in configs/shared.yaml):
- Adam, learning rate 1e-3, batch size 64 (no weight decay, no LR schedule,
  no gradient clipping),
- at most 30 epochs, early stopping after 6 epochs without an improved
  validation macro-F1,
- the checkpoint with the best validation macro-F1 is restored, then the test
  split is evaluated exactly once,
- seed 42 for ``random``, ``numpy`` and ``torch``,
- each run writes ``results/<model>_<sensors>.json`` with exactly the seven
  agreed fields defined in configs/shared.yaml.

The optional ``--robustness`` mode re-runs seeds 0, 1 and 2 with the same
protocol and writes them to ``results/robustness/`` (seed in the file name).
It is an extra check on initialisation/shuffling variance only: the reported
result in the report is the single seed-42 run chosen by the fixed rule.

This temporary trainer lives in ``scripts/`` because ``src/train.py`` is owned
by the leader. It follows the same shape as the CNN/LSTM harness
(``scripts/train_cnn_lstm.py``) so all models are trained the same way.
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
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

# Allow `python scripts/train_main_model.py` to import from src/ (project root).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_processed  # noqa: E402
from src.models.cnn_lstm import CNNLSTM, describe  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "shared.yaml"
VERSIONS = ("acc", "acc_gyro")
# Defaults for the shared settings that are not part of configs/shared.yaml.
EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 1e-3
BATCH_SIZE = 64


def load_config() -> dict:
    """Read the shared fixed rules (seed, paths, classes) instead of hard-coding them."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    """Seed every RNG that affects a run, so a re-run on the same machine repeats it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, true labels) for a whole loader; never used during training."""
    model.eval()  # dropout off; validation/test batches are not shuffled
    predictions = []
    labels = []
    for features, targets in loader:
        logits = model(features.to(device))
        predictions.append(logits.argmax(dim=1).cpu())
        labels.append(targets)
    return torch.cat(predictions).numpy(), torch.cat(labels).numpy()


def train_one(
    sensors: str,
    seed: int,
    *,
    epochs: int,
    patience: int,
    learning_rate: float,
    batch_size: int,
    out_path: Path,
    preds_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> dict:
    """Train and evaluate one run: CNN-LSTM x sensor version x seed.

    Selection uses the validation split only; the test split is touched once,
    after the best-validation checkpoint has been restored.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== CNN-LSTM | {sensors} | seed {seed} | device {device} ===")

    train_loader, val_loader, test_loader, n_channels = load_processed(
        sensors, batch_size=batch_size
    )
    print(describe(n_channels))

    model = CNNLSTM(n_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    epochs_without_gain = 0
    epochs_run = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)

        # Model selection uses validation only, never the test split.
        val_pred, val_true = predict(model, val_loader, device)
        val_f1 = f1_score(val_true, val_pred, average="macro")
        val_acc = accuracy_score(val_true, val_pred)
        print(
            f"epoch {epoch:2d}/{epochs} | train loss "
            f"{running_loss / len(train_loader.dataset):.4f}"
            f" | val acc {val_acc:.4f} | val macro-F1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Copy the weights so the next epoch cannot overwrite the checkpoint.
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                print(f"early stopping at epoch {epoch}: no val macro-F1 gain for {patience} epochs")
                break

    train_time = time.perf_counter() - started

    # Restore the best-validation checkpoint and evaluate the test split once.
    model.load_state_dict(best_state)
    test_pred, test_true = predict(model, test_loader, device)
    test_acc = accuracy_score(test_true, test_pred)
    test_f1 = f1_score(test_true, test_pred, average="macro")

    # Optionally keep the best-validation weights (saved to checkpoints/ for later inference).
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)
        print(f"  saved -> {checkpoint_path.relative_to(ROOT)}")

    # Save the test predictions (true and predicted labels for the 2,947 test
    # windows) so the analysis/visualization person can build confusion matrices
    # without re-training. Saved for every reported run.
    if preds_path is not None:
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(preds_path, y_true=test_true, y_pred=test_pred)

    result = {
        "model": "cnn_lstm",
        "sensors": sensors,
        "seed": seed,
        "accuracy": round(float(test_acc), 4),
        "macro_f1": round(float(test_f1), 4),
        "epochs": int(epochs_run),
        "train_time_sec": round(float(train_time), 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"  val macro-F1 {best_val_f1:.4f} | test acc {test_acc:.4f} | "
          f"macro-F1 {test_f1:.4f} | {epochs_run} epochs | {train_time:.0f}s")
    print(f"  saved -> {out_path.relative_to(ROOT)}")
    if preds_path is not None:
        print(f"  saved -> {preds_path.relative_to(ROOT)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNN-LSTM main model on UCI-HAR.")
    parser.add_argument("--sensors", nargs="+", default=list(VERSIONS), choices=list(VERSIONS))
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default: the fixed seed in configs/shared.yaml)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--save-model", action="store_true",
                        help="save the best weights to checkpoints/cnn_lstm_<sensors>.pt")
    parser.add_argument("--robustness", action="store_true",
                        help="extra sweep over seeds 0, 1, 2, written to results/robustness/")
    args = parser.parse_args()

    config = load_config()
    results_dir = ROOT / config["paths"]["results"]
    seed = args.seed if args.seed is not None else int(config["seed"])

    # Guard: a run with a changed protocol must not overwrite the reported results.
    protocol = (EPOCHS, PATIENCE, LEARNING_RATE, BATCH_SIZE)
    changed = (args.epochs, args.patience, args.lr, args.batch_size) != protocol
    if changed or (seed != int(config["seed"]) and not args.robustness):
        results_dir = results_dir / "smoke"
        print(f"NOTE: protocol changed -> smoke test, results go to {results_dir.relative_to(ROOT)}/ "
              "and must not be used in the report.")

    if args.robustness:
        print("ROBUSTNESS sweep (seeds 0, 1, 2). The reported result is still the "
              "single seed-42 run; this only quantifies init/shuffling variance.")
        for sensors in args.sensors:
            for seed in (0, 1, 2):
                train_one(
                    sensors, seed,
                    epochs=args.epochs, patience=args.patience,
                    learning_rate=args.lr, batch_size=args.batch_size,
                    out_path=results_dir / "robustness" / f"cnn_lstm_{sensors}_seed{seed}.json",
                )
        return

    for sensors in args.sensors:
        train_one(
            sensors, seed,
            epochs=args.epochs, patience=args.patience,
            learning_rate=args.lr, batch_size=args.batch_size,
            out_path=results_dir / f"cnn_lstm_{sensors}.json",
            preds_path=results_dir / "preds" / f"cnn_lstm_{sensors}_preds.npz",
            checkpoint_path=(ROOT / "checkpoints" / f"cnn_lstm_{sensors}.pt"
                             if args.save_model else None),
        )


if __name__ == "__main__":
    main()
