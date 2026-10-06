"""Train and evaluate one configuration: model x sensor version x seed.

Only what the main model needs is implemented here: the CNN-LSTM of Section 4 on the two input
versions, with the protocol fixed in Table 5 of the Main Method. The loop is deliberately small
and self-contained so the numbers in the report can be traced to a single file.

Protocol
--------
Adam, learning rate 1e-3, weight decay 1e-4, batch size 64, at most `--epochs` epochs, gradient
norm clipping at 1.0, `ReduceLROnPlateau` halving the learning rate after 3 epochs without
validation-loss improvement, and early stopping on validation accuracy after `--patience` epochs
of no gain. The weights from the best validation epoch are restored before the test set is
touched, and the test set is then evaluated exactly once per run.

Each seed fixes the weight initialisation and the batch order. The train/validation subject
split is fixed once by `src/prepare_data.py` (see DATA.md) and does not vary with the seed, so a
seed re-draw measures initialisation and shuffling variance only.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__:
    from .data import CHANNELS, load_processed
else:
    from data import CHANNELS, load_processed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 6
VERSIONS = ("acc", "acc_gyro")
# "Normal" is the configuration reported in the Main Method (Table 5). "Fast" exists only for
# smoke tests: it trains fewer epochs on one seed and must never be reported.
PRESETS = {
    "normal": {"epochs": 40, "patience": 8, "seeds": [0, 1, 2]},
    "fast": {"epochs": 15, "patience": 3, "seeds": [0]},
}


def set_seed(seed: int) -> None:
    """Make a run repeatable: same seed gives same initialisation and same batch order.

    Every RNG that affects the result is seeded: `random` and NumPy for data-side shuffling,
    torch for weight init and the DataLoader's shuffling, and CUDA for kernels run on the GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """One pass over a loader. Trains when an optimizer is given, otherwise only evaluates.

    Returns average loss, accuracy, predictions and true labels.
    """
    training = optimizer is not None
    model.train(training)                   # toggles dropout and batch-norm behaviour
    # Keep the loss sum on the compute device: converting to a Python float every batch would
    # force a device-to-host synchronisation in the inner loop.
    total_loss = torch.zeros((), device=device)
    predictions = []
    labels = []

    with torch.set_grad_enabled(training):
        for features, labels_batch in loader:
            # BatchNorm1d cannot form batch statistics from a single window, and a DataLoader
            # with drop_last=False can emit a final batch of size 1. Skipping it costs at most
            # one window per epoch; without this guard the run aborts on such a batch.
            if training and features.size(0) < 2:
                continue
            features = features.to(device)
            labels_batch = labels_batch.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels_batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.detach() * len(labels_batch)
            # detach: otherwise the argmax tensors keep the whole epoch's autograd graph alive.
            predictions.append(logits.detach().argmax(1))
            labels.append(labels_batch)

    predictions = torch.cat(predictions).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    accuracy = float((predictions == labels).mean())
    return (total_loss / len(labels)).item(), accuracy, predictions, labels


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Macro-averaged F1 over the six classes, computed without a sklearn dependency.

    Macro rather than weighted: every class counts equally, so a class the model fails on pulls
    the score down instead of being hidden by the three large dynamic classes.
    """
    scores = []
    for class_id in range(N_CLASSES):
        true_positive = int(((predictions == class_id) & (labels == class_id)).sum())
        false_positive = int(((predictions == class_id) & (labels != class_id)).sum())
        false_negative = int(((predictions != class_id) & (labels == class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        # A class absent from the test set has no denominator; count it as zero rather than NaN.
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    """Confusion matrix as nested lists, rows true, columns predicted."""
    # Flattening each (true, predicted) pair to true * K + predicted lets bincount do the tally,
    # which avoids a Python loop over every window.
    counts = np.bincount(labels * N_CLASSES + predictions, minlength=N_CLASSES ** 2)
    return counts.reshape(N_CLASSES, N_CLASSES).tolist()


def train_and_evaluate(
    model: nn.Module,
    version: str,
    seed: int,
    *,
    epochs: int = 40,
    patience: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    data_dir: str | Path | None = None,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Train one model on one sensor version with one seed and score it once on the test set."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    # Only pass data_dir when it was given, so the loader keeps its own default otherwise.
    extra = {} if data_dir is None else {"data_dir": data_dir}
    train_loader, val_loader, test_loader, channels = load_processed(
        version, batch_size=batch_size, **extra
    )

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # Halve the learning rate once the validation loss has stalled for 3 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_loss = float("inf")
    best_weights = None
    epochs_without_gain = 0
    started = time.time()

    for _ in range(epochs):
        # Train, then score on validation; the test loader is deliberately untouched here.
        train_loss, _, _, _ = _epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc, _, _ = _epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Better accuracy wins; an exact tie is broken by lower validation loss.
        # deepcopy the weights, not the model: the next epoch would otherwise overwrite them.
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:      # no progress for `patience` epochs
                break
    train_time = time.time() - started

    # Restore the best-validation checkpoint before the single test evaluation.
    if best_weights is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_weights)
    _, test_acc, predictions, labels = _epoch(model, test_loader, loss_fn, device)

    return {
        "model": "cnnlstm_main",
        "version": version,
        "seed": seed,
        "channels": channels,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs_run": len(history["val_acc"]),
        "train_time": train_time,
        "val_acc": best_acc,
        "test_acc": test_acc,
        "test_f1": _macro_f1(labels, predictions),
        "confusion_matrix": _confusion(labels, predictions),
        "history": history,
    }


def save_result(result: dict[str, object], out_dir: str | Path | None = None) -> Path:
    """Write one run's metrics to `<out_dir>/main/runs/<model>__<version>__s<seed>.json`.

    Defaults to `results/main/runs`, which is where src/report_main_model.py reads from.
    """
    # One file per run; report_main_model.py reads them back by globbing this directory.
    runs_dir = Path(out_dir or PROJECT_ROOT / "results") / "main" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{result['model']}__{result['version']}__s{result['seed']}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface. The normal preset matches Table 5 of the Main Method."""
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train the CNN-LSTM main model on the Acc and Acc+Gyro input versions.",
    )
    parser.add_argument("--versions", nargs="+", default=list(VERSIONS), choices=list(VERSIONS),
                        help=f"input versions to train (default: {' '.join(VERSIONS)})")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"random seeds (default: {PRESETS['normal']['seeds']})")
    parser.add_argument("--epochs", type=int, default=None,
                        help="maximum epochs; early stopping may end sooner "
                             f"(default: {PRESETS['normal']['epochs']})")
    parser.add_argument("--patience", type=int, default=None,
                        help="epochs without a validation gain before stopping "
                             f"(default: {PRESETS['normal']['patience']})")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fast", action="store_true",
                        help="smoke test only: 15 epochs, patience 3, seed 0. "
                             "Not reportable, and written to results/main_fast "
                             "so it cannot disturb a real run.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="directory holding processed/*.npz (default: <project>/data)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="results root; runs land in <out-dir>/main/runs "
                             "(default: <project>/results)")
    return parser


def resolve_settings(args: argparse.Namespace) -> tuple[int, int, list[int], Path]:
    """Fill in preset values for anything not given explicitly.

    Returns (epochs, patience, seeds, results_root).
    """
    # An explicit flag always beats the preset; the preset only fills in what was left unset.
    preset = PRESETS["fast" if args.fast else "normal"]
    epochs = args.epochs if args.epochs is not None else preset["epochs"]
    patience = args.patience if args.patience is not None else preset["patience"]
    seeds = list(args.seeds if args.seeds is not None else preset["seeds"])

    # A smoke test must not overwrite report-quality runs; both use the same file names.
    if args.out_dir is not None:
        results_root = args.out_dir
    elif args.fast:
        results_root = PROJECT_ROOT / "results" / "main_fast"
    else:
        results_root = PROJECT_ROOT / "results"
    return epochs, patience, seeds, results_root


def main() -> None:
    args = build_parser().parse_args()
    epochs, patience, seeds, results_root = resolve_settings(args)

    if args.fast:
        destination = results_root / "main" / "runs"
        print("FAST MODE: smoke test only (one seed, reduced budget). Not reportable.\n"
              f"Writing to {destination} so reported results are left alone.")

    if __package__:
        from .models.main_model import CNNLSTMMain, describe
    else:
        from models.main_model import CNNLSTMMain, describe

    runs = 0
    for version in args.versions:
        channels = len(CHANNELS[version])
        print(f"--- {version} ({channels}-channel input) ---\n{describe(channels)}\n")
        for seed in seeds:
            runs += 1
            print(f"=== CNN-LSTM | {version} ({channels} channels) | seed {seed} ===")
            result = train_and_evaluate(
                CNNLSTMMain(channels), version, seed,
                epochs=epochs, patience=patience, batch_size=args.batch_size,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                data_dir=args.data_dir,
            )
            print(f"  val {result['val_acc']:.4f}  test acc {result['test_acc']:.4f}  "
                  f"F1 {result['test_f1']:.4f}  "
                  f"({result['epochs_run']} epochs, {result['train_time']:.0f}s)")
            print(f"  saved -> {save_result(result, results_root)}")

    print(f"\n{runs} run(s) done. Build the tables and figures with:")
    print("  python src/report_main_model.py")


if __name__ == "__main__":
    main()