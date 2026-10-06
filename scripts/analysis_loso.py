"""Ablation 3: leave-one-subject-out (LOSO) for the CNN-LSTM (Acc+Gyro).

All 30 subjects are pooled (train + val + test of the processed data). For each
subject, that subject is the test set and the model is trained on the other 29
subjects. Four of those 29 are used as the validation set for early stopping
(chosen with seed 42), so the held-out subject is never used for selection.

Note: the z-score statistics were fitted on the official training subjects only
(by src/prepare_data.py) and are reused unchanged, so the held-out subjects of
the official train set are normalised with statistics that include them. This
is a small effect and is mentioned in the report.

Usage (from the repository root):
    python scripts/analysis_loso.py                  # all 30 folds (about an hour on CPU)
    python scripts/analysis_loso.py --subjects 2 4   # only some folds
Output: results/analysis/loso.json (saved after every fold; finished folds are skipped on re-run)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from analysis_common import OUT_DIR, SEED, evaluate, fit, predict_proba, save_json, set_seed
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_data
from src.models.cnn_lstm import CNNLSTM

ROOT_NPZ = Path(__file__).resolve().parents[1] / "data" / "processed" / "acc_gyro.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data("acc_gyro")
    # Subject ids are stored next to the arrays by src/prepare_data.py.
    with np.load(ROOT_NPZ, allow_pickle=False) as stored:
        subj = np.concatenate([stored["subj_train"], stored["subj_val"], stored["subj_test"]])
    X = np.concatenate([X_tr, X_va, X_te])
    y = np.concatenate([y_tr, y_va, y_te]).astype(np.int64)
    all_subjects = sorted(int(s) for s in np.unique(subj))
    todo = args.subjects or all_subjects

    out = OUT_DIR / "loso.json"
    folds = json.loads(out.read_text())["folds"] if out.exists() else []
    done = {f["subject"] for f in folds}

    for s in todo:
        if s in done:
            continue
        others = [x for x in all_subjects if x != s]
        val_subjects = np.random.RandomState(SEED).choice(others, 4, replace=False)
        is_test, is_val = subj == s, np.isin(subj, val_subjects)
        is_train = ~is_test & ~is_val
        print(f"\n=== LOSO: held-out subject {s} | train {is_train.sum()} "
              f"val {is_val.sum()} test {is_test.sum()} ===", flush=True)

        set_seed(SEED)
        model = CNNLSTM(X.shape[2])
        epochs, seconds = fit(model, X[is_train], y[is_train], X[is_val], y[is_val],
                              epochs=args.epochs, verbose=False)
        metrics = evaluate(model, X[is_test], y[is_test])
        pred = predict_proba(model, X[is_test]).argmax(axis=1)
        folds.append({"subject": s, "n_windows": int(is_test.sum()), **metrics,
                      "epochs": epochs, "train_time_sec": round(seconds, 1),
                      "y_true": y[is_test].tolist(), "y_pred": pred.tolist()})
        print(f"  subject {s}: acc {metrics['accuracy']:.4f} | {epochs} epochs | {seconds:.0f}s",
              flush=True)
        save_json("loso.json", summarise(folds))  # saved after every fold

    print("\nLOSO summary:", {k: v for k, v in summarise(folds).items() if k != "folds"})


def summarise(folds: list[dict]) -> dict:
    """Mean/SD of per-subject accuracy plus pooled accuracy and macro-F1 over all folds."""
    folds = sorted(folds, key=lambda f: f["subject"])
    accs = np.array([f["accuracy"] for f in folds])
    y_true = np.concatenate([f["y_true"] for f in folds])
    y_pred = np.concatenate([f["y_pred"] for f in folds])
    return {
        "model": "cnn_lstm", "sensors": "acc_gyro", "seed": SEED, "n_subjects": len(folds),
        "mean_subject_accuracy": round(float(accs.mean()), 4),
        "std_subject_accuracy": round(float(accs.std(ddof=1)), 4) if len(accs) > 1 else 0.0,
        "min_subject_accuracy": round(float(accs.min()), 4),
        "max_subject_accuracy": round(float(accs.max()), 4),
        "pooled_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "pooled_macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "folds": folds,
    }


if __name__ == "__main__":
    main()
