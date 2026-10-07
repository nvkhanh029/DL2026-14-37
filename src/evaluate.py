"""Error analysis of the six reported runs (owner: analysis person).

Reads the saved test predictions (results/preds/*.npz: y_true, y_pred), rebuilds
the Random Forest predictions (the RF script saves none), and writes
    results/analysis/error_analysis.json
    results/analysis/confusion_matrices.png
    results/analysis/per_subject_accuracy.png
    results/analysis/ablations.png          (only if the ablation JSON files exist)

Usage (from the repository root, after src/prepare_data.py and the training runs):
    python src/evaluate.py
No model is trained or tuned here, and nothing is selected on the test set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG = yaml.safe_load((ROOT / "configs" / "shared.yaml").read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["paths"]["analysis_results"]
CLASSES = [c.upper() for c in CONFIG["dataset"]["classes"]]
RUNS = [("rf", "acc"), ("rf", "acc_gyro"), ("cnn", "acc_gyro"), ("lstm", "acc_gyro"),
        ("cnn_lstm", "acc"), ("cnn_lstm", "acc_gyro")]


def rf_predictions(sensors: str) -> tuple[np.ndarray, np.ndarray]:
    """Re-run the RF baseline with its shared settings (seed 42, 300 trees) to get test predictions."""
    from sklearn.ensemble import RandomForestClassifier

    from src.data import load_data
    from src.models.rf import extract_features

    X_tr, y_tr, _, _, X_te, y_te = load_data(sensors)
    forest = RandomForestClassifier(n_estimators=300, random_state=int(CONFIG["seed"]), n_jobs=-1)
    forest.fit(extract_features(X_tr), y_tr)
    return y_te, forest.predict(extract_features(X_te))


def load_predictions() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{run name: (y_true, y_pred)}; the accuracy is checked against results/<run>.json."""
    preds = {}
    for model, sensors in RUNS:
        name = f"{model}_{sensors}"
        if model == "rf":
            y_true, y_pred = rf_predictions(sensors)
        else:
            path = ROOT / "results" / "preds" / f"{name}_preds.npz"
            if not path.is_file():
                print(f"skip {name}: {path.name} not found")
                continue
            with np.load(path) as stored:
                y_true, y_pred = stored["y_true"], stored["y_pred"]
        reported = json.loads((ROOT / "results" / f"{name}.json").read_text())["accuracy"]
        assert abs(accuracy_score(y_true, y_pred) - reported) < 1e-3, f"{name}: preds do not match JSON"
        preds[name] = (y_true.astype(int), y_pred.astype(int))
    return preds


def per_class(y_true, y_pred) -> dict:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=range(6), zero_division=0)
    return {CLASSES[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                         "f1": round(float(f[i]), 4), "support": int(n[i])} for i in range(6)}


def top_confusions(y_true, y_pred, k: int = 6) -> list[dict]:
    """The k most frequent (true -> predicted) mistakes."""
    cm = confusion_matrix(y_true, y_pred, labels=range(6))
    np.fill_diagonal(cm, 0)
    order = np.argsort(cm, axis=None)[::-1][:k]
    return [{"true": CLASSES[i // 6], "predicted": CLASSES[i % 6], "count": int(cm[i // 6, i % 6]),
             "share_of_true_class": round(float(cm[i // 6, i % 6] / (y_true == i // 6).sum()), 4)}
            for i in order]


def per_subject(y_true, y_pred, subjects) -> dict:
    return {int(s): round(float(accuracy_score(y_true[subjects == s], y_pred[subjects == s])), 4)
            for s in np.unique(subjects)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = load_predictions()
    with np.load(ROOT / "data" / "processed" / "acc_gyro.npz") as stored:
        subjects = stored["subj_test"]
        y_test = stored["y_test"].astype(int)

    report = {"runs": {}}
    for name, (y_true, y_pred) in preds.items():
        assert (y_true == y_test).all(), "test order differs from the processed data"
        report["runs"][name] = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "per_class": per_class(y_true, y_pred),
            "top_confusions": top_confusions(y_true, y_pred),
            "per_subject_accuracy": per_subject(y_true, y_pred, subjects),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(6)).tolist(),
        }

    # Windows the models get wrong together: shared errors point at the data, not the model.
    deep = [n for n in ("cnn_acc_gyro", "lstm_acc_gyro", "cnn_lstm_acc_gyro") if n in preds]
    wrong = {n: preds[n][0] != preds[n][1] for n in deep}
    if deep:
        all_wrong = np.logical_and.reduce([wrong[n] for n in deep])
        any_wrong = np.logical_or.reduce([wrong[n] for n in deep])
        by_class = {CLASSES[c]: int((all_wrong & (y_test == c)).sum()) for c in range(6)}
        report["shared_errors"] = {
            "models": deep, "wrong_in_all": int(all_wrong.sum()), "wrong_in_any": int(any_wrong.sum()),
            "wrong_in_all_by_true_class": by_class,
            "windows": int(len(y_test)),
        }
    (OUT / "error_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    plot_confusions(preds)
    plot_subjects(report)
    plot_ablations()
    for name, run in report["runs"].items():
        print(f"{name:20s} acc {run['accuracy']:.4f}  macro-F1 {run['macro_f1']:.4f}  "
              f"top error: {run['top_confusions'][0]['true']} -> {run['top_confusions'][0]['predicted']}")
    print("saved ->", OUT.relative_to(ROOT))


def plot_confusions(preds) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, (y_true, y_pred)) in zip(axes.flat, preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=range(6), normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(6), CLASSES, rotation=60, fontsize=7)
        ax.set_yticks(range(6), CLASSES, fontsize=7)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_title(f"{name} (acc {accuracy_score(y_true, y_pred):.3f})", fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=120)
    plt.close(fig)


def plot_subjects(report) -> None:
    names = ["rf_acc_gyro", "cnn_acc_gyro", "lstm_acc_gyro", "cnn_lstm_acc_gyro"]
    names = [n for n in names if n in report["runs"]]
    subjects = list(report["runs"][names[0]]["per_subject_accuracy"])
    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, n in enumerate(names):
        vals = [report["runs"][n]["per_subject_accuracy"][s] for s in subjects]
        ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, label=n)
    ax.set_xticks(x, subjects)
    ax.set_xlabel("test subject")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "per_subject_accuracy.png", dpi=120)
    plt.close(fig)


def plot_ablations() -> None:
    """Window size, BiLSTM and LOSO results in one figure (skipped if not run yet)."""
    window, bilstm, loso = (OUT / "window_size.json", OUT / "bilstm.json", OUT / "loso.json")
    if not (window.exists() or loso.exists()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if window.exists():
        summ = json.loads(window.read_text())["summary"]
        sizes = [r["window_size"] for r in summ]
        axes[0].errorbar(sizes, [r["accuracy_mean"] for r in summ],
                         yerr=[r["accuracy_std"] for r in summ], fmt="o-", capsize=4,
                         label="CNN-LSTM (mean ± SD over seeds)")
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(sizes, sizes)
        axes[0].set_xlabel("window size (samples at 50 Hz)")
        axes[0].set_ylabel("test accuracy")
        if bilstm.exists():
            bi = [r for r in json.loads(bilstm.read_text())["summary"] if r["model"] == "cnn_bilstm"][0]
            axes[0].axhline(bi["accuracy_mean"], ls="--", color="tab:red",
                            label=f"BiLSTM, window 128 ({bi['accuracy_mean']:.3f})")
        axes[0].legend(fontsize=8)
        axes[0].set_title("Window size and BiLSTM ablations")
    if loso.exists():
        folds = json.loads(loso.read_text())["folds"]
        folds = sorted(folds, key=lambda f: f["accuracy"])
        axes[1].bar([str(f["subject"]) for f in folds], [f["accuracy"] for f in folds])
        axes[1].set_ylim(0.5, 1.0)
        axes[1].set_xlabel("held-out subject")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("Leave-one-subject-out")
        axes[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "ablations.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
