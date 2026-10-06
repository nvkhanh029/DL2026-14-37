"""Build the tables and figures for the Main Model results section.

    python src/report_main_model.py

Reads results/main/runs/*.json (written by src/train.py) and fills the block between the
GENERATED markers in report/main_model_results.md, so the prose in that file is preserved and
only the numbers are refreshed. Figures go to figures/main_model_*.png.

Only numpy and matplotlib are required: pandas is deliberately not used, so the script runs with
the project's existing requirements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVITIES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)
VERSION_LABEL = {"acc": "Acc", "acc_gyro": "Acc+Gyro"}
# Duplicated from data.py's CHANNELS on purpose: importing data.py pulls in torch, and this
# script only renders tables and figures, so it deliberately needs numpy + matplotlib alone.
VERSION_CHANNELS = {"acc": 3, "acc_gyro": 6}
BEGIN = "<!-- BEGIN GENERATED -->"
END = "<!-- END GENERATED -->"


def mean_sd(values: list[float]) -> str:
    """Format mean ± sample standard deviation as a percentage.

    ddof=1 (sample, not population) because the seeds are a sample of the possible runs. A
    single value has no spread, so its standard deviation is reported as 0.00 rather than NaN.
    """
    array = np.asarray(values, dtype=float)
    sd = array.std(ddof=1) if len(array) > 1 else 0.0
    return f"{100 * array.mean():.2f} ± {100 * sd:.2f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavoured markdown table.

    Row order is preserved, so callers control the presentation order of the table.
    """
    # First two lines are the header and the |---|---| separator markdown expects.
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def collect(runs_dir: Path) -> list[dict]:
    """Load every run result from `<version>__s<seed>.json` files in `runs_dir`."""
    files = sorted(Path(runs_dir).glob("*.json"))
    if not files:
        # Fail with the fix rather than a bare stack trace; the usual cause is running this
        # before any training, or pointing --runs-dir at the empty fast-test directory.
        raise SystemExit(
            f"No run results found in {runs_dir}.\n"
            "Run `python src/train.py` first."
        )
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    # Guard against publishing a table built from a reduced smoke test.
    seeds = sorted({run["seed"] for run in runs})
    if len(seeds) < 3:
        print(
            f"Warning: only seed(s) {seeds} present. A reported table needs all three seeds "
            "(0, 1, 2) so that the standard deviation is meaningful."
        )
    return runs


def for_version(runs: list[dict], version: str) -> list[dict]:
    """All runs belonging to one sensor version, in file order."""
    return [run for run in runs if run["version"] == version]


def normalised_confusion(runs: list[dict], version: str) -> np.ndarray | None:
    """Sum the per-seed confusion matrices, then normalise each row to sum to one.

    Rows are true classes, so a normalised row is the per-class recall (diagonal) and the
    distribution of mistakes (off-diagonal). Returns None when the version has no runs; callers
    that pass a version taken from `present` always get an array.
    """
    selected = for_version(runs, version)
    if not selected:
        return None
    # Summing counts over seeds before normalising pools evidence across runs.
    matrix = np.sum([run["confusion_matrix"] for run in selected], axis=0).astype(float)
    return matrix / matrix.sum(axis=1, keepdims=True)


def version_stats(runs: list[dict], version: str) -> dict[str, str]:
    """Aggregate the runs of one input version into the cells of a Table M1 row.

    `params` is a property of the architecture, so the first run's value is representative.
    `epochs` and `time` differ between runs, so they are averaged.
    """
    selected = for_version(runs, version)
    if not selected:
        raise ValueError(f"no runs for version {version!r}")
    return {
        "channels": str(VERSION_CHANNELS[version]),
        # The three metrics are the seed-to-seed mean with the sample standard deviation.
        "acc": mean_sd([run["test_acc"] for run in selected]),
        "f1": mean_sd([run["test_f1"] for run in selected]),
        "val": mean_sd([run["val_acc"] for run in selected]),
        "params": f"{selected[0]['params']:,}",                  # architecture, seed-invariant
        "epochs": f"{np.mean([run['epochs_run'] for run in selected]):.1f}",
        "time": f"{np.mean([run['train_time'] for run in selected]):.0f}",
    }


def table_m1(runs: list[dict], present: list[str]) -> str:
    """Table M1: one summary row per input version.

    `present` is built from the versions that actually have runs, so it is never empty here.
    """
    rows = []
    for version in present:
        stats = version_stats(runs, version)
        rows.append([
            VERSION_LABEL[version], stats["channels"], stats["acc"], stats["f1"], stats["val"],
            stats["params"], stats["epochs"], stats["time"],
        ])
    return md_table(
        ["Version", "Channels", "Test acc (%)", "Macro-F1 (%)", "Val acc (%)",
         "Params", "Epochs", "Train time (s)"],
        rows,
    )


def table_m2(runs: list[dict]) -> str:
    """Table M2: one row per individual run, sorted by version then seed."""
    rows = []
    for run in sorted(runs, key=lambda r: (r["version"], r["seed"])):
        rows.append([
            # .get keeps an unexpected version readable instead of raising a KeyError.
            VERSION_LABEL.get(run["version"], run["version"]),
            str(run["seed"]),
            f"{100 * run['val_acc']:.2f}",
            f"{100 * run['test_acc']:.2f}",
            f"{100 * run['test_f1']:.2f}",
            str(run["epochs_run"]),
            f"{run['train_time']:.0f}",
        ])
    return md_table(
        ["Version", "Seed", "Val acc (%)", "Test acc (%)", "Macro-F1 (%)",
         "Epochs", "Train time (s)"],
        rows,
    )


def table_m3(recalls: dict[str, np.ndarray], present: list[str]) -> str:
    """Table M3: per-class recall, one column per input version."""
    # Each row is one activity; `recalls[version][i]` is the i-th diagonal, i.e. its recall.
    rows = [
        [activity, *[f"{recalls[version][i]:.3f}" for version in present]]
        for i, activity in enumerate(ACTIVITIES)
    ]
    return md_table(["True class", *[VERSION_LABEL[v] for v in present]], rows)


# Captions are written next to the tables so the report and the figures cannot drift apart.
FIGURE_CAPTIONS = (
    "**Figure M1.** Confusion matrices of the main model, one per input version "
    "(rows normalised, summed over seeds) — `figures/main_model_confusion.png`.",
    "**Figure M2.** Per-class recall, Acc vs Acc+Gyro — `figures/main_model_recall.png`.",
    "**Figure M3.** Training/validation loss and validation accuracy (first seed) — "
    "`figures/main_model_curves.png`.",
    "**Figure M4.** Test accuracy with standard-deviation error bars — "
    "`figures/main_model_accuracy.png`.",
)


def build_markdown(runs: list[dict]) -> str:
    """Assemble the generated block: Tables M1-M3 followed by the figure captions."""
    present = [version for version in VERSION_LABEL if for_version(runs, version)]
    # Row-normalised recall per version, computed once and shared by Table M3 and Figure M1.
    recalls = {version: np.diag(normalised_confusion(runs, version)) for version in present}

    sections = [
        BEGIN,
        "",
        "**Table M1.** Main model (CNN-LSTM) on the two input versions. "
        "Mean ± standard deviation over the seeds.",
        "",
        table_m1(runs, present),
        "",
        "**Table M2.** Individual runs behind Table M1.",
        "",
        table_m2(runs),
        "",
        "**Table M3.** Per-class recall of the main model "
        "(confusion matrices summed over seeds, then normalised by row).",
        "",
        table_m3(recalls, present),
        "",
        *FIGURE_CAPTIONS,
        END,
    ]
    return "\n".join(sections)


def merge(existing: str, block: str) -> str:
    """Replace the GENERATED block, keeping the surrounding prose. Markers are always kept so a
    re-run refreshes the numbers instead of appending a second copy."""
    if BEGIN in existing and END in existing:
        head, rest = existing.split(BEGIN, 1)       # prose before the block
        _, tail = rest.split(END, 1)                # prose after the block
    else:
        # No markers yet (fresh file): keep whatever text is there and append the block.
        head = (existing.rstrip() + "\n\n") if existing.strip() else ""
        tail = "\n"
    return head + block + tail


def figure_confusion(runs: list[dict], present: list[str], figures_dir: Path) -> None:
    """Figure M1: one confusion matrix per input version, rows normalised."""
    # squeeze=False keeps `axes` 2-D, so one version and several iterate the same way.
    fig, axes = plt.subplots(1, len(present), figsize=(5.6 * len(present), 4.8), squeeze=False)
    for axis, version in zip(axes[0], present):
        matrix = normalised_confusion(runs, version)
        # Fixed 0-1 scale so the two panels stay comparable to each other.
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks(range(len(ACTIVITIES)), ACTIVITIES, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(ACTIVITIES)), ACTIVITIES, fontsize=8)
        # Annotate every cell; white text on dark cells keeps it readable.
        for row in range(len(ACTIVITIES)):
            for column in range(len(ACTIVITIES)):
                value = matrix[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=7, color="w" if value > 0.5 else "k")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(f"{VERSION_LABEL[version]} ({len(for_version(runs, version))} seeds)")
    fig.tight_layout()
    fig.savefig(figures_dir / "main_model_confusion.png", dpi=200)
    plt.close(fig)


def figure_recall(runs: list[dict], present: list[str], figures_dir: Path) -> None:
    """Figure M2: per-class recall, one bar group per input version."""
    fig, axis = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / len(present)          # share the width so versions sit side by side
    positions = np.arange(len(ACTIVITIES))
    for offset, version in enumerate(present):
        # The diagonal of the normalised confusion matrix is the recall of each activity.
        recall = np.diag(normalised_confusion(runs, version))
        axis.bar(positions + offset * width, recall, width, label=VERSION_LABEL[version])
    axis.set_xticks(positions + 0.4 - width / 2,
                    [name.replace("_", " ").title() for name in ACTIVITIES],
                    rotation=20, ha="right")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Recall")
    axis.set_title("Per-class recall, main model")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "main_model_recall.png", dpi=200)
    plt.close(fig)


def figure_curves(runs: list[dict], present: list[str], figures_dir: Path) -> None:
    """Figure M3: loss and validation accuracy of the lowest seed of each version."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for version in present:
        # One representative seed per version: the curves are for shape, not for statistics.
        history = sorted(for_version(runs, version), key=lambda r: r["seed"])[0]["history"]
        axes[0].plot(history["train_loss"], label=f"{VERSION_LABEL[version]} train")
        axes[0].plot(history["val_loss"], label=f"{VERSION_LABEL[version]} val", linestyle="--")
        axes[1].plot(history["val_acc"], label=VERSION_LABEL[version])
    axes[0].set_title("Loss")
    axes[1].set_title("Validation accuracy")
    for axis in axes:
        axis.set_xlabel("epoch")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "main_model_curves.png", dpi=200)
    plt.close(fig)


def figure_accuracy(runs: list[dict], present: list[str], figures_dir: Path) -> None:
    """Figure M4: mean test accuracy per version, error bars at one standard deviation."""
    # Build all three lists in one pass so bar, label and error bar always stay aligned.
    labels = []
    means = []
    deviations = []
    for version in present:
        accuracies = np.asarray([run["test_acc"] for run in for_version(runs, version)])
        labels.append(VERSION_LABEL[version])
        means.append(100 * accuracies.mean())
        # ddof=1 matches Table M1; a single seed has no spread, so report 0 rather than NaN.
        deviations.append(100 * accuracies.std(ddof=1) if len(accuracies) > 1 else 0.0)

    fig, axis = plt.subplots(figsize=(4.4, 3.8))
    axis.bar(labels, means, 0.55, yerr=deviations, capsize=4,
             color=["#7fb3d5", "#2e86c1"][:len(labels)])
    for position, (mean, deviation) in enumerate(zip(means, deviations)):
        axis.text(position, mean + deviation + 0.4, f"{mean:.2f}", ha="center", fontsize=9)
    # The axis is deliberately not anchored at zero: accuracies sit near 90%, where a full 0-100
    # scale would flatten every difference the figure exists to show. The numeric labels and the
    # error bars keep the absolute values readable, and Table M1 carries the exact figures.
    axis.set_ylim(max(0, min(means) - 3 * max(max(deviations), 0.5) - 1), 100)
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("Main model (CNN-LSTM), mean ± sd over seeds")
    fig.tight_layout()
    fig.savefig(figures_dir / "main_model_accuracy.png", dpi=200)
    plt.close(fig)


def make_figures(runs: list[dict], figures_dir: Path) -> None:
    """Write the four figures referenced by the results section.

    The calls are in figure order (M1, M2, M3, M4) so this reads like the caption list.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    present = [version for version in VERSION_LABEL if for_version(runs, version)]
    figure_confusion(runs, present, figures_dir)
    figure_recall(runs, present, figures_dir)
    figure_curves(runs, present, figures_dir)
    figure_accuracy(runs, present, figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Main Model results tables and figures."
    )
    parser.add_argument("--runs-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "main" / "runs",
                        help="directory of run JSON files written by src/train.py")
    parser.add_argument("--results-md", type=Path,
                        default=PROJECT_ROOT / "report" / "main_model_results.md",
                        help="markdown file whose GENERATED block is refreshed")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures",
                        help="directory receiving the main_model_*.png figures")
    args = parser.parse_args()

    runs = collect(args.runs_dir)

    # Read before writing: merge() needs the existing prose to preserve it.
    results_md = args.results_md
    results_md.parent.mkdir(parents=True, exist_ok=True)
    existing = results_md.read_text(encoding="utf-8") if results_md.is_file() else ""
    results_md.write_text(merge(existing, build_markdown(runs)), encoding="utf-8")

    make_figures(runs, args.figures_dir)

    for version in VERSION_LABEL:
        selected = for_version(runs, version)
        if selected:
            accuracy = mean_sd([run["test_acc"] for run in selected])
            macro_f1 = mean_sd([run["test_f1"] for run in selected])
            print(f"{VERSION_LABEL[version]:9s} test acc {accuracy} %"
                  f"   macro-F1 {macro_f1} %   ({len(selected)} seeds)")
    print(f"\nWrote {results_md} and figures in {args.figures_dir}")


if __name__ == "__main__":
    main()