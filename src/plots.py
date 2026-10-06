"""Plots for the DL2026 HAR project. Owner: visualization person.

Run from the repo root:  python src/plots.py

Rules for every figure
- Read numbers from results/*.json. Never type results by hand.
- Title, axis labels, legend, and the value written on each bar.
- Same color for the same thing in every figure (e.g. Acc = blue, Acc+Gyro = orange).
- Save to figures/ with dpi=200, then plt.close(fig).
- If a result file is missing (FileNotFoundError), its owner has not finished yet.
  Skip that figure and tell the leader. Never invent numbers.

How to work: do ONE function at a time, run it, look at the PNG, post a screenshot
in the group chat, then move to the next. Replace each TODO with real code.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

CLASS_NAMES = ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"]  # labels 0-5
MODEL_NAMES = {"rf": "Random Forest", "cnn": "CNN", "lstm": "LSTM", "cnn_lstm": "CNN-LSTM"}
SENSOR_NAMES = {"acc": "Acc", "acc_gyro": "Acc+Gyro"}


def load_result(model, sensors):
    """Return the dict stored in results/<model>_<sensors>.json.

    Fields inside: model, sensors, seed, accuracy, macro_f1, epochs, train_time_sec.
    TODO: open the file with open(), read it with json.load(), return the dict.
    Test it: print(load_result("cnn_lstm", "acc_gyro")["accuracy"])
    """
    raise NotImplementedError


def plot_step1_sensor_effect():
    """Figure 1 -> figures/step1_sensor_effect.png

    Question it answers: does adding the gyroscope help?
    Data: rf_acc, rf_acc_gyro, cnn_lstm_acc, cnn_lstm_acc_gyro (accuracy and macro_f1).
    Layout: two panels side by side (accuracy | macro F1). In each panel, two groups
            (Random Forest, CNN-LSTM), and in each group two bars (Acc, Acc+Gyro).
    Hints:
      - plt.subplots(1, 2, sharey=True) gives two axes.
      - Grouped bars: call ax.bar twice with x positions shifted by the bar width.
      - ax.bar_label(bars) writes the value on top of the bars.
      - The differences are about one point, so ax.set_ylim(0.80, 1.0) makes them visible.
        Say that the axis starts at 0.80 in the report caption.
    Done when: both panels show 4 bars each, labelled, and the legend is readable.
    """
    # TODO
    raise NotImplementedError


def plot_step2_model_comparison():
    """Figure 2 -> figures/step2_model_comparison.png

    Question it answers: which model works best with both sensors?
    Data: rf, cnn, lstm, cnn_lstm, all on acc_gyro (accuracy and macro_f1).
    Layout: one panel, 4 groups (one per model), 2 bars each (accuracy, macro F1).
    Hints: same grouped-bar idea as figure 1. Keep the model order:
           Random Forest, CNN, LSTM, CNN-LSTM (simple to complex).
    Done when: the order matches the report and all 8 bars have values.
    """
    # TODO
    raise NotImplementedError


def plot_confusion_matrices():
    """Figure 3 -> figures/<model>_acc_gyro_confusion.png (one file per model)

    Question it answers: which activities get mixed up?
    Data: results/<model>_acc_gyro_preds.npz (CNN and LSTM exist; ask persons C and A
          to save theirs in the same format).
    Step 0: find the array names inside a file:
            np.load("results/cnn_acc_gyro_preds.npz").files
    Hints:
      - from sklearn.metrics import ConfusionMatrixDisplay
      - ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASS_NAMES,
                                                normalize="true")
      - normalize="true" shows the share of each real class, easier to compare than counts.
      - Rotate the x tick labels if they overlap.
    Check with person D first: D may also draw these for the error analysis.
    Done when: 6x6 matrix, class names on both axes, diagonal clearly visible.
    """
    # TODO
    raise NotImplementedError


def plot_class_distribution():
    """Figure 4 -> figures/class_distribution.png

    Question it answers: are the classes balanced across train, val and test?
    Data (windows per class, in CLASS_NAMES order):
        train: 967, 840, 770, 1017, 1089, 1117
        val:   259, 233, 216,  269,  285,  290
        test:  496, 471, 420,  491,  532,  537
      (These come from DATA.md / prepare_data.py. They are the only numbers you may
       type by hand, because they describe the data, not the results.)
    Layout: grouped bars, 6 groups (classes) x 3 bars (train, val, test).
    Done when: it shows that the classes are mildly imbalanced.
    """
    # TODO
    raise NotImplementedError


def plot_sample_signals():
    """Figure 5 -> figures/sample_signals.png

    Question it answers: what does the raw sensor data look like?
    Data: from src.data import load_data
          X_train, y_train, *_ = load_data("acc_gyro")   # X_train shape (5800, 128, 6)
    Layout: 2 rows x 6 columns. Columns = the 6 activities. Top row = the 3 accelerometer
            channels, bottom row = the 3 gyroscope channels, for ONE example window each.
    Hints:
      - Pick the first window of each class: np.where(y_train == class_id)[0][0]
      - Channels 0-2 are accelerometer x, y, z. Channels 3-5 are gyroscope x, y, z.
      - The data is normalized, so the y-axis has no unit. Say so in the caption.
    Done when: you can see that walking looks very different from laying.
    """
    # TODO
    raise NotImplementedError


def plot_robustness():
    """Figure 6 -> figures/robustness_cnn_lstm.png

    Question it answers: how stable is the CNN-LSTM across random seeds?
    Data: results/robustness/cnn_lstm_acc_seed{0,1,2}.json and
          results/robustness/cnn_lstm_acc_gyro_seed{0,1,2}.json
    Layout: bars with error bars: mean of the 3 seeds, error bar = standard deviation,
            for Acc and Acc+Gyro (accuracy and macro F1).
    Hints:
      - np.mean(values) and np.std(values) for the 3 seeds.
      - ax.bar(x, means, yerr=stds, capsize=4)
      - Optional: draw the 3 single seeds as dots on top (ax.scatter).
    Done when: the figure supports the statement "differences smaller than the error
               bars are not clear wins".
    """
    # TODO
    raise NotImplementedError


def plot_analysis():
    """Figure 7 -> figures/analysis_*.png  (WAIT for person D's results in results/analysis/)

    Likely figures: accuracy vs window size, BiLSTM vs LSTM, per-subject accuracy from
    leave-one-subject-out. Ask D which files and fields to read, then reuse the same
    bar/line plot ideas from the figures above.
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # Uncomment each line when its function is finished.
    # plot_step1_sensor_effect()
    # plot_step2_model_comparison()
    # plot_confusion_matrices()
    # plot_class_distribution()
    # plot_sample_signals()
    # plot_robustness()
    # plot_analysis()
    pass