## 5. Main Model Results

> **Status: complete.** The reported numbers are the two seed-42 runs in
> `results/cnn_lstm_acc.json` and `results/cnn_lstm_acc_gyro.json`. To reproduce from scratch:
> ```bash
> pip install -r requirements.txt
> python src/prepare_data.py              # download UCI HAR and build acc.npz + acc_gyro.npz
> python scripts/train_main_model.py      # Acc + Acc+Gyro, seed 42 -> results/cnn_lstm_*.json
> ```
> An optional robustness sweep over seeds 0, 1 and 2 is written to `results/robustness/` with
> `python scripts/train_main_model.py --robustness`; it is not the reported result.

### 5.1 Setup

The main model of Section 4 (CNN-LSTM, Table 3) was trained on the two input versions of Table 4
under the shared protocol of Section 4.4: seed 42, Adam, learning rate 1e-3, batch size 64, at
most 30 epochs, early stopping on validation macro-F1 with patience 6. Each reported row is a
single seed-42 run. The seed controls the weight initialisation and the batch order only; the
train/validation subject split is fixed once by `src/prepare_data.py` at `--val-seed 42` and
reused, so it is not re-drawn per run. The test set is the official UCI test partition (9 subjects
never seen in training), evaluated once per run with the checkpoint selected on the validation
split.

**Table M1.** Main model (CNN-LSTM), one seed-42 run per input version.

| Version | Channels | Test acc (%) | Macro-F1 (%) | Params | Epochs | Train time (s) |
|---|---|---|---|---|---|---|
| Acc | 3 | 90.26 | 90.31 | 307,462 | 18 | 65 |
| Acc+Gyro | 6 | 91.11 | 91.20 | 308,422 | 12 | 44 |

**Table M2 (robustness only).** The same model and protocol re-run over seeds 0, 1 and 2, to put
a spread around the single reported seed-42 number. These runs are **not** the reported result
(seed 42 is the fixed rule); they only measure initialisation and shuffling variance.

| Version | Seed 0 | Seed 1 | Seed 2 | Mean ± sd |
|---|---|---|---|---|
| Acc | 88.73 | 89.41 | 89.68 | 89.27 ± 0.49 |
| Acc+Gyro | 91.82 | 91.35 | 90.77 | 91.31 ± 0.53 |

### 5.2 Reading the results

**Does adding the gyroscope help the main model?** Compare the Acc+Gyro and Acc rows of Table M1:
91.11% versus 90.26%, a gain of 0.85 points. The robustness sweep in Table M2 shows a seed-to-seed
standard deviation of about 0.5 points, so this difference is roughly one to two seed deviations
and should be read as weak evidence rather than a clear gyroscope benefit. The two runs also
differ in more than information content — adding three channels changes the first convolution's
parameter count (307,462 → 308,422) — so no strong claim about the gyroscope is made here.

**How does it compare with the baselines?** On Acc+Gyro the plain 1D-CNN scores 91.31% and the
LSTM 89.35% (CNN/LSTM section), against 91.11% for the CNN-LSTM. The hybrid is therefore
level with the plain CNN and about 1.8 points above the LSTM. The difference from the CNN is
smaller than the seed-to-seed spread, so the honest reading is that the extra recurrence does not
buy accuracy over the simpler CNN on this dataset; its value, if any, is the LSTM's ability to
model temporal order, which this experiment does not show as a measurable gain.

**Did the model converge?** Training stopped after 18 epochs for Acc and 12 for Acc+Gyro, well
inside the 30-epoch budget, because validation macro-F1 stopped improving for 6 epochs. The
validation curve had therefore flattened before the budget was exhausted, so the reported numbers
are not limited by the number of epochs.

**Was the added accuracy worth the cost?** Table M1 reports parameters and training time next to
accuracy. At roughly 0.31 M parameters the CNN-LSTM is the most expensive of the three
architectures, yet it does not beat the plain CNN's accuracy; the simpler baseline is the better
value on this dataset. This is a result worth stating plainly rather than hiding.

### 5.3 Limitations

These hold regardless of what the numbers turn out to be, and should be stated in the report.

* **Six classes from 21 training subjects.** The training partition contains only 21 people, and
  the validation split holds out 4 of them, so fitting uses the remaining 17 subjects and most of
  the 7,352 official training windows. This is a small-data regime; differences of one or two
  accuracy points between models should be interpreted with that in mind rather than as decisive.
* **Subject variability is not measured.** We report accuracy over unseen subjects, but not
  accuracy *per subject*. A model with a good average can still fail badly on individual people,
  and the confusion matrix cannot show this. With 9 test subjects, per-subject accuracy would be
  a useful addition if time allows.
* **One window length.** The 128-sample window is fixed by the dataset, so no claim is made about
  how accuracy depends on window length. This is a known limitation of working with the windows
  as provided rather than re-segmenting the raw continuous signals.
* **No augmentation.** `DATA.md` records that no augmentation was applied. For a
  subject-generalisation task, adding jitter or rotation to the training windows is the most
  likely source of further improvement.
* **One reported seed, three robustness seeds.** The fixed rule reports seed 42, so the headline
  numbers rest on one run each. The extra sweep in Table M2 gives a spread of about half an
  accuracy point, which is enough to show that the gyro gain is not decisive but too coarse to
  settle differences of a few tenths of a point.
* **Reproducibility caveat.** Seeds fix the Python, NumPy and PyTorch RNGs, but on GPU hardware
  non-deterministic kernels mean a re-run can differ slightly. Reported figures come from a single
  machine (CPU); cross-hardware comparison of training times in particular should be treated as
  indicative.

### References for this section
* D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for
  human activity recognition using smartphones," in *Proc. ESANN*, 2013.
* S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 9(8), 1997.
