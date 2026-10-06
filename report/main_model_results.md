## 5. Main Model Results

> **Status: results are complete.** The six runs (Acc and Acc+Gyro x seeds 0, 1, 2) are in
> `results/main/runs/`, and Tables M1-M3 with Figures M1-M4 below were generated from them.
> To reproduce from scratch:
> ```bash
> pip install -r requirements.txt
> python src/prepare_data.py       # download UCI HAR and build acc.npz + acc_gyro.npz
> python src/train.py              # Acc + Acc+Gyro x 3 seeds -> results/main/runs/
> python src/report_main_model.py  # refreshes Tables M1-M3 and Figures M1-M4 below
> ```
> `python src/train.py --fast` runs a cheaper smoke test into `results/main_fast/` if you only
> want to confirm the pipeline works; it is deliberately kept out of the reported tables.

### 5.1 Setup

The main model of Section 4 (CNN-LSTM, Table 3) was trained on the two input versions of Table 4
under the protocol of Section 4.4: Adam, learning rate 1e-3, weight decay 1e-4, batch size 64, at
most 40 epochs, early stopping on validation accuracy with patience 8, three seeds (0, 1, 2). Each
row below is therefore a mean over three independent runs, where a seed re-draws the weight
initialisation and the batch order.

The train/validation subject split is *not* re-drawn per seed: it is fixed once by
`src/prepare_data.py` at `--val-seed 42` and reused by every run. The reported spread therefore
reflects initialisation and shuffling variance only, and does not include the variance that a
different subject-level split would introduce. With 21 training subjects that second source of
variance is likely to be the larger of the two, so the standard deviations below should be read as
a lower bound on the true run-to-run spread.

The test set is the official UCI test partition (9 subjects never seen in training), evaluated
once per run with the checkpoint selected by validation accuracy.

<!-- BEGIN GENERATED -->

**Table M1.** Main model (CNN-LSTM) on the two input versions. Mean ± standard deviation over the seeds.

| Version | Channels | Test acc (%) | Macro-F1 (%) | Val acc (%) | Params | Epochs | Train time (s) |
|---|---|---|---|---|---|---|---|
| Acc | 3 | 90.23 ± 1.48 | 90.26 ± 1.46 | 93.84 ± 0.35 | 307,462 | 14.7 | 53 |
| Acc+Gyro | 6 | 91.56 ± 1.08 | 91.63 ± 1.09 | 95.04 ± 0.40 | 308,422 | 16.0 | 65 |

**Table M2.** Individual runs behind Table M1.

| Version | Seed | Val acc (%) | Test acc (%) | Macro-F1 (%) | Epochs | Train time (s) |
|---|---|---|---|---|---|---|
| Acc | 0 | 93.81 | 88.77 | 88.82 | 15 | 53 |
| Acc | 1 | 93.49 | 90.19 | 90.21 | 12 | 43 |
| Acc | 2 | 94.20 | 91.72 | 91.74 | 17 | 63 |
| Acc+Gyro | 0 | 94.59 | 92.03 | 92.07 | 15 | 59 |
| Acc+Gyro | 1 | 95.36 | 90.33 | 90.39 | 10 | 42 |
| Acc+Gyro | 2 | 95.17 | 92.33 | 92.43 | 23 | 95 |

**Table M3.** Per-class recall of the main model (confusion matrices summed over seeds, then normalised by row).

| True class | Acc | Acc+Gyro |
|---|---|---|
| WALKING | 0.962 | 0.972 |
| WALKING_UPSTAIRS | 0.900 | 0.957 |
| WALKING_DOWNSTAIRS | 0.948 | 0.965 |
| SITTING | 0.835 | 0.818 |
| STANDING | 0.775 | 0.793 |
| LAYING | 1.000 | 1.000 |

**Figure M1.** Confusion matrices of the main model, one per input version (rows normalised, summed over seeds) — `figures/main_model_confusion.png`.
**Figure M2.** Per-class recall, Acc vs Acc+Gyro — `figures/main_model_recall.png`.
**Figure M3.** Training/validation loss and validation accuracy (first seed) — `figures/main_model_curves.png`.
**Figure M4.** Test accuracy with standard-deviation error bars — `figures/main_model_accuracy.png`.
<!-- END GENERATED -->

### 5.2 Reading the results

The following checks are what the numbers above have to settle. Each one is stated as a question
so that it can be answered from the tables without interpretation drifting into what we hoped to
see.

**Does adding the gyroscope help the main model?** Compare the Acc+Gyro and Acc rows of Table M1.
The comparison is clean in design terms — identical architecture, identical hyper-parameters, only
`C` changes — but note that the two runs differ in more than information content: doubling the
input width also changes the parameter count slightly (the first convolution scales with `C`) and
the optimisation problem the network solves. A difference smaller than the seed-to-seed spread
should be reported as *not* a difference rather than as a small one, and a difference of a few
tenths of a point should be checked against per-seed values in Table M2 to see whether it is one
lucky run.

**Which classes remain hard?** Table M3 and Figure M2 give per-class recall. The classes that are
expected to be confused are the three dynamic ones — walking, walking upstairs, walking
downstairs — because they share the same underlying step cycle and differ mainly in rate and
amplitude; and sitting versus standing, which are separated only by the orientation of the phone
rather than by any motion. If recall is instead lost on a class we did not anticipate, that is
more informative than the headline accuracy and should be stated explicitly.

**Did the model converge?** Figure M3 shows the training and validation curves. Two things to
check: whether the validation curve has visibly flattened (if it is still rising at epoch 40, the
40-epoch budget was the binding constraint and the reported number understates the model), and
whether the training curve has fallen well below the validation curve (if so, the model is
overfitting and more dropout or weight decay, not more epochs, is what would help).

**Was the added accuracy worth the cost?** Table M1 reports parameters and training time next to
accuracy. The fusion model is the most expensive of the three architectures, so the honest
conclusion may well be that the simpler baseline reaches the same accuracy for a fraction of the
cost.

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
* **Three seeds.** Three seeds give a usable but coarse estimate of the standard deviation. With
  differences of a few tenths of a point in play, five seeds would be needed to make the
  comparison between versions conclusive.
* **Reproducibility caveat.** Seeds fix the Python, NumPy and PyTorch RNGs, but on GPU hardware
  non-deterministic kernels mean a re-run can differ slightly. Reported figures come from a single
  machine; cross-hardware comparison of training times in particular should be treated as
  indicative.

### References for this section
* D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for
  human activity recognition using smartphones," in *Proc. ESANN*, 2013.
* S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 9(8), 1997.