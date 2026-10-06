## 5. Main Model Results

> **Status: complete.** The reported results are the two seed-42 runs in `results/cnn_lstm_acc.json` and `results/cnn_lstm_acc_gyro.json`.
>
> To reproduce them from scratch:
>
> ```bash
> pip install -r requirements.txt
> python src/prepare_data.py              # download UCI HAR and build acc.npz + acc_gyro.npz
> python scripts/train_main_model.py      # Acc + Acc+Gyro, seed 42 -> results/cnn_lstm_*.json
> ```
>
> `python scripts/train_main_model.py --robustness` runs the optional sweep over seeds 0, 1 and 2 and writes it to `results/robustness/`. Those runs are not part of the reported results.

### 5.1 Setup

We trained the CNN-LSTM from Section 4 (Table 3) on both input versions in Table 4, using the shared protocol from Section 4.4: seed 42, Adam with a learning rate of 1e-3, batch size 64, at most 30 epochs, and early stopping on validation macro-F1 with a patience of 6 epochs.

Each result in Table M1 is one seed-42 run on machine D. The seed fixes the weight initialisation and the batch order, which makes the run repeat itself on the same machine and software, though not exactly on other machines (Section 5.3). The train/validation subject split is made once by `src/prepare_data.py` with `--val-seed 42` and is the same for every run. The test set is the official UCI test partition: 9 subjects that the model never sees during training. It is evaluated once per run, with the checkpoint chosen on validation.

**Table M1.** Main model (CNN-LSTM), one seed-42 run per input version.

| Version | Channels | Test acc (%) | Macro-F1 (%) | Params | Epochs | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Acc | 3 | 88.09 | 88.23 | 307,462 | 9 | 38 |
| Acc+Gyro | 6 | 91.99 | 92.05 | 308,422 | 23 | 101 |

*Epochs* counts every epoch that was run, including the 6 patience epochs after the best validation macro-F1. The restored checkpoint is therefore from epoch 3 for Acc and epoch 17 for Acc+Gyro.

**Table M2.** The same model and protocol with seeds 0, 1 and 2. These runs are not the reported results; seed 42 stays the reporting seed, and the sweep is only there to show how much initialisation and batch order move the numbers. It was run on a different machine (C) from Table M1, so its values should not be compared one-to-one with Table M1. Only the spread is used.

| Version | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| Acc | 88.73 | 89.41 | 89.68 | 89.27 ± 0.49 |
| Acc+Gyro | 91.82 | 91.35 | 90.77 | 91.31 ± 0.53 |

### 5.2 Interpretation of the Results

**The gyroscope helps.** With Acc+Gyro the model reaches 91.99% test accuracy, against 88.09% with the accelerometer alone, a gain of 3.9 percentage points. The seed sweep in Table M2 moves by only about half a point, so the gain is seven to eight times larger than the run-to-run noise. Rotation adds real information for this model.

One small caveat remains. Three extra input channels also add a few weights to the first convolution (308,422 parameters instead of 307,462), so the experiment does not isolate the gyroscope's information perfectly. Given how small that change is, we attribute the gain to the extra channels.

**The CNN-LSTM and the plain CNN are level.** On Acc+Gyro, the plain 1D-CNN reaches 91.31% and the standalone LSTM 89.35%. The CNN-LSTM's 91.99% is about 0.7 points above the CNN and about 2.6 points above the LSTM.

*Note:* the CNN and LSTM numbers come from machine B and the CNN-LSTM numbers from machine D, so the 0.7-point gap is also a comparison across machines.

A gap of less than one point, close to the half-point seed spread and measured across two machines, is not enough to rank the two models. We treat the CNN-LSTM and the CNN as comparable and do not claim that the recurrent stage improves accuracy on this dataset. It may still be modelling temporal order, but this experiment cannot show a benefit from it. The LSTM on its own, on the other hand, falls clearly behind both.

**Training stopped well before the limit.** The Acc run stopped after 9 epochs and the Acc+Gyro run after 23, both short of the 30-epoch maximum. These counts include the 6 patience epochs: in both runs, validation macro-F1 had not improved for six epochs in a row, which triggered early stopping. Validation performance had levelled off before the budget ran out, so the epoch limit did not hold the results back.

**Accuracy and cost.** The CNN-LSTM has about 0.31 million parameters, fewer than the 1D-CNN (about 0.57 million), and the LSTM is by far the slowest to train (447 s). On neither measure is the CNN-LSTM the most costly of the three. The training times come from different machines and runs and are indicative only. The CNN-LSTM is a sensible choice, as it matches the CNN with roughly half the parameters, but the accuracy gap is too small to call it the better model.

### 5.3 Limitations

- **Few training subjects.** The dataset has only 21 training subjects. Four are held out for validation, which leaves 17 subjects and 5,800 windows for fitting. In a setting this small, a difference of one or two points between models should not be read as decisive.
- **No per-subject breakdown here.** The test subjects are unseen, but this section reports only the average over all of them. A model with a good average can still do badly for particular people, and these aggregate numbers would not show it. With only 9 test subjects, accuracy per subject is the obvious next check; the error analysis section takes it up.
- **Fixed window length.** We use the 128-sample windows supplied with the dataset, so these results say nothing about other window lengths. Testing longer windows would require re-segmenting the raw continuous signals.
- **No data augmentation.** As `DATA.md` states, no augmentation is applied. Because the task is to generalise to new people, jittering or small rotations of the signal would be natural things to try next.
- **One reporting seed.** The headline numbers come from one run per input version. The sweep over seeds 0, 1 and 2 has a standard deviation of about half a point. That is enough to show that the 3.9-point gyroscope gain is real, but not enough to separate models that differ by a few tenths of a point, such as the CNN-LSTM and the CNN.
- **Reproducibility across hardware.** Seeding Python, NumPy and PyTorch makes a run repeat itself on the same machine and software, but not exactly on another one. A different CPU or GPU, PyTorch version or thread count can change the numerical path, and some GPU operations are not deterministic. Table M1 was produced on one CPU machine (D), Table M2 on another (C), and the CNN/LSTM baselines on a third (B). Any comparison across these machines carries that extra uncertainty, and the training times are indicative only.

### References for This Section

- D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for human activity recognition using smartphones," in *Proc. ESANN*, 2013.
- S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 9(8), 1997.
