## 5. Main Model Results

> **Status: complete.** The reported results are based on the two seed-42 runs stored in `results/cnn_lstm_acc.json` and `results/cnn_lstm_acc_gyro.json`.
>
> To reproduce the results from scratch:
>
> ```bash
> pip install -r requirements.txt
> python src/prepare_data.py              # download UCI HAR and build acc.npz + acc_gyro.npz
> python scripts/train_main_model.py      # Acc + Acc+Gyro, seed 42 -> results/cnn_lstm_*.json
> ```
>
> An optional robustness sweep using seeds 0, 1, and 2 is written to `results/robustness/` by running `python scripts/train_main_model.py --robustness`. These runs are not used as the reported results.

### 5.1 Setup

The main model described in Section 4 (CNN-LSTM, Table 3) was trained using the two input versions from Table 4 and the shared training protocol from Section 4.4. The protocol uses seed 42, Adam with a learning rate of 1e-3, a batch size of 64, a maximum of 30 epochs, and early stopping based on validation macro-F1 with a patience of 6 epochs.

Each reported result is from a single seed-42 run. The seed determines weight initialisation and batch ordering. The train/validation subject split, however, is fixed once by `src/prepare_data.py` using `--val-seed 42` and is reused for all runs. The test set is the official UCI test partition, consisting of 9 subjects that are not used for training. It is evaluated once per run using the checkpoint selected on the validation split.

**Table M1.** Main model (CNN-LSTM), with one seed-42 run for each input version.

| Version | Channels | Test acc (%) | Macro-F1 (%) | Params | Epochs | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Acc | 3 | 88.09 | 88.23 | 307,462 | 9 | 38 |
| Acc+Gyro | 6 | 91.99 | 92.05 | 308,422 | 23 | 101 |

*Epochs* is the number of epochs run, including the 6 patience epochs after the best validation macro-F1 (the best checkpoint is therefore 6 epochs earlier: epoch 3 for Acc and epoch 17 for Acc+Gyro).

**Table M2.** Robustness results for the same model and protocol using seeds 0, 1, and 2. These runs are **not** the reported results; seed 42 remains the fixed reporting seed. The additional runs are used only to estimate variation caused by initialisation and batch shuffling. They were run on a different machine from Table M1, so they should not be compared directly with the Table M1 numbers; only their spread is used.

| Version | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| Acc | 88.73 | 89.41 | 89.68 | 89.27 ± 0.49 |
| Acc+Gyro | 91.82 | 91.35 | 90.77 | 91.31 ± 0.53 |

### 5.2 Interpretation of the Results

**Effect of adding gyroscope data.** The Acc+Gyro model achieves 91.99% test accuracy compared with 88.09% for the Acc-only model, an improvement of 3.9 percentage points. This is roughly seven to eight times the seed-to-seed standard deviation of about 0.5 percentage points in Table M2, so the gain is larger than run-to-run noise and the gyroscope clearly helps this model.

The comparison also involves a small change in model size because adding three input channels increases the parameter count from 307,462 to 308,422. Consequently, the experiment does not isolate the information content of the gyroscope perfectly, and the gain is attributed to the added channels with that small caveat.

**Comparison with the baseline models.** On the Acc+Gyro input, the plain 1D-CNN achieves 91.31% accuracy, while the standalone LSTM achieves 89.35%. The CNN-LSTM achieves 91.99%, about 0.7 percentage points above the CNN and about 2.6 points above the LSTM.

The difference between the CNN-LSTM and the plain CNN is below one percentage point and close to the seed-to-seed spread of about 0.5 points. The two models are therefore treated as **comparable**, and no win is claimed for the CNN-LSTM over the CNN. The LSTM is clearly behind both on this dataset.

**Convergence.** Training stopped after 9 epochs for Acc and 23 epochs for Acc+Gyro, both below the 30-epoch maximum. The epoch counts include the 6 patience epochs: in each case, validation macro-F1 had stopped improving for six consecutive epochs, triggering early stopping. The validation performance had therefore plateaued before the epoch budget was exhausted, indicating that the reported results were not limited by the maximum number of training epochs.

**Accuracy versus computational cost.** Table M1 reports parameter count and training time alongside the accuracy metrics. The CNN-LSTM has about 0.31 million parameters, fewer than the 1D-CNN (about 0.57 million), and the LSTM is the slowest of the three to train (447 s). The CNN-LSTM is therefore not the most expensive model by either measure. Training times come from different runs and machines, so they are indicative only. Because the CNN-LSTM is about 0.7 points ahead of the CNN with fewer parameters, it is a reasonable choice, but the accuracy gap is too small to call it better.

### 5.3 Limitations

The following limitations should be considered when interpreting the results.

- **Limited number of training subjects.** The dataset contains six classes and only 21 training subjects. The validation split holds out 4 subjects, leaving 17 subjects for model fitting and most of the 7,352 official training windows. This is a relatively small-data setting, so differences of one or two percentage points should not be treated as decisive evidence of superiority.
- **Subject-level variability is not analysed in detail.** The evaluation is performed on unseen subjects, but accuracy is not reported separately for each test subject. A model with a strong overall average could still perform poorly for particular individuals, and the current aggregate results do not reveal this. With only 9 test subjects, a per-subject accuracy analysis would be a useful extension if time permits.
- **Fixed window length.** The input window is fixed at 128 samples because the experiment uses the dataset's provided windows. Therefore, the results do not establish how performance would change with different window lengths. This is a limitation of working directly with the supplied windows rather than re-segmenting the original continuous signals.
- **No data augmentation.** As documented in `DATA.md`, no augmentation is applied. For a subject-generalisation task, techniques such as jittering or rotation could be promising sources of further improvement.
- **Limited robustness evaluation.** The reporting protocol uses a single fixed seed, 42, so the headline results are based on one run per input version. The additional sweep over seeds 0, 1, and 2 shows a standard deviation of roughly half a percentage point. This is enough to show that the 3.9-point gyroscope improvement is well above run-to-run noise, but it is not enough to resolve differences of only a few tenths of a percentage point, such as the CNN-LSTM versus CNN gap.
- **Reproducibility across hardware.** The random seeds fix the Python, NumPy, and PyTorch random-number generators, but GPU implementations can still contain non-deterministic operations. A repeated run may therefore differ slightly. The Table M1 results were obtained on a single CPU machine, and the Table M2 robustness runs on a different machine, so training-time comparisons across different hardware should be considered indicative rather than absolute.

### References for This Section

- D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for human activity recognition using smartphones," in *Proc. ESANN*, 2013.
- S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 9(8), 1997.
