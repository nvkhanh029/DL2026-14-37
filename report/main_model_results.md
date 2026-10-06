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
| Acc | 3 | 90.26 | 90.31 | 307,462 | 18 | 65 |
| Acc+Gyro | 6 | 91.11 | 91.20 | 308,422 | 12 | 44 |

**Table M2.** Robustness results for the same model and protocol using seeds 0, 1, and 2. These runs are **not** the reported results; seed 42 remains the fixed reporting seed. The additional runs are used only to estimate variation caused by initialisation and batch shuffling.

| Version | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| Acc | 88.73 | 89.41 | 89.68 | 89.27 ± 0.49 |
| Acc+Gyro | 91.82 | 91.35 | 90.77 | 91.31 ± 0.53 |

### 5.2 Interpretation of the Results

**Effect of adding gyroscope data.** The Acc+Gyro model achieves 91.11% test accuracy compared with 90.26% for the Acc-only model, corresponding to an improvement of 0.85 percentage points. However, the robustness sweep shows a seed-to-seed standard deviation of approximately 0.5 percentage points. The observed improvement is therefore only around one to two seed-level standard deviations and should be interpreted as **weak evidence rather than a definitive benefit from the gyroscope**.

The comparison also involves a small change in model size because adding three input channels increases the parameter count from 307,462 to 308,422. Consequently, the experiment does not isolate the information content of the gyroscope perfectly, and no strong causal claim is made about its contribution.

**Comparison with the baseline models.** On the Acc+Gyro input, the plain 1D-CNN achieves 91.31% accuracy, while the standalone LSTM achieves 89.35%. The CNN-LSTM achieves 91.11%, placing it essentially at the same level as the CNN and approximately 1.8 percentage points above the LSTM.

The difference between the CNN-LSTM and the plain CNN is smaller than the observed seed-to-seed variation. The most defensible conclusion is therefore that the additional recurrent component does **not** produce a measurable accuracy improvement over the simpler CNN on this dataset. The LSTM component may still provide a useful mechanism for modelling temporal order, but this experiment does not demonstrate an accuracy benefit from that capability.

**Convergence.** Training stopped after 18 epochs for Acc and 12 epochs for Acc+Gyro, both well below the 30-epoch maximum. In each case, validation macro-F1 had stopped improving for six consecutive epochs, triggering early stopping. The validation performance had therefore plateaued before the epoch budget was exhausted, indicating that the reported results were not limited by the maximum number of training epochs.

**Accuracy versus computational cost.** Table M1 reports parameter count and training time alongside the accuracy metrics. With approximately 0.31 million parameters, the CNN-LSTM is the most expensive of the three architectures. Despite this additional complexity, it does not outperform the simpler CNN in accuracy. On this dataset, the plain CNN therefore provides the better accuracy-to-complexity trade-off.

### 5.3 Limitations

The following limitations should be considered when interpreting the results.

- **Limited number of training subjects.** The dataset contains six classes and only 21 training subjects. The validation split holds out 4 subjects, leaving 17 subjects for model fitting and most of the 7,352 official training windows. This is a relatively small-data setting, so differences of one or two percentage points should not be treated as decisive evidence of superiority.
- **Subject-level variability is not analysed in detail.** The evaluation is performed on unseen subjects, but accuracy is not reported separately for each test subject. A model with a strong overall average could still perform poorly for particular individuals, and the current aggregate results do not reveal this. With only 9 test subjects, a per-subject accuracy analysis would be a useful extension if time permits.
- **Fixed window length.** The input window is fixed at 128 samples because the experiment uses the dataset's provided windows. Therefore, the results do not establish how performance would change with different window lengths. This is a limitation of working directly with the supplied windows rather than re-segmenting the original continuous signals.
- **No data augmentation.** As documented in `DATA.md`, no augmentation is applied. For a subject-generalisation task, techniques such as jittering or rotation could be promising sources of further improvement.
- **Limited robustness evaluation.** The reporting protocol uses a single fixed seed, 42, so the headline results are based on one run per input version. The additional sweep over seeds 0, 1, and 2 shows a standard deviation of roughly half a percentage point. This is sufficient to indicate that the observed gyroscope improvement is not decisive, but it is not enough to resolve differences of only a few tenths of a percentage point.
- **Reproducibility across hardware.** The random seeds fix the Python, NumPy, and PyTorch random-number generators, but GPU implementations can still contain non-deterministic operations. A repeated run may therefore differ slightly. The reported results were obtained on a single CPU machine, so training-time comparisons across different hardware should be considered indicative rather than absolute.

### References for This Section

- D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for human activity recognition using smartphones," in *Proc. ESANN*, 2013.
- S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 9(8), 1997.
