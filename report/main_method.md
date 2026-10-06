## 4. Main Method

This section describes the main model and the protocol used to train and evaluate it. The dataset, data splits, and preprocessing are defined in `DATA.md`. This section focuses on the model architecture and training procedure so that the main model and the baseline models can be compared under the same conditions.

### 4.1 Design Rationale

The task is to classify a 2.56-second window of inertial sensor data into one of six activities. The signal contains two types of information that are not captured equally well by a single model architecture:

- **Local patterns.** Activity identity is expressed through short, recurring waveform patterns, such as heel strikes, individual steps, or turns. These patterns may occur at different positions within a window, making convolutional filters well suited to detecting them.
- **Long-range temporal structure.** The order and rhythm of these local patterns also matter. For example, climbing stairs and walking downstairs can contain similar instantaneous movements but differ in how those movements are arranged and repeated over time. This longer-term temporal structure is well suited to recurrent networks.

The main model therefore uses a **hybrid CNN-LSTM architecture**. A small 1D CNN first extracts local features and compresses each input window into a shorter sequence of learned descriptors. An LSTM then models how those descriptors evolve over time before producing the final classification.

The convolutional front end reduces the original 128-sample window to 32 time steps. The two pooling layers produce an effective stride of 4, so consecutive LSTM inputs are four samples apart. The resulting features have a receptive field of approximately 16 samples, corresponding to roughly 0.3 seconds at 50 Hz. This reduction removes local redundancy and decreases the computational cost of the recurrent stage by a factor of four. At the same time, the resulting sequence remains sufficiently fine-grained: an individual step, which typically spans roughly 25–35 samples (0.5–0.7 seconds), is represented across several LSTM time steps.

The two components are based on the corresponding baseline architectures rather than being chosen independently. The convolutional front end follows the same `Conv1d → ReLU → MaxPool` structure, kernel width, and first two channel widths as the plain 1D-CNN baseline, while adding BatchNorm after each convolution. The recurrent back end follows the standalone LSTM baseline, using a hidden size of 128 and two layers, together with the same last-step readout. The main model uses dropout of 0.3 instead of the baseline's 0.5.

The CNN-LSTM is therefore **not** a strict combination or superset of the two baselines: it adds BatchNorm and uses a different dropout value. Nevertheless, the two branches retain comparable capacity, making the comparison more meaningful because any accuracy difference is less likely to result simply from a substantially larger network. All three models also use the same training protocol described in Section 4.4, avoiding differences caused by the training recipe.

The parameter counts in Table M1 show that most of the model capacity is concentrated in the recurrent component. For the Acc input, the model has 307,462 parameters: 42,496 (14%) in the convolutional front end, 264,192 (86%) in the LSTM, and 774 in the linear classification head. The architecture is therefore dominated by the recurrent branch rather than being evenly balanced between the CNN and LSTM.

This distribution follows directly from the design. A two-layer LSTM with 128 hidden units requires substantially more parameters than the two relatively narrow convolutional layers. Keeping the same hidden size and depth as the standalone LSTM also makes the recurrent component directly comparable to that baseline. In practice, training time is therefore dominated by the recurrent stage, making the CNN's fourfold reduction in sequence length important for keeping the model computationally manageable.

### 4.2 Architecture

**Table 3.** Main model (CNN-LSTM), with input shape `(batch, 128, C)`, where `C` is the number of sensor channels.

| # | Stage | Output shape |
|---|---|---|
| 0 | Input window | (B, 128, C) |
| 1 | Conv1d(C→64, k=5, pad=2) + BatchNorm + ReLU + MaxPool(2) | (B, 64, 64) |
| 2 | Conv1d(64→128, k=5, pad=2) + BatchNorm + ReLU + MaxPool(2) | (B, 128, 32) |
| 3 | Transpose to time-major layout | (B, 32, 128) |
| 4 | LSTM(input 128, hidden 128, 2 layers, dropout 0.3) | (B, 32, 128) |
| 5 | Take the last time step | (B, 128) |
| 6 | Dropout(0.3) + Linear(128→6) | (B, 6) |

Three implementation details are particularly important:

- **Padding.** Each convolution uses kernel size 5 and padding 2, which preserves the temporal dimension. Consequently, the time axis is reduced only by the two pooling layers, and the shapes in Table 3 are exact for any input window whose length is divisible by 4.
- **Batch normalization.** BatchNorm is applied after each convolution to keep feature activations on a stable scale before they are passed to the LSTM. This is especially relevant in the hybrid model because the recurrent weights are sensitive to the scale of their inputs.
- **Last-step readout.** The classification head uses the LSTM output at the final time step rather than averaging outputs across the sequence. The final hidden state has incorporated information from the entire window, making it a natural summary for classification. This also matches the standalone LSTM baseline and keeps the comparison consistent.

Dropout of 0.3 is applied inside the LSTM, between its two layers, and again immediately before the linear classification head. The model is trained with cross-entropy loss on the six output logits, corresponding to standard multi-class classification without class weighting. According to `DATA.md`, the classes are sufficiently balanced across the data splits for class weighting to be unnecessary. Macro-F1 is nevertheless reported alongside accuracy so that performance differences between classes remain visible.

### 4.3 Input Versions

The main model is evaluated using the two input versions defined in `DATA.md`. This allows the effect of adding gyroscope information to be evaluated for the main model as well as for the baseline models.

**Table 4.** Input versions used for the main model.

| Version | Channels | Window shape | What the CNN sees first |
|---|---|---|---|
| Acc | `total_acc` x, y, z | 128 × 3 | 3 |
| Acc+Gyro | `total_acc` x, y, z + `body_gyro` x, y, z | 128 × 6 | 6 |

Only the number of input channels `C` changes between the two versions. All other hyperparameters remain fixed. This controlled setup ensures that any performance difference between the two input versions is attributable to the additional rotational information rather than to a separately tuned model.

### 4.4 Training Protocol

The main model uses the same hyperparameters as the CNN and LSTM baselines. Keeping the training procedure fixed ensures that comparisons between the three architectures are not confounded by different optimisation settings.

**Table 5.** Training hyperparameters shared by the CNN, LSTM, and CNN-LSTM models.

| Setting | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Max epochs | 30 |
| Loss | Cross-entropy |
| Early stopping | Validation macro-F1, patience 6 epochs (the reported epoch count includes these 6 epochs) |
| Model selection | Best validation macro-F1, restored before test evaluation |
| Seed | 42 (fixed by `configs/shared.yaml`) |

Seed 42 is defined in `configs/shared.yaml` and is applied to Python's `random` module, NumPy, and PyTorch before each run. It controls weight initialisation and batch shuffling. It does **not** regenerate the train/validation subject split: that split is created once by `src/prepare_data.py` using `--val-seed 42` and then reused across runs, as described in `DATA.md`.

The reported result for each input version is based on a single seed-42 run. Section 5 additionally reports a three-seed sweep using seeds 0, 1, and 2 as a robustness check. This sweep measures variation caused by initialisation and batch shuffling, but it does not capture the additional variability that could result from changing the subject-level train/validation split. It should therefore be interpreted as a lower bound on total run-to-run variability.

Early stopping and model selection are based on **validation macro-F1**, rather than accuracy or loss. Macro-F1 gives equal weight to all six classes, preventing strong performance on the larger dynamic classes from masking poor performance on a weaker class. The best validation checkpoint is restored before the official test evaluation, ensuring that the test set is not used to select the model.

### 4.5 Evaluation Protocol

The primary evaluation metrics are **test accuracy** and **macro-averaged F1**. Accuracy is the headline metric because it follows the benchmark convention for this dataset. Macro-F1 is reported alongside it to provide a class-balanced view of performance and to ensure that strong results on the three larger dynamic classes do not conceal poor performance on other classes.

The data splits follow the procedure described in `DATA.md`. The official test subjects remain untouched until final evaluation, and normalisation statistics are computed from the training windows only. Each run evaluates the test set exactly once, after early stopping has selected the best validation checkpoint.

The reported result consists of one seed-42 run for each input version, consistent with the fixed seed in `configs/shared.yaml`. Section 5 also reports an additional three-seed robustness sweep; these runs are kept separate from the headline results.

Training time is measured on the same machine for all runs and is reported alongside parameter count. Cost matters when comparing the three architectures: the 1D-CNN has the most parameters (about 0.57 million versus 0.31 million for the CNN-LSTM), and the LSTM is the slowest to train. Any accuracy difference should be read together with these costs.

### 4.6 Implementation

The models are implemented in PyTorch under `src/models/`. The main CNN-LSTM model is defined in `src/models/cnn_lstm.py`. The CNN and LSTM baselines are implemented in `src/models/cnn.py` and `src/models/lstm.py`, respectively, and are covered by the CNN/LSTM section of the report. Each file contains one architecture.

The CNN-LSTM is trained using `scripts/train_main_model.py`. This script is kept under `scripts/` rather than `src/train.py` because `src/train.py` serves as the shared trainer stub maintained by the project leader. The training script reads the seed and result paths from `configs/shared.yaml` and follows the same settings as the CNN/LSTM training harness in `scripts/train_cnn_lstm.py`. As a result, the training recipe described in Section 4.4 is shared rather than separately reimplemented for each model.

The script contains no model-specific training logic beyond instantiating `CNNLSTM`, allowing the same training infrastructure to be reused with a different model module when required.

### 4.7 Reproducing This Section

```bash
cd DL2026-Group14-Project37
pip install -r requirements.txt
python src/prepare_data.py                 # download UCI HAR and build acc.npz + acc_gyro.npz
python scripts/train_main_model.py         # Acc + Acc+Gyro, seed 42 -> results/cnn_lstm_*.json
python scripts/train_main_model.py --save-model   # also save checkpoints/cnn_lstm_<sensors>.pt for src/demo.py
python scripts/train_main_model.py --robustness   # optional: seeds 0, 1, 2 -> results/robustness/
```

`src/prepare_data.py` creates the two processed datasets and is the only step that requires network access. Before committing to a full training run, the pipeline can be checked with `--epochs 3` on a single input version. This is intended only as a smoke test; its resulting metrics must not be used in the report.

All paths are resolved relative to the project root, so the commands above behave consistently regardless of the working directory from which they are executed.

### References for This Section

- D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for human activity recognition using smartphones," in *Proc. ESANN*, 2013.
- S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 9(8), 1997.
- Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, 86(11), 1998.
- S. Ioffe and C. Szegedy, "Batch normalization: Accelerating internal covariate shift," *ICML*, 2015.
- D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *ICLR*, 2015.
