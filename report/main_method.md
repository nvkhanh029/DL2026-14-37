## 4. Main Method

This section describes the main model and how we trained and evaluated it. The dataset, the subject splits and the preprocessing are defined in `DATA.md` and are not repeated here. What matters for this section is that the main model and the two deep baselines see the same data and the same training recipe, so that any difference between them comes from the architecture.

### 4.1 Design Rationale

Each input is a 2.56-second window of inertial data, and the model has to assign it to one of six activities. Two kinds of information in that window are useful, and they suit different kinds of network.

- **Local patterns.** A heel strike, a single step or a turn shows up as a short waveform that can appear anywhere in the window. Convolutional filters are a natural fit: they look for the same shape at every position.
- **Temporal structure.** The order and rhythm of those patterns also carry meaning. Walking upstairs and walking downstairs contain similar individual movements; what separates them is how the movements follow one another. A recurrent network is designed to model exactly that kind of sequence.

We therefore use a hybrid CNN-LSTM. A small 1D CNN turns the raw window into a shorter sequence of learned local features, and an LSTM reads that sequence to decide the activity.

The convolutional front end shortens the window from 128 samples to 32 time steps. Its two pooling layers give an effective stride of 4, and each feature it produces covers about 16 input samples, or roughly 0.3 s at 50 Hz. This removes redundancy between neighbouring samples and makes the recurrent stage four times cheaper. The sequence is still fine enough for its purpose: a single step lasts roughly 25-35 samples (0.5-0.7 s), so it is spread over several LSTM time steps rather than squeezed into one.

We did not design the two halves from scratch. The front end follows the plain 1D-CNN baseline: the same `Conv1d → ReLU → MaxPool` block, the same kernel width and the same first two channel widths, with BatchNorm added after each convolution. The back end follows the standalone LSTM baseline: hidden size 128, two layers and the same last-step readout. The only other change is the dropout rate, 0.3 here instead of 0.5.

The CNN-LSTM is therefore not a strict union of the two baselines, because of BatchNorm and the different dropout. Its parts do, however, have the same capacity as the baselines they come from, so a difference in accuracy cannot simply be explained by a much larger network. All three models are also trained with the protocol in Section 4.4.

Most of the model's parameters sit in the LSTM. With the Acc input the network has 307,462 parameters: 42,496 (14%) in the convolutional front end, 264,192 (86%) in the LSTM and 774 in the classification head. This follows from keeping the LSTM identical in size to the baseline: two layers of 128 hidden units cost far more than two narrow convolutions. For the same reason, the recurrent stage dominates the training time, which is why shortening the sequence by a factor of four in the CNN matters in practice.

### 4.2 Architecture

**Table 3.** Main model (CNN-LSTM). The input has shape `(batch, 128, C)`, where `C` is the number of sensor channels.

| # | Stage | Output shape |
|---|---|---|
| 0 | Input window | (B, 128, C) |
| 1 | Conv1d(C→64, k=5, pad=2) + BatchNorm + ReLU + MaxPool(2) | (B, 64, 64) |
| 2 | Conv1d(64→128, k=5, pad=2) + BatchNorm + ReLU + MaxPool(2) | (B, 128, 32) |
| 3 | Transpose to time-major layout | (B, 32, 128) |
| 4 | LSTM(input 128, hidden 128, 2 layers, dropout 0.3) | (B, 32, 128) |
| 5 | Take the last time step | (B, 128) |
| 6 | Dropout(0.3) + Linear(128→6) | (B, 6) |

Three details are worth spelling out.

- **Padding.** Every convolution uses kernel size 5 with padding 2, so it keeps the length of the time axis. Only the pooling layers shorten it, which is why the shapes in Table 3 hold exactly for any window length divisible by 4.
- **Batch normalisation.** BatchNorm after each convolution keeps the features on a stable scale before they reach the LSTM, whose gates are sensitive to the size of their inputs.
- **Last-step readout.** The classifier uses the LSTM output at the final time step instead of an average over all steps. By then the LSTM has read the whole window, so this state is a reasonable summary of it. The standalone LSTM baseline does the same, which keeps the two comparable.

Dropout of 0.3 is applied between the two LSTM layers and again just before the linear head. The model is trained with plain cross-entropy over the six classes, without class weights; `DATA.md` shows that the classes are balanced enough across the splits for weighting to be unnecessary. We still report macro-F1 next to accuracy so that a weak class cannot hide behind strong ones.

### 4.3 Input Versions

The main model is trained on both input versions from `DATA.md`, so the effect of the gyroscope can be measured for this model as well as for the baselines.

**Table 4.** Input versions used for the main model.

| Version | Channels | Window shape | Input channels to the first convolution |
|---|---|---|---|
| Acc | `total_acc` x, y, z | 128 × 3 | 3 |
| Acc+Gyro | `total_acc` x, y, z + `body_gyro` x, y, z | 128 × 6 | 6 |

The number of input channels is the only thing that changes between the two versions. Every other setting stays the same, and the model is not re-tuned for either input, so a difference between the two runs can be put down to the rotational information the gyroscope adds.

### 4.4 Training Protocol

The CNN-LSTM uses exactly the same training settings as the CNN and LSTM baselines. If the recipe differed between models, a comparison between architectures would also be a comparison between optimisation settings.

**Table 5.** Training settings shared by the CNN, LSTM and CNN-LSTM.

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

The seed comes from `configs/shared.yaml` and is applied to Python's `random`, NumPy and PyTorch before every run. It fixes the weight initialisation and the order of the training batches. It does not touch the train/validation split: that split is made once by `src/prepare_data.py` (with `--val-seed 42`) and then reused by every run, as `DATA.md` explains.

Each headline result is a single seed-42 run. Section 5 adds a small sweep over seeds 0, 1 and 2 to show how much the result moves with initialisation and batch order. Because the subject split is held fixed in that sweep, it does not measure the extra variation that a different choice of validation subjects would bring, and it should be read as a lower bound on the true run-to-run spread.

Both early stopping and model selection use validation macro-F1 rather than accuracy or loss. Macro-F1 weights the six classes equally, so a model cannot look good by doing well on the large walking classes while failing on a smaller one. The best validation checkpoint is restored before the test set is touched, and the test set plays no part in choosing the model.

### 4.5 Evaluation Protocol

We report test accuracy and macro-averaged F1. Accuracy is the usual headline figure for this dataset; macro-F1 gives the class-balanced view next to it.

The splits follow `DATA.md`. The official test subjects are not used until the final evaluation, and the normalisation statistics come from the training windows only. Each run evaluates the test set once, after early stopping has picked the best validation checkpoint. The seed-42 runs are the reported results; the three-seed sweep in Section 5 is kept separate from them.

Training time is listed next to the parameter count, but it was not measured on a single machine. In this report we name the machines after the team member who ran them: **B** ran the CNN and LSTM baselines, **C** the CNN-LSTM robustness sweep, and **D** the reported CNN-LSTM runs. As Section 5 notes, the times are therefore only indicative. Cost is still worth keeping in view: the 1D-CNN has the most parameters (about 0.57 million against 0.31 million for the CNN-LSTM), and the LSTM is by far the slowest to train.

### 4.6 Implementation

All models are written in PyTorch under `src/models/`, one architecture per file. The CNN-LSTM is in `src/models/cnn_lstm.py`; the CNN and LSTM baselines are in `src/models/cnn.py` and `src/models/lstm.py` and are described in the CNN/LSTM section of the report.

The CNN-LSTM is trained by `scripts/train_main_model.py`. It lives in `scripts/` because `src/train.py` is the shared trainer maintained by the project leader. The script reads the seed and the output paths from `configs/shared.yaml` and copies the settings of the CNN/LSTM harness (`scripts/train_cnn_lstm.py`), so the recipe in Section 4.4 is shared rather than re-implemented for each model. Apart from building a `CNNLSTM`, it contains no model-specific logic, and another model could be trained with it by swapping that one line.

### 4.7 Reproducing This Section

```bash
cd DL2026-Group14-Project37
pip install -r requirements.txt
python src/prepare_data.py                        # download UCI HAR and build acc.npz + acc_gyro.npz
python scripts/train_main_model.py                # Acc + Acc+Gyro, seed 42 -> results/cnn_lstm_*.json
python scripts/train_main_model.py --save-model   # also save checkpoints/cnn_lstm_<sensors>.pt for src/demo.py (re-trains, see below)
python scripts/train_main_model.py --robustness   # optional: seeds 0, 1, 2 -> results/robustness/
```

Only `src/prepare_data.py` needs network access. To check the pipeline before a full run, use `--epochs 3` on one input version. The script treats any run with a changed protocol (`--epochs`, `--patience`, `--lr`, `--batch-size` or a non-default `--seed`) as a smoke test and writes it to `results/smoke/` instead of `results/`. Those numbers must not appear in the report.

`--save-model` trains the model again, so it also overwrites `results/cnn_lstm_*.json` and `results/preds/*.npz`. Only machine D, which produced the reported numbers, should commit those files. On any other machine, run it to get the checkpoint and then `git restore results/` before committing.

Seed 42 makes a run repeat itself on the same machine with the same software. It does not guarantee identical numbers elsewhere: a different CPU or GPU, PyTorch version or thread count can change the result. A re-run on another machine should land close to Table M1, but not necessarily on it.

All paths are resolved from the project root, so the commands work from any working directory.

### References for This Section

- D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for human activity recognition using smartphones," in *Proc. ESANN*, 2013.
- S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 9(8), 1997.
- Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, 86(11), 1998.
- S. Ioffe and C. Szegedy, "Batch normalization: Accelerating internal covariate shift," *ICML*, 2015.
- D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *ICLR*, 2015.
