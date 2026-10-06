## 4. Main Method

This section describes the main model of the project and the protocol used to train and evaluate
it. The dataset, its splits and its preprocessing are defined in `DATA.md`; here we fix the
architecture and the training recipe, so that the model and the baselines are compared under
identical conditions.

### 4.1 Design rationale

The task is to classify a 2.56 s window of inertial signal into one of six activities. Two
properties of the signal make a single network type a poor fit on its own:

* **Local patterns.** Activity identity is expressed by short, recurring waveform shapes — a
  heel strike, a step, a turn — that occupy only a few dozen samples. These are what a
  convolutional filter is good at detecting, and they are the same filters regardless of where in
  the window they occur.
* **Long-range structure.** The *order* and *rhythm* of those patterns also matter: climbing
  stairs and walking downstairs contain similar instantaneous motions but differ in how those
  motions repeat over the window. This is temporal structure over the whole window, which is what
  a recurrent network models.

The main model is therefore a **hybrid**: a small 1D CNN compresses each window into a short
sequence of learned local descriptors, and an LSTM models how those descriptors evolve across the
window before classifying. The convolutional stage reduces the 128-sample window to 32 time steps:
the two pooling layers give a stride of 4, so each step advances four samples and has a receptive
field of about 16 samples (roughly 0.3 s at 50 Hz). This removes local redundancy and cuts the
computational cost of the recurrence by a factor of four, while the steps are still fine enough
that a single step, which lasts roughly 25–35 samples (0.5–0.7 s), is still spread over several of
them.

The two halves are not chosen arbitrarily. The convolutional front end follows the same
`Conv1d → ReLU → MaxPool` motif, the same kernel width and the same first two channel widths as
the plain 1D-CNN baseline, but inserts BatchNorm after each convolution so that the LSTM receives
well-scaled inputs. The recurrent back end follows the standalone LSTM: hidden size 128, two
layers and the same last-step readout, with dropout 0.3 instead of the baseline's 0.5. The main
model is therefore *not* a strict superset of the two baselines — it adds BatchNorm and uses its
own dropout — but the two branches keep comparable capacity, so a difference in accuracy can be
read as a property of the fusion rather than of a much larger network. All three models also share
one training protocol (Section 4.4), so the comparison in Section 5 is not confounded by the recipe.

The parameter counts in Table M1 show where the capacity sits. For the Acc input the model has
307,462 parameters, of which the convolutional front end holds 42,496 (14%), the LSTM 264,192
(86%), and the linear head 774. The model is therefore dominated by the recurrent branch, not
evenly balanced between the two. That is a consequence of the design above rather than an accident:
a 2-layer LSTM with 128 hidden units is expensive relative to two narrow convolutions, and reusing
the standalone LSTM's hidden size and depth keeps the recurrence comparable to that baseline. The
practical consequence is that training time is dominated by the recurrence, so the front end's
4x reduction of the sequence length is what keeps the model affordable at all.

### 4.2 Architecture

**Table 3.** Main model (CNN-LSTM), input `(batch, 128, C)` with `C` sensor channels.

| # | Stage | Output shape |
|---|---|---|
| 0 | Input window | (B, 128, C) |
| 1 | Conv1d(C→64, k=5, pad=2) + BatchNorm + ReLU + MaxPool(2) | (B, 64, 64) |
| 2 | Conv1d(64→128, k=5, pad=2) + BatchNorm + ReLU + MaxPool(2) | (B, 128, 32) |
| 3 | Transpose to time-major layout | (B, 32, 128) |
| 4 | LSTM(input 128, hidden 128, 2 layers, dropout 0.3) | (B, 32, 128) |
| 5 | Take the last time step | (B, 128) |
| 6 | Dropout(0.3) + Linear(128→6) | (B, 6) |

Three implementation details are worth noting.

* **Padding.** Kernel size 5 with padding 2 is a *same* convolution, so the time axis is only
  reduced by the two pooling layers and the shape arithmetic in Table 3 is exact for any window
  length divisible by 4.
* **BatchNorm in the front end.** Batch normalisation after each convolution keeps the feature
  activations on a scale the LSTM can be trained on stably. This matters more here than in a
  plain CNN, because the LSTM's recurrent weights are sensitive to the scale of their input.
* **Last-step readout.** The classification head reads the LSTM's output at the final time step
  rather than an average over steps. The last hidden state is the only one that has attended to
  the whole window, so it is the natural summary; using it also matches the standalone LSTM
  baseline, keeping the two comparable.

Dropout of 0.3 is applied inside the LSTM (between its two layers) and again before the linear
head. We use cross-entropy loss on the six logits, i.e. plain softmax classification with no class
weighting. `DATA.md` records that the classes are close enough to balanced across the splits for
this to be appropriate; macro-F1 is reported alongside accuracy precisely so that any residual
per-class imbalance would be visible.

### 4.3 Input versions

The main model is evaluated on the two input versions defined in `DATA.md`, so that the effect of
adding the gyroscope can be measured on the main model as well as on the baselines.

**Table 4.** Input versions used for the main model.

| Version | Channels | Window shape | What the CNN sees first |
|---|---|---|---|
| Acc | `total_acc` x, y, z | 128 × 3 | 3 |
| Acc+Gyro | `total_acc` x, y, z + `body_gyro` x, y, z | 128 × 6 | 6 |

Only the number of input channels `C` changes between the two versions; every other
hyper-parameter is fixed and shared. This is deliberate: any difference in accuracy between the
two rows of Table M1 is attributable to the extra rotational information, not to a re-tuned
model.

### 4.4 Training protocol

The main model is trained with the same hyper-parameters as the CNN and LSTM baselines, so a
comparison between the three is not confounded by the training recipe. The settings below are the
shared ones.

**Table 5.** Training hyper-parameters, shared by the CNN, LSTM and CNN-LSTM models.

| Setting | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Max epochs | 30 |
| Loss | Cross-entropy |
| Early stopping | Validation macro-F1, patience 6 epochs |
| Model selection | Best validation macro-F1, restored before the test evaluation |
| Seed | 42 (fixed by `configs/shared.yaml`) |

Seed 42 is fixed by `configs/shared.yaml` and is set for `random`, NumPy and PyTorch before each
run. It controls the weight initialisation and the batch shuffling; it does **not** re-draw the
train/validation subject split, which is chosen once by `src/prepare_data.py` at `--val-seed 42`
and reused by every run (see DATA.md). The reported result is a single seed-42 run per input
version; Section 5 additionally lists an extra three-seed sweep (seeds 0, 1, 2) as a robustness
check on initialisation and shuffling variance. That spread is a lower bound on the true
run-to-run variability, because it omits the variance a different subject-level split would
introduce.

Early stopping and model selection monitor **validation macro-F1**, not accuracy or loss.
Macro-F1 weights the six classes equally, so a class the model is failing cannot be hidden by the
three large dynamic classes — which is exactly the failure mode early stopping should guard
against. Because the official test set must not be consulted for model selection, the weights from
the best validation epoch are restored before the single test evaluation.

### 4.5 Evaluation protocol

Metrics are test accuracy and macro-averaged F1. Accuracy is the headline number because it
matches the benchmark convention for this dataset; macro-F1 is reported next to it so that
performance cannot be carried by the three dynamic classes while a static class fails.

Splits follow `DATA.md`: the official test subjects are untouched until the final evaluation of a
run, normalisation statistics come from the training windows only, and each run touches the test
set exactly once, after early stopping has selected the checkpoint. The reported result is a single
seed-42 run per input version, matching the fixed seed in `configs/shared.yaml`; Section 5
additionally lists an extra three-seed sweep as a robustness check, kept separate from the reported
numbers.

Training time is measured on the same machine for all runs and is reported next to the parameter
count, because the fusion model is the most expensive of the three and the accuracy it buys has
to be weighed against that cost.

### 4.6 Implementation

The model is implemented in PyTorch under `src/models/`. The main model is
`src/models/cnn_lstm.py`; the CNN and LSTM baselines are `src/models/cnn.py` and
`src/models/lstm.py` respectively and are owned and reported by the CNN/LSTM section of the
report. Each file defines one architecture only.

The CNN-LSTM is trained by `scripts/train_main_model.py`. It is deliberately kept in `scripts/`
rather than `src/train.py`, because `src/train.py` is the leader's shared-trainer stub. The script
reads the seed and the result paths from `configs/shared.yaml` and uses the same settings as the
CNN/LSTM harness (`scripts/train_cnn_lstm.py`), so the training recipe of Section 4.4 is shared
rather than re-implemented per model. It has no model-specific logic beyond instantiating
`CNNLSTM`, so a baseline can be run through the same code by passing a different module.

### 4.7 Reproducing this section

```bash
cd dl-data
pip install -r requirements.txt
python src/prepare_data.py                 # download UCI HAR and build acc.npz + acc_gyro.npz
python scripts/train_main_model.py         # Acc + Acc+Gyro, seed 42 -> results/cnn_lstm_*.json
python scripts/train_main_model.py --robustness   # optional: seeds 0, 1, 2 -> results/robustness/
```

`src/prepare_data.py` writes the two processed datasets and is the only step that needs network
access. To confirm the pipeline runs before committing to a run, use `--epochs 3` with a single
input version; that is a smoke test and its numbers are not reportable.

All paths resolve relative to the project root, so the commands behave identically from any
working directory.

### References for this section
* D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for
  human activity recognition using smartphones," in *Proc. ESANN*, 2013.
* S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 9(8), 1997.
* Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document
  recognition," *Proc. IEEE*, 86(11), 1998.
* S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing
  internal covariate shift," *ICML*, 2015.
* D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *ICLR*, 2015.