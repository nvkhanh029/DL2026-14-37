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

The two halves are not chosen arbitrarily. The convolutional front end keeps the kernel width,
channel widths, and layer order (`Conv1d → BatchNorm → ReLU → MaxPool`) of the 1D-CNN already
implemented for this project, and the recurrent back end keeps the hidden size, depth, dropout
and last-step readout of the standalone LSTM. The main model is thus a strict superset of the two
baselines: any difference in accuracy is attributable to the fusion rather than to a change of
capacity in one branch, which is what makes the comparison in Section 5 interpretable.

The parameter counts in Table M1 show where the capacity sits. For the Acc input the model has
307,462 parameters, of which the convolutional front end holds 42,496 (14%), the LSTM 264,192
(86%), and the linear head 774. The model is therefore dominated by the recurrent branch, not
evenly balanced between the two. That is a consequence of the design above rather than an accident:
a 2-layer LSTM with 128 hidden units is expensive relative to two narrow convolutions, and matching
the standalone LSTM's capacity is what makes the comparison against that baseline meaningful. The
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

The main model is trained by a single model-agnostic loop, so the same hyper-parameters below apply
to the baselines unchanged and a comparison is not confounded by the training recipe.

**Table 5.** Training hyper-parameters, identical for every model and sensor version.

| Setting | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Max epochs | 40 |
| LR schedule | `ReduceLROnPlateau`, factor 0.5, patience 3, on validation loss |
| Gradient clipping | Global norm 1.0 |
| Loss | Cross-entropy |
| Early stopping | Validation accuracy, patience 8 epochs |
| Model selection | Best validation accuracy; ties broken by lower validation loss |
| Seeds | 0, 1, 2 (mean ± standard deviation reported) |

Each seed controls the weight initialisation and the batch shuffling. It does **not** re-draw the
train/validation subject split: that is chosen once by `src/prepare_data.py` at `--val-seed 42` and
reused by every run (see DATA.md). The reported standard deviation therefore measures
initialisation and shuffling variance only; it is a lower bound on the true run-to-run spread,
because it omits the variance a different subject-level split would introduce. With 21 training
subjects that omitted term is plausibly the larger of the two.

Gradient clipping at norm 1.0 is applied on every step; the recurrent part of the model is the part
most at risk of exploding gradients, and clipping removes that failure mode without needing to
lower the learning rate.

Early stopping monitors **validation accuracy**, not validation loss. Loss keeps improving after
the accuracy has plateaued, so stopping on loss would routinely train past the point where the
model generalises best. Because the official test set must not be consulted for model selection,
the weights from the best validation epoch are restored before the single test evaluation.

### 4.5 Evaluation protocol

Metrics are test accuracy and macro-averaged F1, plus per-class recall from the confusion matrix.
Accuracy is the headline number because it matches the benchmark convention for this dataset;
macro-F1 is reported next to it so that performance cannot be carried by the three dynamic
classes while a static class fails.

Splits follow `DATA.md`: the official test subjects are untouched until the final evaluation of a
run, normalisation statistics come from the training windows only, and each run touches the test
set exactly once, after early stopping has selected the checkpoint. Every reported figure is a mean
over three seeds with the sample standard deviation, and per-seed values are listed in Table M2 so
the spread can be inspected rather than taken on trust.

Training time is measured on the same machine for all runs and is reported next to the parameter
count, because the fusion model is the most expensive of the three and the accuracy it buys has
to be weighed against that cost.

### 4.6 Implementation

The model is implemented in PyTorch, one file per model, under `src/models/`:

| File | Model |
|---|---|
| `src/models/main_model.py` | the main CNN-LSTM of Table 3 |
| `src/models/cnn_baseline.py` | 1D CNN baseline |
| `src/models/lstm_baseline.py` | LSTM baseline |

The CNN and the LSTM were originally implemented earlier in the project inside a single shared
module that also contained a GRU, an early 1-layer CNN-LSTM prototype and a Transformer. For this
deliverable the CNN and the LSTM are split into their own files so that each can be read, run and
cited on its own. An earlier 1-layer CNN-LSTM prototype was deliberately not adopted as the main
model: with fewer recurrent layers than the LSTM baseline it would have been a weaker model, and
the comparison would have answered a different question than the one Section 5 asks.

`train_and_evaluate` in `src/train.py` takes the model as an argument and contains no
model-specific logic, so a baseline can be run through exactly the same loop and the same
hyper-parameters as the main model by passing a different module. The baseline files therefore
define architecture only; the training recipe of Section 4.4 is shared rather than re-implemented.
As shipped, the command-line entry point wires up the main model, since that is what this section
reports.

### 4.7 Reproducing this section

```bash
cd dl-data
pip install -r requirements.txt
python src/prepare_data.py           # download UCI HAR and build acc.npz + acc_gyro.npz
python src/train.py                  # Acc and Acc+Gyro x 3 seeds -> results/main/runs/*.json
python src/report_main_model.py      # Tables M1-M3 -> report/, Figures M1-M4 -> figures/
```

`src/prepare_data.py` writes the two processed datasets and is the only step that needs network
access. For a quick check that the pipeline runs before committing to the full grid:

```bash
python src/train.py --fast           # 15 epochs, one seed: a smoke test, not reportable
```

Regenerating the report is idempotent: it replaces only the block between the GENERATED markers, so
the prose in `report/main_model_results.md` is never overwritten by a re-run. All paths resolve
relative to the project root, so the commands behave identically from any working directory.

### References for this section
* D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for
  human activity recognition using smartphones," in *Proc. ESANN*, 2013.
* S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 9(8), 1997.
* Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document
  recognition," *Proc. IEEE*, 86(11), 1998.
* S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing
  internal covariate shift," *ICML*, 2015.
* D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *ICLR*, 2015.