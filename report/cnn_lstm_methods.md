## Methods: CNN and LSTM (Model B)

Both models are trained on the **Acc+Gyro** input version (6 channels:
`total_acc_x/y/z` + `body_gyro_x/y/z`, windows of 128 steps = 2.56 s at 50 Hz)
and predict one of the six activities. All preprocessing (subject-wise split,
per-channel z-score normalization fitted on training data only) is done by the
shared data pipeline and is described in the Dataset section; the models
receive the normalized windows directly.

### Plain 1D-CNN

The 1D-CNN treats a sensor window as a one-dimensional signal with six
channels, mirroring how a 2D-CNN treats an image. Convolutions slide along the
time axis and learn local motion patterns (e.g., a specific arm swing or
posture transition inside the window), while max-pooling makes the learned
features robust to small shifts of a pattern within the window.

| Layer | Output shape | Notes |
|---|---|---|
| Input | (128, 6) | normalized window |
| Conv1d, 64 filters, kernel 5, padding 2 | (128, 64) | ReLU |
| MaxPool1d, kernel 2 | (64, 64) | halves the time axis |
| Conv1d, 128 filters, kernel 5, padding 2 | (64, 128) | ReLU |
| MaxPool1d, kernel 2 | (32, 128) | halves the time axis again |
| Flatten + Linear(4096 → 128) | (128,) | ReLU, Dropout 0.5 |
| Linear(128 → 6) | (6,) | class logits |

The two convolutions use the same kernel width (5 steps = 0.1 s), chosen to
capture short motion primitives; the receptive field grows with depth, so the
second block sees patterns spanning roughly 0.2 s of raw signal. Dropout 0.5
before the final layer reduces overfitting on the 5,800 training windows. The
model has about 0.57 M parameters.

### LSTM

The LSTM reads the same window as a sequence of 128 time steps and maintains a
hidden state that can carry information across long spans of the window, which
complements the CNN's purely local filters.

| Layer | Output shape | Notes |
|---|---|---|
| Input | (128, 6) | normalized window |
| LSTM, hidden 128, 2 layers | (128, 128) | Dropout 0.5 between layers |
| Take last time step | (128,) | hidden state after the full window |
| Linear(128 → 6) | (6,) | class logits |

We classify from the hidden state at the final time step: after reading the
whole window it summarizes the activity trajectory. The model has about
0.20 M parameters.

### Training protocol (shared by both models)

- **Optimizer:** Adam, learning rate 1e-3; loss: cross-entropy.
- **Batch size:** 64 (training batches shuffled; validation/test in fixed order).
- **Epochs:** at most 30, with early stopping after 6 epochs without
  improvement in validation macro-F1; the checkpoint with the best validation
  macro-F1 is kept. The `epochs` value in each result file is the number of
  epochs actually run, **including** those last 6 patience epochs that brought
  no improvement, so the saved checkpoint comes from an earlier epoch.
- **Seeding:** seed 42 for `random`, `numpy`, and `torch` before each run.
- **Model selection:** validation split only (subjects 17, 25, 26, 30). The
  test split is evaluated exactly once, after training is finished.

The training loop lives in `scripts/train_cnn_lstm.py` (a temporary harness
until the shared trainer in `src/train.py` lands); the models are
`src/models/cnn.py` and `src/models/lstm.py`.

## Main research experiment: CNN and LSTM on Acc+Gyro

In the main experiment's model comparison (Random Forest → CNN / LSTM →
CNN-LSTM, all on Acc+Gyro), our two models occupy the middle of the ladder:
they test whether replacing hand-crafted features (Random Forest baseline)
with **learned** temporal representations improves recognition, before the
CNN-LSTM hybrid combines both ideas.

| Model | Sensors | Accuracy | Macro-F1 |
|---|---|---|---|
| Random Forest (baseline) | acc_gyro | 0.8039 | 0.8016 |
| **CNN (ours)** | acc_gyro | 0.9131 | 0.9125 |
| **LSTM (ours)** | acc_gyro | 0.8935 | 0.8936 |
| CNN-LSTM (main model) | acc_gyro | 0.9199 | 0.9205 |

**What we observe.** Both learned models reach about 0.9 accuracy on the
unseen test subjects, so replacing hand-crafted features with learned
representations already pays off; the final comparison against Random Forest
and CNN-LSTM is made in their sections. The CNN (0.9131 accuracy, 0.9125
macro-F1) is about 2 points ahead of the LSTM (0.8935 accuracy,
0.8936 macro-F1), while using roughly three times more parameters
(0.57 M vs 0.20 M). Both runs stopped early (CNN after 9, LSTM after 14 of
the 30 allowed epochs — those counts include the 6 patience epochs that brought
no improvement), which means validation macro-F1 stopped improving long before
the epoch budget and there is no sign that either model was still learning on
the training split.

**Where the errors are.** The two confusion matrices show the same picture.
Both models classify `LAYING` perfectly (recall 1.000) and are strong on the
three dynamic activities — walking, walking upstairs and walking downstairs —
with recalls between 0.87 and 0.99. The largest error cluster is the swap
between the two static postures `SITTING` and `STANDING`: 149 of the CNN's 256
test errors and 192 of the LSTM's 314 are that single confusion. This is
expected, because the two activities share the same body orientation and
differ mainly in fine acceleration structure rather than in gross motion. The
second cluster is `WALKING` vs `WALKING_DOWNSTAIRS`, where both models
over-predict downstairs (precision ≈ 0.85 for both despite recall ≥ 0.97): the
gait patterns are close and only the slope distinguishes them.

**Why the CNN wins.** A short kernel already sees the local acceleration
signature that separates sitting from standing, and max-pooling makes that
detection insensitive to where in the 2.56 s window it happens. The LSTM has
to build the same evidence step by step, which costs an order of magnitude
more compute (447 s vs 44 s of training on this machine) and, with only
5,800 training windows, appears to be the more data-hungry of the two. The
gap is small, so the LSTM is not a failure: its last hidden state already
summarizes the window well enough to stay within 2 points of the CNN.

**Limitation.** These are single runs at seed 42 on CPU. The ranking
CNN > LSTM is a 2-point difference from one seed each, so it should be read as
"the plain 1D-CNN is at least as good here", not as a precisely measured
margin; the CNN-LSTM section and the analysis runs add the context needed to
interpret it.
