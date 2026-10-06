"""The main model of the project: a CNN-LSTM for raw inertial HAR.

This is the deliverable for the "Main model" part of the report. Each model lives in its own
file: the two baselines it is built from are `cnn_baseline.py` and `lstm_baseline.py` in this
folder, and this module composes their two halves rather than restating them.

How it relates to the baselines
------------------------------
  * `cnn_baseline.py`  - Conv1d-BN-ReLU-MaxPool x3 + global average pool.
  * `lstm_baseline.py` - nn.LSTM, hidden 128, 2 layers, dropout 0.3, last-step readout.
  * an earlier 1-layer CNN-LSTM prototype is not used: its single LSTM layer made it a weaker
    model than either baseline, so the comparison would have been unfair.

The main model keeps the convolutional front end of the CNN baseline (same kernel width, same
channel widths, same Conv-BN-ReLU-Pool order) and the recurrent back end of the LSTM baseline
(same hidden size, same depth, same dropout, same last-step readout), and connects the two
instead of pooling the CNN's features away with a global average.

Shape contract, for C input channels and a window of T = 128 samples:
    (B, T, C)
      -> transpose                (B, C, T)
      -> Conv1d(C,   64, k=5, pad=2) + BN + ReLU + MaxPool2   (B, 64, T/2)     local features
      -> Conv1d(64, 128, k=5, pad=2) + BN + ReLU + MaxPool2   (B, 128, T/4)    local features
      -> transpose                (B, T/4, 128)                            32 time steps
      -> LSTM(128, 128, 2 layers, dropout 0.3)                 (B, 32, 128)     temporal model
      -> last time step                                        (B, 128)
      -> Dropout(0.3) + Linear(128, 6)                        (B, 6)            logits
"""
import torch
import torch.nn as nn

# Design constants. Kept in one place so the report and the code cannot drift apart.
KERNEL_SIZE = 5      # same as cnn_baseline.py
CONV_CHANNELS = (64, 128)
HIDDEN = 128         # same as lstm_baseline.py
LSTM_LAYERS = 2      # same as lstm_baseline.py
DROPOUT = 0.3        # same as both baselines
N_CLASSES = 6        # WALKING ... LAYING
WINDOW = 128         # samples per window (2.56 s at 50 Hz), fixed by the dataset


def conv_block(c_in, c_out, pool=True):
    """One Conv1d-BN-ReLU-(pool) stage.

    `padding=2` with `kernel_size=5` keeps the time axis unchanged ('same' convolution), so the
    pooling is the only thing that reduces the length. BatchNorm after the convolution keeps the
    activations scaled for the LSTM that follows.
    """
    # Order matters: normalising the convolution output before the non-linearity is what keeps
    # the activations well scaled. Pooling is optional so the block can also be used unpooled.
    layers = [nn.Conv1d(c_in, c_out, KERNEL_SIZE, padding=KERNEL_SIZE // 2),
              nn.BatchNorm1d(c_out), nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class CNNLSTMMain(nn.Module):
    """CNN front end -> LSTM back end -> linear classifier. Input (batch, time, channels)."""

    def __init__(self, c, n_classes=N_CLASSES, dropout=DROPOUT, hidden=HIDDEN,
                 layers=LSTM_LAYERS, window=WINDOW):
        super().__init__()
        c1 = CONV_CHANNELS[0]
        c2 = CONV_CHANNELS[1]
        self.window = window
        # Two conv blocks, each followed by a pooling layer that halves the time axis.
        self.features = nn.Sequential(conv_block(c, c1), conv_block(c1, c2))
        # The two pooling layers divide the window by 4, which is how many steps the LSTM sees.
        self.n_steps = window // 4
        # dropout=0 is required for a single-layer LSTM; PyTorch warns if dropout is set there.
        self.rnn = nn.LSTM(c2, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"expected (batch, time, channels), got {tuple(x.shape)}")
        # Conv1d wants (batch, channels, time), so swap the axes, then swap back for the LSTM.
        z = self.features(x.transpose(1, 2)).transpose(1, 2)
        outputs, _ = self.rnn(z)
        # Read the final step only: it is the only one that has seen the whole window.
        return self.head(outputs[:, -1])


def count_parameters(model):
    """Total trainable parameters, split into the convolutional and recurrent parts.

    Reported in the Main Model results so the accuracy gain can be read against the cost.
    """
    # Split by sub-module so the conv/LSTM balance is visible; requires_grad skips frozen parts.
    conv = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
    rnn = sum(p.numel() for p in model.rnn.parameters() if p.requires_grad)
    head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    return {"conv": conv, "lstm": rnn, "head": head, "total": conv + rnn + head}


def describe(c):
    """Human-readable architecture summary, one line per stage.

    Printed by src/train.py before training and quoted in the Main Method, so this must stay in
    step with Table 3 of report/main_method.md.
    """
    model = CNNLSTMMain(c)
    params = count_parameters(model)
    first = CONV_CHANNELS[0]
    second = CONV_CHANNELS[1]
    pad = KERNEL_SIZE // 2
    pooled_once = model.window // 2
    return "\n".join([
        f"Input: (batch, {model.window}, {c}) - {c} channel(s), {model.window}-sample window",
        f"Conv1d({c} -> {first}, k={KERNEL_SIZE}, pad={pad})"
        f" + BatchNorm + ReLU + MaxPool(2)  -> (batch, {first}, {pooled_once})",
        f"Conv1d({first} -> {second}, k={KERNEL_SIZE}, pad={pad})"
        f" + BatchNorm + ReLU + MaxPool(2)  -> (batch, {second}, {model.n_steps})",
        f"LSTM(input {second}, hidden {HIDDEN}, {LSTM_LAYERS} layers, dropout {DROPOUT})"
        f"  -> (batch, {model.n_steps}, {HIDDEN})",
        f"Last time step -> Dropout({DROPOUT}) -> Linear({HIDDEN} -> {N_CLASSES})"
        f"  -> (batch, {N_CLASSES}) logits",
        f"Parameters: conv {params['conv']:,} + LSTM {params['lstm']:,}"
        f" + head {params['head']:,} = {params['total']:,}",
    ])


if __name__ == "__main__":
    # Shape check + parameter count. Run with:  python src/models/main_model.py
    torch.manual_seed(0)
    for name, c in (("Acc", 3), ("Acc+Gyro", 6)):
        model = CNNLSTMMain(c)
        x = torch.randn(4, WINDOW, c)
        y = model(x)
        assert y.shape == (4, N_CLASSES), y.shape
        print(f"--- CNN-LSTM main model, {name} input ({c} channels) ---")
        print(describe(c))
        print(f"Forward check: input {tuple(x.shape)} -> logits {tuple(y.shape)}  OK\n")