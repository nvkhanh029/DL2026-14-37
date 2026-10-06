"""Baseline model: LSTM for raw inertial HAR. Input (batch, time, channels).

Hidden size, depth, dropout and the last-step readout match the LSTM implemented earlier in the
project, so the main model can be compared against a baseline that differs from it in one
architectural respect only. Kept in its own file so it can be read, run and cited on its own. A
GRU variant also exists in the project's shared model module; only the LSTM is needed here.

Shape contract, for C input channels and a window of T = 128 samples:
    (B, T, C)  -- passed straight to the LSTM, no downsampling --
      -> LSTM(C -> 128, 2 layers, dropout 0.3)   (B, T, 128)
      -> last time step                          (B, 128)
      -> Dropout(0.3) -> Linear(128 -> 6)        (B, 6)  logits

The main model reuses exactly this recurrent back end and differs only in what happens before
it: here the full 128-step window is fed to the LSTM unmodified, whereas the main model
compresses the window to 32 steps with a CNN first. So the comparison between this baseline and
the main model isolates the contribution of the convolutional front end, and the comparison
against `cnn_baseline.py` isolates the contribution of the recurrent back end.
"""
import torch
import torch.nn as nn

# Shared design constants; mirrored by main_model.py so the two stay comparable.
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.3
N_CLASSES = 6
WINDOW = 128                # samples per window (2.56 s at 50 Hz), fixed by the dataset


class LSTMBaseline(nn.Module):
    """LSTM baseline. Input (batch, time, channels) -> logits (batch, n_classes)."""

    def __init__(self, c, hidden=HIDDEN, layers=LAYERS, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        # batch_first=True because the project's inputs are (batch, time, channels).
        # dropout applies between stacked layers, so it needs layers > 1 to take effect.

        # feature extractor: responsible for processing the sequence across time.
        self.rnn = nn.LSTM(c, hidden, layers, batch_first=True, dropout=dropout)

        # classifier: converts the LSTM's learned feature representation into final activity predictions.
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"expected (batch, time, channels), got {tuple(x.shape)}")
        # The second return value is (hidden, cell); only the output sequence is needed here.
        outputs, _ = self.rnn(x)
        # The last hidden state is the only one that has attended to the whole window.
        return self.head(outputs[:, -1])


def count_parameters(model):
    """Trainable parameters, split into the recurrent and classification parts."""
    rnn = sum(p.numel() for p in model.rnn.parameters() if p.requires_grad)
    head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    return {"lstm": rnn, "head": head, "total": rnn + head}


def describe(c):
    """Architecture summary, one line per stage. Quoted in the report."""
    model = LSTMBaseline(c)
    params = count_parameters(model)
    return "\n".join([
        f"Input: (batch, {WINDOW}, {c}) - fed to the LSTM without downsampling",
        f"LSTM(input {c} -> hidden {HIDDEN}, {LAYERS} layers, dropout {DROPOUT})"
        f"  -> (batch, {WINDOW}, {HIDDEN})",
        f"Last time step -> Dropout({DROPOUT}) -> Linear({HIDDEN} -> {N_CLASSES})"
        f"  -> (batch, {N_CLASSES}) logits",
        f"Parameters: LSTM {params['lstm']:,} + head {params['head']:,}"
        f" = {params['total']:,}",
    ])


if __name__ == "__main__":
    torch.manual_seed(0)
    for name, c in (("Acc", 3), ("Acc+Gyro", 6)):
        model = LSTMBaseline(c)
        x = torch.randn(4, WINDOW, c)
        y = model(x)
        assert y.shape == (4, N_CLASSES), y.shape
        print(f"--- LSTM baseline, {name} input ({c} channels) ---")
        print(describe(c))
        print(f"Forward check: input {tuple(x.shape)} -> logits {tuple(y.shape)}  OK\n")