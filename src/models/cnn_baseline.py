"""Baseline model: 1D CNN for raw inertial HAR. Input (batch, time, channels).

The layer sizes, kernel width, layer order and dropout match the 1D-CNN implemented earlier in
the project, so the main model can be compared against a baseline that differs from it in one
architectural respect only. Kept in its own file so it can be read, run and cited on its own.

Shape contract, for C input channels and a window of T = 128 samples:
    (B, T, C)
      -> transpose                                    (B, C, T)
      -> Conv1d(C, 64)   + BN + ReLU + MaxPool(2)      (B,  64, T/2)
      -> Conv1d(64, 128) + BN + ReLU + MaxPool(2)      (B, 128, T/4)
      -> Conv1d(128, 128)+ BN + ReLU + MaxPool(2)      (B, 128, T/8)
      -> AdaptiveAvgPool1d(1)                         (B, 128, 1)
      -> Flatten -> Dropout(0.3) -> Linear(128 -> 6)  (B, 6)  logits
    every convolution uses kernel=5, padding=2

The global average pool is what separates this baseline from the main model: it collapses the
time axis completely, so the classifier sees a single averaged descriptor and the network has no
representation of *when* something happened in the window. The main model keeps a sequence and
models its order, which is the difference Section 5 is about.
"""
import torch
import torch.nn as nn

# Shared design constants; mirrored by main_model.py so the two stay comparable.
KERNEL_SIZE = 5
CHANNELS = (64, 128, 128)   # output width of the three convolution blocks
DROPOUT = 0.3
N_CLASSES = 6
WINDOW = 128                # samples per window (2.56 s at 50 Hz), fixed by the dataset


def conv_block(c_in, c_out):
    """One Conv1d-BatchNorm-ReLU-MaxPool stage. `padding=2` with `k=5` keeps the time axis."""
    # padding=2 with kernel 5 is a 'same' convolution, so MaxPool(2) is the only downsampling.
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, KERNEL_SIZE, padding=KERNEL_SIZE // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),                  # f(x) = max(0, x)
        nn.MaxPool1d(2),
    )


class CNN1DBaseline(nn.Module):
    """1D CNN baseline. Input (batch, time, channels) -> logits (batch, n_classes)."""

    def __init__(self, c, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        c1 = CHANNELS[0]
        c2 = CHANNELS[1]
        c3 = CHANNELS[2]
        # Three conv blocks (128 -> 64 -> 32 -> 16 steps), then average over time to one vector. (convolutional feature block)
        self.features = nn.Sequential(conv_block(c, c1), conv_block(c1, c2), conv_block(c2, c3),
                                     nn.AdaptiveAvgPool1d(1))

        # head: classification layer
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c3, n_classes))

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"expected (batch, time, channels), got {tuple(x.shape)}")
        # Conv1d wants (batch, channels, time), so move the channel axis in front of time.
        channels_first = x.transpose(1, 2)
        # The global average pool removes the time axis, so the classifier sees one vector.
        return self.head(self.features(channels_first))


def count_parameters(model):
    """Trainable parameters, split into the convolutional and classification parts."""
    conv = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
    head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    return {"conv": conv, "head": head, "total": conv + head}


def describe(c):
    """Architecture summary, one line per stage. Quoted in the report."""
    model = CNN1DBaseline(c)
    params = count_parameters(model)
    first = CHANNELS[0]
    second = CHANNELS[1]
    third = CHANNELS[2]
    pad = KERNEL_SIZE // 2
    return "\n".join([
        f"Input: (batch, {WINDOW}, {c})",
        f"Conv1d({c} -> {first}, k={KERNEL_SIZE}, pad={pad}) + BN + ReLU + MaxPool(2)"
        f"  -> (batch, {first}, {WINDOW // 2})",
        f"Conv1d({first} -> {second}, k={KERNEL_SIZE}, pad={pad}) + BN + ReLU + MaxPool(2)"
        f"  -> (batch, {second}, {WINDOW // 4})",
        f"Conv1d({second} -> {third}, k={KERNEL_SIZE}, pad={pad}) + BN + ReLU + MaxPool(2)"
        f"  -> (batch, {third}, {WINDOW // 8})",
        f"AdaptiveAvgPool1d(1) -> Flatten -> Dropout({DROPOUT})"
        f" -> Linear({third} -> {N_CLASSES})",
        f"Parameters: conv {params['conv']:,} + head {params['head']:,} = {params['total']:,}",
    ])


if __name__ == "__main__":
    torch.manual_seed(0)
    for name, c in (("Acc", 3), ("Acc+Gyro", 6)):
        model = CNN1DBaseline(c)
        x = torch.randn(4, WINDOW, c)
        y = model(x)
        assert y.shape == (4, N_CLASSES), y.shape
        print(f"--- 1D CNN baseline, {name} input ({c} channels) ---")
        print(describe(c))
        print(f"Forward check: input {tuple(x.shape)} -> logits {tuple(y.shape)}  OK\n")