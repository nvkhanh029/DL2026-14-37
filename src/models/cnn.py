"""Plain 1D-CNN for UCI-HAR activity recognition.

Input:  (batch, 128, n_channels) windows, already normalized by the data pipeline.
Output: (batch, 6) class logits, one per activity.
"""

from __future__ import annotations

from torch import nn


class CNN1D(nn.Module):
    """Two 1D convolution blocks + a small fully-connected head.

    The convolutions slide over time and learn local motion patterns
    (e.g. a specific arm-swing inside the 128-step window). Max-pooling
    halves the time dimension after each block, so the head sees a
    shorter, more abstract sequence.
    """

    def __init__(self, n_channels: int, n_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (B, n_channels, 128) -> (B, 64, 64)
            nn.Conv1d(n_channels, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Block 2: (B, 64, 64) -> (B, 128, 32)
            nn.Conv1d(64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # regularizes the large fully-connected layer
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x comes as (B, 128, C) from the loader; Conv1d wants (B, C, 128),
        # so swap the time and channel axes.
        x = x.transpose(1, 2)
        return self.classifier(self.features(x))
