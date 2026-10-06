"""LSTM classifier for UCI-HAR activity recognition.

Input:  (batch, 128, n_channels) windows, already normalized by the data pipeline.
Output: (batch, 6) class logits, one per activity.
"""

from __future__ import annotations

from torch import nn


class LSTMClassifier(nn.Module):
    """Stacked LSTM that reads the window step by step.

    Unlike the CNN, the LSTM processes the whole window as a sequence and
    can in principle connect motion patterns that are far apart in time.
    We take its hidden state at the last time step and classify from it.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # input shape is (B, 128, C), not (128, B, C)
            dropout=dropout,   # dropout between LSTM layers (no effect if num_layers == 1)
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        # x: (B, 128, C) -> out: (B, 128, hidden_size), one hidden state per step.
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # hidden state after the final of the 128 steps
        return self.head(last_step)
