import torch
import torch.nn as nn
from typing import Dict, Any

class MusicRNN(nn.Module):
    def __init__ (
        self,
        num_pitches: int = 88,
        hidden_size: int=256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=num_pitches,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc_out = nn.Linear(hidden_size, num_pitches)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            x: (batch_size, seq_len, 3)
            returns a dict with:
            - 'logits': (batch_size, num_pitches)
            """
            _, (h_n, _) = self.lstm(x)
            h_last = h_n[-1]  # (B, hidden_size)
            logits = self.fc_out(h_last)  # (B, num_pitches)
            return logits
