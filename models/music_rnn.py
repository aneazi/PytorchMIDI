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
            output, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
            last_output = output[:, -1, :]  # Take only the last timestep: (batch_size, hidden_size)
            logits = self.fc_out(last_output)  # (batch_size, num_pitches)
            return logits  # Remove the Dict wrapper - just return tensor directly