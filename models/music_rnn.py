import torch
import torch.nn as nn
from typing import Dict, Any

class MusicRNN(nn.Module):
    def __init__ (
        self,
        input_size: int=4,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: int = 0.0,
    ):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc_pitch = nn.Linear(hidden_size, 128)  # 128 MIDI classes
        self.fc_step = nn.Linear(hidden_size, 1)    # scalar time-gap
        self.fc_duration = nn.Linear(hidden_size, 1)    # scalar length
        self.fc_velocity = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            x: (batch_size, seq_len, 3)
            returns a dict with:
              'pitch':    (batch_size, 128)  — raw logits
              'step':     (batch_size,   1)  — real prediction
              'duration': (batch_size,   1)  — real prediction
            """
            _, (h_n, _) = self.lstm(x)  
            features = h_n[-1]
            return {
                'pitch': self.fc_pitch(features),
                'step': self.fc_step(features),
                'duration': self.fc_duration(features),
                'velocity': self.fc_velocity(features),
            }
