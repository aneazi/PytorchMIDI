import torch
import torch.nn as nn
from typing import Dict
from .qlstm import QLSTM


class QuantumMusicRNN(nn.Module):
    def __init__(self,
                 n_qubits,
                 n_qlayers,
                 input_size=4,
                 hidden_size=128,
                 dropout: float = 0.0):
        super().__init__()
        
        print(f"Initializing QuantumMusicRNN: input_size={input_size}, hidden_size={hidden_size}, n_qubits={n_qubits}")
        
        # Use QLSTM instead of regular LSTM
        self.qlstm = QLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
        )
        
        # Output heads remain the same
        self.fc_pitch = nn.Linear(hidden_size, 128)     # 128 MIDI pitch classes
        self.fc_step = nn.Linear(hidden_size, 1)        # scalar time-gap
        self.fc_duration = nn.Linear(hidden_size, 1)    # scalar duration
        self.fc_velocity = nn.Linear(hidden_size, 1)    # scalar duration
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch_size, seq_len, 3)
        returns a dict with:
          'pitch':    (batch_size, 128)  — raw logits
          'step':     (batch_size, 1)    — real prediction  
          'duration': (batch_size, 1)    — real prediction
          'velocity': (batch_size, 1)  — real prediction
        """
        # x: (batch_size, seq_len, 4)
        B, T, _ = x.shape
        # initialize hidden + cell
        h_t = torch.zeros(B, self.qlstm.hidden_size, device=x.device)
        c_t = torch.zeros(B, self.qlstm.hidden_size, device=x.device)

        # step through the sequence
        for t in range(T):
            x_t = x[:, t, :]                   # (B, 4)
            h_t, (h_t, c_t) = self.qlstm(x_t, (h_t, c_t))

        # h_t now holds your final hidden state
        features = h_t                       # (B, hidden_size)

        if self.dropout:
            features = self.dropout(features)

        return {
            'pitch':    self.fc_pitch(features),
            'step':     self.fc_step(features),
            'duration': self.fc_duration(features),
            'velocity': self.fc_velocity(features),
        }