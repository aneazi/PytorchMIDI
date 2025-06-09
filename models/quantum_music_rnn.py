import torch
import torch.nn as nn
from typing import Dict
from .qlstm import QLSTM


class QuantumMusicRNN(nn.Module):
    def __init__(self, 
                 input_size: int = 3,
                 hidden_size: int = 128,
                 n_qubits: int = 4,
                 n_qlayers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        
        print(f"Initializing QuantumMusicRNN: input_size={input_size}, hidden_size={hidden_size}, n_qubits={n_qubits}")
        
        # Use QLSTM instead of regular LSTM
        self.qlstm = QLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            batch_first=True,
            return_sequences=True
        )
        
        # Output heads remain the same
        self.fc_pitch = nn.Linear(hidden_size, 128)     # 128 MIDI pitch classes
        self.fc_step = nn.Linear(hidden_size, 1)        # scalar time-gap
        self.fc_duration = nn.Linear(hidden_size, 1)    # scalar duration
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch_size, seq_len, 3)
        returns a dict with:
          'pitch':    (batch_size, 128)  — raw logits
          'step':     (batch_size, 1)    — real prediction  
          'duration': (batch_size, 1)    — real prediction
        """
        # Pass through QLSTM
        lstm_out, (h_n, c_n) = self.qlstm(x)
        
        # Use the last hidden state
        features = h_n  # Shape: (batch_size, hidden_size)
        
        # Apply dropout if specified
        if self.dropout:
            features = self.dropout(features)
        
        return {
            'pitch': self.fc_pitch(features),
            'step': self.fc_step(features),
            'duration': self.fc_duration(features),
        }