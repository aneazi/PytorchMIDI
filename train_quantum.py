import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.midi_dataset import MidiSequenceDataset
from models.quantum_music_rnn import QuantumMusicRNN


def main():
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    """
    - Takes length of sequence and number of files to load.
    """
    seq_len=25
    max_files=1
    batch_size=32
    learning_rate=0.0001
    num_epochs=50
    """
    - Loads MIDI dataset from MAESTRO v3.0.0.
    - Each sequence is of length 'seq_len'.
    - Contains 3 features: pitch, step, duration.
    """
    dataset = MidiSequenceDataset(
        midi_dir = "../maestro-v3.0.0",
        seq_len = seq_len,
        max_files = max_files
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} sequences → {len(loader)} batches per epoch")
    
    # Use QuantumMusicRNN instead of regular MusicRNN
    model = QuantumMusicRNN(
        input_size=3, 
        hidden_size=128,
        n_qubits=4,
        n_qlayers=1
    ).to(device)
    
    criterion_pitch = nn.CrossEntropyLoss()  # for 128-way pitch classification
    criterion_step = nn.MSELoss()            # for scalar step prediction
    criterion_duration = nn.MSELoss()        # for scalar duration prediction

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    """
    - Training loop.
    - Saves the best model weights to 'quantum_music_rnn.pt'.
    """
    for epoch in range(1, num_epochs + 1):
        model.train()
        sum_loss = 0.0
        sum_pitch = 0.0
        sum_step = 0.0
        sum_duration = 0.0

        for batch_seq, batch_nxt in loader:
            batch_seq = batch_seq.to(device)   # (B, SEQ_LEN, 3)
            batch_nxt = batch_nxt.to(device)   # (B, 3)
            optimizer.zero_grad()
            preds = model(batch_seq)
            # unpack predictions & targets
            pitch_logits = preds['pitch']             # (B, 128)
            step_pred = preds['step'].squeeze(-1)     # (B,)
            dur_pred = preds['duration'].squeeze(-1)  # (B,)

            true_pitch = batch_nxt[:, 0].long()       # (B,)
            true_step = batch_nxt[:, 1]               # (B,)
            true_duration = batch_nxt[:, 2]           # (B,)

            # compute individual losses
            loss_p = criterion_pitch(pitch_logits, true_pitch)
            loss_s = criterion_step(step_pred,    true_step)
            loss_d = criterion_duration(dur_pred, true_duration)

            # weighted sum
            loss = 0.05 * loss_p + 1.0 * loss_s + 1.0 * loss_d
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            sum_pitch += loss_p.item()
            sum_step += loss_s.item()
            sum_duration += loss_d.item()
        # report averages
        batches = len(loader)
        print(f"Epoch {epoch:2d}/{num_epochs}  "
              f"loss={sum_loss/batches:.4f}  "
              f"pitch={sum_pitch/batches:.4f}  "
              f"step={sum_step/batches:.4f}  "
              f"dur={sum_duration/batches:.4f}")
    
    torch.save(model.state_dict(), "quantum_music_rnn.pt")
    print("Quantum model weights saved to quantum_music_rnn.pt")


if __name__ == "__main__":
    main()