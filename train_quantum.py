import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.midi_dataset import MidiSequenceDataset
from models.quantum_music_rnn import QuantumMusicRNN
import time

def main():
    print("At main...")
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    """
    - Takes length of sequence and number of files to load.
    """
    seq_len=32
    max_files=1
    batch_size=32
    learning_rate=0.0005
    num_epochs=50
    """
    - Loads MIDI dataset from MAESTRO v3.0.0.
    - Each sequence is of length 'seq_len'.
    - Contains 4 features: pitch, step, duration, velocity.
    """
    dataset = MidiSequenceDataset(
        midi_dir = "../maestro-v3.0.0",
        seq_len = seq_len,
        max_files = max_files
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} sequences → {len(loader)} batches per epoch")
    
    # Use QuantumMusicRNN instead of regular MusicRNN
    print("Creating QuantumMusicRNN model...")
    model = QuantumMusicRNN(
        input_size=4,
        hidden_size=128,
        n_qubits=8,
        n_qlayers=1
    ).to(device)
    print("\nModel parameters:")
    
    criterion_pitch = nn.CrossEntropyLoss()  # for 128-way pitch classification
    criterion_step = nn.MSELoss()            # for scalar step prediction
    criterion_duration = nn.MSELoss()        # for scalar duration prediction
    criterion_velocity = nn.MSELoss()        # for scalar velocity prediction (if needed)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Starting training for {num_epochs} epochs...")
    """
    - Training loop.
    - Saves the best model weights to 'quantum_music_rnn.pt'.
    """
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        sum_loss = 0.0
        sum_pitch = 0.0
        sum_step = 0.0
        sum_duration = 0.0
        sum_velocity = 0.0
        
        for batch_seq, batch_nxt in loader:
            batch_seq = batch_seq.to(device)   # (B, SEQ_LEN, 4)
            batch_nxt = batch_nxt.to(device)   # (B, 4)
            optimizer.zero_grad()
            preds = model(batch_seq)
            # unpack predictions & targets
            pitch_logits = preds['pitch']             # (B, 128)
            step_pred = preds['step'].squeeze(-1)     # (B,)
            dur_pred = preds['duration'].squeeze(-1)  # (B,)
            vel_pred = preds['velocity'].squeeze(-1)      # (B,)

            true_pitch = batch_nxt[:, 0].long()       # (B,)
            true_step = batch_nxt[:, 1]               # (B,)
            true_duration = batch_nxt[:, 2]           # (B,)
            true_velocity = batch_nxt[:, 3]           # (B,) if needed
            
            # compute individual losses
            loss_p = criterion_pitch(pitch_logits, true_pitch)
            loss_s = criterion_step(step_pred,    true_step)
            loss_d = criterion_duration(dur_pred, true_duration)
            loss_v = criterion_velocity(vel_pred, true_velocity)
            # weighted sum
            loss = 1.0 * loss_p + 1.0 * loss_s + 1.0 * loss_d + 1.0 * loss_v
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            sum_pitch += loss_p.item()
            sum_step += loss_s.item()
            sum_duration += loss_d.item()
            sum_velocity += loss_v.item()
        epoch_time = time.time() - epoch_start_time
        # report averages
        batches = len(loader)
        print(f"TIME: {time.strftime('%H:%M:%S')} "
              f"Epoch {epoch:2d}/{num_epochs} took {epoch_time:.2f}s "
              f"loss={sum_loss/batches:.4f}  "
              f"pitch={sum_pitch/batches:.4f}  "
              f"step={sum_step/batches:.4f}  "
              f"dur={sum_duration/batches:.4f} "
              f"vel={sum_velocity/batches:.4f}")
    
    torch.save(model.state_dict(), "quantum_music_rnn.pt")
    print("Quantum model weights saved to quantum_music_rnn.pt")


if __name__ == "__main__":
    main()
