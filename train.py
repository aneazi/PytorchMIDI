import random
import numpy as np
import yaml
from argparse import Namespace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.midi_dataset import MidiSequenceDataset
from models.music_rnn    import MusicRNN
import time

def main():
    cfg_dict = yaml.safe_load(open("config.yml", "r"))
    cfg = Namespace(**cfg_dict)
    print("At main...")
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    """
    - Takes length of sequence and number of files to load.
    """
    seq_len=cfg.SEQ_LEN
    max_files=cfg.MAX_FILES
    batch_size=cfg.BATCH_SIZE
    learning_rate=cfg.LEARNING_RATE
    num_epochs=cfg.EPOCHS
    model_path=cfg.CLASSICAL_MODEL_PATH
    """
    - Loads MIDI dataset from MAESTRO v3.0.0.
    - Each sequence is of length 'seq_len'.
    - Contains 4 features: pitch, step, duration, velocity.
    """
    dataset = MidiSequenceDataset(
        midi_dir = cfg.DATASET,
        seq_len = seq_len,
        max_files = max_files
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} sequences → {len(loader)} batches per epoch")
    model = MusicRNN(input_size=4, hidden_size=128).to(device)
    criterion_pitch = nn.CrossEntropyLoss()  # for 128-way pitch classification
    criterion_step = nn.MSELoss()            # for scalar step prediction
    criterion_duration = nn.MSELoss()        # for scalar duration prediction
    criterion_velocity = nn.MSELoss()        # for scalar velocity prediction

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    """
    - Training loop.
    - Saves the best model weights to 'music_rnn.pt'.
    """
    for epoch in range(1, num_epochs + 1):
        start_time=time.time()
        model.train()
        sum_loss = 0.0
        sum_pitch = 0.0
        sum_step = 0.0
        sum_duration = 0.0
        sum_velocity = 0.0

        for batch_idx, (batch_seq, batch_nxt) in enumerate(loader):
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
            true_velocity = batch_nxt[:, 3]           # (B,)

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
            
            # Show batch progress every 100 batches or at the end
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                batch_progress = (batch_idx + 1) / len(loader) * 100
                print(f"  Epoch {epoch}/{num_epochs} - Batch {batch_idx + 1}/{len(loader)} ({batch_progress:.1f}%)")
        
        # report averages
        end_time=time.time()-start_time
        batches = len(loader)
        print(f"TIME: {time.strftime('%H:%M:%S')} "
              f"Epoch {epoch:2d}/{num_epochs} took {end_time:.2f}s "
              f"loss={sum_loss/batches:.4f}  "
              f"pitch={sum_pitch/batches:.4f}  "
              f"step={sum_step/batches:.4f}  "
              f"dur={sum_duration/batches:.4f} "
              f"vel={sum_velocity/batches:.4f}")
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")

if __name__ == "__main__":
    main()
