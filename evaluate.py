# evaluate.py

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from data.midi_dataset import MidiSequenceDataset
from models.music_rnn    import MusicRNN

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
WEIGHTS = "music_rnn.pt"

def load_model(path: str) -> MusicRNN:
    model = MusicRNN(input_size=3, hidden_size=128).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def main():
    # 1) Prepare the toy dataset (C-major scale of length 8)
    seq_len   = 7
    ds = MidiSequenceDataset(midi_dir="ToySet", seq_len=seq_len, max_files=1)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # 2) Load your best model
    model = load_model(WEIGHTS)

    # metrics accumulators
    total = 0
    correct_pitches = 0
    sum_step_sq  = 0.0
    sum_dur_sq   = 0.0

    # 3) Slide over every window in the one file
    for seq, nxt in loader:
        # seq: (1, seq_len, 3), nxt: (1, 3)
        seq = seq.to(DEVICE)
        nxt = nxt.to(DEVICE)
        with torch.no_grad():
            out = model(seq)
        # GREEDY pitch (take argmax instead of sampling)
        pred_pitch = out['pitch'].argmax(dim=-1).item()
        true_pitch = nxt[0,0].long().item()
        correct_pitches += (pred_pitch == true_pitch)

        # step & duration predictions
        pred_step = out['step'].squeeze(-1).item()
        pred_dur  = out['duration'].squeeze(-1).item()
        true_step = nxt[0,1].item()
        true_dur  = nxt[0,2].item()

        sum_step_sq += (pred_step - true_step)**2
        sum_dur_sq  += (pred_dur  - true_dur )**2

        total += 1

    # 4) Compute and print
    pitch_acc = correct_pitches / total if total>0 else 0.0
    step_mse   = sum_step_sq   / total if total>0 else 0.0
    dur_mse    = sum_dur_sq    / total if total>0 else 0.0

    print("Evaluation on C-major scale:")
    print(f"  Windows tested : {total}")
    print(f"  Pitch accuracy : {pitch_acc*100:5.1f}%")
    print(f"  Step    MSE    : {step_mse:.6f}")
    print(f"  Duration MSE   : {dur_mse:.6f}")

if __name__ == "__main__":
    main()
