import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.midi_dataset import MidiSequenceDataset
from models.music_rnn    import MusicRNN


def main():
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    best_loss = float('inf')
    seq_len=5
    max_files=10
    pitch_range=(21, 109)  # MIDI pitch range
    fs=20  # frames per second
    batch_size=64
    learning_rate=0.001
    num_epochs=50
    dataset = MidiSequenceDataset(
        midi_dir = "ToySet",
        seq_len = seq_len,
        fs = fs,
        pitch_range = pitch_range,
        max_files = max_files,
        max_frames = 10  # load all frames
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} sequences -> {len(loader)} batches per epoch")
    
    num_pitches = pitch_range[1] - pitch_range[0]  # 88 pitches in MIDI
    model = MusicRNN(num_pitches=num_pitches, hidden_size=256, num_layers=2, dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()  # for 128-way pitch classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    best_model = None
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)   # (B, seq_len, num_pitches)
            batch_y = batch_y.to(device)   # (B, num_pitches)

            optimizer.zero_grad()
            logits = model(batch_x)        # (B, num_pitches)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model.state_dict()
    torch.save(best_model, "music_rnn.pt")

if __name__ == "__main__":
    main()