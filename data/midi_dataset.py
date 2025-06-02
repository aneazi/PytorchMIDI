import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from data.preprocess import load_all_rolls

class MidiSequenceDataset(Dataset):
    def __init__(
        self,
        midi_dir: str,
        seq_len: int,
        fs: int=100,
        pitch_range: tuple=(21, 109),
        max_files: int = None,
        max_frames: int = None
):
        """Dataset of pianorolls

        Args:
            midi_dir (str): Directory of dataset
            seq_len (int): Number of timesteps per sequence
            fs (int, optional): Sampling rate. Defaults to 100.
            pitch_range (tuple, optional): Pitch range. Defaults to (21, 109).
            max_files (int, optional): Limits MIDI files loaded. Defaults to None.
            max_frames (int, optional): Limites frames loaded. Defaults to None.

        Raises:
            ValueError: incorrect number of notes
        """
        notes = load_all_rolls(midi_dir, fs, pitch_range, max_files=max_files)
        if max_frames:
            notes = notes[:max_frames]
        total_frames, _ = notes.shape
        self.seq_len = seq_len
        self.num_sequences = total_frames - seq_len
        # build overlapping sequences
        sequences = []
        next_frames = []
        for i in range(self.num_sequences):
            sequences.append(notes[i:i + seq_len])
            next_frames.append(notes[i + seq_len])
        self.X = torch.from_numpy(np.stack(sequences))  # (N, seq_len, num_pitches)
        self.y = torch.from_numpy(np.stack(next_frames))  # (N, num_pitches)

    def __len__(self) -> int:
        return self.num_sequences
    def __getitem__(self, idx: int):
        # returns one (sequence, next_note) pair
        return self.X[idx], self.y[idx]
