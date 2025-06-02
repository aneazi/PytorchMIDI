import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple

from data.preprocess import load_all_notes

class MidiSequenceDataset(Dataset):
    def __init__(
        self,
        midi_dir: str,
        seq_len: int,
        max_files: Optional[int] = None
):
        """
        - Gets all MIDI files from a directory, converts them to sequences of notes
        - Preps for model training.
        Args:
            midi_dir (str): Directory containing MIDI files.
            seq_len (int): Length of each sequence to be used for training.
            max_files (Optional[int], optional): Number of files to train on. Defaults to None.
        """
        notes = load_all_notes(midi_dir, max_files=max_files)
        total_notes, features = notes.shape
        self.seq_len=seq_len
        self.num_sequences = max(0, total_notes - seq_len)
        sequences = np.stack([
            notes[i : i + seq_len]
            for i in range(self.num_sequences)
        ], axis=0)
        next_notes = notes[seq_len : seq_len + self.num_sequences]
        self.sequences = torch.from_numpy(sequences).float()
        self.next_notes = torch.from_numpy(next_notes).float()
    def __len__(self) -> int:
        return self.num_sequences
    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # returns one (sequence, next_note) pair
        return self.sequences[idx], self.next_notes[idx]
