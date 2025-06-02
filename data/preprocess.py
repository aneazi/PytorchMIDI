import glob
from pathlib import Path
import pretty_midi
import pandas as pd
import numpy as np
import collections
from typing import List, Optional, Tuple, Union



def midi_to_pianoroll(midi_file: str, fs: int = 100, pitch_range: tuple = (21, 109)) -> np.ndarray:
    """Parses MIDI file and returns a piano-roll matrix with columns:
    Arguments: filename, frames per second/sampling rate
    Returns array of shape (T, numpitches)"""
    
    pm = pretty_midi.PrettyMIDI(midi_file)
    roll = pm.get_piano_roll()[pitch_range[0]:pitch_range[1]]
    piano_roll=(roll.T).astype(np.float32)
    piano_roll[piano_roll > 0] = 100.0
    print(piano_roll.shape)
    return piano_roll
 
 
def load_all_rolls(midi_dir: str, fs: int = 100, pitch_range: tuple=(21, 109), max_files: Optional[int]=None) -> np.ndarray:
    """Load max_files into one pianoroll

    Args:
        midi_dir (str): Directory path
        fs (int, optional): frames per second. Defaults to 100.
        pitch_range (tuple, optional): MIDI pitch range. Defaults to (21, 109).
        max_files (Optional[int], optional): Number of files to train on. Defaults to None.

    Returns:
        np.ndarray: (total_frames, num_pitches)
    """
    rolls=[]
    count = 0
    for mid_path in Path(midi_dir).rglob("*.mid*"):
        if max_files is not None and count >= max_files:
            break
        try:
            pr = midi_to_pianoroll(str(mid_path), fs, pitch_range)
            rolls.append(pr)
            count += 1
        except Exception as e:
            print(f"Failed to parse {mid_path}: {e}")
    if not rolls:
        return np.zeros((0, pitch_range[1] - pitch_range[0]), dtype=np.float32)
    return np.vstack(rolls)