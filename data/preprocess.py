import glob
from pathlib import Path
import pretty_midi
import pandas as pd
import numpy as np
import collections
from typing import List, Optional, Tuple, Union



def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """Parses MIDI file and returns a dataframe with columns:
    pitch | start | end | step | duration | velocity"""
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        velocity = note.velocity
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        notes['velocity'].append(velocity)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_df_to_array(
    notes_df: pd.DataFrame,
    feature_cols: List[str] = ['pitch','step','duration', 'velocity']
) -> np.ndarray:
    """
    Convert the notes DataFrame into a (N, len(feature_cols)) float32 array.
    By default returns (N,4) with [pitch, step, duration, velocity].
    """
    return notes_df[feature_cols].to_numpy(dtype=np.float32)


def load_all_notes(midi_dir: str, max_files: Optional[int]=None) -> np.ndarray:
    """
    - Load all MIDIs from a directory and convert them to a numpy array.
    - Each MIDI is converted to a (N, 4) array where N is the number of notes. 
    - And the columns are: [pitch, step, duration, velocity].
    Args:
        midi_dir (str): Path to the directory containing MIDI files.
        max_files (Optional[int], optional): Number of files to train on. Defaults to None.
    Returns:
        np.ndarray: A 2D numpy array of shape (N, 4) where N is the total number of notes across all MIDI files.
    """
    print(f"Loading MIDI files from directory: {midi_dir}")
    midi_dir = Path(midi_dir)
    paths = list(midi_dir.rglob('*.mid')) + list(midi_dir.rglob('*.midi'))
    if max_files:
        paths = paths[:max_files]
    arrays = []
    # Paths loop
    for path in paths:
        df = midi_to_notes(str(path))
        if df.empty:
            continue
        arr = notes_df_to_array(df)
        arrays.append(arr)

    return np.vstack(arrays)