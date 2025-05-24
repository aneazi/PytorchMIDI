import glob
from pathlib import Path
import pretty_midi
import pandas as pd
import numpy as np
import collections
from typing import List, Optional, Tuple, Union



def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """Parses MIDI file and returns a dataframe with columns:
    pitch | start | end | step | duration"""
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_df_to_array(
    notes_df: pd.DataFrame,
    feature_cols: List[str] = ['pitch','step','duration']
) -> np.ndarray:
    """
    Convert the notes DataFrame into a (N, len(feature_cols)) float32 array.
    By default returns (N,3) with [pitch, step, duration].
    """
    return notes_df[feature_cols].to_numpy(dtype=np.float32)


def load_all_notes(midi_dir: str, max_files: Optional[int]=None) -> np.ndarray:
    midi_dir = Path(midi_dir)
    paths = list(midi_dir.rglob('*.mid')) + list(midi_dir.rglob('*.midi'))
    if max_files:
        paths = paths[:max_files]
    arrays = []
    for p in paths:
        df = midi_to_notes(str(p))
        if df.empty:
            continue
        arr = notes_df_to_array(df)
        arrays.append(arr)

    return np.vstack(arrays)


if __name__ == "__main__":
    sample = load_all_notes("../maestro-v3.0.0", 100)
    print(sample.shape)
    print(sample)