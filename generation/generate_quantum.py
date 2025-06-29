import torch
import numpy as np
import pygame
import yaml
from argparse import Namespace
import pandas as pd
import random
import pretty_midi
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocess import midi_to_notes, notes_df_to_array
from models.quantum_music_rnn import QuantumMusicRNN

cfg_dict = yaml.safe_load(open("config.yml", "r"))
cfg = Namespace(**cfg_dict)

# Parameters
weight_path = cfg.QUANTUM_MODEL_PATH  # Updated to match quantum model weights
midi_dir = Path(cfg.QUANTUM_SEED_DIR)
output_midi = cfg.QUANTUM_OUTPUT_PATH
instrument_name = cfg.INSTRUMENT_NAME
seq_len = cfg.SEQ_LEN
num_predictions = cfg.NUM_PREDICTIONS
temperature = cfg.TEMPERATURE
time_stretch = cfg.TIME_STRETCH

def play_music(midi_filename):
    """
    Play a MIDI file using pygame.
    
    Args:
        midi_filename (_type_): Filename of the MIDI file to play.
    """
    clock = pygame.time.Clock()
    pygame.mixer.music.load(midi_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30) # check if playback has finished

def load_quantum_model(device: torch.device) -> QuantumMusicRNN:
    """Load the quantum model with weights."""
    model = QuantumMusicRNN(
        input_size=4, 
        hidden_size=128,
        n_qubits=4,
        n_qlayers=1
    ).to(device)
    
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def sample_sequence(
    model: QuantumMusicRNN,
    seed: np.ndarray,
    num_steps: int,
    temperature: float,
    device: torch.device
) -> pd.DataFrame:
    """
    Autoregressively sample next notes.
    seed: array shape (SEQ_LEN, 4) of [pitch, step, duration, velocity].
    returns a DataFrame with columns
      ['pitch','step','duration','start','end','velocity'] of length num_steps.
    """
    gen = []
    prev_start = 0.0
    seq = seed.copy()

    for _ in range(num_steps):
        # to tensor (1, SEQ_LEN, 4)
        x = torch.from_numpy(seq).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
        # pitch: sample from softmax
        logits = out['pitch'] / temperature           # (1,128)
        probs = torch.softmax(logits, dim=-1)        # (1,128)
        pitch = torch.multinomial(probs, num_samples=1).item()
        # step & duration: direct scalars
        step = out['step'].item()
        duration = out['duration'].item()
        velocity = out['velocity'].item()
        # clamp to non-negative
        step = max(step, 0.0) * time_stretch
        duration = max(duration, 0.0) * time_stretch

        start = prev_start + step
        end = start + duration
        gen.append((pitch, step, duration, start, end, velocity))
        prev_start = start

        # shift window, append new note
        next_feat = np.array([pitch, step, duration, velocity], dtype=np.float32)
        seq = np.vstack([seq[1:], next_feat])

    return pd.DataFrame(
        gen, columns=['pitch','step','duration','start','end', 'velocity']
    )

def notes_to_midi(
    notes_df: pd.DataFrame,
    out_path: str,
    instrument_name: str,
) -> pretty_midi.PrettyMIDI:
    """
    Turn a DataFrame of (pitch,step,duration,start,end) into a .mid file.
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )
    for _, row in notes_df.iterrows():
        note = pretty_midi.Note(
            velocity=int(row['velocity']),
            pitch=int(row['pitch']),
            start=float(row['start']),
            end=float(row['end'])
        )
        inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(out_path)
    return pm


def main():
    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Load model
    model = load_quantum_model(device)
    
    paths = list(midi_dir.rglob('*.mid')) + list(midi_dir.rglob('*.midi'))
    print(len(paths))
    random_file = random.choice(paths)
    seed_file = paths[0]
    print("Seeding from:", seed_file)
    df_seed = midi_to_notes(str(seed_file))
    arr = notes_df_to_array(df_seed)
    # use the first SEQ_LEN notes
    seed = arr[:seq_len]

    # 3) Sample
    print(f"Sampling {num_predictions} notes (T={temperature})…")
    gen_df = sample_sequence(model, seed, num_predictions, temperature, device)
    print(gen_df.head(10))

    # 4) Export to MIDI
    notes_to_midi(gen_df, output_midi, instrument_name)
    print("Wrote generated MIDI to", output_midi)

    """ # 5) Play via pygame
    freq = 44100  # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 1  # 1 is mono, 2 is stereo
    buffer = 1024   # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(1.0)
    play_music(output_midi) """

if __name__ == "__main__":
    main()
