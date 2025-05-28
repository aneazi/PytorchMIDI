import torch
import numpy as np
import pygame
import pandas as pd
import random
import pretty_midi
from pathlib import Path
from data.preprocess   import midi_to_pianoroll
from models.music_rnn  import MusicRNN

weight_path = "music_rnn.pt"
midi_dir = "../maestro-v3.0.0"
output_midi= "output.mid"
instrument_name = "Acoustic Grand Piano"
seq_len = 5
num_predictions = 120
temperature = 2.0
fs = 20
pitch_range = (21, 109)  # MIDI pitch range
threshold = 0.0

def play_music(midi_filename):
  '''Stream music_file in a blocking manner'''
  clock = pygame.time.Clock()
  pygame.mixer.music.load(midi_filename)
  pygame.mixer.music.play()
  while pygame.mixer.music.get_busy():
    clock.tick(30) # check if playback has finished

def load_model(device: torch.device) -> MusicRNN:
    """Instantiate the model, load weights, switch to eval."""
    num_pitches = pitch_range[1] - pitch_range[0]
    model = MusicRNN(num_pitches=num_pitches, hidden_size=256, num_layers=2, dropout=0.2).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def sample_sequence(
    model: MusicRNN,
    seed: np.ndarray,
    num_steps: int,
    temperature: float,
    device: torch.device
) -> np.ndarray:
    """
    Autoregressively sample next notes.
    seed: array shape (SEQ_LEN, 3) of [pitch, step, duration].
    returns a DataFrame with columns
      ['pitch','step','duration','start','end'] of length num_steps.
    """
    seq = seed.copy()
    gen = []
    for _ in range(num_steps):
        x = torch.from_numpy(seq).float().unsqueeze(0).to(device)  # (1, seq_len, P)
        with torch.no_grad():
            logits = model(x)                            # (1, P)
        logits = logits / temperature
        probs  = torch.sigmoid(logits).cpu().numpy()[0] 
        gen.append(probs.astype(np.float32))
        # slide window
        seq = np.vstack([seq[1:], probs])
    return np.stack(gen, axis=0)  # (num_steps, P)


def roll_to_midi(
    roll: np.ndarray,
    fs: int,
    out_path: str,
    instrument_name: str,
    threshold: float = 0.5
) -> None:
    """
    Convert a piano-roll (num_frames × num_pitches, [0,1]) to a MIDI file.
    Writes to out_path.
    """
    pm = pretty_midi.PrettyMIDI()
    program = pretty_midi.instrument_name_to_program(instrument_name)
    inst = pretty_midi.Instrument(program=program)
    num_frames, num_pitches = roll.shape
    time_per_frame = 1.0 / fs
    lows, _ = pitch_range

    # for each pitch, track on/off state
    for i in range(num_pitches):
        pitch = i + lows
        active = roll[:, i] > threshold
        state = False
        start_time = 0.0
        velocity = 100
        for t, is_on in enumerate(active):
            if is_on and not state:
                # note-on
                state = True
                start_time = t * time_per_frame
                # velocity proportional to intensity
                velocity = int(np.clip(roll[t, i] * 127, 1, 127))
            elif not is_on and state:
                # note-off
                end_time = t * time_per_frame
                note = pretty_midi.Note(velocity=velocity, pitch=pitch,
                                        start=start_time, end=end_time)
                inst.notes.append(note)
                state = False
        # close any lingering note at end
        if state:
            end_time = num_frames * time_per_frame
            note = pretty_midi.Note(velocity=velocity, pitch=pitch,
                                    start=start_time, end=end_time)
            inst.notes.append(note)

    pm.instruments.append(inst)
    pm.write(out_path)

def main():
    # Device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Load model
    model = load_model(device)

    # Pick a MIDI file to seed from
    midis = sorted(Path(midi_dir).rglob("*.mid")) + sorted(Path(midi_dir).rglob("*.midi"))
    seed_file = random.choice(midis)
    print("Seeding from:", seed_file)

    # Load piano-roll seed
    roll = midi_to_pianoroll(str(seed_file), fs=fs, pitch_range=pitch_range)
    if roll.shape[0] < seq_len:
        raise ValueError(f"Seed too short ({roll.shape[0]} < {seq_len})")
    seed = roll[:seq_len]  # (seq_len, P)

    # Sample new frames
    print(f"Sampling {num_predictions} frames at T={temperature}…")
    gen_roll = sample_sequence(model, seed, num_predictions, temperature, device)

    # Export generated roll to MIDI
    roll_to_midi(gen_roll, fs, output_midi, instrument_name, threshold)
    print("Wrote generated MIDI to", output_midi)

    # Play via pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)
    pygame.mixer.music.set_volume(0.8)
    play_music(output_midi)

if __name__ == "__main__":
    main()