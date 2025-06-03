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
seq_len = 10  # Match training
num_predictions = 500
temperature = 0.8  # Lower for more focused sampling
fs = 100
time_stretch = 100.0  # Slow down by this factor (3x slower)
pitch_range = (21, 109)  # MIDI pitch range
threshold = 0.4  # Low since we'll use binary sampling

def play_music(midi_filename):
    """Stream music_file in a blocking manner"""
    clock = pygame.time.Clock()
    pygame.mixer.music.load(midi_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)

def load_model(device: torch.device) -> MusicRNN:
    """Instantiate the model, load weights, switch to eval."""
    num_pitches = pitch_range[1] - pitch_range[0]
    model = MusicRNN(num_pitches=num_pitches, hidden_size=256, num_layers=2, dropout=0.2).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def create_seed_from_notes(notes: list, num_pitches: int) -> np.ndarray:
    """
    Create a seed sequence from a list of note pitches.
    
    Args:
        notes: List of MIDI note numbers (e.g., [60, 64, 67] for C major chord)
        num_pitches: Total number of pitches in the model
    
    Returns:
        Seed array of shape (seq_len, num_pitches)
    """
    seed = np.zeros((seq_len, num_pitches), dtype=np.float32)
    low_pitch = pitch_range[0]
    
    # Fill the seed with the specified notes
    for note in notes:
        if low_pitch <= note < pitch_range[1]:
            pitch_idx = note - low_pitch
            # Add the notes to all timesteps in the seed
            seed[:, pitch_idx] = 1.0
    
    return seed

def sample_from_probabilities(probs: np.ndarray, temperature: float = 1.0, top_k: int = 5) -> np.ndarray:
    """
    Sample notes from probabilities using controlled stochastic sampling.
    
    Args:
        probs: Probability array of shape (num_pitches,)
        temperature: Sampling temperature
        top_k: Only sample from top k most likely pitches
    
    Returns:
        Binary array indicating which notes to play
    """
    # Apply temperature scaling to logits
    logits = np.log(np.clip(probs, 1e-8, 1.0)) / temperature
    
    # Get top-k indices
    top_indices = np.argsort(logits)[-top_k:]
    
    # Create output array
    output = np.zeros_like(probs)
    
    # Sample from top-k probabilities
    top_probs = probs[top_indices]
    if np.sum(top_probs) > 0:
        top_probs = top_probs / np.sum(top_probs)  # Renormalize
        
        # Sample 1-4 notes based on probabilities (max 4 notes at once)
        num_notes = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])  # Allow up to 4 notes
        
        for _ in range(num_notes):
            if np.sum(top_probs) > 0:
                sampled_idx = np.random.choice(top_indices, p=top_probs)
                output[sampled_idx] = 1.0
                
                # Remove sampled note from future sampling
                idx_pos = np.where(top_indices == sampled_idx)[0]
                if len(idx_pos) > 0:
                    top_probs[idx_pos[0]] = 0
                    if np.sum(top_probs) > 0:
                        top_probs = top_probs / np.sum(top_probs)
    
    return output

def sample_sequence(
    model: MusicRNN,
    seed: np.ndarray,
    num_steps: int,
    temperature: float,
    device: torch.device
) -> np.ndarray:
    """
    Autoregressively sample next piano roll frames using controlled sampling with note sustaining.
    
    Args:
        model: Trained MusicRNN model
        seed: Initial sequence of shape (seq_len, num_pitches)
        num_steps: Number of frames to generate
        temperature: Sampling temperature
        device: PyTorch device
        
    Returns:
        Generated piano roll of shape (num_steps, num_pitches)
    """
    model.eval()
    seq = seed.copy()  # (seq_len, num_pitches)
    generated = []
    
    # Track active notes and their remaining duration
    active_notes = np.zeros(seq.shape[1], dtype=int)  # Duration left for each pitch
    note_decay = 0.95  # How much notes decay over time
    
    print(f"Starting generation with seed shape: {seq.shape}")
    print(f"Seed contains {np.sum(seed)} active notes")
    
    for step in range(num_steps):
        # Convert to tensor and add batch dimension
        x = torch.from_numpy(seq).float().unsqueeze(0).to(device)  # (1, seq_len, num_pitches)
        
        with torch.no_grad():
            logits = model(x)  # (1, num_pitches)
            
        # Convert to probabilities
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # (num_pitches,)
        
        # Sample new notes to start
        new_notes = sample_from_probabilities(probs, temperature, top_k=8)
        
        # Create frame with both sustained and new notes
        frame = np.zeros_like(probs)
        
        # Add sustained notes (decay over time)
        sustained_notes = []
        for i in range(len(active_notes)):
            if active_notes[i] > 0:
                frame[i] = max(0.3, 1.0 - (20 - active_notes[i]) * 0.03)  # Gradual decay
                active_notes[i] -= 1
                sustained_notes.append(i)
        
        # Add new notes and set their duration
        new_note_indices = []
        for i in range(len(new_notes)):
            if new_notes[i] > 0:
                frame[i] = 1.0  # New notes at full volume
                # Random duration between 5-20 frames, scaled by time_stretch
                base_duration = np.random.randint(8, 25)
                active_notes[i] = int(base_duration * time_stretch)
                new_note_indices.append(i)
        
        # Enforce 4-note maximum with smart priority system
        all_active_indices = np.where(frame > 0)[0]
        
        if len(all_active_indices) > 4:
            # Create priority list: new notes first, then sustained by remaining duration
            priority_list = []
            
            # High priority: new notes (they were just selected)
            for i in new_note_indices:
                if frame[i] > 0:
                    priority_list.append((i, 1000 + frame[i]))  # High priority value
            
            # Lower priority: sustained notes (by remaining duration)
            for i in sustained_notes:
                if frame[i] > 0 and i not in new_note_indices:
                    priority_list.append((i, active_notes[i]))  # Priority = remaining duration
            
            # Sort by priority (descending)
            priority_list.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only top 4
            keep_indices = [i for i, _ in priority_list[:4]]
            
            # Update frame and active_notes to only include kept notes
            new_frame = np.zeros_like(frame)
            new_active_notes = np.zeros_like(active_notes)
            
            for i in keep_indices:
                new_frame[i] = frame[i]
                new_active_notes[i] = active_notes[i]
            
            frame = new_frame
            active_notes = new_active_notes
        
        generated.append(frame)
        
        # Update sequence: slide window and add new frame
        seq = np.vstack([seq[1:], frame.reshape(1, -1)])
        
        if step % 50 == 0:
            active_count = np.sum(frame > 0)
            sustained_count = np.sum(active_notes > 0)
            print(f"Generated {step}/{num_steps} frames... (Active notes: {active_count}, Sustained: {sustained_count})")
    
    return np.stack(generated, axis=0)  # (num_steps, num_pitches)


def roll_to_midi(
    roll: np.ndarray,
    fs: int,
    out_path: str,
    instrument_name: str,
    threshold: float = 0.5
) -> None:
    """
    Convert a piano-roll (num_frames × num_pitches, [0,1]) to a MIDI file.
    """
    pm = pretty_midi.PrettyMIDI()
    program = pretty_midi.instrument_name_to_program(instrument_name)
    inst = pretty_midi.Instrument(program=program)
    num_frames, num_pitches = roll.shape
    time_per_frame = 1.0 / fs
    low_pitch, _ = pitch_range

    # Track note states for each pitch
    for i in range(num_pitches):
        pitch = i + low_pitch
        active = roll[:, i] > threshold
        
        # Find note-on and note-off events
        note_changes = np.diff(np.concatenate([[False], active, [False]]).astype(int))
        note_ons = np.where(note_changes == 1)[0]
        note_offs = np.where(note_changes == -1)[0]
        
        # Create MIDI notes
        for on, off in zip(note_ons, note_offs):
            start_time = on * time_per_frame
            end_time = off * time_per_frame
            
            # Ensure minimum note duration
            if end_time - start_time < 0.05:  # 50ms minimum
                end_time = start_time + 0.05
            
            velocity = int(np.clip(80 + np.random.randint(-20, 21), 60, 100))  # Vary velocity
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            inst.notes.append(note)

    pm.instruments.append(inst)
    pm.write(out_path)

def main():
    # Device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Load model
    model = load_model(device)
    num_pitches = pitch_range[1] - pitch_range[0]
    
    # Option 2: Alternatively, load from a MIDI file (uncomment to use)
    midis = sorted(Path(midi_dir).rglob("*.mid")) + sorted(Path(midi_dir).rglob("*.midi"))
    if midis:
        seed_file = random.choice(midis)
        print("Seeding from:", seed_file)
        roll = midi_to_pianoroll(str(seed_file), fs=fs, pitch_range=pitch_range)
        if roll.shape[0] >= seq_len:
            seed = roll[:seq_len]
        else:
            print("MIDI file too short, using chord seed instead")
            seed = create_seed_from_notes([60, 64, 67], num_pitches)

    # Sample new frames
    print(f"Sampling {num_predictions} frames at T={temperature}...")
    print(f"Time stretch factor: {time_stretch}x (notes will be {time_stretch}x longer)")
    gen_roll = sample_sequence(model, seed, num_predictions, temperature, device)

    # Debug: inspect generated roll
    print(f"Generated roll stats – min: {gen_roll.min():.4f}, max: {gen_roll.max():.4f}, mean: {gen_roll.mean():.4f}")
    n_active = (gen_roll > threshold).sum()
    total_frames = gen_roll.shape[0] * gen_roll.shape[1]
    print(f"Total activations above threshold {threshold}: {n_active}/{total_frames} ({100*n_active/total_frames:.2f}%)")

    # Export generated roll to MIDI
    roll_to_midi(gen_roll, fs, output_midi, instrument_name, threshold)
    print("Wrote generated MIDI to", output_midi)

    # Play via pygame
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)
        pygame.mixer.music.set_volume(0.8)
        play_music(output_midi)
    except Exception as e:
        print(f"Could not play audio: {e}")

if __name__ == "__main__":
    main()