import argparse
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

def midi_to_pianoroll(midi_file, output_file, fs=100, time_start=None, time_end=None):
    """
    Convert a MIDI file to a piano roll and save as a PNG image.

    Parameters:
    - midi_file: Path to the input MIDI (.mid) file.
    - output_file: Path to the output PNG file.
    - fs: Sampling frequency (frames per second) for the piano roll.
    - time_start: Start time in seconds (None for beginning of track).
    - time_end: End time in seconds (None for end of track).
    """
    # Load MIDI file into PrettyMIDI object
    pm = pretty_midi.PrettyMIDI(midi_file)
    
    # Get the piano roll (numpy array) with shape (128, T)
    piano_roll = pm.get_piano_roll(fs=fs)
    
    # Apply time range filtering if specified
    if time_start is not None or time_end is not None:
        total_duration = piano_roll.shape[1] / fs
        
        # Convert time to frame indices
        start_frame = int(time_start * fs) if time_start is not None else 0
        end_frame = int(time_end * fs) if time_end is not None else piano_roll.shape[1]
        
        # Ensure indices are within bounds
        start_frame = max(0, start_frame)
        end_frame = min(piano_roll.shape[1], end_frame)
        
        # Slice the piano roll
        piano_roll = piano_roll[:, start_frame:end_frame]
        
        print(f"Time range: {time_start or 0:.1f}s to {time_end or total_duration:.1f}s")
        print(f"Frame range: {start_frame} to {end_frame}")
    
    # Convert to boolean: note is on if velocity > 0
    piano_roll_bool = piano_roll > 0
    
    # Create time axis for x-axis labels
    time_duration = piano_roll.shape[1] / fs
    time_axis = np.linspace(time_start or 0, (time_start or 0) + time_duration, piano_roll.shape[1])
    
    # Plot the piano roll
    plt.figure(figsize=(12, 6))
    plt.imshow(piano_roll_bool, aspect='auto', origin='lower', cmap='gray_r', 
               extent=[time_axis[0], time_axis[-1], 0, 127])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Note Number')
    
    # Update title to show time range if specified
    title = 'qLSTM Generated Piano Roll'
    plt.title(title)
    
    plt.tight_layout()
    
    # Save to PNG
    plt.savefig(output_file, dpi=300)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a MIDI file to a piano roll PNG image.'
    )
    parser.add_argument(
        'input_midi',
        help='Path to the input MIDI file (e.g., output.mid)'
    )
    parser.add_argument(
        'output_png',
        help='Path to the output PNG file (e.g., pianoroll.png)'
    )
    parser.add_argument(
        '--fs',
        type=int,
        default=100,
        help='Sampling frequency for piano roll frames per second (default: 100)'
    )
    parser.add_argument(
        '--time-start',
        type=float,
        default=None,
        help='Start time in seconds (default: beginning of track)'
    )
    parser.add_argument(
        '--time-end',
        type=float,
        default=None,
        help='End time in seconds (default: end of track)'
    )
    args = parser.parse_args()
    
    midi_to_pianoroll(args.input_midi, args.output_png, fs=args.fs, 
                     time_start=args.time_start, time_end=args.time_end)
