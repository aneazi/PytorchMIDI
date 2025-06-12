import argparse
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

def midi_to_pianoroll(midi_file, output_file, fs=100):
    """
    Convert a MIDI file to a piano roll and save as a PNG image.

    Parameters:
    - midi_file: Path to the input MIDI (.mid) file.
    - output_file: Path to the output PNG file.
    - fs: Sampling frequency (frames per second) for the piano roll.
    """
    # Load MIDI file into PrettyMIDI object
    pm = pretty_midi.PrettyMIDI(midi_file)
    
    # Get the piano roll (numpy array) with shape (128, T)
    piano_roll = pm.get_piano_roll(fs=fs)
    
    # Convert to boolean: note is on if velocity > 0
    piano_roll_bool = piano_roll > 0
    
    # Plot the piano roll
    plt.figure(figsize=(12, 6))
    plt.imshow(piano_roll_bool, aspect='auto', origin='lower', cmap='gray_r')
    plt.xlabel('Time'.format(fs))
    plt.ylabel('Note Number')
    plt.title(f'qLSTM Generated Piano Roll')
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
    args = parser.parse_args()
    
    midi_to_pianoroll(args.input_midi, args.output_png, fs=args.fs)
