#!/usr/bin/env python3
"""
Convert a MIDI file to a piano roll image with an annotated keyboard on the left.
"""
import argparse
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def midi_to_pianoroll_with_keys(midi_file, output_file, fs=100,
                                time_start=None, time_end=None,
                                note_low=40, note_high=90,
                                key_frac=0.2,
                                title='qLSTM Generated Pianoroll'):
    """
    Convert a MIDI file to a piano roll PNG with piano keys on the left.

    Parameters:
    - midi_file: Path to the input MIDI file.
    - output_file: Path to save the output PNG.
    - fs: Frames per second for the piano roll.
    - time_start: Start time in seconds (None for 0).
    - time_end: End time in seconds (None for end).
    - note_low: Lowest MIDI note to display.
    - note_high: Highest MIDI note to display.
    - key_frac: Fraction of figure width allocated to the keyboard.
    - title: Figure title displayed above the plot.
    """
    # Load MIDI and build boolean piano roll
    pm = pretty_midi.PrettyMIDI(midi_file)
    pr = pm.get_piano_roll(fs=fs) > 0

    # Slice time if needed
    start = time_start or 0
    end   = time_end if time_end is not None else pr.shape[1] / fs
    pr = pr[:, int(start*fs):int(end*fs)]

    # Restrict note range
    pr = pr[note_low:note_high+1, :]
    _, T = pr.shape
    time_axis = np.linspace(start, end, T)

    # Figure and axes setup
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.97)
    gs = GridSpec(1, 2, width_ratios=[key_frac, 1-key_frac], wspace=0)
    ax_keys = fig.add_subplot(gs[0])
    ax_roll = fig.add_subplot(gs[1], sharey=ax_keys)
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0)

    # Draw keys: white keys blank, black keys filled
    white_pcs = {0, 2, 4, 5, 7, 9, 11}
    for note in range(note_low, note_high+1):
        pc = note % 12
        y = note
        if pc in white_pcs:
            rect = patches.Rectangle((0, y), 1, 1,
                                     facecolor='white', edgecolor=None)
        else:
            rect = patches.Rectangle((0, y+0.1), 0.6, 0.8,
                                     facecolor='black', edgecolor=None)
        ax_keys.add_patch(rect)

    # Draw horizontal separators at centers of black keys
    black_pcs = {1, 3, 6, 8, 10}
    for note in range(note_low, note_high+1):
        if note % 12 in black_pcs:
            y_mid = note + 0.5
            ax_keys.hlines(y_mid, 0, 1, color='black', linewidth=1.0)

    # Draw separators between E-F and B-C white-key gaps
    gap_pcs = {4, 11}  # E and B
    for note in range(note_low, note_high):
        if note % 12 in gap_pcs:
            y_gap = note + 1
            ax_keys.hlines(y_gap, 0, 1, color='black', linewidth=1.0)

    # Add a vertical border line at left edge
    ax_keys.vlines(0, note_low, note_high+1, color='black', linewidth=1.0)
    # Add horizontal caps at top and bottom of the vertical line
    ax_keys.hlines(note_low, 0, 1, color='black', linewidth=1.0)
    ax_keys.hlines(note_high+1, 0, 1, color='black', linewidth=1.0)

    # Finalize key axis
    ax_keys.set_xlim(0, 1)
    ax_keys.set_ylim(note_low, note_high+1)
    ax_keys.axis('off')

    # Plot piano roll
    ax_roll.imshow(
        pr, aspect='auto', origin='lower',
        extent=[time_axis[0], time_axis[-1], note_low, note_high+1],
        cmap='gray_r', interpolation='nearest'
    )
    ax_roll.set_xlabel('Time (s)')
    ax_roll.set_yticks([])

    # Save and close
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MIDI to piano roll PNG with keys.')
    parser.add_argument('input_midi', help='Input MIDI file path')
    parser.add_argument('output_png', help='Output PNG image path')
    parser.add_argument('--fs', type=int, default=100, help='Frames/sec piano roll')
    parser.add_argument('--time-start', type=float, default=None, help='Start time (s)')
    parser.add_argument('--time-end', type=float, default=None, help='End time (s)')
    parser.add_argument('--note-low', type=int, default=40, help='Lowest MIDI note')
    parser.add_argument('--note-high', type=int, default=90, help='Highest MIDI note')
    parser.add_argument('--key-frac', type=float, default=0.1, help='Keyboard width fraction')
    parser.add_argument('--title', type=str, default='qLSTM Generated Pianoroll', help='Figure title')
    args = parser.parse_args()
    midi_to_pianoroll_with_keys(
        args.input_midi,
        args.output_png,
        fs=args.fs,
        time_start=args.time_start,
        time_end=args.time_end,
        note_low=args.note_low,
        note_high=args.note_high,
        key_frac=args.key_frac,
        title=args.title
    )
