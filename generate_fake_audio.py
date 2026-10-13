#!/usr/bin/env python3
"""
Generate Fake Audio Samples
============================

This script creates "fake" audio by applying different signal processing
transformations to bonafide (real) audio. Each transformation simulates
a different TTS/voice conversion system.

Systems simulated:
- System A: Vocoder simulation (phase vocoder with artifacts)
- System B: Pitch-shifted version (simulates voice conversion)
- System C: Time-stretched + low-pass filtered (simulates neural vocoder)

Note: In real research, these would be actual TTS/VC system outputs.
This is a simplified demonstration for understanding the attribution pipeline.
"""

import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def apply_system_a_transform(audio, sr):
    """
    System A: Phase vocoder with artifacts

    Simulates vocoder-based synthesis by:
    1. Time-stretching slightly (1.05x)
    2. Adding subtle artifacts via LPC residual modification

    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate

    Returns:
        Transformed audio
    """
    # Time stretch with phase vocoder (common in neural vocoders)
    transformed = librosa.effects.time_stretch(audio, rate=1.05)

    # Add slight high-frequency attenuation (vocoder artifact)
    # Apply mild low-pass filter at 7kHz
    transformed = librosa.effects.preemphasis(transformed, coef=-0.95)

    # Match original length
    if len(transformed) > len(audio):
        transformed = transformed[:len(audio)]
    else:
        transformed = np.pad(transformed, (0, len(audio) - len(transformed)))

    return transformed


def apply_system_b_transform(audio, sr):
    """
    System B: Pitch-shifted voice conversion

    Simulates voice conversion by:
    1. Pitch shifting (+2 semitones)
    2. Formant preservation attempt (incomplete, leaves artifacts)

    Args:
        audio: Audio signal
        sr: Sample rate

    Returns:
        Transformed audio
    """
    # Pitch shift (+2 semitones) - common VC artifact
    transformed = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2.0)

    # Slight time compression to compensate
    transformed = librosa.effects.time_stretch(transformed, rate=0.98)

    # Match length
    if len(transformed) > len(audio):
        transformed = transformed[:len(audio)]
    else:
        transformed = np.pad(transformed, (0, len(audio) - len(transformed)))

    return transformed


def apply_system_c_transform(audio, sr):
    """
    System C: Neural vocoder with bandwidth limitation

    Simulates bandwidth-limited neural vocoder:
    1. Aggressive low-pass filter (6 kHz cutoff)
    2. Slight time dilation
    3. Add subtle quantization noise

    Args:
        audio: Audio signal
        sr: Sample rate

    Returns:
        Transformed audio
    """
    # Apply aggressive low-pass filter (neural vocoder bandwidth limit)
    from scipy.signal import butter, filtfilt
    nyquist = sr / 2
    cutoff = 6000 / nyquist
    b, a = butter(4, cutoff, btype='low')
    transformed = filtfilt(b, a, audio)

    # Time stretch slightly
    transformed = librosa.effects.time_stretch(transformed, rate=0.97)

    # Add subtle quantization noise (vocoder artifact)
    noise_scale = 0.001
    transformed = transformed + np.random.normal(0, noise_scale, len(transformed))

    # Match length
    if len(transformed) > len(audio):
        transformed = transformed[:len(audio)]
    else:
        transformed = np.pad(transformed, (0, len(audio) - len(transformed)))

    return transformed


def generate_fake_samples(bonafide_dir, output_dirs, max_samples=50):
    """
    Generate fake audio for multiple systems.

    Args:
        bonafide_dir: Path to bonafide audio directory
        output_dirs: Dictionary mapping system name to output directory
        max_samples: Maximum number of samples to process
    """
    # Get list of bonafide files
    bonafide_files = [f for f in os.listdir(bonafide_dir) if f.endswith('.flac')][:max_samples]

    print(f"Generating fake audio from {len(bonafide_files)} bonafide samples...")
    print(f"Systems: {list(output_dirs.keys())}")

    transform_functions = {
        'system_A': apply_system_a_transform,
        'system_B': apply_system_b_transform,
        'system_C': apply_system_c_transform,
    }

    for filename in tqdm(bonafide_files, desc="Processing"):
        try:
            # Load bonafide audio
            filepath = os.path.join(bonafide_dir, filename)
            audio, sr = librosa.load(filepath, sr=16000)

            # Generate fake version for each system
            for system_name, output_dir in output_dirs.items():
                transform_fn = transform_functions.get(system_name)
                if transform_fn is None:
                    print(f"Warning: No transform for {system_name}, skipping")
                    continue

                # Apply transformation
                fake_audio = transform_fn(audio, sr)

                # Save to output directory
                output_filename = f"{system_name}_{filename.replace('.flac', '.wav')}"
                output_path = os.path.join(output_dir, output_filename)
                sf.write(output_path, fake_audio, sr)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print("\nGeneration complete!")
    for system_name, output_dir in output_dirs.items():
        count = len([f for f in os.listdir(output_dir) if f.endswith('.wav')])
        print(f"  {system_name}: {count} samples")


if __name__ == '__main__':
    # Paths
    bonafide_dir = 'dataset/bonafide'
    output_dirs = {
        'system_A': 'dataset/fake_system_A',
        'system_B': 'dataset/fake_system_B',
        'system_C': 'dataset/fake_system_C',
    }

    # Generate fake audio (limit to 50 samples to save space)
    generate_fake_samples(bonafide_dir, output_dirs, max_samples=50)
