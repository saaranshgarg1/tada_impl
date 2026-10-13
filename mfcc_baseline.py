#!/usr/bin/env python3
"""
MFCC Baseline: Classical Feature Extraction for Deepfake Attribution
======================================================================

This script implements a CLASSICAL MFCC-based baseline for comparison
against the SSL embedding approach (attribution.py).

Why a Baseline?
---------------
Research papers need comparison against simpler approaches.
A good paper shows:
1. The proposed method (SSL + kNN) works
2. Classical features (MFCC + kNN) work less well → proving SSL features are better

MFCC Features (from course material):
--------------------------------------
1. Pre-emphasis: boost high frequencies (α ≈ 0.97)
   y[n] = x[n] - α * x[n-1]

2. Framing: split into 25ms windows, 10ms hop
   Each frame is quasi-stationary (speech doesn't change much in 25ms)

3. Hamming window: reduce edge effects
   w[n] = 0.54 - 0.46 * cos(2πn/(N-1))

4. STFT: Short-Time Fourier Transform
   X_t[k] = Σ x[n] * w[n] * e^(-j2πkn/N)

5. Mel filterbank: 40 triangular filters on Mel scale
   mel = 2595 * log10(1 + f/700)
   (Matches human auditory perception)

6. Log compression: log(filterbank energy)
   Compresses dynamic range

7. DCT: Discrete Cosine Transform
   Decorrelates filterbank channels
   Keep 13 coefficients (captures spectral envelope)

8. Delta features: first + second derivatives
   Capture how features change over time
   Concatenated: 13 static + 13 Δ + 13 ΔΔ = 39-dim

9. Aggregate: mean + std across frames → 78-dim vector per clip

This baseline uses librosa for computation.
"""

import os
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json


# ─── MFCC Parameters ───────────────────────────────────────────────────────────
N_MFCC = 13         # Number of cepstral coefficients
N_MELS = 40         # Number of Mel filterbank channels
HOP_LENGTH = 160    # 10ms hop at 16kHz
WIN_LENGTH = 400    # 25ms window at 16kHz
PREEMPHASIS = 0.97  # Pre-emphasis coefficient


def extract_mfcc(audio_path, n_mfcc=N_MFCC):
    """
    Extract MFCC features from an audio file.

    Pipeline:
        1. Load audio at 16kHz
        2. Pre-emphasis filter
        3. MFCC via librosa (handles framing, windowing, Mel, log, DCT)
        4. Compute delta and delta-delta
        5. Aggregate: mean + std over time

    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients to keep (default: 13)

    Returns:
        features: 1D numpy array of shape (3 * n_mfcc * 2,) = (78,)
                  78 = 13 MFCC + 13 Δ + 13 ΔΔ, each with mean+std
    """
    # Step 1: Load audio at 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)

    # Step 2: Pre-emphasis (boost high frequencies)
    # Compensates for natural rolloff in speech spectrum
    audio = librosa.effects.preemphasis(audio, coef=PREEMPHASIS)

    # Step 3: Compute MFCCs using librosa
    # librosa handles: framing, windowing, power spectrum, mel filter, log, DCT
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=80,    # Minimum frequency (below fundamental)
        fmax=7600   # Maximum frequency (Nyquist for 16kHz)
    )
    # mfcc shape: (n_mfcc, n_frames)

    # Step 4: Delta features
    # Δ captures rate of change (first derivative over 9 frames)
    # ΔΔ captures acceleration (second derivative)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Step 5: Aggregate over time
    # Compute mean and std for each coefficient dimension
    # This reduces variable-length audio to fixed-length vector
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.std(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1),
        np.std(delta2_mfcc, axis=1),
    ])

    return features


def extract_all_mfcc(audio_dir, label, n_mfcc=N_MFCC):
    """
    Extract MFCC features from all audio files in a directory.

    Args:
        audio_dir: Directory containing audio files
        label: Label string for this group
        n_mfcc: Number of MFCC coefficients

    Returns:
        embeddings: (N, feature_dim) array
        labels: List of N label strings
    """
    files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.flac', '.mp3'))]

    embeddings = []
    labels = []

    for filename in tqdm(files, desc=f"MFCC {label}"):
        filepath = os.path.join(audio_dir, filename)
        try:
            feat = extract_mfcc(filepath, n_mfcc)
            embeddings.append(feat)
            labels.append(label)
        except Exception as e:
            print(f"  Error: {filename}: {e}")

    return np.array(embeddings), labels


def run_mfcc_baseline(features_dir_ssl="features", k=5, seed=42):
    """
    Run MFCC-based attribution and compare against SSL embeddings.

    Args:
        features_dir_ssl: Directory with SSL features (for comparison)
        k: Number of kNN neighbors
        seed: Random seed

    Returns:
        dict with MFCC results
    """
    print("=" * 60)
    print("MFCC BASELINE")
    print("=" * 60)

    # Datasets to process
    datasets = [
        ("dataset/fake_system_A", "system_A"),
        ("dataset/fake_system_B", "system_B"),
        ("dataset/fake_system_C", "system_C"),
    ]

    all_embeddings = []
    all_labels = []

    for audio_dir, label in datasets:
        if not os.path.exists(audio_dir):
            print(f"  Skipping {audio_dir} (not found)")
            continue

        embeddings, labels = extract_all_mfcc(audio_dir, label)
        all_embeddings.append(embeddings)
        all_labels.extend(labels)
        print(f"  {label}: {embeddings.shape[0]} samples, {embeddings.shape[1]}-dim features")

    all_embeddings = np.vstack(all_embeddings)

    # Convert labels to integers
    label_names = sorted(set(all_labels))
    label2idx = {n: i for i, n in enumerate(label_names)}
    y = np.array([label2idx[l] for l in all_labels])

    print(f"\nTotal: {all_embeddings.shape[0]} samples, {all_embeddings.shape[1]}-dim features")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        all_embeddings, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Standardize MFCC features (important for MFCC, less for SSL)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit kNN
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=label_names)

    print(f"\nResults (MFCC Baseline):")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"\n{report}")

    # Save results
    results = {
        "method": "MFCC",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "feature_dim": int(all_embeddings.shape[1]),
        "k": k,
        "label_names": label_names,
        "report": report,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/mfcc_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open("results/mfcc_report.txt", 'w') as f:
        f.write(f"MFCC Baseline Results\n")
        f.write(f"Feature dim: {all_embeddings.shape[1]}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(report)

    # Compare MFCC vs SSL (if SSL results exist)
    ssl_results_path = "results/attribution_results.json"
    if os.path.exists(ssl_results_path):
        with open(ssl_results_path) as f:
            ssl_results = json.load(f)

        print("\n" + "=" * 60)
        print("COMPARISON: MFCC vs SSL Embeddings")
        print("=" * 60)
        print(f"  MFCC (78-dim) Accuracy: {acc:.4f}")
        print(f"  SSL  (1024-dim) Accuracy: {ssl_results['accuracy']:.4f}")
        print(f"  MFCC Macro F1: {macro_f1:.4f}")
        print(f"  SSL  Macro F1: {ssl_results['macro_f1']:.4f}")
        print(f"  Difference (Acc): {ssl_results['accuracy'] - acc:+.4f}")

        # Plot comparison
        plot_comparison(acc, macro_f1, ssl_results['accuracy'], ssl_results['macro_f1'])

    return results


def plot_comparison(mfcc_acc, mfcc_f1, ssl_acc, ssl_f1):
    """
    Bar chart comparing MFCC baseline vs SSL embeddings.

    Args:
        mfcc_acc: MFCC accuracy
        mfcc_f1: MFCC macro F1
        ssl_acc: SSL accuracy
        ssl_f1: SSL macro F1
    """
    methods = ['MFCC (78d)\nBaseline', 'w2v-BERT-2.0\n(1024d, frozen)']
    accs = [mfcc_acc, ssl_acc]
    f1s = [mfcc_f1, ssl_f1]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1s, width, label='Macro F1', color='coral')

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Score')
    ax.set_title('Deepfake Attribution: MFCC vs SSL Embeddings')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=150, bbox_inches='tight')
    print("Saved comparison plot to results/comparison.png")
    plt.close()


if __name__ == '__main__':
    run_mfcc_baseline()
