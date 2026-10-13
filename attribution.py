#!/usr/bin/env python3
"""
Checkpoint Attribution via k-Nearest Neighbors (kNN)
======================================================

This is the core of the TADA attribution method.

Problem:
    Given an audio clip, identify WHICH TTS/voice-conversion system generated it.
    This is like a "fingerprinting" task — each system leaves subtle acoustic
    traces in its output, and we want to identify the source.

Method:
    1. Use frozen SSL embeddings (from extract_features.py)
    2. Train a kNN classifier on 80% of data
    3. Predict system labels for remaining 20%
    4. Evaluate with precision, recall, F1

Why kNN works here:
    Each TTS/VC system produces audio with consistent spectral characteristics.
    These characteristics cluster together in the embedding space.
    kNN finds the closest known examples and votes on the label.

Connection to Course Material:
    - kNN is a simple non-parametric classifier (distances in feature space)
    - Compare to GMM: kNN is more flexible, no Gaussian assumption
    - Compare to DTW: kNN on fixed-dim vectors vs sequence alignment
    - Embeddings are like MFCC but learned by a transformer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.decomposition import PCA
import json


def load_features(features_dir="features"):
    """
    Load pre-extracted embeddings and labels from disk.

    Args:
        features_dir: Directory containing .npy and .txt files

    Returns:
        X: Embedding matrix (N x 1024)
        y: Label array (N,)
        label_names: List of unique system names
    """
    embeddings_path = os.path.join(features_dir, "embeddings.npy")
    labels_path = os.path.join(features_dir, "labels.txt")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"No embeddings found at {embeddings_path}. "
            "Run extract_features.py first."
        )

    # Load embeddings
    X = np.load(embeddings_path)
    print(f"Loaded embeddings: {X.shape}")  # (N, 1024)

    # Load labels
    with open(labels_path) as f:
        y_raw = [line.strip() for line in f.readlines()]
    print(f"Loaded labels: {len(y_raw)}")

    # Sanity check
    assert len(y_raw) == X.shape[0], \
        f"Mismatch: {X.shape[0]} embeddings, {len(y_raw)} labels"

    # Convert string labels to integers
    # But keep label_names for plotting
    label_names = sorted(set(y_raw))
    label2idx = {name: idx for idx, name in enumerate(label_names)}
    y = np.array([label2idx[label] for label in y_raw])

    print(f"Classes ({len(label_names)}): {label_names}")

    # Print class distribution
    from collections import Counter
    counts = Counter(y_raw)
    for label in label_names:
        print(f"  {label}: {counts[label]} samples")

    return X, y, label_names


def filter_fakes_only(X, y, label_names):
    """
    Remove bonafide samples for attribution (we only attribute fakes).

    In deepfake attribution, we only care about classifying WHICH SYSTEM
    generated the fake audio. Bonafide samples are real and have no system.

    Args:
        X: Full embedding matrix
        y: Full label array
        label_names: List of label names

    Returns:
        X_fakes: Embeddings of fake samples only
        y_fakes: Labels of fake samples only
        fake_label_names: Names of fake systems only
    """
    # Find bonafide index
    bonafide_idx = label_names.index("bonafide") if "bonafide" in label_names else None

    if bonafide_idx is None:
        print("No bonafide samples found, using all data")
        return X, y, label_names

    # Create mask for fake samples
    fake_mask = y != bonafide_idx
    X_fakes = X[fake_mask]
    y_fakes_raw = y[fake_mask]

    # Remap labels to exclude bonafide
    fake_label_names = [n for n in label_names if n != "bonafide"]
    old2new = {}
    new_idx = 0
    for i, name in enumerate(label_names):
        if name != "bonafide":
            old2new[i] = new_idx
            new_idx += 1

    y_fakes = np.array([old2new[label] for label in y_fakes_raw])

    print(f"\nFiltered to fakes only: {X_fakes.shape[0]} samples")
    return X_fakes, y_fakes, fake_label_names


def run_attribution(features_dir="features", k=5, use_scaling=False, seed=42):
    """
    Run checkpoint attribution experiment.

    Args:
        features_dir: Directory with pre-extracted features
        k: Number of neighbors in kNN
        use_scaling: Whether to standardize features first
        seed: Random seed for reproducibility

    Returns:
        dict with results including accuracy, F1, confusion matrix
    """
    print("=" * 60)
    print("CHECKPOINT ATTRIBUTION")
    print("=" * 60)
    print(f"  k (neighbors): {k}")
    print(f"  Scaling: {use_scaling}")
    print(f"  Seed: {seed}")

    # Load features
    X, y, label_names = load_features(features_dir)

    # For attribution, use only fake samples
    X, y, label_names = filter_fakes_only(X, y, label_names)

    # Train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y  # Ensure balanced splits
    )

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # Optional: standardize features (zero mean, unit variance)
    if use_scaling:
        print("\nApplying StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Fit kNN classifier
    print(f"\nFitting kNN (k={k})...")
    clf = KNeighborsClassifier(
        n_neighbors=k,
        metric='euclidean',  # Euclidean distance in embedding space
        n_jobs=-1             # Use all CPU cores
    )
    clf.fit(X_train, y_train)

    # Predict on test set
    print("Predicting...")
    y_pred = clf.predict(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=label_names)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    print(f"\nResults:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"\nClassification Report:")
    print(report)

    # Save results
    results = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "k": k,
        "use_scaling": use_scaling,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "label_names": label_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/attribution_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open("results/attribution_report.txt", 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Plot confusion matrix
    plot_confusion_matrix(cm, label_names, save_path="results/confusion_matrix.png")

    # Plot PCA visualization of embeddings
    plot_embedding_space(
        X, y,
        label_names=["bonafide"] + label_names
        if "bonafide" in load_features(features_dir)[2]
        else label_names,
        save_path="results/embedding_space_pca.png"
    )

    return results


def plot_confusion_matrix(cm, label_names, save_path=None):
    """
    Plot normalized confusion matrix as a heatmap.

    Each cell (i,j) shows what fraction of system i's samples
    were predicted as system j.

    Args:
        cm: Confusion matrix (normalized)
        label_names: List of class names
        save_path: Where to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        xticklabels=label_names,
        yticklabels=label_names,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        vmin=0, vmax=1
    )
    plt.xlabel("Predicted System")
    plt.ylabel("True System")
    plt.title("Attribution Confusion Matrix\n(Row-normalized: shows recall per class)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_embedding_space(X, y, label_names, save_path=None, n_components=2):
    """
    Visualize embeddings in 2D using PCA.

    PCA reduces 1024-dimensional embeddings to 2D for visualization.
    Each point is an audio sample, colored by its system label.

    If the method works, samples from the same system should cluster
    together — which is why kNN can classify them!

    Args:
        X: Embeddings (N x 1024)
        y: Integer labels (N,)
        label_names: List of class names
        save_path: Where to save the plot
        n_components: Dimensions for PCA (2 for visualization)
    """
    print(f"\nComputing PCA ({n_components} components)...")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    explained_var = pca.explained_variance_ratio_.sum() * 100
    print(f"  Explained variance: {explained_var:.1f}%")

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))

    for i, (name, color) in enumerate(zip(label_names, colors)):
        mask = y == i
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[color],
            label=name,
            alpha=0.6,
            s=30
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(
        f"w2v-BERT-2.0 Layer 4 Embeddings (PCA)\n"
        f"Explained variance: {explained_var:.1f}%"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PCA plot to {save_path}")
    else:
        plt.show()

    plt.close()


def ablation_k(features_dir="features", k_values=None, seed=42):
    """
    Ablation study: how does accuracy change with different k values?

    This is an important analysis showing model sensitivity to k.
    In the paper, k=21 is used — but optimal k depends on dataset size.

    Args:
        features_dir: Directory with pre-extracted features
        k_values: List of k values to test
        seed: Random seed

    Returns:
        dict mapping k -> accuracy
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 11, 21, 31]

    print("\n" + "=" * 60)
    print("ABLATION STUDY: k values")
    print("=" * 60)

    X, y, label_names = load_features(features_dir)
    X, y, label_names = filter_fakes_only(X, y, label_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    results = {}
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results[k] = {'accuracy': acc, 'f1': f1}
        print(f"  k={k:3d}: Acc={acc:.3f}, F1={f1:.3f}")

    # Plot k ablation
    k_list = sorted(results.keys())
    accs = [results[k]['accuracy'] for k in k_list]
    f1s = [results[k]['f1'] for k in k_list]

    plt.figure(figsize=(8, 5))
    plt.plot(k_list, accs, 'o-', label='Accuracy', color='blue')
    plt.plot(k_list, f1s, 's--', label='Macro F1', color='orange')
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Score")
    plt.title("kNN Attribution: Effect of k")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/k_ablation.png", dpi=150, bbox_inches='tight')
    print("Saved k ablation plot to results/k_ablation.png")
    plt.close()

    return results


if __name__ == '__main__':
    # Run full attribution experiment
    run_attribution(k=5, use_scaling=False)

    # Run ablation study
    ablation_k()
