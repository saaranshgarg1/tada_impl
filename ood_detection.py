#!/usr/bin/env python3
"""
Out-of-Distribution (OOD) Detection
=====================================

Problem:
    What if we encounter audio from a NEW deepfake system that
    wasn't seen during training?

    Standard kNN would still assign a label — but it would be wrong.
    We need to DETECT when a sample is from an unknown generator.

Method (TADA paper):
    Use kNN DISTANCE as an anomaly score.
    - Known samples: small distance to neighbors (in-distribution)
    - Unknown samples: large distance to neighbors (out-of-distribution)

    Set a threshold (via Equal Error Rate) and classify:
    - score < threshold → in-distribution
    - score > threshold → out-of-distribution

Evaluation Metric: EER (Equal Error Rate)
    Point where False Acceptance Rate = False Rejection Rate
    Lower EER = better system.

    FAR: Fraction of OOD samples classified as in-distribution
    FRR: Fraction of in-distribution samples classified as OOD

Connection to Course:
    - GMM can also be used for OOD: compute log-likelihood instead
      of kNN distance. Low likelihood → OOD.
    - This is similar to VAD: distinguish speech (in-dist) vs silence (OOD)
      using a distance/likelihood threshold.
"""

import os
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    f1_score,
    classification_report
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json


def load_features_with_metadata(features_dir="features"):
    """
    Load embeddings, labels, and file metadata.

    Returns:
        X: Embeddings (N x 1024)
        y_raw: List of system name strings
        filenames: List of file paths
    """
    X = np.load(os.path.join(features_dir, "embeddings.npy"))

    with open(os.path.join(features_dir, "labels.txt")) as f:
        y_raw = [line.strip() for line in f.readlines()]

    with open(os.path.join(features_dir, "metadata.txt")) as f:
        filenames = [line.strip() for line in f.readlines()]

    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]}-dim embeddings")

    return X, y_raw, filenames


def compute_eer(y_true, scores):
    """
    Compute Equal Error Rate (EER).

    EER is the operating point where FAR = FRR.
    It's a single number summarizing the ROC curve quality.

    Lower EER → better system.

    Args:
        y_true: Binary labels (1=OOD, 0=in-distribution)
        scores: Anomaly scores (higher → more likely OOD)

    Returns:
        eer: EER value (0 to 1)
        threshold: Decision threshold at EER point
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)

    # EER: point where FPR = (1 - TPR), i.e., FAR = FRR
    # FPR = False Positive Rate = FAR
    # FRR = 1 - TPR = 1 - True Positive Rate
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = float(interp1d(fpr, thresholds)(eer))

    print(f"  EER: {eer*100:.2f}% | Threshold: {threshold:.4f}")

    return eer, threshold


class KNNOODDetector:
    """
    Out-of-Distribution detector using kNN distances.

    When a sample is far from ALL training samples (high kNN distance),
    it's likely from an unknown distribution.

    Attributes:
        k: Number of neighbors to use
        knn: Fitted NearestNeighbors model
        threshold: Decision boundary (set by calibration)
    """

    def __init__(self, k=1):
        """
        Args:
            k: Number of neighbors. k=1 is recommended in TADA paper
               (smaller k → more sensitive to individual outliers)
        """
        self.k = k
        self.knn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
        self.threshold = None

    def fit(self, X_train):
        """
        Fit the kNN index on training (in-distribution) data.

        Args:
            X_train: In-distribution training embeddings (N x D)
        """
        print(f"  Fitting kNN (k={self.k}) on {X_train.shape[0]} training samples...")
        self.knn.fit(X_train)

    def anomaly_score(self, X_test):
        """
        Compute anomaly score for each test sample.

        Score = mean distance to k nearest training neighbors.
        Higher score → more anomalous (farther from known data).

        Args:
            X_test: Test embeddings (M x D)

        Returns:
            scores: Anomaly scores (M,)
        """
        distances, _ = self.knn.kneighbors(X_test)
        return np.mean(distances, axis=1)  # Average over k neighbors

    def calibrate(self, X_val, y_val):
        """
        Set decision threshold using a calibration (validation) set.

        Uses EER to find optimal threshold.

        Args:
            X_val: Validation embeddings (M x D)
            y_val: Binary labels (1=OOD, 0=in-distribution)
        """
        scores = self.anomaly_score(X_val)
        eer, self.threshold = compute_eer(y_val, scores)
        return eer

    def predict(self, X_test):
        """
        Predict in-distribution (0) or OOD (1) for each sample.

        Args:
            X_test: Test embeddings (M x D)

        Returns:
            predictions: Binary array (1=OOD, 0=in-distribution)
        """
        if self.threshold is None:
            raise ValueError("Call calibrate() first to set the threshold")

        scores = self.anomaly_score(X_test)
        return (scores > self.threshold).astype(int)


def run_ood_detection(features_dir="features", k=1, seed=42, n_ood_per_system=2):
    """
    Run OOD detection experiment.

    Setup:
        - 3 systems total (A, B, C)
        - Hold out 1 system as "unknown" (OOD)
        - Train on 2 systems
        - Test: known samples (in-dist) + held-out samples (OOD)
        - Calibrate threshold on validation set
        - Report EER and accuracy

    Args:
        features_dir: Directory with pre-extracted features
        k: kNN neighbors for OOD scoring
        seed: Random seed
        n_ood_per_system: Number of systems to treat as OOD

    Returns:
        dict with OOD detection results
    """
    print("=" * 60)
    print("OUT-OF-DISTRIBUTION (OOD) DETECTION")
    print("=" * 60)
    print(f"  k: {k}")
    print(f"  Seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)

    # Load features
    X, y_raw, filenames = load_features_with_metadata(features_dir)

    # Get unique systems (exclude bonafide)
    all_systems = sorted(set(y_raw) - {'bonafide'})
    print(f"  Systems: {all_systems}")

    # We'll do leave-one-out: each system as OOD once
    all_results = {}
    eer_list = []

    for ood_system in all_systems:
        print(f"\n--- OOD System: {ood_system} ---")

        known_systems = [s for s in all_systems if s != ood_system]

        # Create masks
        in_dist_mask = np.array([y in known_systems for y in y_raw])
        ood_mask = np.array([y == ood_system for y in y_raw])

        # In-distribution data
        X_known = X[in_dist_mask]
        y_known_binary = np.zeros(X_known.shape[0])  # All in-dist → 0

        # OOD data
        X_ood = X[ood_mask]
        y_ood_binary = np.ones(X_ood.shape[0])  # All OOD → 1

        # Split in-dist into train/val/test
        n_known = X_known.shape[0]
        n_train = int(0.6 * n_known)
        n_val = int(0.2 * n_known)

        indices = list(range(n_known))
        random.shuffle(indices)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx_known = indices[n_train + n_val:]

        X_train = X_known[train_idx]
        X_val_known = X_known[val_idx]
        X_test_known = X_known[test_idx_known]

        # Split OOD into val/test
        n_ood = X_ood.shape[0]
        n_ood_val = n_ood // 2
        ood_indices = list(range(n_ood))
        random.shuffle(ood_indices)
        X_val_ood = X_ood[ood_indices[:n_ood_val]]
        X_test_ood = X_ood[ood_indices[n_ood_val:]]

        # Calibration set (mix of known + OOD)
        X_val = np.vstack([X_val_known, X_val_ood])
        y_val = np.concatenate([
            np.zeros(len(X_val_known)),  # in-dist
            np.ones(len(X_val_ood))      # OOD
        ])

        # Test set
        X_test = np.vstack([X_test_known, X_test_ood])
        y_test = np.concatenate([
            np.zeros(len(X_test_known)),
            np.ones(len(X_test_ood))
        ])

        print(f"  Train: {X_train.shape[0]} (in-dist)")
        print(f"  Val: {X_val.shape[0]} ({X_val_known.shape[0]} in-dist + {X_val_ood.shape[0]} OOD)")
        print(f"  Test: {X_test.shape[0]} ({X_test_known.shape[0]} in-dist + {X_test_ood.shape[0]} OOD)")

        # Fit and calibrate OOD detector
        detector = KNNOODDetector(k=k)
        detector.fit(X_train)
        print("  Calibrating threshold on validation set:")
        val_eer = detector.calibrate(X_val, y_val)

        # Evaluate on test set
        test_scores = detector.anomaly_score(X_test)
        test_eer, _ = compute_eer(y_test, test_scores)
        y_pred = detector.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"\n  Test Results (OOD={ood_system}):")
        print(f"    EER: {test_eer*100:.2f}%")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    Macro F1: {f1:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=['in-dist', f'OOD ({ood_system})']))

        all_results[ood_system] = {
            'eer': float(test_eer),
            'accuracy': float(acc),
            'f1': float(f1),
            'ood_system': ood_system,
            'known_systems': known_systems,
        }
        eer_list.append(test_eer)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: OOD Detection")
    print("=" * 60)
    for system, res in all_results.items():
        print(f"  {system}: EER={res['eer']*100:.2f}%, Acc={res['accuracy']:.4f}, F1={res['f1']:.4f}")
    print(f"\n  Average EER: {np.mean(eer_list)*100:.2f}%")

    # Save results
    summary = {
        "results_per_system": all_results,
        "mean_eer": float(np.mean(eer_list)),
        "mean_accuracy": float(np.mean([r['accuracy'] for r in all_results.values()])),
        "mean_f1": float(np.mean([r['f1'] for r in all_results.values()])),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/ood_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot EER per held-out system
    plot_ood_results(all_results)

    print(f"\nResults saved to results/ood_results.json")
    return summary


def plot_ood_results(results_per_system):
    """
    Plot EER and accuracy for each OOD system configuration.

    Args:
        results_per_system: dict mapping system_name -> result dict
    """
    systems = list(results_per_system.keys())
    eers = [results_per_system[s]['eer'] * 100 for s in systems]
    accs = [results_per_system[s]['accuracy'] for s in systems]

    x = np.arange(len(systems))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # EER plot
    bars = ax1.bar(x, eers, color='tomato')
    ax1.set_xlabel("OOD (unseen) System")
    ax1.set_ylabel("EER (%)")
    ax1.set_title("OOD Detection: EER per Held-Out System\n(Lower = Better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax1.legend()
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)

    # Accuracy plot
    bars2 = ax2.bar(x, accs, color='steelblue')
    ax2.set_xlabel("OOD (unseen) System")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("OOD Detection: Accuracy per Held-Out System")
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("results/ood_results.png", dpi=150, bbox_inches='tight')
    print("Saved OOD results plot to results/ood_results.png")
    plt.close()


if __name__ == '__main__':
    run_ood_detection(k=1)
