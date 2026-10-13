#!/usr/bin/env python3
"""
TADA-lite: Main Orchestration Script
======================================

Runs the complete pipeline:
    1. MFCC baseline (classical features)
    2. Attribution (SSL embeddings + kNN)
    3. OOD detection (kNN distance threshold)
    4. Comparison summary

Usage:
    python run_all.py              # Run full pipeline
    python run_all.py --skip-extract   # If features already extracted

Steps:
    1. Extract features → features/
    2. Run MFCC baseline → results/mfcc_results.json
    3. Run kNN attribution → results/attribution_results.json
    4. Run OOD detection → results/ood_results.json
    5. Print comparison table
"""

import os
import json
import argparse
import sys


def print_summary():
    """Print a comparison table of all results."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    results_files = {
        "MFCC Baseline": "results/mfcc_results.json",
        "SSL Attribution (k=5)": "results/attribution_results.json",
        "OOD Detection": "results/ood_results.json",
    }

    for name, path in results_files.items():
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"\n{name}:")

            if "accuracy" in data:
                print(f"  Accuracy:  {data['accuracy']:.4f} ({data['accuracy']*100:.1f}%)")
            if "macro_f1" in data:
                print(f"  Macro F1:  {data['macro_f1']:.4f}")
            if "mean_eer" in data:
                print(f"  Mean EER:  {data['mean_eer']*100:.2f}%")
            if "mean_accuracy" in data:
                print(f"  Mean Acc:  {data['mean_accuracy']:.4f}")
        else:
            print(f"\n{name}: results not found at {path}")

    print("\n" + "=" * 70)
    print("Output files:")
    output_files = [
        ("results/confusion_matrix.png", "Attribution confusion matrix"),
        ("results/embedding_space_pca.png", "PCA embedding visualization"),
        ("results/k_ablation.png", "kNN k-value ablation"),
        ("results/comparison.png", "MFCC vs SSL comparison"),
        ("results/ood_results.png", "OOD detection per system"),
    ]
    for filepath, desc in output_files:
        status = "[✓]" if os.path.exists(filepath) else "[ ]"
        print(f"  {status} {filepath}: {desc}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete TADA-lite attribution + OOD pipeline"
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip feature extraction (use if features/ already populated)"
    )
    parser.add_argument(
        "--skip-mfcc",
        action="store_true",
        help="Skip MFCC baseline"
    )
    parser.add_argument(
        "--skip-ood",
        action="store_true",
        help="Skip OOD detection"
    )
    args = parser.parse_args()

    # Check prerequisites
    if not os.path.exists("dataset/bonafide"):
        print("ERROR: dataset/bonafide not found")
        print("Run generate_fake_audio.py first, or prepare dataset manually")
        sys.exit(1)

    os.makedirs("results", exist_ok=True)
    os.makedirs("features", exist_ok=True)

    # ─── Step 1: Feature Extraction ──────────────────────────────────────────
    if not args.skip_extract:
        if os.path.exists("features/embeddings.npy"):
            print("\n[Step 1] Features already extracted, skipping")
            print("  (Delete features/ to re-extract)")
        else:
            print("\n[Step 1] Extracting SSL embeddings...")
            from extract_features import EmbeddingExtractor
            import numpy as np

            extractor = EmbeddingExtractor(layer_index=4)
            datasets = [
                ("dataset/bonafide", "bonafide"),
                ("dataset/fake_system_A", "system_A"),
                ("dataset/fake_system_B", "system_B"),
                ("dataset/fake_system_C", "system_C"),
            ]

            all_embeddings = []
            all_labels = []
            all_filenames = []

            for audio_dir, label in datasets:
                if not os.path.exists(audio_dir):
                    print(f"  Skipping {audio_dir}")
                    continue
                emb, labs, fns = extractor.extract_from_directory(audio_dir, label, "features")
                all_embeddings.append(emb)
                all_labels.extend(labs)
                all_filenames.extend(fns)

            all_embeddings = np.vstack(all_embeddings)
            np.save("features/embeddings.npy", all_embeddings)
            with open("features/labels.txt", 'w') as f:
                f.write('\n'.join(all_labels))
            with open("features/metadata.txt", 'w') as f:
                f.write('\n'.join(all_filenames))
            print(f"  Saved {all_embeddings.shape[0]} embeddings")
    else:
        print("\n[Step 1] Skipping feature extraction (--skip-extract)")

    # ─── Step 2: MFCC Baseline ───────────────────────────────────────────────
    if not args.skip_mfcc:
        print("\n[Step 2] Running MFCC baseline...")
        from mfcc_baseline import run_mfcc_baseline
        run_mfcc_baseline()
    else:
        print("\n[Step 2] Skipping MFCC baseline (--skip-mfcc)")

    # ─── Step 3: Attribution (SSL + kNN) ─────────────────────────────────────
    print("\n[Step 3] Running SSL attribution...")
    from attribution import run_attribution, ablation_k
    run_attribution(k=5)
    ablation_k()

    # ─── Step 4: OOD Detection ───────────────────────────────────────────────
    if not args.skip_ood:
        print("\n[Step 4] Running OOD detection...")
        from ood_detection import run_ood_detection
        run_ood_detection(k=1)
    else:
        print("\n[Step 4] Skipping OOD detection (--skip-ood)")

    # ─── Step 5: Print Summary ───────────────────────────────────────────────
    print_summary()


if __name__ == '__main__':
    main()
