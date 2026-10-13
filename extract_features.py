#!/usr/bin/env python3
"""
Feature Extraction: w2v-BERT-2.0 Embeddings
============================================

This script extracts frozen speech embeddings using the pretrained
w2v-BERT-2.0 model from Facebook/Meta AI.

Key Concepts:
-------------
1. **Self-Supervised Learning (SSL)**: w2v-BERT-2.0 was trained on
   unlabeled speech data to learn general speech representations.

2. **Frozen Features**: We do NOT fine-tune the model. We only use it
   as a feature extractor (like using MFCC, but learned from data).

3. **Layer Selection**: The paper uses Layer 4 (early-middle layer).
   Early layers capture acoustic details; later layers capture semantics.

4. **Average Pooling**: Each audio produces a sequence of frame-level
   embeddings. We average across time to get one vector per audio file.

Architecture:
-------------
Audio (16kHz) → w2v-BERT-2.0 → Hidden States (13 layers)
              → Select Layer 4 → Average Pool → 1024-dim embedding

Output:
-------
- features/embeddings_<dataset>.npy: Stacked embeddings (N x 1024)
- features/labels_<dataset>.txt: Corresponding labels (one per line)
- features/metadata_<dataset>.txt: File paths for tracking
"""

import os
import sys
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import argparse


class EmbeddingExtractor:
    """
    Extracts frozen embeddings from pretrained w2v-BERT-2.0 model.

    The model has 13 transformer layers (0-12). Layer indices:
    - Layer 0: CNN feature extraction output
    - Layers 1-12: Transformer layers
    - Layer 4: Early-middle layer (used in TADA paper)

    Attributes:
        device: 'cuda' if GPU available, else 'cpu'
        feature_extractor: Preprocessor (resampling, normalization)
        model: w2v-BERT-2.0 model in eval mode
        layer_index: Which layer to extract (4 in paper)
    """

    def __init__(self, model_name="facebook/w2v-bert-2.0", layer_index=4):
        """
        Initialize the embedding extractor.

        Args:
            model_name: HuggingFace model identifier
            layer_index: Which transformer layer to use (0-12)
        """
        print(f"Initializing EmbeddingExtractor...")
        print(f"  Model: {model_name}")
        print(f"  Layer: {layer_index}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {self.device}")

        # Load feature extractor (handles audio preprocessing)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        # Load model with hidden state outputs enabled
        self.model = Wav2Vec2BertModel.from_pretrained(
            model_name,
            output_hidden_states=True  # Return all layer outputs
        )
        self.model.eval()  # Set to evaluation mode (disables dropout)
        self.model.to(self.device)

        self.layer_index = layer_index

        print(f"Model loaded successfully!")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

    def extract_embedding(self, audio_path):
        """
        Extract embedding from a single audio file.

        Pipeline:
        1. Load audio at 16kHz (model's expected sample rate)
        2. Preprocess with feature extractor
        3. Forward pass through model (no gradients)
        4. Extract Layer 4 hidden states
        5. Average pool across time dimension

        Args:
            audio_path: Path to audio file

        Returns:
            embedding: 1024-dimensional numpy array
        """
        # Load audio at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)

        # Preprocess: normalize, pad, convert to tensors
        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",  # Return PyTorch tensors
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass (no gradient computation)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract hidden states
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Shape of each: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.hidden_states

        # Select the specified layer
        # +1 because hidden_states[0] is the CNN output before transformers
        layer_output = hidden_states[self.layer_index + 1]

        # Average pool across time dimension (dim=1)
        # Shape: (batch_size, hidden_size)
        embedding = torch.mean(layer_output, dim=1)

        # Convert to numpy and move to CPU
        embedding = embedding.cpu().numpy()

        # Return as 1D array (squeeze batch dimension)
        return embedding.squeeze()

    def extract_from_directory(self, audio_dir, label, output_dir):
        """
        Extract embeddings from all audio files in a directory.

        Args:
            audio_dir: Directory containing audio files
            label: Label for this dataset (e.g., "bonafide", "system_A")
            output_dir: Where to save the extracted features

        Returns:
            embeddings: Stacked numpy array (N x 1024)
            labels: List of labels (length N)
            filenames: List of file paths (length N)
        """
        # Find all audio files
        audio_files = [
            f for f in os.listdir(audio_dir)
            if f.endswith(('.wav', '.flac', '.mp3'))
        ]

        print(f"\nProcessing {audio_dir}")
        print(f"  Files: {len(audio_files)}")
        print(f"  Label: {label}")

        embeddings = []
        labels = []
        filenames = []
        failed_files = []

        # Process each file
        for filename in tqdm(audio_files, desc=f"Extracting {label}"):
            filepath = os.path.join(audio_dir, filename)
            try:
                embedding = self.extract_embedding(filepath)
                embeddings.append(embedding)
                labels.append(label)
                filenames.append(filepath)
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                failed_files.append(filename)

        if failed_files:
            print(f"  Failed: {len(failed_files)} files")

        # Stack embeddings
        embeddings = np.vstack(embeddings)

        print(f"  Extracted: {embeddings.shape}")

        return embeddings, labels, filenames


def main():
    parser = argparse.ArgumentParser(
        description="Extract w2v-BERT-2.0 embeddings for TADA deepfake detection"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=4,
        help="Which layer to extract (0-12, default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="features",
        help="Output directory for features"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize extractor
    extractor = EmbeddingExtractor(layer_index=args.layer)

    # Define datasets
    datasets = [
        ("dataset/bonafide", "bonafide"),
        ("dataset/fake_system_A", "system_A"),
        ("dataset/fake_system_B", "system_B"),
        ("dataset/fake_system_C", "system_C"),
    ]

    # Extract features for each dataset
    all_embeddings = []
    all_labels = []
    all_filenames = []

    for audio_dir, label in datasets:
        if not os.path.exists(audio_dir):
            print(f"Warning: {audio_dir} does not exist, skipping")
            continue

        embeddings, labels, filenames = extractor.extract_from_directory(
            audio_dir, label, args.output_dir
        )

        all_embeddings.append(embeddings)
        all_labels.extend(labels)
        all_filenames.extend(filenames)

    # Combine all datasets
    all_embeddings = np.vstack(all_embeddings)

    # Save features
    print(f"\nSaving features...")
    np.save(
        os.path.join(args.output_dir, "embeddings.npy"),
        all_embeddings
    )

    with open(os.path.join(args.output_dir, "labels.txt"), 'w') as f:
        f.write('\n'.join(all_labels))

    with open(os.path.join(args.output_dir, "metadata.txt"), 'w') as f:
        f.write('\n'.join(all_filenames))

    print(f"  embeddings.npy: {all_embeddings.shape}")
    print(f"  labels.txt: {len(all_labels)} lines")
    print(f"  metadata.txt: {len(all_filenames)} lines")

    # Print summary
    print("\nSummary:")
    from collections import Counter
    label_counts = Counter(all_labels)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} samples")

    print("\nFeature extraction complete!")


if __name__ == '__main__':
    main()
