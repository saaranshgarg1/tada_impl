# TADA-lite: Training-Free Audio Deepfake Attribution

A simplified, well-documented implementation of the TADA (Training-free Attribution and Out-of-Domain Detection of Audio Deepfakes) method from Interspeech 2025.

**Original Paper**: [TADA: Training-free Attribution and Out-of-Domain Detection of Audio Deepfake](https://arxiv.org/abs/2506.05802) by Adriana Stan et al.

## What This Project Does

This project demonstrates **deepfake attribution** and **out-of-distribution detection** for synthetic speech:

1. **Attribution**: Given a fake audio clip, identify WHICH TTS/voice-conversion system generated it
2. **OOD Detection**: Detect when audio comes from an UNKNOWN generator not seen during training
3. **Baseline Comparison**: Compare modern SSL embeddings against classical MFCC features

**Key Innovation**: The method uses **frozen (pretrained)** speech embeddings from w2v-BERT-2.0 + simple k-nearest neighbors (kNN) — NO neural network training required!

## Project Structure

```
tada_impl/
├── dataset/
│   ├── bonafide/          # Real human speech (LibriSpeech)
│   ├── fake_system_A/     # Fake audio from System A
│   ├── fake_system_B/     # Fake audio from System B
│   └── fake_system_C/     # Fake audio from System C
├── features/
│   ├── embeddings.npy     # Extracted w2v-BERT embeddings (N x 1024)
│   ├── labels.txt         # System labels (one per line)
│   └── metadata.txt       # File paths
├── results/
│   ├── attribution_results.json
│   ├── mfcc_results.json
│   ├── ood_results.json
│   ├── confusion_matrix.png
│   ├── embedding_space_pca.png
│   ├── comparison.png
│   └── ood_results.png
└── scripts/
    ├── generate_fake_audio.py      # Create fake audio via signal processing
    ├── extract_features.py         # Extract w2v-BERT-2.0 embeddings
    ├── attribution.py              # kNN-based system attribution
    ├── mfcc_baseline.py            # Classical MFCC baseline
    ├── ood_detection.py            # Out-of-distribution detection
    └── run_all.py                  # Main orchestration script
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment with uv (fast)
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers librosa scikit-learn numpy tqdm joblib matplotlib seaborn scipy soundfile
```

### 2. Generate Dataset

Since large deepfake datasets require authentication, we generate a small demo dataset using signal processing transformations:

```bash
python generate_fake_audio.py
```

This creates ~150 samples:
- 50 bonafide (real LibriSpeech)
- 49 System A (vocoder simulation)
- 49 System B (pitch-shifted)
- 49 System C (bandwidth-limited)

**Total size**: ~20 MB audio + ~1.5 GB model weights

### 3. Extract Features

Extract frozen w2v-BERT-2.0 embeddings (Layer 4, average-pooled):

```bash
python extract_features.py --layer 4 --output-dir features
```

This downloads the pretrained model (~1.5 GB) and extracts 1024-dimensional embeddings for each audio file. **CPU inference takes ~5-10 minutes** for 150 samples.

### 4. Run Experiments

**Option A: Run everything**
```bash
python run_all.py
```

**Option B: Run individual experiments**
```bash
# MFCC baseline (classical features)
python mfcc_baseline.py

# Attribution (SSL + kNN)
python attribution.py

# OOD detection (kNN distance)
python ood_detection.py
```

## Method Overview

### 1. Feature Extraction (`extract_features.py`)

```
Audio (16kHz) → w2v-BERT-2.0 (frozen) → Layer 4 Hidden States → Average Pool → 1024-dim embedding
```

**Why w2v-BERT-2.0?**
- Self-supervised learning on 4.5M hours of unlabeled speech
- Captures rich acoustic + linguistic features
- Frozen (no training) → fast and reproducible

**Why Layer 4?**
- Early layers: low-level acoustics (pitch, voicing, formants)
- Middle layers: phonetic features
- Late layers: semantic content
- **Layer 4**: Best balance for detecting generation artifacts

### 2. Attribution (`attribution.py`)

**Problem**: Which TTS/VC system generated this audio?

**Method**:
1. Train/test split (80/20)
2. Fit k-nearest neighbors (k=5) on embeddings
3. Predict system label by majority vote of nearest neighbors

**Why kNN works**: Each TTS system has consistent spectral characteristics that cluster in embedding space.

**Evaluation**:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- k-value ablation (k=1, 3, 5, 7, 11, 21, 31)

### 3. MFCC Baseline (`mfcc_baseline.py`)

**Purpose**: Compare SSL embeddings against classical speech features.

**Pipeline**:
```
Audio → Pre-emphasis → Framing (25ms/10ms) → STFT → Mel Filterbank (40 channels)
      → Log → DCT → 13 MFCC + Δ + ΔΔ → Mean/Std aggregation → 78-dim vector
```

**Connection to Course**:
- Pre-emphasis: compensate for high-frequency rolloff
- Mel scale: matches human perception
- Cepstrum: separates source (pitch) from filter (vocal tract)
- Delta features: capture temporal dynamics

**Expected Result**: MFCC should work but be outperformed by SSL embeddings (demonstrating the value of learned representations).

### 4. OOD Detection (`ood_detection.py`)

**Problem**: What if we encounter a NEW deepfake system?

**Method**:
1. Hold out one system (e.g., System C) as "unknown"
2. Train kNN on remaining systems (A + B)
3. Compute kNN distance for test samples
4. Threshold distance → in-distribution vs out-of-distribution

**Anomaly Score**: Mean distance to k=1 nearest neighbors
- Low distance → similar to training data → in-distribution
- High distance → dissimilar → OOD (unknown system)

**Calibration**: Set threshold via Equal Error Rate (EER) on validation set
- EER: point where False Acceptance Rate = False Rejection Rate
- Lower EER = better detector

## Results Interpretation

### Attribution Metrics

**Accuracy**: Overall correct predictions  
**Macro F1**: Average F1 across systems (accounts for class imbalance)  
**Confusion Matrix**: Shows which systems are confused with each other

Example:
```
              precision  recall  f1-score
system_A        0.95      0.92     0.93
system_B        0.89      0.94     0.91
system_C        0.93      0.90     0.91
```

**Good result**: >90% accuracy (systems are distinguishable)  
**Poor result**: ~33% accuracy (random guessing for 3 classes)

### OOD Detection Metrics

**EER (Equal Error Rate)**: Single number summarizing detector quality
- 0% = perfect (no errors)
- 50% = random guessing
- Typical good result: 5-15%

**Accuracy**: Fraction of correct in-dist / OOD classifications

### MFCC vs SSL Comparison

**Expected pattern**:
- MFCC: 70-85% accuracy (decent, uses hand-crafted features)
- SSL: 85-95% accuracy (better, learned from massive data)

If SSL is NOT better → either:
1. Dataset too small/simple (both methods saturate)
2. Bug in feature extraction
3. Need more training data for kNN

## Connection to Course Material

| Course Topic | Where It Appears |
|-------------|------------------|
| **Source-Filter Model** | MFCC pre-emphasis compensates for source+filter; fake systems have different filter characteristics |
| **Cepstral Analysis** | MFCC uses cepstrum (log spectrum → DCT) to separate pitch from vocal tract |
| **LPC / Formants** | Fake systems may have abnormal formant transitions detectable in embeddings |
| **VAD** | Short-time energy/spectral features similar to MFCC; both use framing + windowing |
| **DTW** | kNN is a fixed-vector analog of DTW template matching |
| **GMM** | Alternative to kNN: fit Gaussian per system, classify via log-likelihood |
| **HMM** | Could model temporal dynamics of fake speech (extension idea) |
| **ASR/TTS** | We're attributing TTS output — understanding synthesis helps understand artifacts |
| **SSL / Foundation Models** | w2v-BERT-2.0 is a self-supervised speech foundation model |
| **Ethics** | Deepfake detection is a defense against malicious synthesis |

## Extending This Project

### Research Extensions

1. **Add Real TTS Systems**: Replace signal-processing fakes with actual TTS (Tacotron2, FastSpeech2, VITS)
2. **Larger Datasets**: Use ASVspoof 2019 LA (~25k samples, 6 systems)
3. **Layer Analysis**: Compare all 13 w2v-BERT layers, plot attribution accuracy per layer
4. **GMM Baseline**: Replace kNN with Gaussian Mixture Models
5. **Temporal Modeling**: Use LSTM on frame-level embeddings instead of average pooling
6. **Cross-Dataset Generalization**: Train on ASVspoof, test on In-The-Wild or FoR
7. **Adversarial Robustness**: Add noise, compression, resampling and measure degradation
8. **Explainability**: Which frequency bands / phonemes are most discriminative?

### Classical Signal Processing Extensions

1. **LPC Residual**: Extract LPC residual, check if fake systems have abnormal excitation
2. **Formant Tracking**: Compare formant trajectories of real vs fake
3. **Pitch Analysis**: Do fake systems have unnatural F0 contours?
4. **Spectral Flux**: Measure spectral change rate (higher in some vocoders)

### Machine Learning Extensions

1. **Fine-Tuning**: Unfreeze w2v-BERT layers and fine-tune on deepfake detection
2. **Contrastive Learning**: Learn embeddings where same-system samples attract, different-system repel
3. **Few-Shot Learning**: Can we detect a new system with only 5 examples?
4. **Multi-Modal**: Combine audio + text (if transcripts available)

## Troubleshooting

### Feature extraction is slow
- **Expected**: CPU inference on w2v-BERT-2.0 is ~1-2 seconds per audio file
- **Solution**: Use GPU if available (automatic via `torch.cuda.is_available()`)
- **Alternative**: Pre-extract features once, then run experiments multiple times

### Out of disk space
- Model weights: ~1.5 GB
- LibriSpeech test-clean: ~350 MB (uncompressed)
- Virtual environment: ~1.5 GB
- **Total**: ~3.5 GB minimum

### kNN accuracy is low (~33%)
- Check if all systems have samples: `cat features/labels.txt | sort | uniq -c`
- Verify embeddings shape: `python -c "import numpy as np; print(np.load('features/embeddings.npy').shape)"`
- Try different k values: `python attribution.py` runs k-ablation automatically

### MFCC extraction fails
- Ensure `librosa` and `soundfile` are installed
- Check audio files are not corrupted: `file dataset/bonafide/*.flac`

## Citation

If you use this code for your project, please cite the original TADA paper:

```bibtex
@inproceedings{stan_interspeech25,
  author={Stan, Adriana and Combei, David and Oneata, Dan and Cucu, Horia},
  booktitle={Proc. of Interspeech},
  title={{TADA: Training-free Attribution and Out-of-Domain Detection of Audio Deepfake}},
  year={2025}
}
```

## License

This implementation is for educational purposes. See original TADA repository for research use.

## Acknowledgements

- **Original TADA paper**: Adriana Stan et al., Interspeech 2025
- **w2v-BERT-2.0**: Meta AI / Facebook Research
- **LibriSpeech**: OpenSLR
- **Course**: AI553 Speech and Audio Processing

---

**Author**: Implementation for AI553 course project  
**Date**: 2026-10-13  
**Goal**: Demonstrate training-free deepfake attribution using frozen SSL embeddings
