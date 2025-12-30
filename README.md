# VDE (Vocal Design Ecosystem)

**Source-Fidelity Vocal Extraction Pipeline for DiffSVC/RVC Training**

A Python-based audio processing pipeline designed to prepare "dry" and "clean" vocal datasets for AI training.

## The Problem

Standard separation models prioritize "listening quality" (stereo width, air, smoothing) over "data purity." This introduces high-frequency noise and reverb that degrade downstream model training.

## The Solution

VDE chains multiple models (RoFormer, MDX-Net, De-Reverb) to aggressively isolate the vocal core. It prioritizes artifact reduction and transient stability over psychoacoustic enhancement.

**Best for:** Diff-SVC/RVC dataset preparation, J-Pop/Rock/Ballad genres.

---

## Features

- **Multi-model vocal separation** (Mel-RoFormer + BS-RoFormer blend)
- **De-chorus & De-reverb** (Kara2, De-Echo-Aggressive)
- **Transient Restore** (consonant attacks recovered from pre-Kara source)
- **Harmonic Exciter** (DSP-based high freq enhancement)
- **Mono normalization** (48kHz, -2dB peak, 32-bit float)
- **CUDA 12 acceleration** for ONNX models

---

## Default Pipeline

```
1. HPF 100Hz
2. Vocal: Mel low + BS mid/high (1kHz-3kHz gradual crossover)
3. Cleanup: (skipped)
4. De-Chorus: Kara2                    → vocal/
5. De-Reverb: Aggressive
6. Transient Restore
7. Harmonic Exciter                    → dry_vocal/
8. Mono & Normalize                    → mono_vocal/
```

---

## Output Structure

```
output_dataset/<singer>/
├── vocal/          Vocal_<song>.wav         (Step 4: dechorus)
├── dry_vocal/      dry vocal_<song>.wav     (Step 7: excited)
├── mono_vocal/     mono vocal_<song>.wav    (Step 8: mono normalized)
└── temp/           (intermediate files)
```

---

## Requirements

- **Anaconda/Miniconda** (conda in PATH)
- **NVIDIA GPU** with CUDA support
- **FFmpeg** installed and in PATH

---

## Quick Start

### 1. Setup Main Environment

```bash
conda create -n vde_env python=3.10 -y
conda activate vde_env
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Setup CUDA 12 Environment

```bash
setup_cuda12_env.bat
```

### 3. Download Models

Place in `models/` directory:
- `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt`
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
- `Kim_Vocal_2.onnx`
- `UVR_MDXNET_KARA_2.onnx`
- `UVR-De-Echo-Aggressive.pth`

### 4. Run Pipeline

```bash
# Add YouTube URLs to urls.txt, then:
python cleansing.py <singer_name>        # Download + process
python cleansing.py <singer_name> d      # Default pipeline (skip download)
python cleansing.py <singer_name> n      # Manual mode
```

---

## Configuration

In `process_lib.py`:
```python
CUDA12_ENV_NAME = "cuda12_env"
USE_CUDA12_FOR_ONNX = True
```

---

## Troubleshooting

**ONNX models running on CPU:**
1. Run `setup_cuda12_env.bat`
2. Verify: `conda activate cuda12_env && python -c "import torch; print(torch.cuda.is_available())"`
