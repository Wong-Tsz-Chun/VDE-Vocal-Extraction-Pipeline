# VDE (Vocal Design Ecosystem)

**Source-Fidelity Vocal Extraction Pipeline for DiffSVC/RVC Training**

VDE is a specialized Python audio processing pipeline designed to prepare "dry" and "clean" vocal datasets for AI voice model training.

---

## 🎧 Demos

[![VDE Demo Comparison](https://img.youtube.com/vi/7ayRvN_Pxzc/0.jpg)](https://www.youtube.com/watch?v=7ayRvN_Pxzc)

*Click above to watch the spectrogram analysis and comparison.*

## The Engineering Gap

Standard separation models prioritize "listening quality" (stereo width, psychoacoustic air, smoothing) over "data purity." While pleasing to the human ear, these features introduce artifacts—such as high-frequency hallucination and reverb tails—that degrade downstream model training (Diff-SVC, RVC).

### Technical Challenges vs. VDE Solutions

| Limitation | Affected Models | The VDE Solution |
|------------|----------------|------------------|
| **Spectral Hallucination** | Demucs v4, AudioSR | **Source Fidelity**: Standard models often generate artificial noise above the source cutoff (>16kHz) to mimic "air." VDE strictly respects source bandwidth to prevent models from learning "metallic" noise. |
| **Volume Pumping** | Mel-Band RoFormer | **Multi-Band Hybrid**: RoFormers can suffer from amplitude fluctuation. VDE chains Mel-Band (for body stability) with BS-Roformer (for high-end transients) to ensure consistent energy. |
| **Reverb & Bleed** | Commercial Tools, MDX-Net | **Dry Priority**: Commercial extractors intentionally leave reverb/chorus for naturalness. VDE uses aggressive, cascaded de-reverberation (FoxJoy) to isolate the dry vocal core. |
| **Silence Artifacts** | Gaudio, Kim Vocal 2 | **Noise Gating**: Mitigates the "silence hallucination" issue where normalization layers amplify the noise floor during empty sections. |

**Best for:** Diff-SVC/RVC dataset preparation, J-Pop/Rock/Ballad genres.

---

## Features

- **Multi-Model Chaining**: Hybrid architecture blending Mel-RoFormer (Lows) and BS-RoFormer (Highs).
- **Aggressive Cleaning**: Dedicated De-chorus (Kara2) and De-reverb (FoxJoy) stages.
- **Transient Restoration**: Preserves consonant attacks often lost during aggressive de-reverberation.
- **Harmonic Exciter**: DSP-based high-frequency enhancement (mathematically derived air, not random noise).
- **Training-Ready Output**: Mono normalization (48kHz, -2dB peak, 32-bit float).
- **Hardware Acceleration**: Dedicated CUDA 12 environment support for ONNX models.

---

## Default Pipeline Architecture

The pipeline uses a "Frankenstein" crossover approach to maximize stability and clarity:

```
1. Pre-Process:      HPF 100Hz (Remove rumble)
2. Hybrid Separation:
   |-- Lows (<10kHz): Mel-Band RoFormer (Stability focus)
   |-- Highs (>10kHz): BS-Roformer (Transient/Air focus)
       -> (Crossover: 1kHz-3kHz gradual blend)
3. De-Chorus:        Kara2
4. De-Reverb:        FoxJoy / Aggressive MDX
5. Post-Process:     Transient Restoration
6. Enhancement:      Harmonic Exciter (DSP-based)
7. Finalize:         Mono Mix -> Normalize -2dB -> 48kHz
```

---

## Output Structure

```
output_dataset/<singer>/
|-- vocal/             Vocal_<song>.wav      (Step 3: Clean, Stereo, Natural Reverb)
|-- dry_vocal/         dry_vocal_<song>.wav  (Step 6: De-reverbed, Excited)
|-- mono_vocal/        mono_vocal_<song>.wav (Step 7: Final Training Data)
|-- temp/              (Intermediate stems)
```

---

## Requirements

- Anaconda/Miniconda (conda in PATH)
- NVIDIA GPU with CUDA support
- FFmpeg installed and in PATH

---

## Quick Start

### 1. Setup Main Environment

Handles logic and PyTorch-based models.

```bash
conda create -n vde_env python=3.10 -y
conda activate vde_env
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Setup CUDA 12 Environment

Handles ONNX Runtime acceleration.

```bash
# Run the provided setup script
setup_cuda12_env.bat
```

### 3. Download Models

Place the following in the `models/` directory:

- `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt`
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
- `Kim_Vocal_2.onnx` (or `BS-Roformer-ViperX-1296.onnx`)
- `UVR_MDXNET_KARA_2.onnx`
- `Reverb_HQ_By_FoxJoy.onnx`
- `UVR-De-Echo-Aggressive.pth`

### 4. Run Pipeline

```bash
# Option A: Download from YouTube list and process
python cleansing.py <singer_name>

# Option B: Process local files (skip download)
python cleansing.py <singer_name> d

# Option C: Manual Mode (Interactive step selection)
python cleansing.py <singer_name> n
```

---

## Configuration

In `process_lib.py`, you can toggle the secondary environment usage:

```python
CUDA12_ENV_NAME = "cuda12_env"
USE_CUDA12_FOR_ONNX = True
```

---

## Troubleshooting

**Issue: ONNX models running on CPU**

1. Ensure `setup_cuda12_env.bat` completed successfully.
2. Verify CUDA visibility in the sub-environment:

```bash
conda activate cuda12_env
python -c "import torch; print(torch.cuda.is_available())"
```

---

## License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this software for personal or commercial purposes (e.g., training commercial voice models), provided that the original copyright notice is included.

**Note:** This software uses third-party models (UVR, RoFormer) which may have their own licenses. Please respect the licenses of the model weights you download.
