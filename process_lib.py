"""
process_lib.py - Audio Processing Library
Contains all processing functions for vocal separation pipeline.
Includes standard separation, cleanup, and new EQ/Dynamics processors.
"""

import os
import gc
import torch
import soundfile as sf
import numpy as np
import librosa
import scipy.signal as signal
import warnings
from pathlib import Path
from audio_separator.separator import Separator
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio
import subprocess
import shutil

warnings.filterwarnings("ignore")

# --- Configuration ---
BASE_DIR = Path(__file__).parent.absolute()
MODELS_DIR = BASE_DIR / "models"
DEBUG_OUTPUT = BASE_DIR / "debug_output"

# CUDA 12 Environment for ONNX models
# Uses `conda run -n ENV_NAME` instead of hardcoded paths for portability
CUDA12_ENV_NAME = "cuda12_env"  # Name of the conda environment with CUDA 12 + onnxruntime-gpu
CUDA_CLI_SCRIPT = BASE_DIR / "cuda_CLI.py"
USE_CUDA12_FOR_ONNX = True  # Set to False to use main env for ONNX models

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DEBUG_OUTPUT.mkdir(exist_ok=True)

# --- Model Definitions ---
VOCAL_SEP_MODELS = {
    1: {"name": "MelBand_RoFormer", "file": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"},
    2: {"name": "BS_RoFormer", "file": "model_bs_roformer_ep_317_sdr_12.9755.ckpt"},
    3: {"name": "Kim_Vocal_2", "file": "Kim_Vocal_2.onnx"},
    4: {"name": "MelBand_Karaoke", "file": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"},
}

CLEANUP_MODELS = {
    1: {"name": "MDX_Inst_HQ_3 (Inst Removal)", "type": "inst", "id": 1, "file": "UVR-MDX-NET-Inst_HQ_3.onnx"},
    2: {"name": "htdemucs (Demucs)", "type": "demucs", "id": 1},
    3: {"name": "htdemucs_ft (Demucs)", "type": "demucs", "id": 2},
}

DEMUCS_MODELS = {
    1: {"name": "htdemucs", "model_id": "htdemucs"},
    2: {"name": "htdemucs_ft", "model_id": "htdemucs_ft"},
}

INST_REMOVAL_MODELS = {
    1: {"name": "MDX_Inst_HQ_3", "file": "UVR-MDX-NET-Inst_HQ_3.onnx"},
}

DECHORUS_MODELS = {
    1: {"name": "MDXNET_KARA_2", "file": "UVR_MDXNET_KARA_2.onnx"},
}

DEREVERB_MODELS = {
    1: {"name": "De-Echo-Aggressive", "file": "UVR-De-Echo-Aggressive.pth", "aggression": 10},
    2: {"name": "Reverb_HQ_FoxJoy", "file": "Reverb_HQ_By_FoxJoy.onnx", "aggression": None}, 
    3: {"name": "De-Echo-Normal", "file": "UVR-De-Echo-Normal.pth", "aggression": 5},
}


def clean_vram():
    """Clean VRAM after processing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _find_conda_executable():
    """Find conda executable in common locations."""
    import os
    import shutil
    
    # First try PATH
    conda_path = shutil.which("conda")
    if conda_path:
        return conda_path
    
    # Common conda locations on Windows
    home = os.path.expanduser("~")
    common_paths = [
        os.path.join(home, "anaconda3", "Scripts", "conda.exe"),
        os.path.join(home, "miniconda3", "Scripts", "conda.exe"),
        os.path.join(home, "Anaconda3", "Scripts", "conda.exe"),
        os.path.join(home, "Miniconda3", "Scripts", "conda.exe"),
        "C:/ProgramData/anaconda3/Scripts/conda.exe",
        "C:/ProgramData/miniconda3/Scripts/conda.exe",
        "C:/anaconda3/Scripts/conda.exe",
        "C:/miniconda3/Scripts/conda.exe",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def _check_conda_env_exists(env_name):
    """Check if a conda environment exists."""
    conda_exe = _find_conda_executable()
    if not conda_exe:
        return False
    
    try:
        result = subprocess.run(
            [conda_exe, "env", "list"],
            capture_output=True, text=True, check=True
        )
        return env_name in result.stdout
    except Exception:
        return False


def run_onnx_cuda12(model_file, input_path, output_dir, stem=None):
    """
    Run ONNX model via cuda12_env subprocess using `conda run`.
    Falls back to local execution if cuda12_env is not available.
    
    Args:
        model_file: ONNX model filename (e.g., 'UVR_MDXNET_KARA_2.onnx')
        input_path: Input audio file path
        output_dir: Output directory
        stem: Output stem name (e.g., 'Vocals', 'No Reverb')
    
    Returns:
        Path to output file
    """
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find conda executable
    conda_exe = _find_conda_executable()
    
    # Check if CUDA 12 env is available and enabled
    if USE_CUDA12_FOR_ONNX and CUDA_CLI_SCRIPT.exists() and conda_exe and _check_conda_env_exists(CUDA12_ENV_NAME):
        print(f"   -> Using CUDA 12 env for {model_file}")
        
        # Build command using conda run
        cmd = [
            conda_exe, "run", "-n", CUDA12_ENV_NAME, "--no-capture-output",
            "python", str(CUDA_CLI_SCRIPT),
            "--model", model_file,
            "--input", input_path,
            "--output_dir", str(output_dir),
        ]
        
        if stem:
            cmd.extend(["--stem", stem])
        
        try:
            # Don't capture output so user sees progress
            result = subprocess.run(cmd, check=True)
            
            # Find the output file
            import glob
            pattern = str(output_dir / f"*{Path(model_file).stem}*")
            matches = glob.glob(pattern)
            if matches:
                return matches[-1]  # Most recent
            else:
                # Fallback: look for any wav file
                wav_files = list(output_dir.glob("*.wav"))
                if wav_files:
                    return str(wav_files[-1])
                raise Exception("No output file found")
        except subprocess.CalledProcessError as e:
            print(f"   [WARN] CUDA12 subprocess failed: {e}")
            print(f"   -> Falling back to local execution...")
    
    # Fallback: run locally in current env
    print(f"   -> Using local env for {model_file} (CPU fallback)")
    separator = Separator(
        log_level=20,
        model_file_dir=str(MODELS_DIR),
        output_dir=str(output_dir),
        output_single_stem=stem,
        mdx_params={
            "hop_length": 1024,
            "segment_size": 256,
            "overlap": 0.75,
            "batch_size": 1,
            "enable_denoise": True
        }
    )
    separator.load_model(model_file)
    out = separator.separate(input_path)
    if isinstance(out, str): out = [out]
    return str(output_dir / out[0])


def highpass_filter(audio, sr, cutoff=80, order=5):
    """80Hz Lowcut Filter"""
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    if normalized_cutoff >= 1: return audio
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    if audio.ndim == 1:
        return signal.filtfilt(b, a, audio)
    else:
        return np.array([signal.filtfilt(b, a, ch) for ch in audio])


def high_shelf_eq(audio, sr, cutoff=8000, gain_db=2.0):
    """High Shelf EQ (+2dB >8kHz) to brighten sound"""
    # Simple IIR implementation
    # Based on RBJ Audio-EQ-Cookbook
    w0 = 2 * np.pi * cutoff / sr
    amp = 10 ** (gain_db / 40.0)
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / 2 * np.sqrt((amp + 1/amp) * (1/0.707 - 1) + 2)
    
    # Coefficients for High Shelf
    b0 = amp * ((amp + 1) + (amp - 1) * cos_w0 + 2 * np.sqrt(amp) * alpha)
    b1 = -2 * amp * ((amp - 1) + (amp + 1) * cos_w0)
    b2 = amp * ((amp + 1) + (amp - 1) * cos_w0 - 2 * np.sqrt(amp) * alpha)
    a0 = (amp + 1) - (amp - 1) * cos_w0 + 2 * np.sqrt(amp) * alpha
    a1 = 2 * ((amp - 1) - (amp + 1) * cos_w0)
    a2 = (amp + 1) - (amp - 1) * cos_w0 - 2 * np.sqrt(amp) * alpha
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    if audio.ndim == 1:
        return signal.filtfilt(b, a, audio)
    else:
        return np.array([signal.filtfilt(b, a, ch) for ch in audio])


def upward_compressor(audio, sr, threshold=-20, ratio=2.0):
    """
    Simple Upward Compressor to boost quiet parts.
    Boosts signals below threshold.
    """
    # Convert threshold to linear
    thresh_lin = 10 ** (threshold / 20.0)
    
    # Envelope follower
    if audio.ndim > 1:
        # Use mixture for envelope
        env = np.mean(np.abs(audio), axis=0)
    else:
        env = np.abs(audio)
        
    # Apply compression curve
    # gain = (thresh / env) ^ (1 - 1/ratio) for env < thresh
    # gain = 1 for env >= thresh
    
    epsilon = 1e-6
    gain_mask = env < thresh_lin
    
    # Calculate gain factor
    # Protect against div by zero with epsilon
    safe_env = np.maximum(env, epsilon)
    
    gain = np.ones_like(env)
    gain[gain_mask] = (thresh_lin / safe_env[gain_mask]) ** (1.0 - 1.0/ratio)
    
    # Cap gain to avoid massive noise boost (e.g. max +6dB)
    max_gain = 2.0 
    gain = np.minimum(gain, max_gain)
    
    # Smooth gain
    b, a = signal.butter(1, 0.01) # Simple smoothing
    gain_smooth = signal.filtfilt(b, a, gain)
    
    if audio.ndim > 1:
        return audio * gain_smooth
    else:
        return audio * gain_smooth


def _save_safe_wav(audio_data, path, sample_rate):
    """Save audio with normalization"""
    if torch.is_tensor(audio_data):
        audio_data = audio_data.cpu().numpy()
    if audio_data.ndim > 1 and audio_data.shape[0] < audio_data.shape[1]:
        audio_data = audio_data.T

    max_val = np.abs(audio_data).max()
    if max_val > 0:
        audio_data = audio_data / max_val * 0.95
    sf.write(path, audio_data, sample_rate, subtype='PCM_16')


# =============================================================================
# Helper: Pipeline Steps
# =============================================================================

def vocal_separation(input_path, output_dir, model_id=1, add_leak=True, leak_percent=0.02):
    """Step 1: Vocal Separation - Uses CUDA 12 env for ONNX models"""
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_info = VOCAL_SEP_MODELS.get(model_id, VOCAL_SEP_MODELS[1])
    basename = Path(input_path).stem
    
    print(f"[Step 1] Vocal Separation: {model_info['name']}")
    
    try:
        if model_info["file"].endswith(".onnx"):
            # ONNX Model (e.g., Kim_Vocal_2) - use CUDA 12 env
            output_path = run_onnx_cuda12(
                model_file=model_info["file"],
                input_path=input_path,
                output_dir=output_dir,
                stem=None  # Don't use single_stem to get both stems
            )
            vocal_path = Path(output_path)
            # Find vocals file if multiple outputs
            if not ("Vocals" in str(vocal_path) or "vocals" in str(vocal_path)):
                # Check for vocals file in output dir
                vocal_files = list(output_dir.glob(f"*Vocals*"))
                if vocal_files:
                    vocal_path = vocal_files[0]
        else:
            # RoFormer models (.ckpt) - run locally
            sep_params = {
                "output_dir": str(output_dir),
                "output_format": "WAV",
                "normalization_threshold": 0.9,
                "model_file_dir": str(MODELS_DIR),
                "log_level": 20,
                "mdxc_params": {
                    "segment_size": 256,
                    "overlap": 8,
                    "batch_size": 1,
                }
            }
            
            separator = Separator(**sep_params)
            separator.load_model(model_info["file"])
            output_files = separator.separate(input_path)
            
            if not output_files: raise Exception("No output from separator")
            if isinstance(output_files, str): output_files = [output_files]
            vocal_file = next((f for f in output_files if "Vocals" in f or "vocals" in f), output_files[0])
            vocal_path = output_dir / vocal_file

        # Load & Filter
        vocal_audio, sr = librosa.load(str(vocal_path), sr=None, mono=False)
        if vocal_audio.ndim == 1: vocal_audio = vocal_audio[np.newaxis, :]
        
        print("   -> Applying 80Hz lowcut filter...")
        vocal_filtered = highpass_filter(vocal_audio, sr, cutoff=80)
        
        final_audio = vocal_filtered
        
        final_path = output_dir / f"{basename}_01_{model_info['name']}.wav"
        _save_safe_wav(final_audio, str(final_path), sr)
        
        if vocal_path.exists() and vocal_path != final_path:
            vocal_path.unlink()
            
        print(f"   -> Saved: {final_path.name}")
        return str(final_path)
        
    finally:
        clean_vram()


def demucs_cleanup(input_path, output_dir, model_id=1, soft_mix=True, mix_percent=0.10):
    """Step 2: Demucs Cleanup"""
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_info = DEMUCS_MODELS.get(model_id, DEMUCS_MODELS[1])
    basename = Path(input_path).stem
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Step 2] Demucs Cleanup: {model_info['name']}")
    
    try:
        demucs_model = get_model(model_info["model_id"])
        demucs_model.to(device)
        
        wav, sr = sf.read(input_path)
        wav = torch.from_numpy(wav).float()
        if wav.ndim == 1: wav = wav.unsqueeze(0).expand(2, -1)
        elif wav.shape[0] > wav.shape[1]: wav = wav.t()
        wav = wav.unsqueeze(0).to(device)
        
        if sr != demucs_model.samplerate:
            wav = convert_audio(wav, sr, demucs_model.samplerate, demucs_model.audio_channels)
            
        with torch.no_grad():
            sources = apply_model(demucs_model, wav, device=device, shifts=2)
            
        vocal_wav = sources[0, demucs_model.sources.index('vocals')]
        vocal_wav_np = vocal_wav.cpu().numpy()
        
        if soft_mix:
            # simple input mix
            input_wav_np = wav.squeeze(0).cpu().numpy()
            vocal_wav_np = (vocal_wav_np * (1 - mix_percent)) + (input_wav_np * mix_percent)

        final_path = output_dir / f"{basename}_02_demucs_{model_info['name']}.wav"
        _save_safe_wav(vocal_wav_np, str(final_path), demucs_model.samplerate)
        
        print(f"   -> Saved: {final_path.name}")
        return str(final_path)
    finally:
        clean_vram()


def inst_removal(input_path, output_dir, model_id=1):
    """Step 2 Alt: Inst Removal (MDX) - Uses CUDA 12 env"""
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_info = INST_REMOVAL_MODELS.get(model_id, INST_REMOVAL_MODELS[1])
    basename = Path(input_path).stem
    
    print(f"[Step 2] Inst Removal: {model_info['name']}")
    
    try:
        # Use CUDA 12 environment for ONNX model
        output_path = run_onnx_cuda12(
            model_file=model_info["file"],
            input_path=input_path,
            output_dir=output_dir,
            stem="Vocals"
        )
        
        # Rename to standard naming
        temp_path = Path(output_path)
        final_path = output_dir / f"{basename}_02_inst_clean.wav"
        
        if final_path.exists(): final_path.unlink()
        if temp_path.exists():
            temp_path.rename(final_path)
        
        print(f"   -> Saved: {final_path.name}")
        return str(final_path)
    finally:
        clean_vram()


def dechorus(input_path, output_dir, model_id=1):
    """Step 3: De-Chorus - Uses CUDA 12 env for ONNX"""
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_info = DECHORUS_MODELS.get(model_id, DECHORUS_MODELS[1])
    basename = Path(input_path).stem
    
    print(f"[Step 3] De-Chorus: {model_info['name']}")
    try:
        # Use CUDA 12 environment for ONNX model
        output_path = run_onnx_cuda12(
            model_file=model_info["file"],
            input_path=input_path,
            output_dir=output_dir,
            stem="Vocals"
        )
        
        # Rename to standard naming
        src = Path(output_path)
        dst = output_dir / f"{basename}_03_dechorus.wav"
        if dst.exists(): dst.unlink()
        if src.exists():
            src.rename(dst)
        print(f"   -> Saved: {dst.name}")
        return str(dst)
    finally:
        clean_vram()


def dereverb(input_path, output_dir, model_id=1):
    """Step 4: De-Reverb - Uses CUDA 12 env for ONNX models"""
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_info = DEREVERB_MODELS.get(model_id, DEREVERB_MODELS[1])
    basename = Path(input_path).stem
    
    print(f"[Step 4] De-Reverb: {model_info['name']}")
    try:
        if model_info["file"].endswith(".pth"):
            # VR Model - run locally (not ONNX)
            aggression = model_info.get("aggression", 10)
            separator = Separator(
                log_level=20,
                model_file_dir=str(MODELS_DIR),
                output_dir=str(output_dir),
                output_single_stem="No Echo",
                vr_params={"aggression": aggression, "window_size": 512}
            )
            separator.load_model(model_info["file"])
            out = separator.separate(input_path)
            if isinstance(out, str): out = [out]
            src = output_dir / out[0]
        else:
            # ONNX Model (e.g., FoxJoy) - use CUDA 12 env
            output_path = run_onnx_cuda12(
                model_file=model_info["file"],
                input_path=input_path,
                output_dir=output_dir,
                stem="No Reverb"
            )
            src = Path(output_path)
        
        # Rename to standard naming
        dst = output_dir / f"{basename}_04_dereverb.wav"
        if dst.exists(): dst.unlink()
        if src.exists():
            src.rename(dst)
        
        print(f"   -> Saved: {dst.name}")
        return str(dst)
    finally:
        clean_vram()


def mono_normalize(input_path, output_dir, apply_high_shelf=False, to_mono=True, verbose=False):
    """
    Mono normalize audio for DiffSVC training.
    - Resamples to 48kHz, converts to mono, peak normalizes to -2dB
    
    Args:
        input_path: Input audio file
        output_dir: Output directory
        apply_high_shelf: Apply high shelf EQ (+2dB >8kHz)
        to_mono: Convert to mono (if False, copies as-is)
        verbose: Print progress messages
    """
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    basename = Path(input_path).stem
    
    final_path = output_dir / f"{basename}_05_final.wav"
    
    # Stereo mode: just copy file as-is
    if not to_mono:
        if verbose:
            print(f"   -> Stereo output (no processing)")
        shutil.copy(input_path, final_path)
        return str(final_path)
    
    # Mono mode: full processing
    TARGET_SR = 48000
    TARGET_PEAK_DB = -2.0
    
    # Load audio
    data, sr = sf.read(input_path)
    
    # Convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        if verbose:
            print(f"   -> Converted to mono")
    
    # Resample to 48kHz if needed
    if sr != TARGET_SR:
        if verbose:
            print(f"   -> Resampling {sr}Hz -> {TARGET_SR}Hz...")
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    
    # Optional High Shelf EQ
    if apply_high_shelf:
        if verbose:
            print("   -> Applying High Shelf EQ (+2dB >8kHz)")
        data = high_shelf_eq(data, sr, cutoff=8000, gain_db=2.0)
    
    # Peak normalize to -2dB
    current_peak = np.max(np.abs(data))
    if current_peak > 0:
        target_peak = 10 ** (TARGET_PEAK_DB / 20.0)
        data = data * (target_peak / current_peak)
        if verbose:
            print(f"   -> Peak normalized to {TARGET_PEAK_DB}dB")
    
    # Save as 32-bit float
    sf.write(str(final_path), data, sr, subtype='FLOAT')
    if verbose:
        print(f"   -> Saved: {final_path.name}")
    return str(final_path)


# =============================================================================
# Dispatch & CLI
# =============================================================================


def add_pink_noise(audio, level_db=-75):
    """
    Add Pink Noise at specified dB level to avoid 'Vacuum' silence.
    audio: np.array (channels, samples) or (samples,)
    """
    if audio.ndim == 1:
        samples = len(audio)
        channels = 1
    else:
        channels, samples = audio.shape

    # Pink Noise Generation
    white = np.random.randn(channels, samples) if channels > 1 else np.random.randn(samples)
    
    if channels > 1:
        noise = []
        for ch in range(channels):
            X_white = np.fft.rfft(white[ch])
            S = np.sqrt(np.arange(len(X_white)) + 1.)
            X_pink = X_white / S
            noise.append(np.fft.irfft(X_pink))
        pink = np.array(noise)
    else:
        X_white = np.fft.rfft(white)
        S = np.sqrt(np.arange(len(X_white)) + 1.)
        X_pink = X_white / S
        pink = np.fft.irfft(X_pink)
        
    if pink.shape[-1] > samples: pink = pink[..., :samples]
    if pink.shape[-1] < samples: 
        pad = np.zeros((channels, samples - pink.shape[-1])) if channels > 1 else np.zeros(samples - pink.shape[-1])
        pink = np.concatenate([pink, pad], axis=-1)

    pink = pink / np.max(np.abs(pink))
    target_amp = 10 ** (level_db / 20.0)
    pink = pink * target_amp
    
    return audio + pink

def weighted_average(path_a, path_b, weight_a, output_path):
    """
    Weighted average of two audio files.
    """
    print(f"[Mix] Averaging: {Path(path_a).name} ({weight_a}) + {Path(path_b).name} ({1.0-weight_a:.1f})")
    y_a, sr_a = librosa.load(str(path_a), sr=None, mono=False)
    y_b, sr_b = librosa.load(str(path_b), sr=None, mono=False)
    
    target_sr = max(sr_a, sr_b)
    if sr_a != target_sr: y_a = librosa.resample(y_a, orig_sr=sr_a, target_sr=target_sr)
    if sr_b != target_sr: y_b = librosa.resample(y_b, orig_sr=sr_b, target_sr=target_sr)
    
    min_len = min(y_a.shape[-1], y_b.shape[-1])
    y_a = y_a[..., :min_len]
    y_b = y_b[..., :min_len]
    
    weight_b = 1.0 - weight_a
    y_avg = (y_a * weight_a) + (y_b * weight_b)
    
    if output_path:
        _save_safe_wav(y_avg, str(output_path), target_sr)
    
    return str(output_path)


def spectral_blend(path_body, path_high, output_path, crossover_hz=8000, verbose=False):
    """
    Spectral blend: Use body source for full range, add highpassed highs from second source.
    
    Args:
        path_body: Path to body audio - used for low/mid
        path_high: Path to high freq audio - highpassed and added
        output_path: Output file path
        crossover_hz: Crossover frequency (default 8000Hz)
        verbose: Print progress messages
    """
    from scipy.signal import butter, sosfilt
    
    # Load both files
    y_body, sr_body = librosa.load(str(path_body), sr=None, mono=False)
    y_high, sr_high = librosa.load(str(path_high), sr=None, mono=False)
    
    # Ensure same sample rate
    target_sr = max(sr_body, sr_high)
    if sr_body != target_sr:
        y_body = librosa.resample(y_body, orig_sr=sr_body, target_sr=target_sr)
    if sr_high != target_sr:
        y_high = librosa.resample(y_high, orig_sr=sr_high, target_sr=target_sr)
    
    # Ensure same length
    min_len = min(y_body.shape[-1], y_high.shape[-1])
    y_body = y_body[..., :min_len]
    y_high = y_high[..., :min_len]
    
    # Handle mono/stereo dimensions
    if y_body.ndim == 1:
        y_body = y_body[np.newaxis, :]
    if y_high.ndim == 1:
        y_high = y_high[np.newaxis, :]
    
    # Filter bands
    nyquist = target_sr / 2
    normalized_cutoff = min(crossover_hz / nyquist, 0.99)
    sos_low = butter(8, normalized_cutoff, btype='low', output='sos')
    y_body_low = sosfilt(sos_low, y_body, axis=-1)
    
    sos_high = butter(8, normalized_cutoff, btype='high', output='sos')
    y_high_hp = sosfilt(sos_high, y_high, axis=-1)
    
    # Combine
    y_blended = y_body_low + y_high_hp
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(y_blended))
    if peak > 0.95:
        y_blended = y_blended * (0.95 / peak)
    
    # Save output
    _save_safe_wav(y_blended, str(output_path), target_sr)
    
    return str(output_path)


def spectral_blend_3band(path_low, path_mid_high, output_path, crossover_hz=300, blend_width=1000, verbose=False):
    """
    3-band spectral blend with gradual crossover transition.
    
    Args:
        path_low: Path to low frequency source (e.g., Mel-RoFormer)
        path_mid_high: Path to mid/high source (e.g., BS-RoFormer)
        output_path: Output file path
        crossover_hz: Center of crossover (default 300Hz)
        blend_width: Width of gradual transition in Hz (default 1000Hz)
        verbose: Print progress messages
    
    Returns:
        Path to output file
    """
    low_start = max(crossover_hz - blend_width // 2, 50)
    high_end = crossover_hz + blend_width // 2
    
    if verbose:
        print(f"   Blend: {low_start}Hz → gradual → {high_end}Hz")
    
    # Load both files
    y_low, sr_low = librosa.load(str(path_low), sr=None, mono=False)
    y_mid, sr_mid = librosa.load(str(path_mid_high), sr=None, mono=False)
    
    # Ensure same sample rate
    target_sr = max(sr_low, sr_mid)
    if sr_low != target_sr:
        y_low = librosa.resample(y_low, orig_sr=sr_low, target_sr=target_sr)
    if sr_mid != target_sr:
        y_mid = librosa.resample(y_mid, orig_sr=sr_mid, target_sr=target_sr)
    
    # Ensure same length
    min_len = min(y_low.shape[-1], y_mid.shape[-1])
    y_low = y_low[..., :min_len]
    y_mid = y_mid[..., :min_len]
    
    # Handle mono/stereo dimensions
    if y_low.ndim == 1:
        y_low = y_low[np.newaxis, :]
    if y_mid.ndim == 1:
        y_mid = y_mid[np.newaxis, :]
    
    # FFT-based blending with gradual crossover
    n_fft = 4096
    hop_length = 1024
    
    y_blended = np.zeros_like(y_low)
    
    for ch in range(y_low.shape[0]):
        # STFT
        D_low = librosa.stft(y_low[ch], n_fft=n_fft, hop_length=hop_length)
        D_mid = librosa.stft(y_mid[ch], n_fft=n_fft, hop_length=hop_length)
        
        # Create frequency-dependent blend mask
        freqs = librosa.fft_frequencies(sr=target_sr, n_fft=n_fft)
        blend_mask = np.zeros(len(freqs))
        
        for i, f in enumerate(freqs):
            if f <= low_start:
                blend_mask[i] = 0.0  # 100% low source
            elif f >= high_end:
                blend_mask[i] = 1.0  # 100% mid/high source
            else:
                # Gradual blend (cosine interpolation for smoothness)
                t = (f - low_start) / (high_end - low_start)
                blend_mask[i] = 0.5 * (1 - np.cos(np.pi * t))  # Smooth S-curve
        
        # Apply blend: low * (1-mask) + mid * mask
        blend_mask_2d = blend_mask[:, np.newaxis]
        D_blended = D_low * (1 - blend_mask_2d) + D_mid * blend_mask_2d
        
        # Inverse STFT
        y_blended[ch] = librosa.istft(D_blended, hop_length=hop_length, length=min_len)
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(y_blended))
    if peak > 0.95:
        y_blended = y_blended * (0.95 / peak)
    
    # Save output
    _save_safe_wav(y_blended, str(output_path), target_sr)
    
    return str(output_path)


def harmonic_exciter(input_path, output_path, drive=0.3, high_shelf_db=3.0, cutoff_hz=6000, verbose=False):
    """
    Harmonic Exciter: Generate natural harmonics using soft saturation.
    
    Args:
        input_path: Input audio file
        output_path: Output file path
        drive: Saturation amount (0.0-1.0, default 0.3 = subtle)
        high_shelf_db: High shelf boost in dB (default 3.0)
        cutoff_hz: Frequency above which to apply exciter (default 6000Hz)
        verbose: Print progress messages
    """
    from scipy.signal import butter, sosfilt
    
    # Load audio
    y, sr = librosa.load(str(input_path), sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    
    # Extract high frequencies only for saturation
    nyquist = sr / 2
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    sos_high = butter(4, normalized_cutoff, btype='high', output='sos')
    y_highs = sosfilt(sos_high, y, axis=-1)
    
    # Apply soft saturation to generate harmonics (tanh saturation)
    y_saturated = np.tanh(y_highs * (1 + drive * 5)) / (1 + drive * 3)
    
    # Blend back: original highs + subtle saturated harmonics
    y_excited = y_highs + (y_saturated - y_highs) * drive
    
    # Apply high-shelf boost
    shelf_gain = 10 ** (high_shelf_db / 20)
    y_excited = y_excited * shelf_gain
    
    # Lowpass the original for the body
    sos_low = butter(4, normalized_cutoff, btype='low', output='sos')
    y_body = sosfilt(sos_low, y, axis=-1)
    
    # Combine body + excited highs
    y_output = y_body + y_excited
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(y_output))
    if peak > 0.95:
        y_output = y_output * (0.95 / peak)
    
    # Save output
    _save_safe_wav(y_output, str(output_path), sr)
    
    return str(output_path)
    
    return str(output_path)


def transient_restore(processed_path, source_path, output_path, attack_ms=30, highpass_hz=4000, blend_amount=0.2, verbose=False, save_debug=True):
    """
    Restore consonant transients from source (pre-Kara) audio.
    
    Args:
        processed_path: Path to processed audio (post-Kara/dereverb)
        source_path: Path to source audio (pre-Kara, has original transients)
        output_path: Output file path
        attack_ms: Attack window duration in ms (default 30ms)
        highpass_hz: Only restore frequencies above this (default 4000Hz)
        blend_amount: How much transient to blend in (default 0.2 = 20%)
        verbose: Print progress messages
        save_debug: Save extracted transients to debug file
    """
    from scipy.signal import butter, sosfilt
    
    # Load both files
    y_proc, sr_proc = librosa.load(str(processed_path), sr=None, mono=False)
    y_src, sr_src = librosa.load(str(source_path), sr=None, mono=False)
    
    # Ensure same sample rate
    target_sr = sr_proc
    if sr_src != target_sr:
        y_src = librosa.resample(y_src, orig_sr=sr_src, target_sr=target_sr)
    
    # Ensure same length
    min_len = min(y_proc.shape[-1], y_src.shape[-1])
    y_proc = y_proc[..., :min_len]
    y_src = y_src[..., :min_len]
    
    # Handle mono/stereo
    if y_proc.ndim == 1:
        y_proc = y_proc[np.newaxis, :]
    if y_src.ndim == 1:
        y_src = y_src[np.newaxis, :]
    
    # Highpass the source to get only consonant frequencies
    nyquist = target_sr / 2
    normalized_cutoff = min(highpass_hz / nyquist, 0.99)
    sos_high = butter(4, normalized_cutoff, btype='high', output='sos')
    y_src_hp = sosfilt(sos_high, y_src, axis=-1)
    
    # Detect onsets (transients) using mono version
    y_mono = np.mean(y_src, axis=0)
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=target_sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=target_sr)
    onset_samples = librosa.frames_to_samples(onset_frames)
    
    if verbose:
        print(f"   Detected {len(onset_samples)} transients")
    
    # Create transient mask - gate that only opens around attacks
    attack_samples = int(attack_ms * target_sr / 1000)
    transient_mask = np.zeros(min_len)
    
    for onset in onset_samples:
        start = max(0, onset - attack_samples // 4)  # Slight pre-attack
        end = min(min_len, onset + attack_samples)
        window_len = end - start
        if window_len > 0:
            envelope = np.hanning(window_len * 2)[:window_len]
            transient_mask[start:end] = np.maximum(transient_mask[start:end], envelope)
    
    # Apply mask to highpassed source transients
    y_transients = np.zeros_like(y_src_hp)
    for ch in range(y_src_hp.shape[0]):
        y_transients[ch] = y_src_hp[ch] * transient_mask
    
    # Voice Activity Gate - only blend where processed audio has content
    y_proc_mono = np.mean(np.abs(y_proc), axis=0)
    window_size = int(0.02 * target_sr)  # 20ms window
    voice_envelope = np.convolve(y_proc_mono, np.ones(window_size)/window_size, mode='same')
    
    # Gate threshold: only blend where vocal energy is above threshold
    voice_threshold = np.max(voice_envelope) * 0.02
    voice_gate = (voice_envelope > voice_threshold).astype(float)
    
    # Smooth the gate to avoid clicks
    gate_smooth = np.convolve(voice_gate, np.hanning(window_size), mode='same')
    gate_smooth = np.clip(gate_smooth, 0, 1)
    
    # Apply voice gate to transients
    for ch in range(y_transients.shape[0]):
        y_transients[ch] = y_transients[ch] * gate_smooth
    
    if verbose:
        voice_pct = np.sum(voice_gate > 0.5) / len(voice_gate) * 100
        print(f"   Voice gate: {voice_pct:.1f}% active")
    
    # Save extracted transients to debug file
    if save_debug:
        debug_transients_path = str(output_path).replace('.wav', '_EXTRACTED.wav')
        _save_safe_wav(y_transients, debug_transients_path, target_sr)
    
    # Blend: processed + scaled transients
    y_output = y_proc + y_transients * blend_amount
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(y_output))
    if peak > 0.95:
        y_output = y_output * (0.95 / peak)
    
    # Save output
    _save_safe_wav(y_output, str(output_path), target_sr)
    
    return str(output_path)


def audiosr_upscale(input_path, output_dir, blend_alpha=0.3):
    """
    Step X: AudioSR Upscale (via subprocess wrapper in audiosr_env).
    """
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    basename = Path(input_path).stem
    
    print(f"[Enhance] AudioSR (Blend {blend_alpha*100:.0f}%)")
    
    temp_sr_path = output_dir / f"{basename}_audiosr_raw.wav"
    
    # Use ../audio_recovery/sr_recover.py
    wrapper_script = BASE_DIR.parent / "audio_recovery" / "sr_recover.py"
    
    if not wrapper_script.exists():
        print(f"[ERROR] sr_recover.py not found at {wrapper_script}")
        return input_path
        
    # Direct path to audiosr_env Python (faster, no conda overhead)
    audiosr_python = Path("C:/Users/asus/anaconda3/envs/audiosr_env/python.exe")
    
    if not audiosr_python.exists():
        print(f"[ERROR] audiosr_env python not found at {audiosr_python}")
        return input_path
        
    cmd = [
        str(audiosr_python),
        str(wrapper_script),
        str(input_path),
        str(temp_sr_path),
        "--guidance", "3.5"
    ]
    
    print(f"   -> Running external process (audiosr_env)...")
    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except Exception as e:
        print(f"[ERROR] AudioSR failed: {e}")
        return input_path
        
    if not temp_sr_path.exists():
        print("[ERROR] AudioSR did not produce output.")
        return input_path
        
    print(f"   -> Blending with original...")
    final_path = output_dir / f"{basename}_audiosr_blended.wav"
    weighted_average(input_path, temp_sr_path, 1.0 - blend_alpha, final_path)
    
    return str(final_path)

def step2_unified_cleanup(input_path, output_dir, model_id=1):
    """Dispatch to correct cleanup function"""
    model_info = CLEANUP_MODELS.get(model_id, CLEANUP_MODELS[1])
    if model_info["type"] == "inst":
        return inst_removal(input_path, output_dir, model_id=model_info["id"])
    else:
        return demucs_cleanup(input_path, output_dir, model_id=model_info["id"])

def get_step_options():
    return {
        1: {"name": "Vocal Separation", "models": VOCAL_SEP_MODELS, "default": 1, "func": vocal_separation},
        2: {"name": "Cleanup", "models": CLEANUP_MODELS, "default": 1, "func": step2_unified_cleanup},
        3: {"name": "De-Chorus", "models": DECHORUS_MODELS, "default": 1, "func": dechorus},
        4: {"name": "De-Reverb", "models": DEREVERB_MODELS, "default": 1, "func": dereverb},
        5: {"name": "Mono/Norm", "models": None, "default": None, "func": mono_normalize},
    }

def save_spectrogram(audio_path, output_path, title):
    import librosa.display
    import matplotlib.pyplot as plt
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        plt.figure(figsize=(12, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] Sepctrogram error: {e}")

if __name__ == "__main__":
    print("This is a library file.")
