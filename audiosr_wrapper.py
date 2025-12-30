import argparse
import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

# Try importing AudioSR, provide helpful error if missing
try:
    from audiosr import build_model, super_resolution
except ImportError:
    print("[ERROR] 'audiosr' not found. Please install it in this environment:")
    print("pip install audiosr")
    sys.exit(1)

def save_audio(waveform, path, sr=48000):
    """Save audio waveform to file."""
    # AudioSR returns float array approx [-1, 1]
    # Squeeze extra dimensions if present
    if waveform.ndim > 1 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    
    sf.write(str(path), waveform, sr, subtype='PCM_16')

def main():
    parser = argparse.ArgumentParser(description="AudioSR Wrapper for Separate Env Execution")
    parser.add_argument("--input", required=True, help="Input audio path")
    parser.add_argument("--output", required=True, help="Output audio path")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale (default: 3.5)")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps (default: 50)")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print(f"[AudioSR_Wrapper] Processing: {Path(args.input).name}")
    
    # Device Setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"[AudioSR_Wrapper] Device: {device}")

    # Load Model
    try:
        model = build_model(model_name="basic", device=device)
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        sys.exit(1)
        
    # Run Inference
    try:
        waveform = super_resolution(
            model,
            args.input,
            seed=42, # Fixed seed for consistency
            guidance_scale=args.guidance_scale,
            ddim_steps=args.ddim_steps,
            latent_t_per_second=12.8
        )
        
        # AudioSR returns a list or tensor. Usually list of waveforms.
        # We assume batch size 1.
        if isinstance(waveform, list):
            audio_data = waveform[0]
        else:
            audio_data = waveform
            
        save_audio(audio_data, args.output)
        print(f"[AudioSR_Wrapper] Success. Saved to {args.output}")
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
