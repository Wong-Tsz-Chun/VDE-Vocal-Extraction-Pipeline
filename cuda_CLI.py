"""
cuda_CLI.py - CUDA 12 ONNX Model Runner

Standalone CLI script that runs ONNX-based audio-separator models
in a CUDA 12 environment. Called via subprocess from process_lib.py.

Usage:
    python cuda_CLI.py --model <model_file> --input <input.wav> --output_dir <dir> [--stem <stem_name>]
    
Example:
    python cuda_CLI.py --model UVR_MDXNET_KARA_2.onnx --input vocals.wav --output_dir temp --stem Vocals
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure models directory is found
BASE_DIR = Path(__file__).parent.absolute()
MODELS_DIR = BASE_DIR / "models"

def run_separation(model_file, input_path, output_dir, stem=None, enable_denoise=True):
    """
    Run audio-separator with the specified ONNX model.
    Returns the path to the output file.
    """
    from audio_separator.separator import Separator
    
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configure separator with denoise enabled
    sep_kwargs = {
        "log_level": 20,
        "model_file_dir": str(MODELS_DIR),
        "output_dir": str(output_dir),
        "output_format": "WAV",
        "mdx_params": {
            "hop_length": 1024,
            "segment_size": 256,
            "overlap": 0.75,
            "batch_size": 1,
            "enable_denoise": enable_denoise
        }
    }
    
    if stem:
        sep_kwargs["output_single_stem"] = stem
    
    separator = Separator(**sep_kwargs)
    separator.load_model(model_file)
    output_files = separator.separate(input_path)
    
    if isinstance(output_files, str):
        output_files = [output_files]
    
    if not output_files:
        raise RuntimeError("No output files produced")
    
    # Return the first output file path
    output_path = output_dir / output_files[0]
    print(str(output_path))  # Print path for subprocess capture
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="CUDA 12 ONNX Model Runner")
    parser.add_argument("--model", required=True, help="Model filename (e.g., UVR_MDXNET_KARA_2.onnx)")
    parser.add_argument("--input", required=True, help="Input audio file path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--stem", default=None, help="Output stem name (e.g., Vocals, No Reverb)")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoise")
    
    args = parser.parse_args()
    
    try:
        output_path = run_separation(
            model_file=args.model,
            input_path=args.input,
            output_dir=args.output_dir,
            stem=args.stem,
            enable_denoise=not args.no_denoise
        )
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
