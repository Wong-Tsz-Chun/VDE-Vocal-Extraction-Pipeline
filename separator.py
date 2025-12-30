"""
separator.py - Batch Vocal Processing Pipeline
Processes all files in downloads/ with interactive model selection.

Usage:
    python separator.py       # Download + interactive model selection
    python separator.py n     # No download, interactive model selection
    python separator.py d     # No download, use default models
"""

import os
import sys
import gc
import torch
import shutil
from pathlib import Path

import batch
import process_lib as proc

# --- Configuration ---
BASE_DIR = Path(__file__).parent.absolute()
INPUT_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "output_dataset"
TEMP_DIR = OUTPUT_DIR / "temp"


def clean_vram():
    """Clean VRAM after processing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def select_model(step_num, step_info):
    """Interactive model selection for a step"""
    print(f"\nStep {step_num}: {step_info['name']}")
    
    if step_info["models"] is None:
        print("   (No model selection - always runs)")
        return None
    
    models = step_info["models"]
    for idx, info in models.items():
        default_marker = " (default)" if idx == step_info["default"] else ""
        print(f"  {idx}. {info['name']}{default_marker}")
    print(f"  0. Skip this step")
    
    while True:
        try:
            choice = input(f"> ").strip()
            if choice == "":
                return step_info["default"]
            choice = int(choice)
            if choice == 0:
                return 0  # Skip
            if choice in models:
                return choice
            print(f"Invalid choice. Enter 1-{len(models)} or 0 to skip.")
        except ValueError:
            print("Enter a number.")


def get_default_choices():
    """Return default model choices for all steps"""
    step_options = proc.get_step_options()
    return {
        1: step_options[1]["default"],
        2: step_options[2]["default"],
        3: step_options[3]["default"],
        4: step_options[4]["default"],
        5: "mono"  # Default: mono output
    }


def get_interactive_choices():
    """Interactive model selection for all steps"""
    step_options = proc.get_step_options()
    choices = {}
    
    print("\n" + "="*50)
    print("SELECT MODELS FOR EACH STEP")
    print("="*50)
    print("(Press Enter for default, 0 to skip)")
    
    for step_num in range(1, 5):  # Steps 1-4
        step_info = step_options[step_num]
        choices[step_num] = select_model(step_num, step_info)
    
    # Step 5: Mono/Stereo selection
    print(f"\nStep 5: Output Format")
    print("  1. Mono (default)")
    print("  2. Stereo")
    while True:
        choice = input("> ").strip()
        if choice == "" or choice == "1":
            choices[5] = "mono"
            break
        elif choice == "2":
            choices[5] = "stereo"
            break
        else:
            print("Enter 1 or 2.")
    
    # Display selection summary
    print("\n" + "="*50)
    print("SELECTED CONFIGURATION")
    print("="*50)
    for step_num in range(1, 6):
        step_info = step_options[step_num]
        if step_num == 5:
            print(f"  Step 5: {choices[5].upper()} & Normalize")
        elif choices[step_num] == 0:
            print(f"  Step {step_num}: SKIP")
        else:
            model_name = step_info["models"][choices[step_num]]["name"]
            print(f"  Step {step_num}: {model_name}")
    print("")
    
    return choices


def process_file(input_path, choices):
    """Process a single file through the pipeline"""
    current_path = str(input_path)
    basename = Path(input_path).stem
    
    TEMP_DIR.mkdir(exist_ok=True)
    
    try:
        # Step 1: Vocal Separation
        if choices[1] and choices[1] != 0:
            current_path = proc.vocal_separation(
                current_path, TEMP_DIR, model_id=choices[1]
            )
        
        # Step 2: Artifact Cleanup / Inst Removal
        if choices[2] and choices[2] != 0:
            current_path = proc.step2_unified_cleanup(
                current_path, TEMP_DIR, model_id=choices[2]
            )
        
        # Step 3: De-Chorus
        if choices[3] and choices[3] != 0:
            current_path = proc.dechorus(
                current_path, TEMP_DIR, model_id=choices[3]
            )
        
        # Step 4: De-Reverb
        if choices[4] and choices[4] != 0:
            current_path = proc.dereverb(
                current_path, TEMP_DIR, model_id=choices[4]
            )
        
        # Step 5: Normalize (mono or stereo based on choice)
        is_mono = choices[5] == "mono"
        current_path = proc.mono_normalize(current_path, TEMP_DIR, to_mono=is_mono)
        
        # Copy final output
        final_output = OUTPUT_DIR / f"{basename}.wav"
        shutil.copy(current_path, final_output)
        
        print(f"[OK] Finished: {final_output.name}")
        return str(final_output)
        
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Parse arguments
    skip_download = False
    use_defaults = False
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "n":
            skip_download = True
        elif arg == "d":
            skip_download = True
            use_defaults = True
    
    # Setup directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print("VOCAL SEPARATOR PIPELINE")
    print("="*50)
    
    # Download if needed
    if not skip_download:
        print("\n[INFO] Running batch download...")
        batch.download_from_file("urls.txt")
    else:
        print("\n[INFO] Skipping download")
    
    # Get files (MP3 and WAV)
    audio_files = list(INPUT_DIR.glob("*.mp3")) + list(INPUT_DIR.glob("*.wav"))
    
    if not audio_files:
        print(f"[ERROR] No MP3 or WAV files found in '{INPUT_DIR}'")
        sys.exit(1)
    
    print(f"\n[INFO] Found {len(audio_files)} files to process")
    
    # Get model choices
    if use_defaults:
        print("[INFO] Using default models")
        choices = get_default_choices()
    else:
        choices = get_interactive_choices()
    
    # Process all files
    print("\n" + "="*50)
    print("PROCESSING")
    print("="*50)
    
    results = []
    for idx, input_file in enumerate(audio_files, 1):
        print(f"\n[{idx}/{len(audio_files)}] {input_file.name}")
        print("-"*40)
        
        result = process_file(input_file, choices)
        results.append({"file": input_file.name, "success": result is not None})
        
        # Delete source file after processing
        if result:
            input_file.unlink()
            print(f"[INFO] Deleted source: {input_file.name}")
        
        # Clean VRAM
        clean_vram()
        print("[INFO] VRAM cleaned")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    success = sum(1 for r in results if r["success"])
    failed = len(results) - success
    
    print(f"  Processed: {len(results)}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\n  Failed files:")
        for r in results:
            if not r["success"]:
                print(f"    - {r['file']}")
    
    print(f"\n[DONE] Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
