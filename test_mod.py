
import sys
import os
from pathlib import Path
import logging

# Ensure local imports work
BASE_DIR = Path(__file__).parent.absolute()
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import process_lib as proc

# Set logs to info
sys.stdout.reconfigure(encoding='utf-8')

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_valid_input(prompt, options):
    """
    Get valid integer input from user.
    options: list of valid integers or range
    """
    while True:
        try:
            choice = input(prompt + " > ").strip()
            if not choice: continue
            val = int(choice)
            if val in options:
                return val
            print(f"Invalid option. Please choose from {options}")
        except ValueError:
            print("Please enter a number.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_vocal_separation.py <input_audio_path>")
        print("Example: python test_vocal_separation.py downloads/song.mp3")
        sys.exit(1)

    input_path = Path(sys.argv[1]).absolute()
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    # Output dir
    output_dir = BASE_DIR / "test_results"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*50}")
    print(f" VDE Interactive Processor")
    print(f" Target: {input_path.name}")
    print(f"{'='*50}")

    # 1. Select Step
    step_options = proc.get_step_options()
    
    print("\nAvailable Steps:")
    valid_steps = []
    
    for step_id in sorted(step_options.keys()):
        info = step_options[step_id]
        print(f" {step_id}. {info['name']}")
        valid_steps.append(step_id)

    step_choice = get_valid_input("\nSelect a Step", valid_steps)
    selected_step = step_options[step_choice]
    
    print(f"\n[Selected: {selected_step['name']}]")

    # 2. Select Model (if applicable)
    model_id = 1
    models = selected_step.get("models")
    
    if models:
        print("\nAvailable Models:")
        valid_models = []
        for mid in sorted(models.keys()):
            model = models[mid]
            print(f" {mid}. {model['name']}")
            valid_models.append(mid)
            
        model_choice = get_valid_input("\nSelect a Model", valid_models)
        model_id = model_choice
    else:
        print(" (No model selection for this step)")

    # 3. Confirm and Run
    print(f"\n{'='*50}")
    print(f"READY TO RUN:")
    print(f" Function: {selected_step['name']}")
    if models:
        print(f" Model:    {models[model_id]['name']}")
    print(f" Input:    {input_path.name}")
    print(f" Output:   {output_dir}")
    print(f"{'='*50}")
    
    input("Press Enter to start...")
    
    try:
        # Construct args based on function signature (mostly unified in process_lib)
        # func(input_path, output_dir, model_id=...) or similar
        
        func = selected_step["func"]
        
        if models:
            result_path = func(input_path, output_dir, model_id=model_id)
        else:
            # Steps like mono_normalize don't take model_id
            # Step 5 signature: mono_normalize(input_path, output_dir, apply_high_shelf=False)
            result_path = func(input_path, output_dir)
            
        print(f"\n[SUCCESS] Saved to:\n{result_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
