import os
import sys
import torch
import soundfile as sf
import numpy as np
import scipy.signal as signal
import librosa
import warnings
import shutil
import glob
import contextlib  # For silencing librosa warnings during scan
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio
from audio_separator.separator import Separator
import batch

warnings.filterwarnings("ignore")

# --- Configuration ---
INPUT_DIR = "downloads"  # Folder containing your MP3s
BASE_OUTPUT = r"output_dataset"
MODELS_DIR = r"models"  # SHARED folder for models


class UltimateVocalPipeline:
    def __init__(self, output_dir, temp_dir, models_dir):
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.models_dir = models_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        print(f"[INFO] Pipeline ready on: {self.device}")
        print(f"[INFO] Output Dir: {self.output_dir}")
        print(f"[INFO] Models Dir: {self.models_dir}")

        print("[INFO] Loading Demucs model...")
        self.demucs_model = get_model('htdemucs')
        self.demucs_model.to(self.device)

    def _save_safe_wav(self, audio_data, path, sample_rate):
        if torch.is_tensor(audio_data):
            audio_data = audio_data.cpu().numpy()
        if audio_data.ndim > 1 and audio_data.shape[0] < audio_data.shape[1]:
            audio_data = audio_data.T

        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95
        sf.write(path, audio_data, sample_rate, subtype='PCM_16')


    def step_1_roformer(self, input_path, basename):
        print(f"\n[Step 1] Roformer: Separating Vocals (with Natural Leak)...")
        
        # Adjusting params to be slightly less aggressive
        separator = Separator(
            output_dir=self.temp_dir,
            output_format="WAV",
            normalization_threshold=0.9,
            # use_autocast REMOVED for version 0.19.4 compatibility
            model_file_dir=self.models_dir,
            mdxc_params={
                "segment_size": 256, 
                "overlap": 4,
                "batch_size": 1,
                "margin": 44100
            }
        )

        model_id = 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt'
        separator.load_model(model_id)

        # 1. Perform standard separation
        output_files = separator.separate(input_path)
        vocal_file_raw = next((f for f in output_files if "Vocals" in f or "vocals" in f), output_files[0])
        vocal_path_raw = os.path.join(self.temp_dir, vocal_file_raw)

        # 2. Introduce "Human Leak" (Mixing 2% of original back in)
        # This fills the "hollow" artifacts common in AI separation
        print(f"   -> Mixing 2% original audio back in for natural harmonics...")
        
        # Load both files (keeping original sample rate)
        orig_audio, sr = librosa.load(input_path, sr=None, mono=False)
        vocal_audio, _ = librosa.load(vocal_path_raw, sr=sr, mono=False)

        # Ensure lengths match (tiny differences can happen during processing)
        min_len = min(orig_audio.shape[-1], vocal_audio.shape[-1])
        orig_audio = orig_audio[..., :min_len]
        vocal_audio = vocal_audio[..., :min_len]

        # The Mix Logic: 98% Clean Vocal + 2% Original Song
        # This acts as a "dither" to prevent robotic RVC training
        mixed_vocal = (vocal_audio * 0.98) + (orig_audio * 0.02)

        # 3. Save and Cleanup
        dst = os.path.join(self.temp_dir, f"{basename}_01_roformer_vocals.wav")
        if os.path.exists(dst): os.remove(dst)
        
        # Save using your existing safe_wav method or soundfile directly
        self._save_safe_wav(mixed_vocal, dst, sr)

        # Cleanup the raw separation file to save space
        if os.path.exists(vocal_path_raw): os.remove(vocal_path_raw)
        # Cleanup other files (like the Instrumental stem) if they exist
        for f in output_files:
            p = os.path.join(self.temp_dir, f)
            if os.path.exists(p): os.remove(p)

        print(f"   -> Saved (Leaked): {dst}")
        return dst


    def step_2_demucs_cleanup(self, input_path, basename):
        print(f"\n[Step 2] Demucs: Soft Artifact Cleanup...")
        # We increase 'shifts' to 2. This makes the model run multiple passes 
        # and average them, which significantly reduces "robotic" cutting.
        
        wav, sr = sf.read(input_path)
        wav = torch.from_numpy(wav).float()

        if wav.ndim == 1:
            wav = wav.unsqueeze(0).expand(2, -1)
        elif wav.shape[0] > wav.shape[1]:
            wav = wav.t()
        
        wav = wav.unsqueeze(0).to(self.device)
        model_sr = self.demucs_model.samplerate

        if sr != model_sr:
            wav = convert_audio(wav, sr, model_sr, self.demucs_model.audio_channels)

        with torch.no_grad():
            # 'shifts=2' is the magic for naturalness. It takes longer but prevents crops.
            sources = apply_model(self.demucs_model, wav, device=self.device, shifts=2)

        vocal_wav = sources[0, self.demucs_model.sources.index('vocals')]
        
        # Soft mixing: Mix 10% of the Step 1 input back into the Step 2 output
        # to ensure Step 2 doesn't "over-clean" what Roformer already did well.
        vocal_wav_np = vocal_wav.cpu().numpy()
        input_wav_np = wav.squeeze(0).cpu().numpy()
        vocal_wav_np = (vocal_wav_np * 0.90) + (input_wav_np * 0.10)

        output_path = os.path.join(self.temp_dir, f"{basename}_02_demucs_clean.wav")
        self._save_safe_wav(vocal_wav_np, output_path, model_sr)
        print(f"   -> Saved (Soft Clean): {output_path}")
        return output_path


    def step_3_remove_chorus(self, input_path, basename):
        print("\n[Step 3] MDX-Net: Removing Chorus (Kara 2)...")
        separator = Separator(
            log_level=40,
            model_file_dir=self.models_dir,
            output_dir=self.temp_dir,
            output_single_stem="Vocals",
            # MDX params to fix buzzing noise in silence
            mdx_params={
                "hop_length": 1024,
                "segment_size": 256,
                "overlap": 0.25,
                "batch_size": 1,
                "enable_denoise": True
            }
        )
        separator.load_model(model_filename='UVR_MDXNET_KARA_2.onnx')
        out = separator.separate(input_path)

        clean_name = os.path.join(self.temp_dir, f"{basename}_03_no_chorus.wav")
        if os.path.exists(clean_name): os.remove(clean_name)
        os.rename(os.path.join(self.temp_dir, out[0]), clean_name)
        print(f"   -> Saved: {clean_name}")
        return clean_name


    def step_4_remove_reverb(self, input_path, basename):
        print("\n[Step 4] VR Arch: Gentle De-Echo...")
        # We lower the 'aggression' to 5 (default is often higher/auto).
        # This keeps the vocal "body" while still removing the room echo.
        separator = Separator(
            log_level=40,
            model_file_dir=self.models_dir,
            output_dir=self.temp_dir,
            output_single_stem="No Echo",
            vr_params={"aggression": 5, "window_size": 512}
        )
        separator.load_model(model_filename='UVR-De-Echo-Aggressive.pth')
        out = separator.separate(input_path)

        clean_name = os.path.join(self.temp_dir, f"{basename}_04_dry.wav")
        if os.path.exists(clean_name): os.remove(clean_name)
        os.rename(os.path.join(self.temp_dir, out[0]), clean_name)
        print(f"   -> Saved: {clean_name}")
        return clean_name


    def step_5_mono_compress(self, input_path, basename):
        print("\n[Step 5] Processing: Mono & Normalization...")
        data, sr = sf.read(input_path)

        # 1. Force Mono (RVC requires mono)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # 2. Soft Limiter (Safe gain boost without clipping)
        drive_db = 2.0
        drive_linear = 10 ** (drive_db / 20.0)
        boosted_data = data * drive_linear

        # Tanh limits peaks softly to prevent distortion
        limited_data = np.tanh(boosted_data)

        # Final safety ceiling at -1.0 dB
        final_data = limited_data * 0.90

        output_path = os.path.join(self.temp_dir, f"{basename}_05_processed_full.wav")
        sf.write(output_path, final_data, sr, subtype='PCM_16')
        print(f"   -> Saved Full Processed (Flat EQ): {output_path}")
        return output_path


    def step_6_smart_slice(self, input_path, basename, target_sr=48000, slice_len_sec=3.5, silence_thresh=-50,
                           min_len_sec=1.0):
        # CHANGED: silence_thresh from -60 to -50 (Filters out static breath better)
        print(f"\n[Step 6] Smart Slicing (Silence Detection)...")

        slice_dir = os.path.join(self.output_dir, "dataset_slices")
        os.makedirs(slice_dir, exist_ok=True)

        try:
            # Normalize to target SR immediately
            y, sr = librosa.load(input_path, sr=target_sr, mono=True)

            # Trim leading/trailing silence aggressively
            y, _ = librosa.effects.trim(y, top_db=25)

            # Split logic
            intervals = librosa.effects.split(y, top_db=abs(silence_thresh), frame_length=4096, hop_length=512)

            chunk_count = 0
            for start_idx, end_idx in intervals:
                # Add a tiny bit of padding (0.05s) so words aren't cut too sharp
                pad = int(0.05 * sr)
                start_idx = max(0, start_idx - pad)
                end_idx = min(len(y), end_idx + pad)

                segment = y[start_idx:end_idx]
                duration = len(segment) / sr

                if duration < min_len_sec: continue

                # Slice strategy
                if duration <= (slice_len_sec + 2.0):
                    out_name = os.path.join(slice_dir, f"{basename}_{chunk_count:04d}.wav")
                    sf.write(out_name, segment, sr)
                    chunk_count += 1
                else:
                    # Overlap splitting for long segments
                    stride = int(3.0 * sr)
                    window = int(slice_len_sec * sr)

                    for i in range(0, len(segment) - window, stride):
                        sub = segment[i: i + window]
                        if np.max(np.abs(sub)) < 0.05: continue  # Skip silent sub-chunks

                        out_name = os.path.join(slice_dir, f"{basename}_{chunk_count:04d}_sub.wav")
                        sf.write(out_name, sub, sr)
                        chunk_count += 1

            print(f"[OK] Slicing Complete! Created {chunk_count} clean chunks in: {slice_dir}")
        except Exception as e:
            print(f"⚠️ Error slicing: {e}")


        """Scans the output folder for broken/short files and deletes them."""
        slice_dir = os.path.join(self.output_dir, "dataset_slices")

        print(f"\n[Step 7] Ghost Buster: Scanning for broken files in {slice_dir}...")

        if not os.path.exists(slice_dir):
            print("[WARN] No slices directory found. Skipping scan.")
            return

        min_size_kb = 10  # Files < 10KB are suspicious
        min_duration = 0.5  # RVC crashes on clips < 0.4s

        bad_files = []

        files_to_scan = [f for f in os.listdir(slice_dir) if f.lower().endswith(".wav")]
        print(f"   -> Scanning {len(files_to_scan)} files...")

        for filename in files_to_scan:
            path = os.path.join(slice_dir, filename)
            reason = ""

            # Check 1: File Size
            if os.path.getsize(path) < (min_size_kb * 1024):
                reason = f"Too Small ({os.path.getsize(path)} bytes)"

            # Check 2: Audio Duration (Only if size passed)
            else:
                try:
                    with contextlib.redirect_stderr(None):  # Silence librosa warnings
                        dur = librosa.get_duration(path=path)
                    if dur < min_duration:
                        reason = f"Too Short ({dur:.2f}s)"
                except Exception:
                    reason = "Corrupted / Unreadable"

            # Action: Delete if bad
            if reason:
                print(f"   [DEL] DELETING BROKEN FILE: {filename} [{reason}]")
                try:
                    os.remove(path)
                    bad_files.append(filename)
                except Exception as e:
                    print(f"      [WARN] Failed to delete {filename}: {e}")

        if len(bad_files) == 0:
            print("[OK] Ghost Buster: No broken files found. Dataset is clean!")
        else:
            print(f"[OK] Ghost Buster: Removed {len(bad_files)} broken files that would cause training crashes.")


# --- Main Execution Loop ---
if __name__ == "__main__":
    import process_lib as proc
    from pathlib import Path
    from scipy.signal import butter, sosfilt
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python cleansing.py <singer_name> [mode]")
        print("  singer_name: Name of the singer (required)")
        print("  mode:")
        print("    (none) - Run batch download + default pipeline")
        print("    d      - No download, use default pipeline (Re-Air from ab_test)")
        print("    n      - No download, manually choose each step")
        sys.exit(1)
    
    singer_name = sys.argv[1].strip()
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else ""
    
    if not singer_name:
        print("[ERROR] Singer name cannot be empty.")
        sys.exit(1)

    OUTPUT_DIR = os.path.join(BASE_OUTPUT, singer_name)
    TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    print(f"\n[INFO] Starting processing for: {singer_name}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    
    # Handle download based on mode
    if mode in ['d', 'n']:
        print("[INFO] Skipping batch download (mode provided)")
    else:
        print("[INFO] Running batch download...")
        batch.download_from_file("urls.txt")

    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] Input directory '{INPUT_DIR}' not found.")
        sys.exit(1)

    all_files = os.listdir(INPUT_DIR)
    audio_files = [f for f in all_files if f.lower().endswith(('.mp3', '.wav'))]

    if not audio_files:
        print(f"[ERROR] No MP3 or WAV files found in '{INPUT_DIR}'.")
        sys.exit(1)

    # Calculate total duration of all input files
    print(f"[INFO] Scanning {len(audio_files)} files...")
    total_duration_sec = 0
    for f in audio_files:
        try:
            dur = librosa.get_duration(path=os.path.join(INPUT_DIR, f))
            total_duration_sec += dur
        except Exception:
            pass  # Skip files that can't be read
    total_duration_min = total_duration_sec / 60.0
    print(f"[INFO] Found {len(audio_files)} files to process (total: {total_duration_min:.1f} minutes)")
    
    # --- Step Selection for Manual Mode ---
    def prompt_step_choice(step_name, models_dict, default_id):
        """Prompt user to select a model for a step."""
        print(f"\n[{step_name}] Select model:")
        for mid, minfo in models_dict.items():
            default_marker = " (default)" if mid == default_id else ""
            print(f"  {mid}: {minfo['name']}{default_marker}")
        print(f"  0: Skip this step")
        
        while True:
            choice = input(f"Enter choice [{default_id}]: ").strip()
            if choice == "":
                return default_id
            if choice == "0":
                return None
            try:
                choice_int = int(choice)
                if choice_int in models_dict:
                    return choice_int
            except ValueError:
                pass
            print("Invalid choice, try again.")

    # --- Default Pipeline (Re-Air from ab_test) ---
    def run_default_pipeline(input_path, temp_dir, basename):
        """
        Default pipeline matching ab_test's Re-Air:
        1. HPF 100Hz on original
        2. Mel + Kimi blend (80%/20%)
        3. Inst_HQ_3 cleanup
        4. Kara2 + De-Reverb (FoxJoy)
        5. Mono & Normalize -> <basename>_processed.wav
        """
        import time
        start_time = time.time()
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)
        
        print("\n[Pipeline] Running Default (Re-Air)...")
        
        # Step 1: HPF 100Hz
        print("\n[Step 1] Applying HPF 100Hz...")
        y_orig, sr = librosa.load(str(input_path), sr=None, mono=False)
        if y_orig.ndim == 1: y_orig = y_orig[np.newaxis, :]
        sos = butter(4, 100, btype='high', fs=sr, output='sos')
        y_hpf = sosfilt(sos, y_orig, axis=-1)
        step1_path = temp_path / f"{basename}_01_hpf.wav"
        if y_hpf.ndim > 1: y_hpf_save = y_hpf.T
        else: y_hpf_save = y_hpf
        sf.write(str(step1_path), y_hpf_save, sr, subtype='PCM_16')
        print(f"   -> Saved: {step1_path.name}")
        
        # Step 2: Mel low + BS mid/high (gradual 1kHz-3kHz crossover)
        print("\n[Step 2] Vocal extraction (Mel low + BS mid/high)...")
        p_mel = proc.vocal_separation(str(step1_path), temp_path, model_id=1)  # Mel-RoFormer
        p_bs = proc.vocal_separation(str(step1_path), temp_path, model_id=2)   # BS-RoFormer
        step2_path = temp_path / f"{basename}_02_blend.wav"
        current_path = proc.spectral_blend_3band(p_mel, p_bs, step2_path, crossover_hz=2000, blend_width=2000)
        print(f"   -> Saved: {step2_path.name}")
        
        # Step 3: Cleanup (SKIPPED - uncomment to restore)
        # print("\n[Step 3] Cleanup with MDX (Inst_HQ_3)...")
        # proc_output = proc.inst_removal(current_path, temp_path, model_id=1)
        # step3_path = temp_path / f"{basename}_03_cleanup.wav"
        # shutil.move(proc_output, str(step3_path))
        # current_path = str(step3_path)
        # print(f"   -> Saved: {step3_path.name}")
        
        # Save pre-Kara path for transient restore later
        pre_kara_path = current_path
        
        # Step 4: De-Chorus (Kara2)
        print("\n[Step 4] De-Chorus (Kara2)...")
        proc_output = proc.dechorus(current_path, temp_path, model_id=1)
        step4_path = temp_path / f"{basename}_04_dechorus.wav"
        shutil.move(proc_output, str(step4_path))
        current_path = str(step4_path)
        print(f"   -> Saved: {step4_path.name}")
        
        # Step 5: De-Reverb (Aggressive - using transient restore to recover consonants)
        print("\n[Step 5] De-Reverb (Aggressive)...")
        proc_output = proc.dereverb(current_path, temp_path, model_id=1)
        step5_path = temp_path / f"{basename}_05_dereverb.wav"
        shutil.move(proc_output, str(step5_path))
        current_path = str(step5_path)
        print(f"   -> Saved: {step5_path.name}")
        
        # Step 6: Transient Restore (recover consonant attacks from pre-Kara source)
        print("\n[Step 6] Transient Restore...")
        step6_path = temp_path / f"{basename}_06_transients.wav"
        current_path = proc.transient_restore(current_path, pre_kara_path, step6_path, 
                                               attack_ms=30, highpass_hz=5000, blend_amount=0.25)
        print(f"   -> Saved: {step6_path.name}")
        
        # Step 7: Harmonic Exciter (DSP-based high freq enhancement)
        print("\n[Step 7] Harmonic Exciter...")
        step7_path = temp_path / f"{basename}_07_excited.wav"
        current_path = proc.harmonic_exciter(current_path, step7_path, drive=0.3, high_shelf_db=3.0, cutoff_hz=6000)
        print(f"   -> Saved: {step7_path.name}")
        
        # Step 8: Mono & Normalize -> final output
        print("\n[Step 8] Mono & Normalize...")
        proc_output = proc.mono_normalize(current_path, temp_path, to_mono=True)
        step8_path = temp_path / f"{basename}_08_mono.wav"
        shutil.move(proc_output, str(step8_path))
        print(f"   -> Saved: {step8_path.name}")
        
        # --- Organize Outputs ---
        print("\n[OUTPUT] Organizing final products...")
        
        # Create output subdirectories
        output_base = temp_path.parent  # output_dataset/singer_name/
        vocal_dir = output_base / "vocal"
        dry_vocal_dir = output_base / "dry_vocal"
        mono_vocal_dir = output_base / "mono_vocal"
        
        vocal_dir.mkdir(exist_ok=True)
        dry_vocal_dir.mkdir(exist_ok=True)
        mono_vocal_dir.mkdir(exist_ok=True)
        
        # Clean basename (remove any step prefixes)
        clean_name = basename.split('_')[0] if '_' in basename else basename
        
        # Copy outputs to organized folders
        # 1. vocal/ <- 04_dechorus (Vocal_xxx.wav)
        vocal_out = vocal_dir / f"Vocal_{clean_name}.wav"
        shutil.copy(str(step4_path), str(vocal_out))
        print(f"   -> vocal/{vocal_out.name}")
        
        # 2. dry_vocal/ <- 07_excited (dry vocal_xxx.wav)
        dry_out = dry_vocal_dir / f"dry vocal_{clean_name}.wav"
        shutil.copy(str(step7_path), str(dry_out))
        print(f"   -> dry_vocal/{dry_out.name}")
        
        # 3. mono_vocal/ <- 08_mono (mono vocal_xxx.wav)
        mono_out = mono_vocal_dir / f"mono vocal_{clean_name}.wav"
        shutil.copy(str(step8_path), str(mono_out))
        print(f"   -> mono_vocal/{mono_out.name}")
        
        elapsed = time.time() - start_time
        return str(mono_out), elapsed

    # --- Manual Pipeline ---
    def run_manual_pipeline(input_path, temp_dir, basename, pipeline_config):
        """Manual pipeline with pre-selected steps."""
        import time
        start_time = time.time()
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)
        
        print("\n[Pipeline] Running Manual Pipeline...")
        current_path = str(input_path)
        step_num = 1
        
        # Step 1: Vocal Separation
        if pipeline_config['vocal_sep']:
            print(f"\n[Step {step_num}] Vocal Separation...")
            proc_output = proc.vocal_separation(current_path, temp_path, model_id=pipeline_config['vocal_sep'])
            step_path = temp_path / f"{basename}_{step_num:02d}_vocal.wav"
            shutil.move(proc_output, str(step_path))
            current_path = str(step_path)
            step_num += 1
        
        # Step 2: Cleanup
        if pipeline_config['cleanup']:
            print(f"\n[Step {step_num}] Cleanup...")
            proc_output = proc.step2_unified_cleanup(current_path, temp_path, model_id=pipeline_config['cleanup'])
            step_path = temp_path / f"{basename}_{step_num:02d}_cleanup.wav"
            shutil.move(proc_output, str(step_path))
            current_path = str(step_path)
            step_num += 1
        
        # Step 3: De-Chorus
        if pipeline_config['dechorus']:
            print(f"\n[Step {step_num}] De-Chorus...")
            proc_output = proc.dechorus(current_path, temp_path, model_id=pipeline_config['dechorus'])
            step_path = temp_path / f"{basename}_{step_num:02d}_dechorus.wav"
            shutil.move(proc_output, str(step_path))
            current_path = str(step_path)
            step_num += 1
        
        # Step 4: De-Reverb
        if pipeline_config['dereverb']:
            print(f"\n[Step {step_num}] De-Reverb...")
            proc_output = proc.dereverb(current_path, temp_path, model_id=pipeline_config['dereverb'])
            step_path = temp_path / f"{basename}_{step_num:02d}_dereverb.wav"
            shutil.move(proc_output, str(step_path))
            current_path = str(step_path)
            step_num += 1
        
        # Final: Mono & Normalize (always run)
        print(f"\n[Step {step_num}] Mono & Normalize...")
        proc_output = proc.mono_normalize(current_path, temp_path)
        final_path = temp_path / f"{basename}_processed.wav"
        shutil.move(proc_output, str(final_path))
        print(f"   -> Saved: {final_path.name}")
        
        elapsed = time.time() - start_time
        return str(final_path), elapsed

    # --- Prompt for Pipeline Config (Mode 'n' only) ---
    pipeline_config = None
    if mode == 'n':
        print("\n" + "=" * 60)
        print("[PIPELINE CONFIGURATION] Select models for each step:")
        print("=" * 60)
        
        pipeline_config = {
            'vocal_sep': prompt_step_choice("Step 1: Vocal Separation", proc.VOCAL_SEP_MODELS, 1),
            'cleanup': prompt_step_choice("Step 2: Cleanup", proc.CLEANUP_MODELS, 1),
            'dechorus': prompt_step_choice("Step 3: De-Chorus", proc.DECHORUS_MODELS, 1),
            'dereverb': prompt_step_choice("Step 4: De-Reverb", proc.DEREVERB_MODELS, 2),
        }
        
        # Show summary
        print("\n" + "-" * 40)
        print("[CONFIG SUMMARY]")
        print(f"  Vocal Separation: {proc.VOCAL_SEP_MODELS[pipeline_config['vocal_sep']]['name'] if pipeline_config['vocal_sep'] else 'SKIP'}")
        print(f"  Cleanup:          {proc.CLEANUP_MODELS[pipeline_config['cleanup']]['name'] if pipeline_config['cleanup'] else 'SKIP'}")
        print(f"  De-Chorus:        {proc.DECHORUS_MODELS[pipeline_config['dechorus']]['name'] if pipeline_config['dechorus'] else 'SKIP'}")
        print(f"  De-Reverb:        {proc.DEREVERB_MODELS[pipeline_config['dereverb']]['name'] if pipeline_config['dereverb'] else 'SKIP'}")
        print(f"  Mono & Normalize: ALWAYS")
        print("-" * 40)
    else:
        # Show summary for default pipeline
        print("\n" + "-" * 40)
        print("[CONFIG SUMMARY] Default Pipeline (Re-Air)")
        print("  1. HPF 100Hz")
        print("  2. Vocal: Mel low + BS mid/high (1kHz-3kHz gradual)")
        print("  3. Cleanup: (skipped)")
        print("  4. De-Chorus: Kara2")
        print("  5. De-Reverb: Aggressive")
        print("  6. Transient Restore (consonant attacks from pre-Kara)")
        print("  7. Harmonic Exciter (drive=0.3, +3dB at 6kHz)")
        print("  8. Mono & Normalize: 48kHz, -2dB, 32-bit float")
        print("-" * 40)
    
    print(f"\n[INFO] Will process {len(audio_files)} files ({total_duration_min:.1f} minutes total)\n")

    # --- Process Files ---
    for idx, filename in enumerate(audio_files, 1):
        input_file_path = os.path.join(INPUT_DIR, filename)
        basename = os.path.splitext(filename)[0]

        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(audio_files)}] Processing: {filename}")
        print(f"{'=' * 60}")

        try:
            # Choose pipeline based on mode
            if mode == 'n':
                final_path, elapsed = run_manual_pipeline(input_file_path, TEMP_DIR, basename, pipeline_config)
            else:
                # mode == 'd' or default (after batch download)
                final_path, elapsed = run_default_pipeline(input_file_path, TEMP_DIR, basename)
            
            # Copy final output to output directory
            final_output = os.path.join(OUTPUT_DIR, f"{basename}.wav")
            if final_path != final_output:
                shutil.copy(final_path, final_output)

            print(f"[OK] Finished: {final_output} ({elapsed:.1f}s)")
            
            # Delete the original input file after successful processing
            os.remove(input_file_path)
            print(f"[INFO] Deleted source: {filename}")

        except Exception as e:
            print(f"\n[ERROR] FAILED on file '{filename}': {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean VRAM after every file
            proc.clean_vram()
            print("[INFO] VRAM cleaned")

    print("\n[DONE] All files processed. Dataset ready.")