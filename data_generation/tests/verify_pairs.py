import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
import os
import shutil
import numpy as np
import h5py
from pedalboard.io import AudioFile
from data_generation.generate_vst_dataset import make_dataset
from data_generation.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC
from data_generation.core import load_plugin, set_params

def verify_pairs():
    print("Initializing verification using generate_vst_dataset pipeline...")
    
    # Configuration
    output_dir = "paired_data_listening"
    temp_h5_path = "verify_data_temp.h5"
    num_samples = 10
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    preset_path = "presets/surge-base.vstpreset"
    
    # Clean up previous runs
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(temp_h5_path):
        os.remove(temp_h5_path)

    # 1. Generate Dataset using existing pipeline
    print(f"Generating {num_samples} samples into {temp_h5_path}...")
    with h5py.File(temp_h5_path, "w") as f:
        make_dataset(
            hdf5_file=f,
            num_samples=num_samples,
            plugin_path=plugin_path,
            preset_path=preset_path,
            sample_rate=44100.0,
            channels=2,
            velocity=100,
            signal_duration_seconds=4.0,
            min_loudness=-50.0,
            param_spec=SURGE_SIMPLE_PARAM_SPEC,
            sample_batch_size=2,  # Small batch size for verification
        )
    
    # 2. Extract and Verify
    print("Extracting samples for verification...")
    
    # Load plugin once for state reconstruction
    plugin = load_plugin(plugin_path)
    
    with h5py.File(temp_h5_path, "r") as f:
        # Load datasets
        target_audio_ds = f["target_audio"]
        ref_audio_ds = f["reference_audio"]
        target_param_ds = f["target_param_array"]
        ref_param_ds = f["reference_param_array"]

        num_generated = target_audio_ds.shape[0]
        print(f"Found {num_generated} samples in dataset.")

        for i in range(min(num_samples, num_generated)):
            print(f"\n--- Extracting Pair {i} ---")
            
            # Extract Audio
            target_audio = target_audio_ds[i].T # (C, T) -> (T, C) if stored as (C, T)? Check logic.
            # generate_vst_dataset.py saves as: 
            # target_audios = np.stack([s.target_audio.T for s in samples], axis=0) -> s.target_audio is (C, T) ?
            # No. Pedalboard returns (C, T). generate_sample returns (C, T) as target_audio.
            # But the code says: `target_audio=target_output.T` in generate_sample
            # if target_output (from pedalboard) is (C, T), then .T makes it (T, C).
            # Then save_samples does: `[s.target_audio.T for s in samples]` which flips is back to (C, T)?
            # Let's re-verify generate_vst_dataset.py logic.
            # render_params returns (C, T) or (T, C)? Pedalboard `process` returns (samples, channels) which is (T, C).
            # So `target_output` is (T, C).
            # `generate_sample` does: `target_audio=target_output.T` -> (C, T).
            # `save_samples` does: `np.stack([s.target_audio.T for s in samples], axis=0)` -> (N, T, C).
            # So h5 has (N, T, C).
            # So `target_audio_ds[i]` is (T, C).
            # So we don't need to transpose for AudioFile writing which expects (T, C).
             
            # WAIT. Let's check `create_datasets_and_get_start_idx`:
            # audio_shape = (num_samples, channels, int(sample_rate * signal_duration_seconds))
            # This implies the shape in H5 is (N, C, T).
            
            # Let's look closely at `save_samples` again.
            # `target_audios = np.stack([s.target_audio.T for s in samples], axis=0)`
            # If `s.target_audio` is (C, T), then .T is (T, C). So stack is (N, T, C).
            # BUT `target_audio_dataset` shape is (N, C, T).
            # There is a mismatch here if `create_datasets` makes (N, C, T) but `save_samples` writes (N, T, C).
            
            # Let's Assume `generate_vst_dataset.py` might have a shape bug I need to account for, OR I misread.
            # `generate_sample`: `target_audio=target_output.T`. Pedalboard `process` output is (T, C). So `target_audio` is (C, T).
            # `save_samples`: `target_audios = np.stack([s.target_audio.T for s in samples], axis=0)`. `s.target_audio.T` flips back to (T, C).
            # So `target_audios` is (Batch, T, C).
            # `target_audio_dataset[start_idx : start_idx + n, :, :] = target_audios`
            # If dataset is (N, C, T), and we write (N, T, C)... that writes assuming dimensions match implicitly or it broadcasts/reshapes weirdly?
            # Actually, `h5py` might complain if dimensions don't match.
            # If `audio_shape` was defined as `(num_samples, channels, samples)`, i.e. (N, C, T).
            # And we write (N, T, C).
            # `T` (samples) is 44100*4 ~ 176k. `C` is 2.
            # If we write (N, 176k, 2) into (N, 2, 176k) -> mismatch.
            
            # Re-reading `save_samples` in the provided file attachment:
            # target_audios = np.stack([s.target_audio.T for s in samples], axis=0)
            # --> If s.target_audio is (C, T) (which it seems to be), s.target_audio.T is (T, C). Stack -> (Batch, T, C).
            
            # Re-reading `create_datasets_and_get_start_idx`:
            # audio_shape = (num_samples, channels, int(sample_rate * signal_duration_seconds)) -> (N, C, T).
            
            # This looks like a bug in `generate_vst_dataset.py` that likely hasn't been hit or I am misinterpreting Pedalboard output.
            # Validated: Pedalboard `process` returns numpy array of shape (samples, channels). (T, C).
            # So `target_output` is (T, C).
            # `generate_sample` -> `target_audio=target_output.T` -> (C, T). Correct for standard audio processing usually, but...
            # `save_samples` -> `s.target_audio.T` -> (T, C).
            # `save_samples` -> writes (Batch, T, C) into (N, C, T).
            # This will definitely crash or write garbage if sizes differed more substantially, but here dimensions are (2) and (176400).
            # It will crash with TypeError/ValueError.
            
            # HOWEVER, the User asked me to use `generate_vst_dataset.py`.
            # If that script is broken, I will find out when I run this.
            # If it runs, maybe my shape analysis is wrong.
            
            # Let's assume the HDF5 file will contain whatever `generate_vst_dataset.py` produces.
            # I will inspect the shape of the dataset when reading.

            target_audio_raw = target_audio_ds[i]
            ref_audio_raw = ref_audio_ds[i]
            
            # Handle Shape
            # We want (T, C) for .wav writing. 
            # If shape is (C, T), transpose params.
            if target_audio_raw.shape[0] < target_audio_raw.shape[1]:
                # Assume (C, T)
                print(f"  Detected (C, T) shape {target_audio_raw.shape}, transposing to (T, C)")
                target_audio_raw = target_audio_raw.T
                ref_audio_raw = ref_audio_raw.T
            else:
                print(f"  Detected (T, C) shape {target_audio_raw.shape}")

            # Normalize Audio to -3.0 dB
            def normalize(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
                peak = np.max(np.abs(audio))
                if peak < 1e-6:
                    return audio
                target_amp = 10 ** (target_db / 20)
                return audio * (target_amp / peak)

            target_audio = normalize(target_audio_raw).astype(np.float32)
            ref_audio = normalize(ref_audio_raw).astype(np.float32)
            
            # Save Audio
            with AudioFile(f"{output_dir}/pair_{i}_target.wav", "w", 44100, 2) as wav:
                wav.write(target_audio)
            with AudioFile(f"{output_dir}/pair_{i}_ref.wav", "w", 44100, 2) as wav:
                wav.write(ref_audio)

            # Decode Params
            target_param_array = target_param_ds[i]
            ref_param_array = ref_param_ds[i]
            
            target_params_dict, _ = SURGE_SIMPLE_PARAM_SPEC.decode(target_param_array)
            ref_params_dict, _ = SURGE_SIMPLE_PARAM_SPEC.decode(ref_param_array)
            # Note: decode returns (synth, note). 
            # Since the array in HDF5 is concatenated (synth + note), decode splits it correctly.
            # Wait, `SURGE_SIMPLE_PARAM_SPEC.decode` returns `synth_params, note_params`.
            # But wait, looking at `save_samples`:
            # target_param_dataset[start_idx : start_idx + n, :] = target_params
            # target_params = np.stack([s.target_param_array for s in samples], axis=0)
            # s.target_param_array comes from `self.param_spec.encode(self.target_synth_params, self.note_params)`
            # So it includes both.
            
            # Separate Note Params
            # `decode` returns two dicts.
            t_synth, t_note = SURGE_SIMPLE_PARAM_SPEC.decode(target_param_array)
            r_synth, r_note = SURGE_SIMPLE_PARAM_SPEC.decode(ref_param_array)

            # --- Full Parameter Dump Logic ---
            # 1. Reset plugin (optional but good for clean state)
            # plugin.reset() # Handled in set_params/render flows usually, but let's just set params.
            
            # 2. Set Target Params to Plugin
            set_params(plugin, t_synth)
            
            # 3. Capture All Plugin Parameters (Raw Values)
            # target_full_state = {
            #     k: p.raw_value for k, p in plugin.parameters.items()
            # }
            
            # 4. Set Reference Params to Plugin
            set_params(plugin, r_synth)
            
            # 5. Capture All Plugin Parameters
            # ref_full_state = {
            #     k: p.raw_value for k, p in plugin.parameters.items()
            # }
            # ---------------------------------
            
            param_dump = {
                "target_spec": t_synth,
                "reference_spec": r_synth,
                # "target_full": target_full_state,
                # "reference_full": ref_full_state,
                "note": t_note # Should be same for both usually
            }
            
            # Convert numpy types
            def convert(o):
                if isinstance(o, np.generic): return o.item()
                raise TypeError
            
            with open(f"{output_dir}/pair_{i}_params.json", "w") as jf:
                json.dump(param_dump, jf, indent=2, default=convert)
            
            print(f"  Saved pair_{i} to {output_dir}/")

    # Clean up
    if os.path.exists(temp_h5_path):
        os.remove(temp_h5_path)
    print("Verification complete.")

if __name__ == "__main__":
    verify_pairs()

