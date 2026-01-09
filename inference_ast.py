import torch
import torchaudio
import numpy as np
import sys
from pathlib import Path
import torchcrepe
import librosa
from model_training.model import Vimtopoeia_AST

# Add root to path to allow imports from data_generation
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

try:
    from data_generation.core import load_plugin, render_params, write_wav
    from data_generation.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC
except ImportError:
    print("⚠️  Warning: Could not import data_generation.core. Audio rendering will be disabled.")
    load_plugin = None

# === 1. Configuration ===
MODEL_PATH = "/Users/wanghuixi/Downloads/model_best.pt"
TARGET_AUDIO_PATH = "/Users/wanghuixi/vimtopoeia/v1_output_samples/output_synth_carlos_007.wav" 
REF_AUDIO_PATH = "/Users/wanghuixi/vimtopoeia/v1_output_samples/output_synth_carlos.wav"      

# Determine N_PARAMS from Spec
FULL_SYNTH_PARAMS = SURGE_SIMPLE_PARAM_SPEC.synth_param_length
MODEL_PARAMS_COUNT = 22

# Indices of parameters actually used in training
ACTIVE_PARAM_INDICES = [
    0, 1, 2, 3,    # Amp Env (A, D, R, S)
    4, 5, 6,       # Filter 1 (Cutoff, FEG Mod, Reso)
    7, 8, 9,       # Filter 2 (Cutoff, FEG Mod, Reso)
    10, 11, 12, 13,# Filter Env (A, D, R, S)
    14,            # Highpass
    15,            # Noise Volume
    18, 19, 20,    # Osc Shapes (Saw, Pulse, Tri)
    21, 22,        # Osc Mods (Width, Sync)
    28             # Unison Detune
]

print(f"ℹ️  Parameter Specification: Full Synth Params: {FULL_SYNTH_PARAMS}, Active Model Params: {MODEL_PARAMS_COUNT}")

def get_default_params():
    """Returns a default 'Init Patch' configuration as a (FULL_SYNTH_PARAMS,) array."""
    arr = np.zeros(FULL_SYNTH_PARAMS)
    
    # Helper to find index of a parameter by name
    # We iterate and accumulate length
    idx_map = {}
    curr = 0
    for p in SURGE_SIMPLE_PARAM_SPEC.synth_params:
        idx_map[p.name] = curr
        curr += len(p)
        
    # Set Defaults:
    # Sustain = 1.0
    if "a_amp_eg_sustain" in idx_map: arr[idx_map["a_amp_eg_sustain"]] = 1.0
    # Open Filter
    if "a_filter_1_cutoff" in idx_map: arr[idx_map["a_filter_1_cutoff"]] = 1.0
    # Oscillator Volume
    if "a_osc_1_volume" in idx_map: arr[idx_map["a_osc_1_volume"]] = 1.0
    # Unison Voices (1 voice = index 0 of the onehot)
    if "a_osc_1_unison_voices" in idx_map: arr[idx_map["a_osc_1_unison_voices"]] = 1.0
    
    return arr

CURRENT_REF_PARAMS = get_default_params()

def get_pitch(audio_path):
    """
    Uses CREPE to get the most dominant pitch (f0) and convert to MIDI.
    """
    device = "cpu" # CREPE is fast enough on M3 CPU for inference
    print(f"🎵 Detecting pitch from: {audio_path}")
    
    try:
        # Load with torchaudio (returns Tensor [Channels, Time])
        audio, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading for pitch detection: {e}")
        return 60 # Default C3
    
    # Ensure mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        
    # audio is [1, T]
    
    # Move to device
    audio = audio.to(device)
    
    # Run Pitch Tracking
    # hop_length=320 corresponds to roughly 20ms granularity (at 16k)
    f0, confidence = torchcrepe.predict(
        audio, 
        sr, 
        hop_length=320, 
        fmin=50, 
        fmax=1000, 
        model='tiny',   # 'tiny' is super fast and accurate enough for MIDI
        decoder=torchcrepe.decode.viterbi, 
        return_periodicity=True,
        device=device
    )
    
    # Filter silence/noise (Confidence check)
    valid_f0 = f0[confidence > 0.4]
    
    if len(valid_f0) == 0:
        print("Warning: No pitch detected. Defaulting to C3.")
        return 60
        
    # Get median pitch (Average pitch of the clip)
    median_f0 = torch.median(valid_f0).item()
    
    # Convert to MIDI
    midi_note = int(librosa.hz_to_midi(median_f0))
    print(f"✅ Detected MIDI Note: {midi_note}")
    return midi_note

def load_and_process_audio(audio_path):
    """读取音频 -> 转单声道 -> 重采样 16k -> Mel 频谱 -> 标准化"""
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

    # 1. 混音成单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 2. 重采样到 16000 Hz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    
    # 3. 转 Mel Spectrogram (参数必须与 dataset.py 完全一致)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, win_length=400, hop_length=160, 
        n_mels=128, f_min=0, f_max=8000
    )
    spec = mel_transform(waveform)
    spec = torchaudio.transforms.AmplitudeToDB()(spec)
    
    # 4. 维度调整 (Freq, Time) -> (Time, Freq)
    spec = spec.squeeze(0).transpose(0, 1) 
    
    # 5. 长度对齐 (Padding/Cropping to 1024 frames)
    max_len = 1024
    if spec.shape[0] < max_len:
        padding = torch.zeros(max_len - spec.shape[0], 128)
        spec = torch.cat([spec, padding], dim=0)
    else:
        spec = spec[:max_len, :]
        
    # 6. 标准化 (Global Mean/Std from Dataset)
    mean = -4.2677393
    std = 4.5689974
    spec = (spec - mean) / (std * 2)
    
    return spec.unsqueeze(0) # Add batch dim -> (1, 1024, 128)

def generate_random_surge_sound():
    """
    Generates a random valid configuration for Surge XT based on our ParamSpec.
    Returns:
        np.array of shape (FULL_SYNTH_PARAMS,)
    """
    print("\n🎲 Generating Random Surge XT Patch...")
    
    # Sample from Spec
    params_dict, note_params_dict = SURGE_SIMPLE_PARAM_SPEC.sample()
    
    # Encode to array (concatenates synth + note)
    full_encoded = SURGE_SIMPLE_PARAM_SPEC.encode(params_dict, note_params_dict)
    
    # Extract only synth params
    synth_encoded = full_encoded[:FULL_SYNTH_PARAMS]

    print("🎲 Random Patch Generated!")
    return synth_encoded

def get_surge_params_from_user():
    """
    Prompt user to input or load current Surge XT parameters if they don't want to use defaults.
    """
    print("\n🎹 Initial Synth State Configuration")
    print("---------------------------------")
    print("1. Use Default Init Patch")
    # print("2. Enter Parameters Manually") # Disabled for 38 params for now
    print("3. Generate Random Patch (🎲)")
    
    choice = input("Select option (1/3): ").strip()
    
    if choice == "3":
        return generate_random_surge_sound()
    
    # if choice == "2":
    #    ...
    
    return CURRENT_REF_PARAMS

def render_audio_result(full_params_array, midi_note=60, output_path="inference_result.wav"):
    """
    Renders the predicted parameters using Surge XT.
    Expects the FULL parameter vector (size 38).
    """
    sr = 44100 # Default sample rate
    if load_plugin is None:
        print("❌ Cannot render audio: dependencies missing.")
        return

    print(f"\n🎹 Rendering result audio to {output_path} (MIDI Note: {midi_note})...")
    
    if len(full_params_array) != FULL_SYNTH_PARAMS:
        print(f"❌ Error: render_audio_result expected {FULL_SYNTH_PARAMS} params, got {len(full_params_array)}")
        return
    
    # Clip parameters to 0-1 (Model space)
    # The model predicts normalized deltas on the [0,1] representation of the sub-range
    full_params_array = np.clip(full_params_array, 0.0, 1.0)
    
    # Prepare array for decoding (requires Synth + Note params slots)
    # We create a full zero array and fill the synth part
    full_arr = np.zeros(len(SURGE_SIMPLE_PARAM_SPEC))
    full_arr[:FULL_SYNTH_PARAMS] = full_params_array
    
    # Decode to Dict (Raw Values for VST)
    render_dict, _ = SURGE_SIMPLE_PARAM_SPEC.decode(full_arr)
    
    # === Force Fixed Topology ===
    # User Request: Set Filter 1 & 2 to Lowpass 24 dB (Value ~0.06)
    render_dict["a_filter_1_type"] = 0.04
    render_dict["a_filter_2_type"] = 0.04
    
    # Note: ParamSpec.decode handles the mapping from [0,1] back to [min, max]
    # And discrete/one-hot logic (Argmax) implicitly.

    # Load Plugin
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    try:
        plugin = load_plugin(plugin_path)
    except Exception as e:
        print(f"❌ Failed to load Surge XT at {plugin_path}: {e}")
        return

    # Render
    try:
        # Determine duration roughly from headers or fixed
        audio = render_params(
            plugin=plugin,
            params=render_dict,
            midi_note=midi_note, 
            velocity=100,
            note_start_and_end=(0.0, 3.0), # 3 second note
            signal_duration_seconds=4.0,   # 4 second file
            sample_rate=sr,
            channels=2,
            preset_path=None
        )

        # Normalize Audio (Peak Normalization to -1.0 dB / 0.9)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            target_peak = 0.9
            gain = target_peak / max_val
            audio = audio * gain
            print(f"🔊 Audio Normalized: Peak {max_val:.4f} -> {target_peak} (Gain: {gain:.2f}x)")
        
        write_wav(audio, output_path, sr, 2)
        print(f"✅ Rendered audio saved to {output_path}")
    except Exception as e:
        print(f"❌ Error during rendering: {e}")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Running on device: {device}")

    # --- 1. Load Model ---
    print(f"📂 Loading model from {MODEL_PATH}...")
    try:
        model = Vimtopoeia_AST(n_params=MODEL_PARAMS_COUNT).to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        # Check for size mismatch in FC layer explicitly to give better error
        if model.fc[-1].out_features != state_dict.get('fc.3.weight', torch.zeros(0)).shape[0]:
            print(f"❌ Error: Model structure mismatch. Code expects {MODEL_PARAMS_COUNT} params, but checkpoint has {state_dict['fc.3.weight'].shape[0]}.")
            print("Action: Please retrain the model or update the param spec.")
            return

        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- 1b. Get Reference Params (User Input) ---
    current_ref_params = get_surge_params_from_user() # Returns Full 38 Params
    
    # --- 1c. Pitch Detection ---
    # Detect Pitch first
    detected_pitch = get_pitch(TARGET_AUDIO_PATH)

    # --- 1d. Render Reference Audio (based on Start Params) ---
    print("\n🎧 Rendering Reference Audio (Start State)...")
    # This allows the user to hear what the "Before" state sounds like
    render_audio_result(current_ref_params, midi_note=detected_pitch, output_path="reference_from_params.wav")


    # --- 2. Process Audio ---
    # ... (Audio Loading Code omitted)
    print(f"🎤 Processing Target: {TARGET_AUDIO_PATH}")
    spec_target = load_and_process_audio(TARGET_AUDIO_PATH)
    
    generated_ref_path = "reference_from_params.wav"
    print(f"🎹 Processing Generated Reference: {generated_ref_path}")
    spec_ref = load_and_process_audio(generated_ref_path)

    if spec_target is None or spec_ref is None:
        print("❌ Aborting due to audio error.")
        return

    spec_target = spec_target.to(device)
    spec_ref = spec_ref.to(device)
    
    # Dummy One-Hot Category
    one_hot = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).to(device)

    # --- 3. Inference ---
    print("🧠 Model is thinking...")
    with torch.no_grad():
        predicted_delta = model(spec_target, spec_ref, one_hot)
    
    delta_values = predicted_delta.squeeze().cpu().numpy() # Shape: (22,)

    # --- 4. Calculate Final Params ---
    # Step A: extract active ref
    active_ref = current_ref_params[ACTIVE_PARAM_INDICES] # Shape: (22,)
    
    # Step B: Apply delta
    final_active = active_ref + delta_values
    
    # Step C: Reconstruct full vector
    final_full_params = current_ref_params.copy()
    final_full_params[ACTIVE_PARAM_INDICES] = final_active

    # --- 5. Print Report ---
    print("\n" + "="*80)
    print(f"{'PARAMETER':<30} | {'START':<5} + {'DELTA':<8} = {'RESULT':<6}")
    print("="*80)

    # Generate flattened name list for display
    flat_names = []
    for p in SURGE_SIMPLE_PARAM_SPEC.synth_params:
        if len(p) == 1:
            flat_names.append(p.name)
        else:
            # One-Hot expansion
            for i in range(len(p)):
                subval = p.values[i] if hasattr(p, 'values') else str(i)
                flat_names.append(f"{p.name}[{subval}]")
    
    # We iterate through indices of the Active set to print relevant changes
    # But it might be nice to print everything? No, only active ones matter for the model.
    # Let's print ONLY active params to reduce clutter
    
    for i in range(MODEL_PARAMS_COUNT):
        full_idx = ACTIVE_PARAM_INDICES[i]
        name = flat_names[full_idx]
        start = current_ref_params[full_idx]
        delta = delta_values[i]
        final = final_full_params[full_idx]
        
        final_clipped = max(0.0, min(1.0, final))
        
        indicator = " "
        if abs(delta) > 0.1: indicator = "🔸"
        if abs(delta) > 0.3: indicator = "🔥"
        
        print(f"{name:<30} | {start:.2f}  + {delta:+.4f} {indicator:<2} = {final_clipped:.3f}")
    
    print("="*80)
    print("✅ Done! Adjust your synth knobs to match the 'RESULT' column.")

    # Render Result (Using Full Params)
    render_audio_result(final_full_params, midi_note=detected_pitch, output_path="inference_result.wav")

if __name__ == "__main__":
    main()
