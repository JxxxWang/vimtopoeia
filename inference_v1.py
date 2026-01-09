import torch
import numpy as np
import librosa
import soundfile as sf
from pedalboard import VST3Plugin
from skimage.transform import resize
import sys
from pathlib import Path
import torchcrepe

# Add current directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from model_training.model import AudioParamVAE
from data_generation.surge_xt_param_spec import SURGE_XT_PARAM_SPEC as PS
from data_generation.core import render_params

# === Configuration ===
MODEL_PATH = "model_training/vae_synth.pth"
INPUT_AUDIO = "untitled_007.wav"
OUTPUT_AUDIO = "output_synth_carlos_007.wav"
VST_PATH = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
SAMPLE_RATE = 44100
DURATION = 4.0

def extract_mel_spec(audio, sr):
    """
    Extract Mel Spectrogram matching the training data format: [1, 2, 128, 201]
    """
    # Ensure audio is stereo (2 channels)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    elif audio.shape[0] > 2:
        audio = audio[:2, :]
    
    # 1. Compute Mel Spec
    n_fft = int(0.025 * sr)
    hop_length = int(sr / 100.0)
    
    # librosa.feature.melspectrogram handles multi-channel input (C, N) -> (C, n_mels, T)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # 2. Resize to (2, 128, 201)
    # skimage.transform.resize expects input and output shapes to match in rank
    log_mel_resized = resize(log_mel, (2, 128, 201), anti_aliasing=True)
    
    # Convert to Tensor [1, 2, 128, 201]
    return torch.tensor(log_mel_resized, dtype=torch.float32).unsqueeze(0)

import torchcrepe

def get_pitch(audio, sr):
    """
    Uses CREPE to get the most dominant pitch (f0) and convert to MIDI.
    """
    device = "cpu" # CREPE is fast enough on M3 CPU for inference
    
    # Ensure mono for pitch detection
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    
    # 1. Prepare audio for CREPE (Shape: [1, samples])
    audio_tensor = torch.tensor(audio).unsqueeze(0).float().to(device)
    
    # 2. Run Pitch Tracking
    # hop_length=100 corresponds to roughly 10ms granularity
    f0, confidence = torchcrepe.predict(
        audio_tensor, 
        sr, 
        hop_length=320, # higher hop = faster, lower = more detailed
        fmin=50, 
        fmax=1000, 
        model='tiny',   # 'tiny' is super fast and accurate enough for MIDI
        decoder=torchcrepe.decode.viterbi, 
        return_periodicity=True,
        device=device
    )
    
    # 3. Filter silence/noise (Confidence check)
    # If confidence is low, ignore those frames
    valid_f0 = f0[confidence > 0.4]
    
    if len(valid_f0) == 0:
        print("Warning: No pitch detected. Defaulting to C3.")
        return 60
        
    # 4. Get the median pitch (Average pitch of the clip)
    median_f0 = torch.median(valid_f0).item()
    
    # 5. Convert to MIDI
    midi_note = int(librosa.hz_to_midi(median_f0))
    return midi_note

def main():
    # 1. Load Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    # Calculate n_params using a valid sample
    dummy_synth, dummy_note = PS.sample()
    encoded_len = len(PS.encode(dummy_synth, dummy_note))
    print(f"Model Input/Output Dimension: {encoded_len}")
    
    model = AudioParamVAE(n_params=encoded_len).to(device)
    
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Load and Process Audio
    print(f"Processing {INPUT_AUDIO}...")
    if not Path(INPUT_AUDIO).exists():
        # Create a dummy test file if it doesn't exist
        print(f"Creating dummy {INPUT_AUDIO}...")
        dummy_audio = np.random.uniform(-0.5, 0.5, int(SAMPLE_RATE * DURATION))
        sf.write(INPUT_AUDIO, dummy_audio, SAMPLE_RATE)

    audio, sr = librosa.load(INPUT_AUDIO, sr=SAMPLE_RATE, duration=DURATION, mono=False)
    
    # Extract features
    mel_tensor = extract_mel_spec(audio, sr).to(device)
    midi_note = get_pitch(audio, sr)
    print(f"Detected Pitch: MIDI {midi_note}")

    # 3. Inference
    with torch.no_grad():
        recon_params, mu, logvar = model(mel_tensor)
    
    predicted_array = recon_params.cpu().numpy()[0]
    
    # 4. Decode Parameters
    print("Decoding parameters...")
    decoded_synth_params, decoded_note_params = PS.decode(predicted_array)
    
    for name, val in decoded_synth_params.items():
        print(f"  {name}: {val}")

    # 5. Render with Surge XT
    print("Rendering with Surge XT...")
    
    # Ensure fixed params are set (e.g. turn off other oscillators)
    fixed_params = {
        "a_osc_2_volume": 0.0,
        "a_osc_3_volume": 0.0,
        "b_osc_1_volume": 0.0,
        "global_volume": 1.0,
        "a_filter_1_type": 0.0,
    }
    render_dict = {**fixed_params, **decoded_synth_params}
    
    if not Path(VST_PATH).exists():
        print(f"Warning: VST not found at {VST_PATH}. Skipping rendering.")
        return

    plugin = VST3Plugin(VST_PATH)
    # plugin.load_preset("presets/surge-base.vstpreset") # Optional

    synth_audio = render_params(
        plugin,
        render_dict,
        midi_note=midi_note,
        velocity=100,
        note_start_and_end=(0.1, DURATION-0.1),
        signal_duration_seconds=DURATION,
        sample_rate=SAMPLE_RATE,
        channels=2
    )

    # 6. Save Output
    sf.write(OUTPUT_AUDIO, synth_audio.T, SAMPLE_RATE)
    print(f"Saved to {OUTPUT_AUDIO}! Listen to it.")

if __name__ == "__main__":
    main()
