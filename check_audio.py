# check_audio.py
import torch
import torchaudio
import h5py
import glob
import random
import os
import hdf5plugin


def apply_spectral_convolution(audio, ir, mix_ratio=1.0):
    """Apply IR convolution in frequency domain (same as dataset.py)"""
    if audio.dim() == 1: 
        audio = audio.unsqueeze(0)
    if ir.dim() == 1: 
        ir = ir.unsqueeze(0)
    
    if audio.shape[0] != ir.shape[0]:
        ir = ir.expand(audio.shape[0], -1)

    n_step = audio.shape[-1]
    n_ir = ir.shape[-1]
    n_fft = n_step + n_ir - 1
    
    # FFT convolution
    audio_f = torch.fft.rfft(audio, n=n_fft)
    ir_f = torch.fft.rfft(ir, n=n_fft)
    convolved_f = audio_f * ir_f
    convolved = torch.fft.irfft(convolved_f, n=n_fft)
    convolved = convolved[..., :n_step]
    
    # Normalize
    original_rms = torch.sqrt(torch.mean(audio**2))
    convolved_rms = torch.sqrt(torch.mean(convolved**2))
    convolved = convolved * (original_rms / (convolved_rms + 1e-8))
    
    # Mix
    if mix_ratio < 1.0:
        return (1 - mix_ratio) * audio + mix_ratio * convolved
    
    return convolved


# 配置
H5_PATH = "/Users/wanghuixi/vimtopoeia/dataset_4k_pair.h5"
IR_DIR = "/Users/wanghuixi/vimtopoeia/V1_outputs/v1_input_vocals"

# 1. Load IR files
ir_files = glob.glob(os.path.join(IR_DIR, "*.wav"))
print(f"Found {len(ir_files)} IR files.")

ir_cache = []
for ir_path in ir_files:
    ir_waveform, ir_sr = torchaudio.load(ir_path)
    if ir_waveform.shape[0] > 1:
        ir_waveform = torch.mean(ir_waveform, dim=0, keepdim=True)
    # Resample to 44100 if needed
    if ir_sr != 44100:
        resampler = torchaudio.transforms.Resample(ir_sr, 44100)
        ir_waveform = resampler(ir_waveform)
    ir_cache.append(ir_waveform)

print(f"Loaded {len(ir_cache)} IR files.")

# 2. Open H5 file and extract audio
os.makedirs(" debug_audio", exist_ok=True)

print("Generating 5 test samples...")
with h5py.File(H5_PATH, 'r') as f:
    for i in range(5):
        # Random sample
        idx = random.randint(0, f["target_audio"].shape[0] - 1)
        
        # Load raw audio from h5
        target_audio = torch.from_numpy(f["target_audio"][idx]).float()
        ref_audio = torch.from_numpy(f["reference_audio"][idx]).float()
        
        # Ensure [C, T] format
        if target_audio.dim() == 1:
            target_audio = target_audio.unsqueeze(0)
        if ref_audio.dim() == 1:
            ref_audio = ref_audio.unsqueeze(0)
        
        # Apply IR convolution to target
        ir_sample = random.choice(ir_cache)
        mix_ratio = random.uniform(0.6, 1.0)
        target_augmented = apply_spectral_convolution(target_audio, ir_sample, mix_ratio)
        
        # Save audio files
        torchaudio.save(f"debug_audio/test_{i}_target_clean.wav", target_audio, 44100)
        torchaudio.save(f"debug_audio/test_{i}_target_augmented.wav", target_augmented, 44100)
        torchaudio.save(f"debug_audio/test_{i}_ref_clean.wav", ref_audio, 44100)
        
        print(f"Saved sample {i} (idx={idx}) - target clean, augmented, and ref clean")

print(f"\nDone! Saved 15 audio files to debug_audio/")
print("Listen to compare clean vs augmented versions.")