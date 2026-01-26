"""
Compute normalization statistics (mean and std) for M2 Surge dataset.
Run this ONCE before training to get mean/std values.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

SAMPLE_RATE = 44100

def compute_stats(h5_path, num_samples=5000):
    """
    Compute mean and std of mel spectrograms from dataset.
    
    Args:
        h5_path: Path to HDF5 file
        num_samples: Number of samples to use for stats (default: 5000)
    """
    print(f"Computing normalization stats from: {h5_path}")
    print(f"Using {num_samples} samples...")
    
    mel_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=1024,
            n_mels=64
        ),
        T.AmplitudeToDB()
    )
    
    all_values = []
    
    with h5py.File(h5_path, 'r') as f:
        total_samples = f['audio'].shape[0]
        actual_samples = min(num_samples, total_samples)
        
        print(f"Total samples in dataset: {total_samples}")
        print(f"Computing stats from: {actual_samples} samples")
        
        for idx in tqdm(range(actual_samples)):
            # Load audio
            audio_np = f['audio'][idx]  # Shape: [2, 176400]
            
            # Convert to mono
            audio_mono = audio_np.mean(axis=0, keepdims=True)  # [1, 176400]
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_mono).float()
            
            # Compute mel spec
            with torch.no_grad():
                mel = mel_transform(audio_tensor)
            
            # Collect all values
            all_values.append(mel.numpy().flatten())
    
    # Concatenate all values
    all_values = np.concatenate(all_values)
    
    # Compute statistics
    mean = np.mean(all_values)
    std = np.std(all_values)
    
    print("\n" + "="*60)
    print("NORMALIZATION STATISTICS")
    print("="*60)
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print("="*60)
    print("\nAdd these to your training script:")
    print(f"NORM_MEAN = {mean}")
    print(f"NORM_STD = {std}")
    
    return mean, std


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, required=True, help='Path to training H5 file')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to use')
    
    args = parser.parse_args()
    
    compute_stats(args.h5_path, args.num_samples)
