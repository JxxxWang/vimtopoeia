#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute normalization statistics (mean and std) for Vimtopoeia dataset.
SAFE VERSION: Includes RMS Normalization to match VimSketch volume.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
import hdf5plugin
import torchaudio

# === CONFIGURATION ===
# Target RMS matches the "loudness" we will force on both Vocals and Synths
TARGET_RMS = 0.1  

class VSTDatasetForStats(torch.utils.data.Dataset):
    def __init__(self, h5_path, target_sr=16000, max_len_frames=1024):
        self.h5_path = h5_path
        self.target_sr = target_sr
        self.max_len_frames = max_len_frames
        
        with h5py.File(h5_path, 'r') as f:
            self.length = f["target_audio"].shape[0]
        self.h5_file = None
        
        # 1. Resample to 16k (Correct for AST)
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=target_sr)
        
        # 2. Mel Spectrogram (Standard AST settings)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_fft=1024, win_length=400, hop_length=160, 
            n_mels=128, f_min=0, f_max=8000,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # Load raw audio
        waveform = torch.from_numpy(self.h5_file["target_audio"][idx]).float()
        
        # Process
        spec = self._process_audio(waveform)
        return spec

    def _process_audio(self, waveform):
        # Ensure correct dimensions
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # === CRITICAL STEP: RMS NORMALIZATION ===
        # We normalize the volume BEFORE resampling or spectrograms
        # This ensures our Synths have the same energy profile as our future Vocals
        current_rms = torch.sqrt(torch.mean(waveform**2))
        if current_rms > 0:
            waveform = waveform * (TARGET_RMS / (current_rms + 1e-9))
        # ========================================

        # Now proceed with standard AST processing
        waveform = self.resampler(waveform)
        spec = self.mel_transform(waveform)
        spec = self.amp_to_db(spec)
        spec = spec.squeeze(0).transpose(0, 1)
        
        # Padding
        if spec.shape[0] < self.max_len_frames:
            padding = torch.zeros(self.max_len_frames - spec.shape[0], 128)
            spec = torch.cat([spec, padding], dim=0)
        else:
            spec = spec[:self.max_len_frames, :]
        
        return spec

def compute_normalization_stats(h5_path: str, batch_size: int = 100, num_workers: int = 4):
    print(f"Computing RMS-Aware normalization stats for: {h5_path}")
    
    dataset = VSTDatasetForStats(h5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # We use accumulators for correct math
    total_sum = 0
    total_sq_sum = 0
    total_count = 0
    
    print("\nProcessing batches...")
    for batch_specs in tqdm(dataloader):
        # batch_specs: [batch, time, freq]
        # Flatten everything
        batch_flat = batch_specs.flatten()
        
        total_sum += torch.sum(batch_flat).item()
        total_sq_sum += torch.sum(batch_flat ** 2).item()
        total_count += batch_flat.numel()
    
    # Calculate Global Stats
    final_mean = total_sum / total_count
    final_var = (total_sq_sum / total_count) - (final_mean ** 2)
    final_std = np.sqrt(final_var)
    
    print(f"\n{'='*60}")
    print(f"FINAL STATS (With RMS Normalization)")
    print(f"{'='*60}")
    print(f"Mean: {final_mean:.6f}")
    print(f"Std:  {final_std:.6f}")
    print(f"{'='*60}")
    print("Use these in dataset_dual.py!")
    
    return final_mean, final_std

if __name__ == "__main__":
    h5_file = "dataset_60k.h5" # Make sure this matches your file
    compute_normalization_stats(h5_file, num_workers=0)