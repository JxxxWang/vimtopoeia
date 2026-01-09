#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute normalization statistics (mean and std) for Vimtopoeia dataset.
Based on AST's get_norm_stats.py approach.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the dataset but we'll modify it to skip normalization
import h5py
import hdf5plugin  # Required for compressed HDF5 files
import torchaudio


class VSTDatasetForStats(torch.utils.data.Dataset):
    """Dataset version that skips normalization to compute raw stats."""
    
    def __init__(self, h5_path, target_sr=16000, max_len_frames=1024):
        self.h5_path = h5_path
        self.target_sr = target_sr
        self.max_len_frames = max_len_frames
        
        with h5py.File(h5_path, 'r') as f:
            self.length = f["target_audio"].shape[0]
            
        self.h5_file = None
        
        print(f"Dataset initialized for stats computation. Length: {self.length}")
        
        # Transforms
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_fft=1024, win_length=400, hop_length=160, 
            n_mels=128, f_min=0, f_max=8000,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # Load both target and reference audio
        waveform_target = torch.from_numpy(self.h5_file["target_audio"][idx]).float()
        waveform_ref = torch.from_numpy(self.h5_file["reference_audio"][idx]).float()
        
        # Process both (without augmentation, without normalization)
        spec_target = self._process_audio(waveform_target)
        spec_ref = self._process_audio(waveform_ref)
        
        # Return both spectrograms to compute stats on all audio data
        return torch.stack([spec_target, spec_ref], dim=0)

    def _process_audio(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = self.resampler(waveform)
        spec = self.mel_transform(waveform)
        spec = self.amp_to_db(spec)
        spec = spec.squeeze(0).transpose(0, 1)
        
        if spec.shape[0] < self.max_len_frames:
            padding = torch.zeros(self.max_len_frames - spec.shape[0], 128)
            spec = torch.cat([spec, padding], dim=0)
        else:
            spec = spec[:self.max_len_frames, :]
        
        # NO NORMALIZATION HERE - we need raw stats
        return spec


def compute_normalization_stats(h5_path: str, batch_size: int = 100, num_workers: int = 4):
    """
    Compute mean and standard deviation for the dataset spectrograms.
    
    Args:
        h5_path: Path to the HDF5 dataset file
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers
        
    Returns:
        tuple: (mean, std) computed across all spectrograms
    """
    print(f"Computing normalization stats for: {h5_path}")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")
    
    dataset = VSTDatasetForStats(h5_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    means = []
    stds = []
    
    print("\nProcessing batches...")
    for batch_specs in tqdm(dataloader, desc="Computing stats"):
        # batch_specs shape: [batch_size, 2, max_len_frames, 128]
        # Flatten to compute stats across all spectrograms
        batch_specs = batch_specs.reshape(-1, batch_specs.shape[-2], batch_specs.shape[-1])
        
        cur_mean = torch.mean(batch_specs)
        cur_std = torch.std(batch_specs)
        
        means.append(cur_mean.item())
        stds.append(cur_std.item())
        
        if len(means) % 10 == 0:
            print(f"  Batch {len(means)}: mean={cur_mean.item():.6f}, std={cur_std.item():.6f}")
    
    final_mean = np.mean(means)
    final_std = np.mean(stds)
    
    print(f"\n{'='*60}")
    print(f"FINAL NORMALIZATION STATISTICS")
    print(f"{'='*60}")
    print(f"Mean: {final_mean:.7f}")
    print(f"Std:  {final_std:.7f}")
    print(f"{'='*60}")
    print(f"\nTo use in your dataset, update the normalization lines to:")
    print(f"  mean = {final_mean}")
    print(f"  std = {final_std}")
    print(f"  spec = (spec - mean) / (std * 2)")
    print(f"\nNote: Dividing by (std * 2) aims for ~0.5 std in normalized data,")
    print(f"      which is the convention used by AST.")
    
    return final_mean, final_std


if __name__ == "__main__":
    # Path to your dataset - using pathlib for cross-platform compatibility
    project_root = Path(__file__).parent.parent
    h5_file = project_root / "dataset_4k_pair.h5"
    
    if not h5_file.exists():
        print(f"Error: Dataset file not found at {h5_file}")
        print("Available .h5 files in project root:")
        for f in project_root.glob("*.h5"):
            print(f"  - {f.name}")
        exit(1)
    
    # Compute stats
    # Note: num_workers=0 to avoid HDF5 multiprocessing issues
    mean, std = compute_normalization_stats(
        str(h5_file),
        batch_size=100,  # Adjust based on your memory
        num_workers=0     # Must be 0 for HDF5 compatibility
    )
