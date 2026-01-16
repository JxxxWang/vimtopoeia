"""
Dual-dataset loader: SurgeXT (H5) + VimSketch (WAV pairs).
"""
from pathlib import Path
from typing import Dict, List, Tuple
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio


class DualDataset(Dataset):
    """
    Combines SurgeXT data from H5 with VimSketch vocal/synth WAV pairs.
    
    Args:
        h5_path: Path to SurgeXT HDF5 file
        vimsketch_root: Root folder containing 'vocal' and 'synth' subfolders
        shuffle_surge: Whether to shuffle SurgeXT indices (default: True)
        seed: Random seed for shuffling (default: 42)
    """
    
    def __init__(
        self,
        h5_path: str | Path,
        vimsketch_root: str | Path,
        shuffle_surge: bool = True,
        seed: int = 42
    ):
        self.h5_path = Path(h5_path)
        self.vimsketch_root = Path(vimsketch_root)
        
        # Load SurgeXT dataset size
        with h5py.File(self.h5_path, 'r') as f:
            self.surge_size = len(f['audio'])
        
        # Create shuffled indices for SurgeXT
        self.surge_indices = list(range(self.surge_size))
        if shuffle_surge:
            random.seed(seed)
            random.shuffle(self.surge_indices)
        
        # Build VimSketch pairs
        self.pairs = self._build_sketch_pairs()
        
        print(f"DualDataset initialized:")
        print(f"  SurgeXT samples: {self.surge_size}")
        print(f"  VimSketch pairs: {len(self.pairs)}")
    
    def _build_sketch_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Build list of (vocal_path, synth_path) tuples.
        
        Vocals: 5 digits before first underscore (e.g., '01832_...')
        Synths: 3 digits before first underscore (e.g., '098_...')
        
        Multiple vocals can map to the same synth by matching the suffix.
        """
        vocal_dir = self.vimsketch_root / 'vocal'
        synth_dir = self.vimsketch_root / 'synth'
        
        if not vocal_dir.exists():
            raise ValueError(f"Vocal directory not found: {vocal_dir}")
        if not synth_dir.exists():
            raise ValueError(f"Synth directory not found: {synth_dir}")
        
        # Get all vocal and synth files
        vocal_files = sorted(vocal_dir.glob('*.wav'))
        synth_files = sorted(synth_dir.glob('*.wav'))
        
        # Build mapping: suffix -> synth_path
        synth_map = {}
        for synth_path in synth_files:
            # Extract suffix after 3-digit prefix
            # e.g., '098_097Music_...' -> '097Music_...'
            name = synth_path.stem
            parts = name.split('_', 1)
            if len(parts) == 2 and len(parts[0]) == 3 and parts[0].isdigit():
                suffix = parts[1]
                synth_map[suffix] = synth_path
        
        # Pair each vocal with its corresponding synth
        pairs = []
        for vocal_path in vocal_files:
            # Extract suffix after 5-digit prefix
            # e.g., '01832_097Music_...' -> '097Music_...'
            name = vocal_path.stem
            parts = name.split('_', 1)
            if len(parts) == 2 and len(parts[0]) == 5 and parts[0].isdigit():
                suffix = parts[1]
                if suffix in synth_map:
                    pairs.append((vocal_path, synth_map[suffix]))
                else:
                    print(f"Warning: No matching synth for vocal '{name}'")
        
        if len(pairs) == 0:
            raise ValueError("No valid vocal/synth pairs found!")
        
        return pairs
    
    def __len__(self) -> int:
        """Dataset size is determined by SurgeXT (larger dataset)."""
        return self.surge_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | np.ndarray]:
        """
        Load samples from both datasets.
        
        Args:
            idx: Index for SurgeXT dataset
            
        Returns:
            Dict with keys:
                - surge_audio: (channels, samples) or (samples,)
                - surge_params: (num_params,)
                - sketch_vocal: (channels, samples) or (samples,)
                - sketch_synth: (channels, samples) or (samples,)
        """
        # Load SurgeXT sample
        surge_idx = self.surge_indices[idx]
        with h5py.File(self.h5_path, 'r') as f:
            surge_audio = f['audio'][surge_idx]
            surge_params = f['parameters'][surge_idx]
        
        # Load VimSketch pair (loop over smaller dataset)
        sketch_idx = idx % len(self.pairs)
        vocal_path, synth_path = self.pairs[sketch_idx]
        
        sketch_vocal, sr_vocal = torchaudio.load(vocal_path)
        sketch_synth, sr_synth = torchaudio.load(synth_path)
        
        # Squeeze if mono
        if sketch_vocal.shape[0] == 1:
            sketch_vocal = sketch_vocal.squeeze(0)
        if sketch_synth.shape[0] == 1:
            sketch_synth = sketch_synth.squeeze(0)
        
        return {
            'surge_audio': torch.from_numpy(surge_audio) if isinstance(surge_audio, np.ndarray) else surge_audio,
            'surge_params': torch.from_numpy(surge_params) if isinstance(surge_params, np.ndarray) else surge_params,
            'sketch_vocal': sketch_vocal,
            'sketch_synth': sketch_synth,
        }


if __name__ == '__main__':
    # Quick test
    h5_path = Path(__file__).parent.parent.parent / 'dataset_60k.h5'
    vimsketch_root = Path(__file__).parent.parent.parent / 'vimsketch_synth'
    
    if h5_path.exists() and vimsketch_root.exists():
        dataset = DualDataset(h5_path, vimsketch_root)
        print(f"\nLoading sample 0...")
        sample = dataset[0]
        print(f"  surge_audio: {sample['surge_audio'].shape}")
        print(f"  surge_params: {sample['surge_params'].shape}")
        print(f"  sketch_vocal: {sample['sketch_vocal'].shape}")
        print(f"  sketch_synth: {sample['sketch_synth'].shape}")
    else:
        print("Test data not found. Skipping test.")
