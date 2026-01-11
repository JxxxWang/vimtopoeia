import torch
import torchaudio
import h5py
import numpy as np
from torch.utils.data import Dataset
import hdf5plugin

# ==========================================
# Dataset Class (No Spectral Convolution)
# ==========================================
class VSTDataset(Dataset):
    def __init__(self, h5_path, target_sr=44100, max_len_frames=1024, 
                 subset='train', val_split=0.1):
        """
        Args:
            h5_path: Path to HDF5 dataset file
            target_sr: 44100 (Defaulting to your native SR)
            max_len_frames: Maximum spectrogram length
        """
        self.h5_path = h5_path
        self.target_sr = target_sr
        self.max_len_frames = max_len_frames
        self.subset = subset
        self.is_train = (subset == 'train')
        
        # 1. Init: Get total length
        with h5py.File(h5_path, 'r') as f:
            total_length = f["target_param_array"].shape[0]
        
        # 2. Calculate split
        split_idx = int(total_length * (1 - val_split))
        
        if self.is_train:
            self.start_idx = 0
            self.length = split_idx
        elif subset == 'val':
            self.start_idx = split_idx
            self.length = total_length - split_idx
        else:
            raise ValueError(f"subset must be 'train' or 'val', got '{subset}'")
            
        self.h5_file = None
        
        print(f"Dataset initialized: subset='{subset}', length={self.length}")

        # 3. Transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=128,
            f_min=20.0
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        actual_idx = self.start_idx + idx
        
        # 1. Load Audio (Original 44.1k)
        target_audio = torch.from_numpy(self.h5_file["target_audio"][actual_idx]).float()
        ref_audio = torch.from_numpy(self.h5_file["reference_audio"][actual_idx]).float()
        
        if target_audio.dim() == 1: target_audio = target_audio.unsqueeze(0)
        if ref_audio.dim() == 1: ref_audio = ref_audio.unsqueeze(0)

        # 2. Load Params & Label
        p_target = torch.from_numpy(self.h5_file["target_param_array"][actual_idx]).float()
        p_ref = torch.from_numpy(self.h5_file["reference_param_array"][actual_idx]).float()
        delta = p_target - p_ref
        
        # 3. Generate Spectrograms
        spec_target = self._spec_and_norm(target_audio)
        spec_ref = self._spec_and_norm(ref_audio)
        
        one_hot = torch.tensor([0.0, 1.0, 0.0]) 

        return {
            "spec_target": spec_target,
            "spec_ref": spec_ref,
            "delta": delta,
            "one_hot": one_hot
        }

    def _spec_and_norm(self, waveform):
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Generate Mel Spec
        spec = self.mel_transform(waveform)
        spec = self.db_transform(spec)
        
        # [1, n_mels, time] -> [time, n_mels]
        spec = spec.squeeze(0).transpose(0, 1)
        
        # Padding
        if spec.shape[0] < self.max_len_frames:
            padding = torch.zeros(self.max_len_frames - spec.shape[0], 128)
            spec = torch.cat([spec, padding], dim=0)
        else:
            spec = spec[:self.max_len_frames, :]
            
        # Normalization
        mean = -26.538128995895384
        std = 39.86343679428101
        spec = (spec - mean) / (std * 2)
        
        return spec
