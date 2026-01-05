import torch
import torchaudio
import h5py
import hdf5plugin
import numpy as np
import random
from torch.utils.data import Dataset

class VSTDataset(Dataset):
    def __init__(self, h5_path, target_sr=16000, max_len_frames=1024):
        self.h5_path = h5_path
        self.target_sr = target_sr
        self.max_len_frames = max_len_frames
        
        # 1. Init: Just get length, then CLOSE immediately
        with h5py.File(h5_path, 'r') as f:
            self.length = f["target_param_array"].shape[0]
            
        self.h5_file = None # Placeholder
        
        # 2. Indices
        self.active_indices = [
            0, 1, 2, 3,    # Amp Env
            4, 5, 6,       # Filter 1
            7, 8, 9,       # Filter 2
            10, 11, 12, 13,# Filter Env
            14,            # Highpass
            15,            # Noise Volume
            18, 19, 20,    # Osc Shapes
            21, 22,        # Osc Mods
            28             # Unison Detune
        ]
        
        print(f"Dataset initialized. Length: {self.length}. Params: {len(self.active_indices)}")
        
        # 3. Transforms
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_fft=1024, win_length=400, hop_length=160, 
            n_mels=128, f_min=0, f_max=8000,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    # === ⚠️ 关键修复：添加这个方法解决 Pickle Error ===
    def __getstate__(self):
        # 当 DataLoader 试图复制这个 Dataset 到子进程时，Python 会调用这个方法
        state = self.__dict__.copy()
        # 强制把 h5_file 设为 None，确保不传输打开的文件句柄
        state['h5_file'] = None
        return state
    # ===============================================

    def __len__(self):
        return self.length

    def _augment_audio(self, waveform):
        gain = random.uniform(0.5, 1.2)
        waveform = waveform * gain
        if random.random() < 0.5: 
            noise = torch.randn_like(waveform)
            snr_db = random.uniform(25, 45) 
            waveform = torchaudio.functional.add_noise(waveform, noise, torch.tensor([snr_db]))
        return waveform

    def __getitem__(self, idx):
        # Lazy Loading
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        waveform_target = torch.from_numpy(self.h5_file["target_audio"][idx]).float()
        waveform_ref = torch.from_numpy(self.h5_file["reference_audio"][idx]).float()
        
        waveform_target = self._augment_audio(waveform_target)

        p_target_full = torch.from_numpy(self.h5_file["target_param_array"][idx]).float()
        p_ref_full = torch.from_numpy(self.h5_file["reference_param_array"][idx]).float()
        
        p_target = p_target_full[self.active_indices]
        p_ref = p_ref_full[self.active_indices]
        
        delta = p_target - p_ref
        
        spec_target = self._process_audio(waveform_target)
        spec_ref = self._process_audio(waveform_ref)
        
        one_hot = torch.tensor([0.0, 1.0, 0.0]) 

        return {
            "spec_target": spec_target,
            "spec_ref": spec_ref,
            "delta": delta,
            "one_hot": one_hot
        }

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
            
        mean = -4.2677393
        std = 4.5689974
        spec = (spec - mean) / (std * 2)
        
        return spec