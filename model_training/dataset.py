import torch
import torchaudio
import h5py
import hdf5plugin
import numpy as np
from torch.utils.data import Dataset

class VSTDataset(Dataset):
    def __init__(self, h5_path, target_sr=16000, max_len_frames=1024):
        self.h5_path = h5_path
        self.target_sr = target_sr
        self.max_len_frames = max_len_frames
        
        self.h5_file = h5py.File(h5_path, 'r')
        self.length = self.h5_file["target_param_array"].shape[0]
        
        # === 关键：定义 Active Indices (基于你提供的 ParamSpec) ===
        # 我们跳过了 Index 16 (Osc Type), 17 (Pitch), 23 (Osc Vol), 24-27 (Voices), 29+ (LFOs)
        self.active_indices = [
            0, 1, 2, 3,    # Amp Env (A, D, R, S)
            4, 5, 6,       # Filter 1 (Cut, Mod, Res)
            7, 8, 9,       # Filter 2 (Cut, Mod, Res)
            10, 11, 12, 13,# Filter Env (A, D, R, S)
            14,            # Highpass
            15,            # Noise Volume (重要!)
            18, 19, 20,    # Osc Shapes (Saw, Pulse, Tri)
            21, 22,        # Osc Mods (Width, Sync)
            28             # Unison Detune (非常重要!)
        ]
        
        # 打印一下，让你确认 output_dim 应该是多少
        print(f"Dataset initialized. Filtering params down to {len(self.active_indices)} active params.")
        
        # AST 预处理
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            win_length=400,
            hop_length=160,
            n_mels=128,
            f_min=0,
            f_max=8000,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. 读取音频
        waveform_target = torch.from_numpy(self.h5_file["target_audio"][idx]).float()
        waveform_ref = torch.from_numpy(self.h5_file["reference_audio"][idx]).float()
        
        # 2. 读取参数 (完整版)
        p_target_full = torch.from_numpy(self.h5_file["target_param_array"][idx]).float()
        p_ref_full = torch.from_numpy(self.h5_file["reference_param_array"][idx]).float()
        
        # 3. 参数切片 (Slicing) -> 只取 22 个核心参数
        p_target = p_target_full[self.active_indices]
        p_ref = p_ref_full[self.active_indices]
        
        # 4. 计算 Label (Delta)
        delta = p_target - p_ref
        
        # 5. 音频处理
        spec_target = self._process_audio(waveform_target)
        spec_ref = self._process_audio(waveform_ref)
        
        # 6. 伪造 One-Hot (为了保持 Model 接口不变)
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
            
        # ⚠️ 重要: 如果你的 WAV 是 44.1k，必须打开这一行
        resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
        waveform = resampler(waveform)

        spec = self.mel_transform(waveform)
        spec = self.amp_to_db(spec)
        spec = spec.squeeze(0).transpose(0, 1)
        
        if spec.shape[0] < self.max_len_frames:
            padding = torch.zeros(self.max_len_frames - spec.shape[0], 128)
            spec = torch.cat([spec, padding], dim=0)
        else:
            spec = spec[:self.max_len_frames, :]
            
        # AST Normalization (AudioSet statistics)
        # Target: Mean 0, Std 0.5
        mean = -4.2677393
        std = 4.5689974
        spec = (spec - mean) / (std * 2)
        
        return spec