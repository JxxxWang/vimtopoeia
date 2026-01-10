import torch
import torchaudio
import h5py
import numpy as np
import random
from torch.utils.data import Dataset

# ==========================================
# 1. Helper: Spectral Convolution
# ==========================================
def apply_spectral_convolution(audio, ir, mix_ratio=1.0):
    """
    Applies spectral convolution (Linear Convolution via FFT)
    Audio: [C, T]
    IR: [C, K]
    """
    if audio.dim() == 1: audio = audio.unsqueeze(0)
    if ir.dim() == 1: ir = ir.unsqueeze(0)
    
    # 确保通道匹配
    if audio.shape[0] != ir.shape[0]:
        ir = ir.expand(audio.shape[0], -1)

    n_step = audio.shape[-1]
    n_ir = ir.shape[-1]
    
    n_fft = n_step + n_ir - 1
    
    # FFT
    audio_f = torch.fft.rfft(audio, n=n_fft)
    ir_f = torch.fft.rfft(ir, n=n_fft)
    
    # Multiply
    convolved_f = audio_f * ir_f
    
    # IFFT
    convolved = torch.fft.irfft(convolved_f, n=n_fft)
    
    # Crop to original length
    convolved = convolved[..., :n_step]
    
    # Normalize (Match RMS)
    original_rms = torch.sqrt(torch.mean(audio**2))
    convolved_rms = torch.sqrt(torch.mean(convolved**2))
    convolved = convolved * (original_rms / (convolved_rms + 1e-8))
    
    # Mix
    if mix_ratio < 1.0:
        return (1 - mix_ratio) * audio + mix_ratio * convolved
    
    return convolved

# ==========================================
# 2. Dataset Class (Fixed: All 44.1kHz)
# ==========================================
class VSTDataset(Dataset):
    def __init__(self, h5_path, target_sr=44100, max_len_frames=1024, 
                 subset='train', val_split=0.1, ir_files_list=None):
        """
        Args:
            h5_path: Path to HDF5 dataset file
            target_sr: 44100 (Defaulting to your native SR)
            max_len_frames: Maximum spectrogram length
        """
        self.h5_path = h5_path
        self.target_sr = target_sr # Now 44100
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
        
        # 3. Preload IR files
        self.ir_cache = []
        if ir_files_list is not None and self.is_train:
            print(f"Loading {len(ir_files_list)} impulse response files...")
            for ir_path in ir_files_list:
                try:
                    # Load IR
                    ir_waveform, ir_sr = torchaudio.load(ir_path)
                    
                    # Convert to mono
                    if ir_waveform.shape[0] > 1:
                        ir_waveform = torch.mean(ir_waveform, dim=0, keepdim=True)
                        
                    # 只在不匹配时才 Resample (既然你是 44.1k，这里通常不会触发)
                    if ir_sr != self.target_sr:
                        resampler = torchaudio.transforms.Resample(ir_sr, self.target_sr)
                        ir_waveform = resampler(ir_waveform)
                        
                    self.ir_cache.append(ir_waveform)
                except Exception as e:
                    print(f"Warning: Failed to load IR {ir_path}: {e}")
            print(f"Loaded {len(self.ir_cache)} IR files into cache (Target SR: {self.target_sr}).")
        
        print(f"Dataset initialized: subset='{subset}', length={self.length}")

        # 4. Transforms
        # 移除了 self.resampler，因为我们全程 44.1k
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr, # 44100
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=128
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

        # 2. Augmentation: Spectral Convolution (at full 44.1k resolution)
        if self.is_train and self.ir_cache and random.random() < 0.8:
            ir_sample = random.choice(self.ir_cache)
            mix = random.uniform(0.6, 1.0)
            target_audio = apply_spectral_convolution(target_audio, ir_sample, mix_ratio=mix)

        # 3. Load Params & Label
        p_target = torch.from_numpy(self.h5_file["target_param_array"][actual_idx]).float()
        p_ref = torch.from_numpy(self.h5_file["reference_param_array"][actual_idx]).float()
        delta = p_target - p_ref
        
        # 4. Generate Spectrograms
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
        
        # Generate Mel Spec (Assuming input is 44.1k now)
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