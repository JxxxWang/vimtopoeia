import torch
import torchaudio
import h5py
import numpy as np
import random
from torch.utils.data import Dataset
import hdf5plugin

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
# 1b. Helper: Audio Feature Extraction
# ==========================================
def get_audio_features(waveform, sample_rate):
    """
    Extract spectral centroid and bandwidth from audio.
    
    Args:
        waveform: Audio tensor [C, T] or [T]
        sample_rate: Sample rate
        
    Returns:
        features: Tensor [2] containing [normalized_centroid, normalized_bandwidth]
    """
    if waveform.dim() == 2:
        waveform = torch.mean(waveform, dim=0)  # Convert to mono
    
    # Compute STFT
    n_fft = 2048
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=512,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        return_complex=True
    )
    
    # Magnitude spectrum [freq_bins, time_frames]
    mag = torch.abs(stft)
    
    # Average over time
    mag_avg = torch.mean(mag, dim=1)  # [freq_bins]
    
    # Frequency bins in Hz
    freq_bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    
    # Spectral Centroid: weighted average of frequencies
    total_mag = torch.sum(mag_avg) + 1e-8
    centroid = torch.sum(freq_bins * mag_avg) / total_mag
    
    # Spectral Bandwidth: std dev around centroid
    bandwidth = torch.sqrt(torch.sum(mag_avg * (freq_bins - centroid) ** 2) / total_mag)
    
    # Normalize to roughly 0-1 range
    centroid_norm = centroid / 10000.0
    bandwidth_norm = bandwidth / 5000.0
    
    return torch.tensor([centroid_norm, bandwidth_norm], dtype=torch.float32)

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
        
        # 3. Preload IR files and compute features (for both train and val)
        self.ir_cache = []
        self.ir_features = None
        if ir_files_list is not None:
            subset_name = "train" if self.is_train else "validation"
            print(f"Loading {len(ir_files_list)} impulse response files for {subset_name}...")
            ir_features_list = []
            for ir_path in ir_files_list:
                try:
                    # Load IR
                    ir_waveform, ir_sr = torchaudio.load(ir_path)
                    
                    # Convert to mono
                    if ir_waveform.shape[0] > 1:
                        ir_waveform = torch.mean(ir_waveform, dim=0, keepdim=True)
                        
                    # Resample if needed
                    if ir_sr != self.target_sr:
                        resampler = torchaudio.transforms.Resample(ir_sr, self.target_sr)
                        ir_waveform = resampler(ir_waveform)
                    
                    # Compute spectral features
                    features = get_audio_features(ir_waveform, self.target_sr)
                    ir_features_list.append(features)
                    
                    self.ir_cache.append(ir_waveform)
                except Exception as e:
                    print(f"Warning: Failed to load IR {ir_path}: {e}")
            
            if ir_features_list:
                self.ir_features = torch.stack(ir_features_list)  # [N, 2]
            
            subset_name = "train" if self.is_train else "validation"
            print(f"Loaded {len(self.ir_cache)} IR files with features for {subset_name} (Target SR: {self.target_sr}).")
        
        print(f"Dataset initialized: subset='{subset}', length={self.length}")

        # 4. Transforms
        # 移除了 self.resampler，因为我们全程 44.1k
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr, # 44100
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

        # 2. Augmentation: Spectral Convolution with Soulmate IR Matching
        # Training: 80% probability, mix ratio 0.3-0.7 (centered on 0.5)
        # Validation: 100% probability, deterministic mix ratio 0.5
        if self.ir_cache and self.ir_features is not None:
            # Determine if we should apply augmentation
            should_augment = random.random() < 0.8 if self.is_train else True
            
            if should_augment:
                # Compute features for current synth audio
                synth_features = get_audio_features(target_audio, self.target_sr)  # [2]
                
                # Find soulmate IR: minimum Euclidean distance
                distances = torch.norm(self.ir_features - synth_features.unsqueeze(0), dim=1)  # [N]
                soulmate_idx = torch.argmin(distances).item()
                
                # Use the matched IR
                ir_sample = self.ir_cache[soulmate_idx]
                
                # Mix ratio: training is random (0.3-0.7), validation is deterministic (0.5)
                mix = random.uniform(0.3, 0.7) if self.is_train else 0.5
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