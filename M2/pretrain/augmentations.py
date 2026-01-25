"""
Audio augmentation module for simulating messy vocal recordings from clean synth audio.

This module provides data augmentation techniques to make clean synthesizer audio
sound like real-world vocal recordings with various imperfections.
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional, List, Union
from pathlib import Path


class AudioAugmenter:
    """
    Audio augmentation class for simulating realistic vocal recording conditions.
    
    Applies various augmentations to clean synth audio to simulate:
    - Environmental noise
    - Poor microphone frequency response
    - Room acoustics and reflections
    
    Args:
        sample_rate: Audio sample rate (default: 44100)
        ir_files: Optional list of IR file paths (MIT IRs and/or vocal snippets)
        mit_ir_path: Optional path to directory containing MIT IR .wav files
        vocal_path: Optional path to directory containing vocal .wav files
        p_noise: Probability of applying noise augmentation (default: 0.5)
        p_freq_mask: Probability of applying frequency masking (default: 0.3)
        p_spectral_conv: Probability of applying spectral convolution (default: 0.6)
        p_real_ir: Probability of using real IR vs synthetic (default: 0.7)
        snr_range: Tuple of (min_snr, max_snr) in dB (default: (10, 30))
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        ir_files: Optional[List[Union[str, Path]]] = None,
        mit_ir_path: Optional[Union[str, Path]] = None,
        vocal_path: Optional[Union[str, Path]] = None,
        p_noise: float = 0.5,
        p_freq_mask: float = 0.3,
        p_spectral_conv: float = 0.6,
        p_real_ir: float = 0.7,
        snr_range: Tuple[float, float] = (10.0, 30.0)
    ):
        self.sample_rate = sample_rate
        self.p_noise = p_noise
        self.p_freq_mask = p_freq_mask
        self.p_spectral_conv = p_spectral_conv
        self.p_real_ir = p_real_ir
        self.snr_range = snr_range
        
        # Collect IR files
        self.ir_files: List[Path] = []
        
        if ir_files is not None:
            # Use provided list of IR files
            self.ir_files.extend([Path(f) for f in ir_files])
        
        # Search directories for .wav files if provided
        if mit_ir_path is not None:
            mit_path = Path(mit_ir_path)
            if mit_path.exists():
                self.ir_files.extend(list(mit_path.glob('*.wav')))
                self.ir_files.extend(list(mit_path.glob('*.WAV')))
        
        if vocal_path is not None:
            vocal_path_obj = Path(vocal_path)
            if vocal_path_obj.exists():
                self.ir_files.extend(list(vocal_path_obj.glob('*.wav')))
                self.ir_files.extend(list(vocal_path_obj.glob('*.WAV')))
        
        # Remove duplicates and filter existing files
        self.ir_files = list(set([f for f in self.ir_files if f.exists()]))
        
        if self.ir_files:
            print(f"AudioAugmenter initialized with {len(self.ir_files)} IR/vocal files")
        else:
            print("AudioAugmenter initialized with synthetic IRs only (no files provided)")
        
    def add_noise(
        self,
        waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        Add Gaussian white noise with random SNR between 10dB and 30dB.
        
        Args:
            waveform: Input tensor of shape (channels, samples) or (samples,)
            
        Returns:
            Noisy waveform with same shape as input
        """
        # Handle both (C, T) and (T,) shapes
        original_shape = waveform.shape
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        # Generate white noise
        noise = torch.randn_like(waveform)
        
        # Calculate signal power
        signal_power = torch.mean(waveform ** 2)
        
        # Random SNR from range
        snr_db = np.random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate noise power needed for target SNR
        noise_power = signal_power / snr_linear
        
        # Scale noise to achieve target SNR
        current_noise_power = torch.mean(noise ** 2)
        noise = noise * torch.sqrt(noise_power / (current_noise_power + 1e-8))
        
        # Add noise to signal
        noisy_waveform = waveform + noise
        
        # Restore original shape
        if len(original_shape) == 1:
            noisy_waveform = noisy_waveform.squeeze(0)
            
        return noisy_waveform
    
    def _generate_pink_noise(
        self,
        channels: int,
        samples: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate pink noise (1/f noise) using the Voss-McCartney algorithm.
        
        Args:
            channels: Number of audio channels
            samples: Number of samples to generate
            device: Target device for the tensor
            
        Returns:
            Pink noise tensor of shape (channels, samples)
        """
        # Number of random sources
        num_sources = 16
        
        pink_noise = torch.zeros(channels, samples, device=device)
        
        for i in range(num_sources):
            # Each source updates at different rates
            update_rate = 2 ** i
            noise_source = torch.randn(channels, (samples // update_rate) + 1, device=device)
            
            # Upsample by repeating values
            upsampled = torch.repeat_interleave(noise_source, update_rate, dim=1)
            
            # Add to pink noise (truncate to correct length)
            pink_noise += upsampled[:, :samples]
        
        # Normalize
        pink_noise = pink_noise / num_sources
        
        return pink_noise
    
    def freq_masking(
        self,
        waveform: torch.Tensor,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> torch.Tensor:
        """
        Randomly mask out a frequency band to simulate poor microphone response.
        
        This applies frequency masking in the spectrogram domain and then
        reconstructs the waveform using Griffin-Lim.
        
        Args:
            waveform: Input tensor of shape (channels, samples) or (samples,)
            n_fft: FFT window size (default: 2048)
            hop_length: Hop length for STFT (default: 512)
            
        Returns:
            Frequency-masked waveform with same shape as input
        """
        original_shape = waveform.shape
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        channels, samples = waveform.shape
        
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            window=torch.hann_window(n_fft, device=waveform.device)
        )
        
        # stft shape: (channels, freq_bins, time_frames)
        freq_bins = stft.shape[1]
        
        # Random frequency band to mask (mask width: 5-15% of spectrum)
        mask_width = np.random.randint(int(0.05 * freq_bins), int(0.15 * freq_bins))
        mask_start = np.random.randint(0, freq_bins - mask_width)
        
        # Apply mask (set to very low value, not zero to avoid artifacts)
        stft_masked = stft.clone()
        stft_masked[:, mask_start:mask_start + mask_width, :] *= 0.01
        
        # Inverse STFT
        masked_waveform = torch.istft(
            stft_masked,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=waveform.device),
            length=samples
        )
        
        # Restore original shape
        if len(original_shape) == 1:
            masked_waveform = masked_waveform.squeeze(0)
            
        return masked_waveform
    
    def spectral_convolution(
        self,
        waveform: torch.Tensor,
        ir_length: int = 8192
    ) -> torch.Tensor:
        """
        Perform convolution with a real or synthetic impulse response (IR).
        
        With 70% probability (if IR files available), uses a random IR file from
        the MIT dataset or vocal snippets. Otherwise uses synthetic IR.
        
        Args:
            waveform: Input tensor of shape (channels, samples) or (samples,)
            ir_length: Length of the synthetic IR in samples (default: 8192)
            
        Returns:
            Convolved waveform with same length as input
        """
        original_shape = waveform.shape
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        channels, samples = waveform.shape
        
        # Decide whether to use real IR or synthetic
        use_real_ir = (len(self.ir_files) > 0 and 
                      np.random.rand() < self.p_real_ir)
        
        if use_real_ir:
            # Load random IR file
            ir_file = np.random.choice(self.ir_files)
            ir, ir_sr = torchaudio.load(ir_file)
            
            # Convert to mono if stereo
            if ir.shape[0] > 1:
                ir = torch.mean(ir, dim=0, keepdim=False)
            else:
                ir = ir.squeeze(0)
            
            # Resample if sample rate differs
            if ir_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=ir_sr,
                    new_freq=self.sample_rate
                )
                ir = resampler(ir.unsqueeze(0)).squeeze(0)
            
            # Move to same device as waveform
            ir = ir.to(waveform.device)
            
            # Limit IR length to avoid excessive computation
            max_ir_length = min(len(ir), self.sample_rate)  # Max 1 second
            ir = ir[:max_ir_length]
        else:
            # Generate synthetic impulse response
            ir = self.generate_synthetic_ir(ir_length, waveform.device)
        
        ir_length = len(ir)
        
        # Perform FFT-based convolution for each channel
        convolved = []
        for ch in range(channels):
            # Use FFT convolution for efficiency
            waveform_fft = torch.fft.rfft(waveform[ch], n=samples + ir_length - 1)
            ir_fft = torch.fft.rfft(ir, n=samples + ir_length - 1)
            
            # Convolve in frequency domain
            convolved_fft = waveform_fft * ir_fft
            
            # Convert back to time domain
            convolved_ch = torch.fft.irfft(convolved_fft)
            convolved.append(convolved_ch)
        
        convolved_waveform = torch.stack(convolved)
        
        # Trim to original length to maintain shape consistency
        convolved_waveform = convolved_waveform[:, :samples]
        
        # Normalize to prevent clipping (max amplitude <= 1.0)
        max_val = torch.max(torch.abs(convolved_waveform))
        if max_val > 1.0:
            convolved_waveform = convolved_waveform / max_val
        
        # Restore original shape
        if len(original_shape) == 1:
            convolved_waveform = convolved_waveform.squeeze(0)
            
        return convolved_waveform
    
    def generate_synthetic_ir(
        self,
        length: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate a synthetic impulse response from filtered noise.
        
        Creates a realistic-sounding IR with:
        - Initial impulse
        - Early reflections
        - Exponential decay (reverb tail)
        
        Args:
            length: Length of IR in samples
            device: Target device for the tensor
            
        Returns:
            Synthetic IR tensor of shape (length,)
        """
        # Initial impulse (direct sound)
        ir = torch.zeros(length, device=device)
        ir[0] = 1.0
        
        # Add early reflections (first 50ms)
        early_reflection_samples = min(int(0.05 * self.sample_rate), length)
        num_early_reflections = np.random.randint(5, 15)
        
        for _ in range(num_early_reflections):
            position = np.random.randint(10, early_reflection_samples)
            amplitude = np.random.uniform(0.1, 0.5) * (1 - position / early_reflection_samples)
            if position < length:
                ir[position] += amplitude
        
        # Add reverb tail (exponentially decaying noise)
        decay_rate = np.random.uniform(0.0001, 0.001)  # Slower decay = longer reverb
        
        for i in range(early_reflection_samples, length):
            # Exponential decay envelope
            envelope = np.exp(-decay_rate * (i - early_reflection_samples))
            # Add filtered noise
            ir[i] += envelope * np.random.randn() * 0.3
        
        # Apply low-pass filter to smooth the IR
        ir = self._simple_lowpass(ir, cutoff_ratio=0.7)
        
        # Normalize
        ir = ir / torch.max(torch.abs(ir))
        
        return ir
    
    def _simple_lowpass(
        self,
        signal: torch.Tensor,
        cutoff_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Apply a simple low-pass filter using FFT.
        
        Args:
            signal: Input signal
            cutoff_ratio: Cutoff frequency as ratio of Nyquist (0.5 = half Nyquist)
            
        Returns:
            Filtered signal
        """
        # FFT
        signal_fft = torch.fft.rfft(signal)
        
        # Create low-pass filter
        freqs = len(signal_fft)
        cutoff_bin = int(freqs * cutoff_ratio)
        
        # Zero out high frequencies
        signal_fft[cutoff_bin:] = 0
        
        # IFFT
        filtered = torch.fft.irfft(signal_fft, n=len(signal))
        
        return filtered
    
    def __call__(
        self,
        waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply random augmentations to the input waveform.
        
        Augmentation chain:
        1. Spectral convolution with probability 0.6
        2. Add noise with probability 0.5
        3. Frequency masking with probability 0.3
        
        Args:
            waveform: Input tensor of shape (channels, samples) or (samples,)
            
        Returns:
            Augmented waveform with same shape as input
        """
        augmented = waveform.clone()
        
        # Apply spectral convolution with probability p_spectral_conv (0.6)
        if np.random.rand() < self.p_spectral_conv:
            augmented = self.spectral_convolution(augmented)
        
        # Apply noise with probability p_noise (0.5)
        if np.random.rand() < self.p_noise:
            augmented = self.add_noise(augmented)
        
        # Apply frequency masking with probability p_freq_mask (0.3)
        if np.random.rand() < self.p_freq_mask:
            augmented = self.freq_masking(augmented)
        
        return augmented


def demo_augmentations():
    """
    Demonstration of the AudioAugmenter class.
    
    Creates a synthetic audio signal and applies augmentations.
    """
    # Create a simple test signal (1 second of 440 Hz sine wave)
    sample_rate = 44100
    duration = 1.0
    frequency = 440.0
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * frequency * t)
    
    # Initialize augmenter with vocal path if available
    vocal_path = Path("vimsketch_synth/vocal")
    
    augmenter = AudioAugmenter(
        sample_rate=sample_rate,
        vocal_path=vocal_path if vocal_path.exists() else None,
        p_noise=1.0,  # Always apply for demo
        p_freq_mask=1.0,
        p_spectral_conv=1.0
    )
    
    # Apply augmentations
    print("Applying augmentations...")
    augmented = augmenter(waveform)
    
    print(f"Original shape: {waveform.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print(f"Original range: [{waveform.min():.3f}, {waveform.max():.3f}]")
    print(f"Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    return waveform, augmented


if __name__ == "__main__":
    demo_augmentations()
