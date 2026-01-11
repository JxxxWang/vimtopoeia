# utils.py
import torch
import torch.fft

def apply_spectral_convolution(audio, ir, mix_ratio=1.0):
    """Apply IR convolution in frequency domain (same as dataset.py)"""
    if audio.dim() == 1: 
        audio = audio.unsqueeze(0)
    if ir.dim() == 1: 
        ir = ir.unsqueeze(0)
    
    if audio.shape[0] != ir.shape[0]:
        ir = ir.expand(audio.shape[0], -1)

    n_step = audio.shape[-1]
    n_ir = ir.shape[-1]
    n_fft = n_step + n_ir - 1
    
    # FFT convolution
    audio_f = torch.fft.rfft(audio, n=n_fft)
    ir_f = torch.fft.rfft(ir, n=n_fft)
    convolved_f = audio_f * ir_f
    convolved = torch.fft.irfft(convolved_f, n=n_fft)
    convolved = convolved[..., :n_step]
    
    # Normalize
    original_rms = torch.sqrt(torch.mean(audio**2))
    convolved_rms = torch.sqrt(torch.mean(convolved**2))
    convolved = convolved * (original_rms / (convolved_rms + 1e-8))
    
    # Mix
    if mix_ratio < 1.0:
        return (1 - mix_ratio) * audio + mix_ratio * convolved
    
    return convolved