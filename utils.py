# utils.py
import torch
import torch.fft

def apply_spectral_convolution(audio, ir, mix_ratio=1.0):
    """
    使用 FFT 在频域进行卷积 (Circular Convolution 近似 Linear)
    Audio: [C, T] or [T]
    IR: [C, K] or [K]
    """
    # 确保输入是 2D (Channels, Time)
    if audio.dim() == 1: audio = audio.unsqueeze(0)
    if ir.dim() == 1: ir = ir.unsqueeze(0)
    
    n_step = audio.shape[-1]
    n_ir = ir.shape[-1]
    
    # 卷积后的长度 = N + M - 1
    # 我们使用 FFT 长度为 N + M - 1 的下一个 2 的幂次，以加速计算
    n_fft = n_step + n_ir - 1
    
    # 1. FFT 变换
    audio_f = torch.fft.rfft(audio, n=n_fft)
    ir_f = torch.fft.rfft(ir, n=n_fft)
    
    # 2. 频域相乘 (Spectral Envelope Transfer)
    # 注意：这里我们让 IR 的能量归一化一下，防止卷积后爆音
    convolved_f = audio_f * ir_f
    
    # 3. iFFT 变回时域
    convolved = torch.fft.irfft(convolved_f, n=n_fft)
    
    # 4. 截断与对齐 (Keep original length)
    # 卷积会拖长声音，我们需要截取主要部分。
    # 通常取前 n_step 即可
    convolved = convolved[..., :n_step]
    
    # 5. 归一化 (防止音量过大或过小)
    # 将卷积后的 rms 音量调整到和原 audio 一致
    original_rms = torch.sqrt(torch.mean(audio**2))
    convolved_rms = torch.sqrt(torch.mean(convolved**2))
    convolved = convolved * (original_rms / (convolved_rms + 1e-8))
    
    # 6. Mix (Dry/Wet) - 可选，保留一点点原始的冲击力
    if mix_ratio < 1.0:
        return (1 - mix_ratio) * audio + mix_ratio * convolved
    
    return convolved