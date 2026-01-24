"""
Inference script for Vimtopoeia AST model.

This script:
1. Generates a new set of synth parameters (using data_generation spec)
2. Renders the reference audio with those parameters
3. Loads a vocal imitation (user-provided, pitch-detected with CREPE)
4. Performs two types of inference:
   a. Direct inference on the vocal
   b. Inference on spectral convolution of vocal + reference audio
5. Outputs predicted parameters and renders audio for both approaches
"""

import torch
import torchaudio
import torch.nn as nn
import numpy as np
import sys
import argparse
from pathlib import Path
import json
import torchcrepe
import librosa

# Add root to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from model_training.model import Vimtopoeia_AST
from data_generation.core import load_plugin, render_params, write_wav
from data_generation.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC

from model_training.model import Vimtopoeia_AST
from data_generation.core import load_plugin, render_params, write_wav
from data_generation.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC


# Legacy model architecture for v3 checkpoint compatibility
class Vimtopoeia_AST_v3(nn.Module):
    """Legacy 3-input AST model for loading v3 checkpoints."""
    def __init__(self, n_params=29, ast_model_path=None):
        super().__init__()
        
        if ast_model_path is None:
            raise ValueError("ast_model_path must be provided")
        
        from transformers import ASTModel
        self.ast = ASTModel.from_pretrained(ast_model_path)
        
        # Original v3 architecture: 768 (vocal) + 768 (ref) + 3 (one_hot) = 1539
        self.fc = nn.Sequential(
            nn.Linear(1539, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_params)
        )
    
    def forward(self, vocal_spec, ref_spec, one_hot):
        """
        Args:
            vocal_spec: [Batch, 1024, 128]
            ref_spec: [Batch, 1024, 128]
            one_hot: [Batch, 3] - oscillator type encoding
        """
        vocal_outputs = self.ast(vocal_spec)
        vocal_embedding = vocal_outputs.pooler_output  # [Batch, 768]
        
        ref_outputs = self.ast(ref_spec)
        ref_embedding = ref_outputs.pooler_output  # [Batch, 768]
        
        combined = torch.cat([vocal_embedding, ref_embedding, one_hot], dim=1)  # [Batch, 1539]
        params = self.fc(combined)
        return params


class VimtopoeiaInference:
    def __init__(
        self, 
        model_path: str, 
        ast_model_path: str,
        plugin_path: str,
        device: str = 'cpu'
    ):
        """
        Initialize inference system.
        
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            ast_model_path: Path to AST model directory
            plugin_path: Path to Surge XT VST3 plugin
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.device = torch.device(device)
        self.param_spec = SURGE_SIMPLE_PARAM_SPEC
        self.n_params = self.param_spec.synth_param_length
        
        print(f"🔧 Loading plugin from: {plugin_path}")
        self.plugin = load_plugin(plugin_path)
        
        print(f"🤖 Loading model from: {model_path}")
        
        # Load checkpoint first to detect parameter count
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"❌ Error loading checkpoint with default method: {e}")
            print("Trying alternative loading method...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Check if this is a full model save or just state_dict
        if isinstance(checkpoint, dict) and 'fc.3.weight' in checkpoint:
            # This is a state_dict - detect architecture from it
            checkpoint_n_params = checkpoint['fc.3.weight'].shape[0]
            print(f"ℹ️  Detected {checkpoint_n_params} parameters in checkpoint")
            
            if checkpoint_n_params != self.n_params:
                print(f"⚠️  Warning: Checkpoint has {checkpoint_n_params} params, but spec has {self.n_params} params")
                print(f"Using checkpoint's parameter count: {checkpoint_n_params}")
                self.n_params = checkpoint_n_params
            
            # Try to create model and load
            self.model = Vimtopoeia_AST_v3(
                n_params=self.n_params, 
                ast_model_path=ast_model_path
            ).to(self.device)
            
            try:
                self.model.load_state_dict(checkpoint, strict=True)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"❌ Architecture mismatch detected: {e}")
                    print("\n⚠️  The checkpoint was trained with a different model architecture.")
                    print("This usually means the training used different AST features or concatenation.")
                    print("\nTo fix this, you need to either:")
                    print("  1. Use the same model architecture that was used during training")
                    print("  2. Retrain the model with the current architecture")
                    print("  3. Check if there's a full model save (not just state_dict)")
                    raise
                else:
                    raise
        else:
            # Might be a full model save - try loading directly
            print("⚠️  Checkpoint format not recognized, attempting direct load...")
            raise ValueError("Unable to load checkpoint - unexpected format")
        
        self.model.eval()
        
        # Audio processing settings (match training - 44.1kHz)
        self.target_sr = 44100
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=128,
            f_min=20.0
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        # Normalization stats (from dataset)
        self.mean = -26.538128995895384
        self.std = 39.86343679428101
        self.max_len_frames = 1024
        
        print("✅ Inference system initialized")
    
    def get_pitch(self, audio_path: str):
        """
        Uses CREPE to detect the most dominant pitch (f0) and convert to MIDI.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            midi_note: Detected MIDI note number
        """
        print(f"🎵 Detecting pitch from: {audio_path}")
        
        try:
            audio, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading for pitch detection: {e}")
            return 60  # Default C4
        
        # Ensure mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Run CREPE pitch tracking
        f0, confidence = torchcrepe.predict(
            audio, 
            sr, 
            hop_length=320, 
            fmin=50, 
            fmax=1000, 
            model='tiny',
            decoder=torchcrepe.decode.viterbi, 
            return_periodicity=True,
            device='cpu'
        )
        
        # Filter by confidence
        valid_f0 = f0[confidence > 0.4]
        
        if len(valid_f0) == 0:
            print("Warning: No pitch detected. Defaulting to C4 (60).")
            return 60
        
        # Get median pitch
        median_f0 = torch.median(valid_f0).item()
        midi_note = int(librosa.hz_to_midi(median_f0))
        
        print(f"✅ Detected MIDI Note: {midi_note} ({median_f0:.1f} Hz)")
        return midi_note
        print(f"✅ Detected MIDI Note: {midi_note} ({median_f0:.1f} Hz)")
        return midi_note
    
    def generate_reference_params(self, midi_note: int = 60, duration: float = 4.0):
        """
        Generate a new set of random parameters and render reference audio.
        
        Args:
            midi_note: MIDI note to render (default: C4)
            duration: Audio duration in seconds
            
        Returns:
            ref_params_dict: Dictionary of synth parameters
            ref_param_array: Encoded parameter array
            ref_audio: Rendered audio as numpy array [channels, samples]
        """
        print(f"🎲 Generating random synth parameters...")
        
        # Sample random parameters from spec (returns tuple of synth_params, note_params)
        ref_params_dict, _ = self.param_spec.sample()
        
        # Note parameters (override with our MIDI note)
        # Note duration is slightly shorter to avoid edge effects
        note_duration = max(0.1, duration - 0.2)
        note_params = {
            'pitch': midi_note,
            'note_start_and_end': (0.1, 0.1 + note_duration)
        }
        
        print(f"🎹 Rendering reference audio at MIDI note {midi_note} ({duration:.2f}s)...")
        ref_audio = render_params(
            plugin=self.plugin,
            params=ref_params_dict,
            midi_note=note_params['pitch'],
            velocity=100,
            note_start_and_end=note_params['note_start_and_end'],
            signal_duration_seconds=duration,
            sample_rate=self.target_sr,
            channels=2,
            preset_path=None
        )
        
        # Encode to parameter array
        ref_param_array = self.param_spec.encode(ref_params_dict, note_params)
        
        return ref_params_dict, ref_param_array, ref_audio
    
    def load_vocal(self, vocal_path: str):
        """
        Load and preprocess vocal audio.
        
        Args:
            vocal_path: Path to vocal audio file
            
        Returns:
            vocal_waveform: Torch tensor [1, samples] at target_sr
        """
        print(f"🎤 Loading vocal from: {vocal_path}")
        waveform, sr = torchaudio.load(vocal_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        return waveform
    
    def normalize_audio(self, audio, target_db=-6.0):
        """
        Normalize audio to target peak level in dB.
        
        Args:
            audio: Audio array (numpy or torch)
            target_db: Target peak level in dB (default: -6dB for headroom)
            
        Returns:
            normalized_audio: Normalized audio (same type as input)
        """
        is_torch = isinstance(audio, torch.Tensor)
        
        if is_torch:
            audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
        else:
            audio_np = audio
        
        # Find peak value
        peak = np.abs(audio_np).max()
        
        if peak == 0:
            return audio  # Silent audio, return as-is
        
        # Calculate gain to reach target dB
        target_linear = 10 ** (target_db / 20.0)
        gain = target_linear / peak
        
        normalized = audio_np * gain
        
        if is_torch:
            return torch.from_numpy(normalized).to(audio.device)
        else:
            return normalized
    
    def get_loudness_rms(self, audio):
        """
        Calculate RMS loudness of audio in dB.
        
        Args:
            audio: Audio array (numpy or torch)
            
        Returns:
            loudness_db: RMS loudness in dB
        """
        is_torch = isinstance(audio, torch.Tensor)
        
        if is_torch:
            audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
        else:
            audio_np = audio
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_np ** 2))
        
        # Convert to dB (with small epsilon to avoid log(0))
        loudness_db = 20 * np.log10(rms + 1e-10)
        
        return loudness_db
    
    def match_loudness(self, audio, target_loudness_db):
        """
        Adjust audio to match a target loudness level.
        
        Args:
            audio: Audio array (numpy or torch)
            target_loudness_db: Target loudness in dB (RMS)
            
        Returns:
            matched_audio: Loudness-matched audio (same type as input)
        """
        is_torch = isinstance(audio, torch.Tensor)
        
        if is_torch:
            audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
        else:
            audio_np = audio
        
        # Get current loudness
        current_loudness_db = self.get_loudness_rms(audio_np)
        
        # Calculate required gain in dB
        gain_db = target_loudness_db - current_loudness_db
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Apply gain
        matched = audio_np * gain_linear
        
        if is_torch:
            return torch.from_numpy(matched).to(audio.device)
        else:
            return matched
    
    def apply_spectral_convolution(self, vocal, reference_audio, mix_ratio=0.8):
        """
        Apply spectral convolution between vocal and reference synth.
        
        Args:
            vocal: Vocal waveform [1, samples]
            reference_audio: Reference synth audio [channels, samples] (numpy)
            mix_ratio: Mixing ratio (1.0 = full convolution)
            
        Returns:
            convolved: Convolved audio [1, samples]
        """
        # Convert reference to mono
        if reference_audio.shape[0] > 1:
            ref_mono = np.mean(reference_audio, axis=0, keepdims=True)
        else:
            ref_mono = reference_audio
        
        ref_tensor = torch.from_numpy(ref_mono).float()
        
        # Match lengths
        min_len = min(vocal.shape[-1], ref_tensor.shape[-1])
        vocal = vocal[..., :min_len]
        ref_tensor = ref_tensor[..., :min_len]
        
        n_fft = vocal.shape[-1] + ref_tensor.shape[-1] - 1
        
        # FFT
        vocal_f = torch.fft.rfft(vocal, n=n_fft)
        ref_f = torch.fft.rfft(ref_tensor, n=n_fft)
        
        # Multiply in frequency domain
        convolved_f = vocal_f * ref_f
        
        # IFFT
        convolved = torch.fft.irfft(convolved_f, n=n_fft)
        
        # Crop to original length
        convolved = convolved[..., :min_len]
        
        # Normalize (match RMS)
        original_rms = torch.sqrt(torch.mean(vocal**2))
        convolved_rms = torch.sqrt(torch.mean(convolved**2))
        convolved = convolved * (original_rms / (convolved_rms + 1e-8))
        
        # Mix
        if mix_ratio < 1.0:
            return (1 - mix_ratio) * vocal + mix_ratio * convolved
        
        return convolved
    
    def audio_to_spec(self, waveform):
        """
        Convert audio waveform to normalized mel spectrogram.
        
        Args:
            waveform: Torch tensor [1, samples] or [channels, samples]
            
        Returns:
            spec: Normalized spectrogram [1024, 128]
        """
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        spec = self.mel_transform(waveform)
        spec = self.db_transform(spec)
        
        # [1, n_mels, time] -> [time, n_mels]
        spec = spec.squeeze(0).transpose(0, 1)
        
        # Padding/cropping
        if spec.shape[0] < self.max_len_frames:
            padding = torch.zeros(self.max_len_frames - spec.shape[0], 128)
            spec = torch.cat([spec, padding], dim=0)
        else:
            spec = spec[:self.max_len_frames, :]
        
        # Normalize
        spec = (spec - self.mean) / (self.std * 2)
        
        return spec
    
    def predict(self, vocal_spec, ref_spec):
        """
        Run model inference.
        
        Args:
            vocal_spec: Vocal spectrogram [1024, 128]
            ref_spec: Reference spectrogram [1024, 128]
            
        Returns:
            predicted_delta: Predicted parameter delta [n_params]
        """
        # Use fixed one-hot encoding (model trained with pulse oscillator)
        one_hot = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        
        # Add batch dimension
        vocal_spec = vocal_spec.unsqueeze(0).to(self.device)
        ref_spec = ref_spec.unsqueeze(0).to(self.device)
        one_hot = one_hot.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predicted_delta = self.model(vocal_spec, ref_spec, one_hot)
        
        return predicted_delta.squeeze(0).cpu()
    
    def render_predicted_audio(self, ref_param_array, predicted_delta, midi_note=60, duration=2.0):
        """
        Render audio from predicted parameters.
        
        Args:
            ref_param_array: Reference parameter array
            predicted_delta: Predicted delta
            midi_note: MIDI note to render
            duration: Audio duration in seconds
            
        Returns:
            audio: Rendered audio [channels, samples]
            target_params_dict: Decoded parameter dictionary
        """
        # Apply delta
        target_param_array = ref_param_array + predicted_delta.numpy()
        
        # Decode to parameter dictionary
        target_params_dict, note_params = self.param_spec.decode(target_param_array)
        
        # Note duration is slightly shorter to avoid edge effects
        note_duration = max(0.1, duration - 0.2)
        
        # Render
        audio = render_params(
            plugin=self.plugin,
            params=target_params_dict,
            midi_note=midi_note,
            velocity=100,
            note_start_and_end=(0.1, 0.1 + note_duration),
            signal_duration_seconds=duration,
            sample_rate=self.target_sr,
            channels=2,
            preset_path=None
        )
        
        return audio, target_params_dict

        return audio, target_params_dict
    
    def run_inference(
        self, 
        vocal_path: str, 
        output_dir: str
    ):
        """
        Run full inference pipeline.
        
        Args:
            vocal_path: Path to vocal audio file
            output_dir: Directory to save outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("STEP 1: Detect Pitch from Vocal")
        print("="*60)
        
        midi_note = self.get_pitch(vocal_path)
        
        print("\n" + "="*60)
        print("STEP 2: Load Vocal Audio & Calculate Duration")
        print("="*60)
        
        vocal_waveform = self.load_vocal(vocal_path)
        vocal_duration = vocal_waveform.shape[-1] / self.target_sr
        print(f"📏 Vocal duration: {vocal_duration:.2f} seconds")
        
        # Measure target loudness
        target_loudness_db = self.get_loudness_rms(vocal_waveform)
        print(f"🔊 Target loudness: {target_loudness_db:.2f} dB RMS")
        
        print("\n" + "="*60)
        print("STEP 3: Generate Reference Parameters & Audio")
        print("="*60)
        
        ref_params_dict, ref_param_array, ref_audio = self.generate_reference_params(midi_note, duration=vocal_duration)
        
        # Normalize and save reference audio
        ref_audio_normalized = self.normalize_audio(ref_audio, target_db=-3.0)
        ref_audio_path = output_dir / "reference_synth.wav"
        write_wav(ref_audio_normalized, str(ref_audio_path), self.target_sr, 2)
        print(f"💾 Saved reference audio: {ref_audio_path}")
        
        # Save reference parameters
        ref_params_path = output_dir / "reference_params.json"
        with open(ref_params_path, 'w') as f:
            json.dump(ref_params_dict, f, indent=2)
        print(f"💾 Saved reference params: {ref_params_path}")
        
        print("\n" + "="*60)
        print("STEP 4a: Direct Inference on Vocal")
        print("="*60)
        
        # Process vocal spectrogram
        vocal_spec = self.audio_to_spec(vocal_waveform)
        ref_waveform = torch.from_numpy(ref_audio).float()
        ref_spec = self.audio_to_spec(ref_waveform)
        
        # Predict
        predicted_delta_direct = self.predict(vocal_spec, ref_spec)
        print(f"✅ Predicted delta (direct): shape {predicted_delta_direct.shape}")
        
        # Render
        audio_direct, params_direct = self.render_predicted_audio(
            ref_param_array, predicted_delta_direct, midi_note, duration=vocal_duration
        )
        
        # Match loudness to target, then normalize for headroom
        audio_direct_loudness_matched = self.match_loudness(audio_direct, target_loudness_db)
        audio_direct_normalized = self.normalize_audio(audio_direct_loudness_matched, target_db=-3.0)
        audio_direct_path = output_dir / "predicted_direct.wav"
        write_wav(audio_direct_normalized, str(audio_direct_path), self.target_sr, 2)
        print(f"💾 Saved direct prediction audio: {audio_direct_path}")
        print(f"   Loudness matched: {self.get_loudness_rms(audio_direct_loudness_matched):.2f} dB RMS")
        
        params_direct_path = output_dir / "predicted_direct_params.json"
        with open(params_direct_path, 'w') as f:
            json.dump(params_direct, f, indent=2)
        print(f"💾 Saved direct prediction params: {params_direct_path}")
        
        # Save delta as readable text file
        delta_direct_path = output_dir / "predicted_direct_delta.txt"
        with open(delta_direct_path, 'w') as f:
            f.write("Parameter Deltas (Direct Inference)\n")
            f.write("=" * 50 + "\n\n")
            delta_array = predicted_delta_direct.numpy()
            param_names = self.param_spec.synth_param_names
            for i, val in enumerate(delta_array):
                param_name = param_names[i] if i < len(param_names) else f"unknown_{i}"
                f.write(f"Param {i:2d} ({param_name:30s}): {val:+.6f}\n")
            f.write(f"\nMean delta: {delta_array.mean():.6f}\n")
            f.write(f"Std delta:  {delta_array.std():.6f}\n")
        print(f"💾 Saved direct delta: {delta_direct_path}")
        
        print("\n" + "="*60)
        print("STEP 4b: Inference on Spectral Convolution")
        print("="*60)
        
        # Apply spectral convolution
        convolved_waveform = self.apply_spectral_convolution(
            vocal_waveform, ref_audio, mix_ratio=0.8
        )
        
        # Normalize and save convolved audio for reference
        convolved_normalized = self.normalize_audio(convolved_waveform, target_db=-3.0)
        convolved_audio_path = output_dir / "convolved_vocal.wav"
        torchaudio.save(
            convolved_audio_path, 
            convolved_normalized.cpu(), 
            self.target_sr
        )
        print(f"💾 Saved convolved vocal: {convolved_audio_path}")
        
        # Process convolved spectrogram
        convolved_spec = self.audio_to_spec(convolved_waveform)
        
        # Predict
        predicted_delta_conv = self.predict(convolved_spec, ref_spec)
        print(f"✅ Predicted delta (convolved): shape {predicted_delta_conv.shape}")
        
        # Render
        audio_conv, params_conv = self.render_predicted_audio(
            ref_param_array, predicted_delta_conv, midi_note, duration=vocal_duration
        )
        
        # Match loudness to target, then normalize for headroom
        audio_conv_loudness_matched = self.match_loudness(audio_conv, target_loudness_db)
        audio_conv_normalized = self.normalize_audio(audio_conv_loudness_matched, target_db=-3.0)
        audio_conv_path = output_dir / "predicted_convolved.wav"
        write_wav(audio_conv_normalized, str(audio_conv_path), self.target_sr, 2)
        print(f"💾 Saved convolved prediction audio: {audio_conv_path}")
        print(f"   Loudness matched: {self.get_loudness_rms(audio_conv_loudness_matched):.2f} dB RMS")
        
        params_conv_path = output_dir / "predicted_convolved_params.json"
        with open(params_conv_path, 'w') as f:
            json.dump(params_conv, f, indent=2)
        print(f"💾 Saved convolved prediction params: {params_conv_path}")
        
        # Save delta as readable text file
        delta_conv_path = output_dir / "predicted_convolved_delta.txt"
        with open(delta_conv_path, 'w') as f:
            f.write("Parameter Deltas (Convolved Inference)\n")
            f.write("=" * 50 + "\n\n")
            delta_array = predicted_delta_conv.numpy()
            param_names = self.param_spec.synth_param_names
            for i, val in enumerate(delta_array):
                param_name = param_names[i] if i < len(param_names) else f"unknown_{i}"
                f.write(f"Param {i:2d} ({param_name:30s}): {val:+.6f}\n")
            f.write(f"\nMean delta: {delta_array.mean():.6f}\n")
            f.write(f"Std delta:  {delta_array.std():.6f}\n")
        print(f"💾 Saved convolved delta: {delta_conv_path}")
        
        print("\n" + "="*60)
        print("✅ INFERENCE COMPLETE")
        print("="*60)
        print(f"All outputs saved to: {output_dir.absolute()}")
        print(f"\nDetected pitch: MIDI {midi_note}")
        print(f"Generated files:")
        print(f"  - reference_synth.wav (original random synth)")
        print(f"  - reference_params.json (original parameters)")
        print(f"  - convolved_vocal.wav (vocal * reference)")
        print(f"  - predicted_direct.wav (prediction from vocal)")
        print(f"  - predicted_direct_params.json")
        print(f"  - predicted_convolved.wav (prediction from convolved)")
        print(f"  - predicted_convolved_params.json")


def main():
    parser = argparse.ArgumentParser(description='Vimtopoeia Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--ast_model_path', type=str, required=True,
                        help='Path to AST model directory')
    parser.add_argument('--plugin_path', type=str, required=True,
                        help='Path to Surge XT VST3 plugin')
    parser.add_argument('--vocal_path', type=str, required=True,
                        help='Path to vocal audio file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Initialize inference system
    inferencer = VimtopoeiaInference(
        model_path=args.model_path,
        ast_model_path=args.ast_model_path,
        plugin_path=args.plugin_path,
        device=args.device
    )
    
    # Run inference
    inferencer.run_inference(
        vocal_path=args.vocal_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
