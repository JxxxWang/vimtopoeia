"""
M2 Phase 1 Inference Script

Evaluate trained M2 AST model on audio files.
Predicts synthesizer parameters from audio input.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import argparse
import json
from pathlib import Path
import numpy as np
import sys

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from M2.pretrain.model import M2_AST_Model
from data_generation.core import load_plugin, render_params, write_wav
from data_generation.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC


class M2Inference:
    def __init__(
        self,
        checkpoint_path: str,
        plugin_path: str = None,
        device: str = 'cpu',
        norm_mean: float = None,
        norm_std: float = None
    ):
        """
        Initialize M2 inference system.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            plugin_path: Path to Surge XT plugin (optional, for rendering)
            device: Device to run on ('cpu', 'cuda', 'mps')
            norm_mean: Mel spectrogram normalization mean (if None, uses default)
            norm_std: Mel spectrogram normalization std (if None, uses default)
        """
        self.device = torch.device(device)
        self.param_spec = SURGE_SIMPLE_PARAM_SPEC
        self.sample_rate = 44100
        
        # Normalization stats - use provided values or defaults from training
        if norm_mean is not None and norm_std is not None:
            self.norm_mean = norm_mean
            self.norm_std = norm_std
        else:
            # Default values computed from Surge training set
            self.norm_mean = -73.64360046386719
            self.norm_std = 34.576133728027344
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Detect n_params from checkpoint state_dict (most reliable method)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if 'mlp_head.4.weight' in state_dict:
            n_params = state_dict['mlp_head.4.weight'].shape[0]
            print(f"Detected n_params from checkpoint: {n_params}")
        elif 'config' in checkpoint:
            n_params = checkpoint['config'].get('n_params', 73)
            print(f"Using n_params from config: {n_params}")
        else:
            n_params = 73
            print(f"No param info found, using default n_params={n_params}")
        
        # Initialize model
        self.model = M2_AST_Model(n_params=n_params).to(self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_loss' in checkpoint:
                print(f"Validation loss: {checkpoint['val_loss']:.6f}")
        else:
            # Assume checkpoint is just state_dict
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Mel spectrogram transform (must match training)
        self.mel_transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=80,
                n_mels=64
            ),
            T.AmplitudeToDB()
        ).to(self.device)
        
        # Load plugin if provided
        self.plugin = None
        if plugin_path:
            print(f"Loading Surge XT plugin: {plugin_path}")
            self.plugin = load_plugin(plugin_path)
        
        print(f"M2 inference initialized on {self.device}")
        print(f"Normalization: mean={self.norm_mean:.4f}, std={self.norm_std:.4f}")
    
    def load_audio(self, audio_path: str):
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            audio_tensor: Preprocessed audio [1, samples]
            duration: Audio duration in seconds
        """
        print(f"Loading: {audio_path}")
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            print(f"Resampling from {sr} Hz to {self.sample_rate} Hz")
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        duration = waveform.shape[-1] / self.sample_rate
        print(f"Duration: {duration:.2f}s")
        
        return waveform, duration
    
    def audio_to_mel(self, audio_tensor):
        """
        Convert audio to normalized mel spectrogram.
        
        Args:
            audio_tensor: Audio tensor [1, samples]
            
        Returns:
            mel_spec: Normalized mel spectrogram [1, 64, time]
        """
        # Crop or pad to 4 seconds (176400 samples at 44.1kHz)
        target_samples = int(4.0 * self.sample_rate)
        
        if audio_tensor.shape[-1] > target_samples:
            # Crop to first 4 seconds
            audio_tensor = audio_tensor[..., :target_samples]
            print(f"  Cropped audio to 4 seconds ({target_samples} samples)")
        elif audio_tensor.shape[-1] < target_samples:
            # Pad with zeros
            padding = target_samples - audio_tensor.shape[-1]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
            print(f"  Padded audio to 4 seconds (added {padding} samples)")
        
        with torch.no_grad():
            mel = self.mel_transform(audio_tensor.to(self.device))
            
            # Normalize using training statistics
            mel = (mel - self.norm_mean) / self.norm_std
            
        return mel
    
    def predict(self, audio_path: str):
        """
        Predict synthesizer parameters from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            predicted_params: Predicted parameter array [n_params]
            duration: Audio duration in seconds
        """
        # Load audio
        audio, duration = self.load_audio(audio_path)
        
        # Convert to mel spectrogram
        mel = self.audio_to_mel(audio)
        
        print(f"Mel spectrogram shape: {mel.shape}")
        
        # Predict
        with torch.no_grad():
            predicted_params = self.model(mel)
        
        return predicted_params.squeeze(0).cpu(), duration
    
    def decode_params(self, param_array):
        """
        Decode parameter array to human-readable dictionary.
        
        Args:
            param_array: Parameter tensor or numpy array
            
        Returns:
            params_dict: Dictionary of synth parameters (JSON-serializable)
            note_params: Dictionary of note parameters (JSON-serializable)
        """
        if isinstance(param_array, torch.Tensor):
            param_array = param_array.numpy()
        
        params_dict, note_params = self.param_spec.decode(param_array)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            else:
                return obj
        
        params_dict = convert_to_native(params_dict)
        note_params = convert_to_native(note_params)
        
        return params_dict, note_params
    
    def render_audio(self, params_dict, note_params=None, duration=4.0):
        """
        Render audio from predicted parameters.
        
        Args:
            params_dict: Synth parameter dictionary
            note_params: Note parameters (optional, uses defaults if None)
            duration: Audio duration in seconds
            
        Returns:
            audio: Rendered audio [channels, samples]
        """
        if self.plugin is None:
            raise ValueError("Plugin not loaded. Provide plugin_path during initialization.")
        
        # Use note params if provided, otherwise use defaults
        if note_params is None:
            midi_note = 60  # C4
            note_start_end = (0.1, duration - 0.1)
        else:
            midi_note = note_params.get('pitch', 60)
            note_start_end = note_params.get('note_start_and_end', (0.1, duration - 0.1))
        
        audio = render_params(
            plugin=self.plugin,
            params=params_dict,
            midi_note=midi_note,
            velocity=100,
            note_start_and_end=note_start_end,
            signal_duration_seconds=duration,
            sample_rate=self.sample_rate,
            channels=2,
            preset_path=None
        )
        
        return audio
    
    def evaluate_file(self, audio_path: str, output_dir: str, render_audio: bool = True):
        """
        Run full evaluation pipeline on a single audio file.
        
        Args:
            audio_path: Path to input audio
            output_dir: Directory to save outputs
            render_audio: Whether to render audio from predictions
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_name = Path(audio_path).stem
        
        print("\n" + "="*60)
        print(f"EVALUATING: {audio_name}")
        print("="*60)
        
        # Predict parameters
        predicted_params, duration = self.predict(audio_path)
        
        print(f"Predicted parameters shape: {predicted_params.shape}")
        print(f"Parameter statistics:")
        print(f"  Mean: {predicted_params.mean():.6f}")
        print(f"  Std:  {predicted_params.std():.6f}")
        print(f"  Min:  {predicted_params.min():.6f}")
        print(f"  Max:  {predicted_params.max():.6f}")
        
        # Decode parameters
        params_dict, note_params = self.decode_params(predicted_params)
        
        # Save parameters as JSON
        params_path = output_dir / f"{audio_name}_predicted_params.json"
        with open(params_path, 'w') as f:
            json.dump(params_dict, f, indent=2)
        print(f"\nSaved parameters: {params_path}")
        
        # Save note parameters
        note_params_path = output_dir / f"{audio_name}_note_params.json"
        with open(note_params_path, 'w') as f:
            json.dump(note_params, f, indent=2)
        print(f"Saved note params: {note_params_path}")
        
        # Save raw parameter array
        param_array_path = output_dir / f"{audio_name}_param_array.npy"
        np.save(param_array_path, predicted_params.numpy())
        print(f"Saved param array: {param_array_path}")
        
        # Render audio if requested
        if render_audio and self.plugin is not None:
            print("\nRendering audio from predicted parameters...")
            try:
                rendered_audio = self.render_audio(params_dict, note_params, duration)
                
                # Normalize audio
                peak = np.abs(rendered_audio).max()
                if peak > 0:
                    rendered_audio = rendered_audio / peak * 0.9
                
                audio_out_path = output_dir / f"{audio_name}_predicted.wav"
                write_wav(rendered_audio, str(audio_out_path), self.sample_rate, 2)
                print(f"Saved rendered audio: {audio_out_path}")
            except Exception as e:
                print(f"Error rendering audio: {e}")
        elif render_audio:
            print("\nSkipping audio rendering (no plugin loaded)")
        
        print("="*60)
        print(f"✅ Evaluation complete for {audio_name}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='M2 AST Model Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--audio_files', type=str, nargs='+', required=True,
                        help='Path(s) to audio file(s) to evaluate')
    parser.add_argument('--output_dir', type=str, default='./M2_inference_outputs',
                        help='Output directory for predictions')
    parser.add_argument('--plugin_path', type=str, default=None,
                        help='Path to Surge XT plugin (optional, for audio rendering)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run inference on')
    parser.add_argument('--norm_mean', type=float, default=None,
                        help='Mel spectrogram normalization mean (optional)')
    parser.add_argument('--norm_std', type=float, default=None,
                        help='Mel spectrogram normalization std (optional)')
    parser.add_argument('--no_render', action='store_true',
                        help='Skip audio rendering (only predict parameters)')
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = M2Inference(
        checkpoint_path=args.checkpoint,
        plugin_path=args.plugin_path,
        device=args.device,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std
    )
    
    # Process each audio file
    for audio_path in args.audio_files:
        try:
            inferencer.evaluate_file(
                audio_path=audio_path,
                output_dir=args.output_dir,
                render_audio=not args.no_render
            )
        except Exception as e:
            print(f"\n❌ Error processing {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("✅ ALL EVALUATIONS COMPLETE")
    print("="*60)
    print(f"Outputs saved to: {Path(args.output_dir).absolute()}")


if __name__ == "__main__":
    main()
