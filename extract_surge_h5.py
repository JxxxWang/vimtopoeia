"""
Extract audio and parameters from surge_train_40k.h5 for pretraining.

This script extracts clean SurgeXT audio samples and their corresponding
parameter settings from the HDF5 dataset file.
"""

import h5py
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def extract_surge_dataset(
    h5_path: str = "surge_train_40k.h5",
    output_dir: str = "surge_extracted",
    sample_rate: int = 44100
):
    """
    Extract audio and parameters from HDF5 file.
    
    Args:
        h5_path: Path to the HDF5 file
        output_dir: Directory to save extracted files
        sample_rate: Expected sample rate (default: 44100)
    """
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    
    # Create output directories
    audio_dir = output_dir / "audio"
    params_dir = output_dir / "params"
    audio_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting from: {h5_path}")
    print(f"Output directory: {output_dir}")
    
    with h5py.File(h5_path, 'r') as f:
        # Explore structure
        print("\nDataset structure:")
        print(f"Keys: {list(f.keys())}")
        
        # Common HDF5 structures for audio datasets
        # Adjust based on actual structure
        if 'audio' in f.keys():
            audio_data = f['audio']
            params_data = f.get('params', f.get('parameters', None))
        elif 'waveforms' in f.keys():
            audio_data = f['waveforms']
            params_data = f.get('params', f.get('parameters', None))
        else:
            # Try to find the main data array
            main_key = list(f.keys())[0]
            audio_data = f[main_key]
            params_data = f.get('params', f.get('parameters', None))
        
        print(f"Audio shape: {audio_data.shape}")
        if params_data is not None:
            print(f"Params shape: {params_data.shape}")
        
        num_samples = audio_data.shape[0]
        print(f"\nExtracting {num_samples} samples...")
        
        # Extract each sample
        for i in tqdm(range(num_samples)):
            # Extract audio
            audio = audio_data[i]
            
            # Handle different audio formats
            if audio.ndim == 1:
                # Mono audio
                audio_to_save = audio
            elif audio.ndim == 2:
                # Could be (channels, samples) or (samples, channels)
                if audio.shape[0] <= 2:
                    # (channels, samples) - transpose for soundfile
                    audio_to_save = audio.T
                else:
                    # (samples, channels)
                    audio_to_save = audio
            
            # Normalize if needed
            if audio_to_save.dtype != np.float32:
                audio_to_save = audio_to_save.astype(np.float32)
            
            # Ensure values are in [-1, 1] range
            max_val = np.abs(audio_to_save).max()
            if max_val > 1.0:
                audio_to_save = audio_to_save / max_val
            
            # Save audio as WAV
            audio_path = audio_dir / f"sample_{i:06d}.wav"
            sf.write(audio_path, audio_to_save, sample_rate)
            
            # Extract and save parameters
            if params_data is not None:
                params = params_data[i]
                params_path = params_dir / f"sample_{i:06d}.npy"
                np.save(params_path, params)
        
        print(f"\nExtraction complete!")
        print(f"Audio files: {audio_dir}")
        print(f"Parameter files: {params_dir}")
        
        # Save metadata
        metadata = {
            'num_samples': num_samples,
            'sample_rate': sample_rate,
            'audio_shape': audio_data.shape,
            'params_shape': params_data.shape if params_data is not None else None
        }
        
        metadata_path = output_dir / "metadata.npy"
        np.save(metadata_path, metadata)
        print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract Surge dataset from HDF5")
    parser.add_argument(
        "--h5_path",
        type=str,
        default="surge_train_40k.h5",
        help="Path to HDF5 file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="surge_extracted",
        help="Output directory for extracted files"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Audio sample rate"
    )
    
    args = parser.parse_args()
    
    extract_surge_dataset(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate
    )
