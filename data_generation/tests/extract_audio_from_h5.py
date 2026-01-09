import h5py
import hdf5plugin
import numpy as np
import soundfile as sf
import os
import sys
from pathlib import Path

def normalize(audio):
    """Normalize audio to -1.0 to 1.0 range safely."""
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        return audio / peak
    return audio

def extract_audio(h5_path, output_dir="extracted_audio"):
    if not os.path.exists(h5_path):
        print(f"File not found: {h5_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Opening {h5_path}...")
    
    with h5py.File(h5_path, "r") as f:
        print("Keys:", list(f.keys()))
        
        target_ds = f["target_audio"]
        ref_ds = f.get("reference_audio") # Might not exist if single dataset
        
        num_samples = target_ds.shape[0]
        print(f"Found {num_samples} samples.")
        print(f"Dataset Info: chunks={target_ds.chunks}, compression={target_ds.compression}, compression_opts={target_ds.compression_opts}, shape={target_ds.shape}, dtype={target_ds.dtype}")
        
        # Limit to 10 for safety unless requested otherwise
        limit = min(num_samples, 10)
        
        for i in range(limit):
            # Target
            t_audio = target_ds[i]
            # Check shape (C, T) vs (T, C)
            # Surge/Pedalboard usually 2 channels
            if t_audio.shape[0] == 2 and t_audio.shape[1] > 2:
                # (C, T) -> Transpose to (T, C)
                t_audio = t_audio.T
                print (f"Transposed target audio for sample {i}")
            
            # Ensure float32 for soundfile
            t_audio = t_audio.astype(np.float32)

            print(f"Sample {i} target channels: {t_audio.shape[1] if t_audio.ndim > 1 else 1}")

            t_path = os.path.join(output_dir, f"sample_{i}_target.wav")
            sf.write(t_path, t_audio, 44100)
            print(f"Saved {t_path}")
            
            # Reference
            if ref_ds is not None:
                r_audio = ref_ds[i]
                if r_audio.shape[0] == 2 and r_audio.shape[1] > 2:
                    r_audio = r_audio.T
                
                # Ensure float32 for soundfile
                r_audio = r_audio.astype(np.float32)

                # Normalize reference for better audibility if needed, 
                # but raw is better for comparison. 
                # Let's write raw, but maybe print peak.
                
                r_path = os.path.join(output_dir, f"sample_{i}_ref.wav")
                sf.write(r_path, r_audio, 44100)
                print(f"Saved {r_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_audio.py <h5_file>")
        sys.exit(1)
        
    h5_file = sys.argv[1]
    extract_audio(h5_file)
