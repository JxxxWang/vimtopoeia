import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

class SurgePretrainDataset(Dataset):
    """
    Dataset for SurgeXT pretraining.
    Reads directly from a specific H5 file (train or val).
    """
    
    def __init__(self, h5_path, augmenter=None):
        """
        Args:
            h5_path: Path to the specific .h5 file (e.g., 'train_data.h5')
            augmenter: Instance of AudioAugmenter (optional)
        """
        self.h5_path = h5_path
        self.augmenter = augmenter
        
        # Open file briefly to get length, then close
        with h5py.File(h5_path, 'r') as f:
            self.length = f['audio'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # We open the file inside __getitem__ to be safe with multi-worker DataLoaders
        with h5py.File(self.h5_path, 'r') as f:
            # 1. Load Audio
            # shape in file is (N, 2, 176400) - stereo audio
            # We take the mean across channels to get mono: (176400,)
            audio_np = f['audio'][idx].mean(axis=0)  # Average L/R channels
            audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, 176400)
            
            # 2. Load Params
            params_np = f['param_array'][idx]  # Changed from 'params' to 'param_array'
            params_tensor = torch.from_numpy(params_np).float()
            
            # Debug: Print shapes and parameter statistics (only for first few samples)
            if idx < 3:
                print(f"\n[Dataset Debug] Sample {idx}:", flush=True)
                print(f"  audio_np.shape: {audio_np.shape}", flush=True)
                print(f"  params_tensor.shape: {params_tensor.shape}", flush=True)
                print(f"  params_tensor min/max: [{params_tensor.min():.4f}, {params_tensor.max():.4f}]", flush=True)
                print(f"  params_tensor mean/std: {params_tensor.mean():.4f} / {params_tensor.std():.4f}", flush=True)
                print(f"  Sample param values (first 10): {params_tensor[:10].tolist()}", flush=True)

        # 3. Apply Augmentation (Only if augmenter is provided)
        # Note: Input to augmenter is Tensor, Output is Tensor
        if self.augmenter:
            final_audio = self.augmenter(audio_tensor)
        else:
            final_audio = audio_tensor

        return {
            'input_audio': final_audio, 
            'target_params': params_tensor
        }