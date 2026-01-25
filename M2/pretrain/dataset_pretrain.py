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
            # shape in file is usually (length,), we make it (1, length) for Torch
            audio_np = f['audio'][idx]
            audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
            
            # 2. Load Params
            params_np = f['params'][idx]
            params_tensor = torch.from_numpy(params_np).float()

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