import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import glob
import os
import argparse
from pathlib import Path
import hdf5plugin

# --- IMPORTS ---
# Ensure these files are in the same directory or adjust paths
from dataset_pretrain import SurgePretrainDataset
from augmentations import AudioAugmenter
from model import M2_AST_Model

# --- CONFIGURATION ---
SAMPLE_RATE = 44100
# Normalization statistics (computed from training set)
NORM_MEAN = -73.64360046386719
NORM_STD = 34.576133728027344

# Use MPS (Apple Silicon) if available, otherwise CUDA, otherwise CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

def main():
    parser = argparse.ArgumentParser(description='M2 Phase 1 Pre-training')
    parser.add_argument('--train_h5', type=str, required=True, help='Path to training H5 file')
    parser.add_argument('--val_h5', type=str, required=True, help='Path to validation H5 file')
    parser.add_argument('--test_h5', type=str, default=None, help='Path to test H5 file (optional)')
    parser.add_argument('--mit_ir_dir', type=str, default='./mit_ir_survey', help='Path to MIT IR directory')
    parser.add_argument('--vocal_dir', type=str, default='./vimsketch_synth/vocal', help='Path to vocal IR directory')
    parser.add_argument('--checkpoints_dir', type=str, default='./M2/pretrain/checkpoints', help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    TRAIN_H5_PATH = args.train_h5
    VAL_H5_PATH = args.val_h5
    MIT_IR_PATH = os.path.join(args.mit_ir_dir, '**/*.wav')
    VOCAL_IR_PATH = os.path.join(args.vocal_dir, '*.wav')
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    EPOCHS = args.num_epochs
    print(f"Running on device: {DEVICE}")

    # 1. Prepare Impulse Responses
    mit_irs = glob.glob(MIT_IR_PATH, recursive=True)
    vocal_irs = glob.glob(VOCAL_IR_PATH, recursive=True)
    all_ir_files = mit_irs + vocal_irs
    
    print(f"Found {len(mit_irs)} MIT IRs and {len(vocal_irs)} Vocal IRs.")
    
    # 2. Initialize Augmenter (Train Only)
    # If no IRs found, it will fallback to synthetic noise (safe)
    print("Initializing augmenter...", flush=True)
    augmenter = AudioAugmenter(ir_files=all_ir_files)
    print("Augmenter initialized.", flush=True)

    # 3. Initialize Datasets
    print("Loading datasets...", flush=True)
    train_dataset = SurgePretrainDataset(h5_path=TRAIN_H5_PATH, augmenter=augmenter)
    val_dataset = SurgePretrainDataset(h5_path=VAL_H5_PATH, augmenter=None)

    print("Creating data loaders...", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
    print("Data loaders created.", flush=True)

    # 4. Initialize Model
    # Random Init, 64 Bins
    print("Initializing AST model...", flush=True)
    model = M2_AST_Model(n_params=73).to(DEVICE)
    print("Model initialized.", flush=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 5. Define Feature Extractor (On GPU)
    # Must match ASTConfig: 64 Mels and produce compatible time frames for 16x16 patches
    # For 4s audio at 44.1kHz (176400 samples): hop_length=80 gives ~2205 frames
    # After 16x16 patching: (2205/16) * (64/16) = 137 * 4 = 548 patches
    mel_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=80,  # Adjusted to produce ~2205 time frames
            n_mels=64   # <--- CRITICAL
        ),
        T.AmplitudeToDB()
    ).to(DEVICE)

    # 6. Create checkpoint directory
    checkpoint_dir = Path(args.checkpoints_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 7. Training Loop
    print("Starting training loop...", flush=True)
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS} starting...", flush=True)
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            audio = batch['input_audio'].to(DEVICE)
            targets = batch['target_params'].to(DEVICE)
            
            # Compute Mel Spec on GPU
            with torch.no_grad():
                mels = mel_transform(audio)
                # Normalize mel spectrogram
                mels = (mels - NORM_MEAN) / NORM_STD
            
            # Debug: Print shapes on first iteration to verify dimensions
            if train_loss == 0.0:
                print(f"Audio shape: {audio.shape}")
                print(f"Mel spec shape: {mels.shape}")
            
            optimizer.zero_grad()
            preds = model(mels)
            loss = criterion(preds, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['input_audio'].to(DEVICE)
                targets = batch['target_params'].to(DEVICE)
                
                mels = mel_transform(audio)
                # Normalize mel spectrogram
                mels = (mels - NORM_MEAN) / NORM_STD
                preds = model(mels)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / 'm2_phase1_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'config': {
                    'n_params': 73,
                    'sample_rate': SAMPLE_RATE,
                    'n_mels': 64,
                    'n_fft': 1024,
                    'hop_length': 512,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LR
                }
            }, checkpoint_path)
            print(f"  >>> New Best Model Saved (Val Loss: {avg_val_loss:.6f})")
        
        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_path = checkpoint_dir / f'm2_phase1_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
            }, periodic_path)
            print(f"  >>> Periodic checkpoint saved: epoch {epoch+1}")


if __name__ == "__main__":
    main()