import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import glob
import os
from pathlib import Path

# --- IMPORTS ---
# Ensure these files are in the same directory or adjust paths
from dataset_pretrain import SurgePretrainDataset
from M2.pretrain.augmentations import AudioAugmenter
from model import M2_AST_Model

# --- CONFIGURATION ---
TRAIN_H5_PATH = '/scratch/hw3140/vimtopoeia_m1/data/train.h5'  # Update if in a subfolder
VAL_H5_PATH = '/scratch/hw3140/vimtopoeia_m1/data/val.h5'      # Update if in a subfolder

# Paths for Convolution IRs
MIT_IR_PATH = './mit_ir_survey/**/*.wav' 
VOCAL_IR_PATH = './vimsketch_synth/vocals/*.wav'

BATCH_SIZE = 64
LR = 5e-5           # DAFx24 Paper Spec
EPOCHS = 50
SAMPLE_RATE = 44100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print(f"Running on device: {DEVICE}")

    # 1. Prepare Impulse Responses
    mit_irs = glob.glob(MIT_IR_PATH, recursive=True)
    vocal_irs = glob.glob(VOCAL_IR_PATH, recursive=True)
    all_ir_files = mit_irs + vocal_irs
    
    print(f"Found {len(mit_irs)} MIT IRs and {len(vocal_irs)} Vocal IRs.")
    
    # 2. Initialize Augmenter (Train Only)
    # If no IRs found, it will fallback to synthetic noise (safe)
    augmenter = AudioAugmenter(ir_files=all_ir_files)

    # 3. Initialize Datasets
    print("Loading datasets...")
    train_dataset = SurgePretrainDataset(h5_path=TRAIN_H5_PATH, augmenter=augmenter)
    val_dataset = SurgePretrainDataset(h5_path=VAL_H5_PATH, augmenter=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. Initialize Model
    # Random Init, 64 Bins
    model = M2_AST_Model(n_params=22).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 5. Define Feature Extractor (On GPU)
    # Must match ASTConfig: 64 Mels
    mel_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512, 
            n_mels=64   # <--- CRITICAL
        ),
        T.AmplitudeToDB()
    ).to(DEVICE)

    # 6. Create checkpoint directory
    checkpoint_dir = Path('checkpoints/M2')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 7. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            audio = batch['input_audio'].to(DEVICE)
            targets = batch['target_params'].to(DEVICE)
            
            # Compute Mel Spec on GPU
            with torch.no_grad():
                mels = mel_transform(audio)
            
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
                    'n_params': 22,
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