import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
from tqdm import tqdm

# Add current directory to path to allow imports if running from root
sys.path.append(str(Path(__file__).parent))

from dataset import VSTDataset
from model import Vimtopoeia_AST

def main():
    # 1. Device Selection (MPS Support)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Metal Performance Shaders) acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA acceleration.")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS/CUDA not available. Using CPU.")

    # 2. Hyperparameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 5
    
    # 3. Dataset Setup
    # Locate dataset relative to this script
    root_dir = Path(__file__).parent.parent
    h5_path = root_dir / "dataset_10k_pairs.h5" # Assuming this is the file name based on context
    
    # Fallback to dataset_10k.h5 if pairs not found
    if not h5_path.exists():
        h5_path = root_dir / "dataset_10k.h5"

    print(f"Loading dataset from: {h5_path}")
    try:
        dataset = VSTDataset(str(h5_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the HDF5 dataset exists in the project root.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    # 4. Model Setup
    # Determine n_params from dataset
    sample = dataset[0]
    n_params = sample['delta'].shape[0]
    print(f"Detected n_params: {n_params}")
    
    model = Vimtopoeia_AST(n_params=n_params).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("Starting training...")
    model.train()
    
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in progress_bar:
            # Move data to device
            spec_vocal = batch['spec_target'].to(device)
            spec_ref = batch['spec_ref'].to(device)
            delta = batch['delta'].to(device)
            one_hot = batch['one_hot'].to(device)

            # Forward pass
            predicted_delta = model(spec_vocal, spec_ref, one_hot)
            
            # Compute loss
            loss = criterion(predicted_delta, delta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # Update progress bar
            if (batch_idx + 1) % 10 == 0:
                progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})

        # Average metrics per sample
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")

    # 6. Save Model
    checkpoints_dir = Path(__file__).parent / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    save_path = checkpoints_dir / "demo_model.pt"
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
