import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt  # <--- NEW: For plotting
import csv                         # <--- NEW: For saving data

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from dataset import VSTDataset
from model import Vimtopoeia_AST

def main():
    # 1. Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA acceleration.")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS acceleration.")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU.")

    # === 2. Hyperparameters ===
    BATCH_SIZE = 16         
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    NUM_WORKERS = 8         
    
    # 3. Dataset Setup
    h5_path = "/scratch/hw3140/dataset_10k_pairs.h5" 
    print(f"Loading dataset from: {h5_path}")
    
    try:
        dataset = VSTDataset(h5_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=False
    )

    # 4. Model Setup
    sample = dataset[0]
    n_params = sample['delta'].shape[0]
    print(f"Detected n_params: {n_params}")
    
    model = Vimtopoeia_AST(n_params=n_params).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Setup directories
    checkpoints_dir = Path("/scratch/hw3140/vimtopoeia/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # === NEW: Store loss history ===
    loss_history = [] 
    best_loss = float('inf')

    # 5. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in progress_bar:
            spec_vocal = batch['spec_target'].to(device, non_blocking=True)
            spec_ref = batch['spec_ref'].to(device, non_blocking=True)
            delta = batch['delta'].to(device, non_blocking=True)
            one_hot = batch['one_hot'].to(device, non_blocking=True)

            predicted_delta = model(spec_vocal, spec_ref, one_hot)
            loss = criterion(predicted_delta, delta)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})

        # Calculate Average Loss
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        
        # === NEW: Record Loss ===
        loss_history.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")

        # Save Checkpoints
        latest_path = checkpoints_dir / "model_latest.pt"
        torch.save(model.state_dict(), latest_path)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = checkpoints_dir / "model_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"🌟 New Best Model Saved (Loss: {best_loss:.6f})")
            
        if (epoch + 1) % 5 == 0:
            history_path = checkpoints_dir / f"model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), history_path)

    # === NEW: Save Graph and Data ===
    print("Training Complete! Generating loss graph...")
    
    # 1. Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), loss_history, label='Training Loss', marker='o')
    plt.title('Vimtopoeia Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plot_path = checkpoints_dir / "loss_curve.png"
    plt.savefig(plot_path)
    print(f"📈 Loss graph saved to: {plot_path}")
    
    # 2. Save raw numbers to CSV (for Excel)
    csv_path = checkpoints_dir / "loss_history.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss'])
        for i, loss in enumerate(loss_history):
            writer.writerow([i + 1, loss])
    print(f"📊 Loss data saved to: {csv_path}")

if __name__ == "__main__":
    main()