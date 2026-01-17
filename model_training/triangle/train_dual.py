"""
Dual-objective training: Supervised parameter prediction + Embedding alignment.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import hdf5plugin
import argparse

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from dataset_dual import DualDataset
from model import Vimtopoeia_AST


def main(
    h5_path: str,
    vimsketch_root: str,
    checkpoints_dir: str,
    ast_model_path: str,
    lambda_consistency: float = 1.0
):
    """
    Train with dual objectives:
    1. Supervised: Predict SurgeXT parameters from audio
    2. Alignment: Align VimSketch vocal/synth embeddings
    
    Args:
        h5_path: Path to SurgeXT HDF5 dataset
        vimsketch_root: Path to VimSketch root (contains 'vocal' and 'synth' folders)
        checkpoints_dir: Path to save model checkpoints
        ast_model_path: Path to pretrained AST model
        lambda_consistency: Weight for alignment loss (default: 1.0)
    """
    # 1. Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration.")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # 2. Hyperparameters
    BATCH_SIZE = 8  # Smaller batch for dual training
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    NUM_WORKERS = 0  # Safe default for complex data loading
    
    print(f"Lambda consistency: {lambda_consistency}")
    
    # 3. Dataset Setup
    print(f"Loading SurgeXT data from: {h5_path}")
    print(f"Loading VimSketch pairs from: {vimsketch_root}")
    
    try:
        # Create full dataset
        full_dataset = DualDataset(
            h5_path=h5_path,
            vimsketch_root=vimsketch_root,
            shuffle_surge=True
        )
        
        # Split into train/val (90/10)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 4. Model Setup
    sample = full_dataset[0]
    n_params = sample['surge_params'].shape[0]
    print(f"Detected n_params: {n_params}")
    print(f"Loading AST model from: {ast_model_path}")
    
    model = Vimtopoeia_AST(n_params=n_params, ast_model_path=ast_model_path).to(device)
    
    # 5. Loss Functions
    criterion_task = nn.MSELoss()  # For parameter prediction
    criterion_consistency = nn.MSELoss()  # For embedding alignment
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Setup directories
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss tracking
    train_loss_history = []
    val_loss_history = []
    task_loss_history = []
    align_loss_history = []
    best_loss = float('inf')

    # 6. Training Loop
    print(f"Starting dual-objective training for {EPOCHS} epochs...")
    
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        
        epoch_total_loss = 0.0
        epoch_task_loss = 0.0
        epoch_align_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in progress_bar:
            # Move data to device
            surge_spec = batch['surge_spec'].to(device, non_blocking=True)
            surge_params = batch['surge_params'].to(device, non_blocking=True)
            sketch_vocal_spec = batch['sketch_vocal_spec'].to(device, non_blocking=True)
            sketch_synth_spec = batch['sketch_synth_spec'].to(device, non_blocking=True)
            
            # Dummy osc_one_hot (not used in this training context)
            batch_size = surge_spec.shape[0]
            osc_one_hot = torch.zeros(batch_size, 3).to(device)
            
            # ============================================
            # Step 1: Supervised Task (Parameter Prediction)
            # ============================================
            predicted_params = model(surge_spec, surge_spec, osc_one_hot, return_embedding=False)
            loss_params = criterion_task(predicted_params, surge_params)
            
            # ============================================
            # Step 2: Alignment Task (Embedding Consistency)
            # ============================================
            # Extract embeddings for vocal and synth
            embed_vocal = model(sketch_vocal_spec, sketch_vocal_spec, osc_one_hot, return_embedding=True)
            embed_synth = model(sketch_synth_spec, sketch_synth_spec, osc_one_hot, return_embedding=True)
            
            # Align embeddings (vocal should be close to synth)
            loss_align = criterion_consistency(embed_vocal, embed_synth)
            
            # ============================================
            # Combined Loss
            # ============================================
            total_loss = loss_params + (lambda_consistency * loss_align)
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track losses
            epoch_total_loss += total_loss.item()
            epoch_task_loss += loss_params.item()
            epoch_align_loss += loss_align.item()
            
            if (batch_idx + 1) % 10 == 0:
                progress_bar.set_postfix({
                    'total': f"{total_loss.item():.4f}",
                    'task': f"{loss_params.item():.4f}",
                    'align': f"{loss_align.item():.4f}"
                })

        # Calculate average losses
        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_task_loss = epoch_task_loss / len(train_loader)
        avg_align_loss = epoch_align_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # Record train losses
        train_loss_history.append(avg_total_loss)
        task_loss_history.append(avg_task_loss)
        align_loss_history.append(avg_align_loss)
        
        # ============================================
        # Validation Loop
        # ============================================
        model.eval()
        val_total_loss = 0.0
        
        with torch.no_grad():
            for val_batch in val_loader:
                v_surge_spec = val_batch['surge_spec'].to(device, non_blocking=True)
                v_surge_params = val_batch['surge_params'].to(device, non_blocking=True)
                v_sketch_vocal_spec = val_batch['sketch_vocal_spec'].to(device, non_blocking=True)
                v_sketch_synth_spec = val_batch['sketch_synth_spec'].to(device, non_blocking=True)
                
                v_batch_size = v_surge_spec.shape[0]
                v_osc_one_hot = torch.zeros(v_batch_size, 3).to(device)
                
                # Task loss
                v_predicted_params = model(v_surge_spec, v_surge_spec, v_osc_one_hot, return_embedding=False)
                v_loss_params = criterion_task(v_predicted_params, v_surge_params)
                
                # Align loss
                v_embed_vocal = model(v_sketch_vocal_spec, v_sketch_vocal_spec, v_osc_one_hot, return_embedding=True)
                v_embed_synth = model(v_sketch_synth_spec, v_sketch_synth_spec, v_osc_one_hot, return_embedding=True)
                v_loss_align = criterion_consistency(v_embed_vocal, v_embed_synth)
                
                v_total_loss = v_loss_params + (lambda_consistency * v_loss_align)
                val_total_loss += v_total_loss.item()
        
        avg_val_loss = val_total_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        model.train()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train: {avg_total_loss:.6f} | "
              f"Val: {avg_val_loss:.6f} | "
              f"Task: {avg_task_loss:.6f} | "
              f"Align: {avg_align_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")

        # Save checkpoints
        latest_path = checkpoints_dir / "model_latest_dual.pt"
        torch.save(model.state_dict(), latest_path)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_path = checkpoints_dir / "model_best_dual.pt"
            torch.save(model.state_dict(), best_path)
            print(f"🌟 New Best Model Saved (Val Loss: {best_loss:.6f})")
            
        if (epoch + 1) % 5 == 0:
            history_path = checkpoints_dir / f"model_dual_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), history_path)

    # 7. Save Training Metrics
    print("Training Complete! Generating loss graphs...")
    
    # Plot combined loss curves
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Train vs Val
    plt.subplot(1, 3, 1)
    plt.plot(range(1, EPOCHS + 1), train_loss_history, label='Train Loss', marker='o')
    plt.plot(range(1, EPOCHS + 1), val_loss_history, label='Val Loss', marker='s')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Subplot 2: Task vs Align
    plt.subplot(1, 3, 2)
    plt.plot(range(1, EPOCHS + 1), task_loss_history, label='Task Loss (Params)', marker='s')
    plt.plot(range(1, EPOCHS + 1), align_loss_history, label='Align Loss (Embeddings)', marker='^')
    plt.title('Loss Components (Train)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Subplot 3: All losses
    plt.subplot(1, 3, 3)
    plt.plot(range(1, EPOCHS + 1), train_loss_history, label='Total Train', marker='o', alpha=0.7)
    plt.plot(range(1, EPOCHS + 1), val_loss_history, label='Total Val', marker='s', alpha=0.7)
    plt.plot(range(1, EPOCHS + 1), task_loss_history, label='Task', marker='^', alpha=0.5)
    plt.plot(range(1, EPOCHS + 1), align_loss_history, label='Align', marker='v', alpha=0.5)
    plt.title('All Losses Combined')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = checkpoints_dir / "loss_curve_dual.png"
    plt.savefig(plot_path)
    print(f"📈 Loss graph saved to: {plot_path}")
    
    # Save CSV
    csv_path = checkpoints_dir / "loss_history_dual.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Task_Loss', 'Align_Loss'])
        for i in range(len(train_loss_history)):
            writer.writerow([
                i + 1,
                train_loss_history[i],
                val_loss_history[i],
                task_loss_history[i],
                align_loss_history[i]
            ])
    print(f"📊 Loss data saved to: {csv_path}")
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Vimtopoeia with dual objectives: parameter prediction + embedding alignment'
    )
    
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to SurgeXT HDF5 dataset file')
    parser.add_argument('--vimsketch_root', type=str, required=True,
                        help='Path to VimSketch root directory (contains vocal/ and synth/)')
    parser.add_argument('--checkpoints_dir', type=str, required=True,
                        help='Path to directory for saving model checkpoints')
    parser.add_argument('--ast_model_path', type=str, required=True,
                        help='Path to pretrained AST model directory')
    parser.add_argument('--lambda_consistency', type=float, default=1.0,
                        help='Weight for embedding alignment loss (default: 1.0)')
    
    args = parser.parse_args()
    
    main(
        h5_path=args.h5_path,
        vimsketch_root=args.vimsketch_root,
        checkpoints_dir=args.checkpoints_dir,
        ast_model_path=args.ast_model_path,
        lambda_consistency=args.lambda_consistency
    )
