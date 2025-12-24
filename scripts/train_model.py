import os
import sys

# Fix OpenMP conflict FIRST
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import yaml
import torch
import numpy as np
import traceback
import time
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.koopman_model import DeepBilinearKoopman
from src.training import train_epoch, validate
from src.data_normalization import StandardScaler

class KoopmanDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.X = data['states']   # [N, 31, 15]
        self.U = data['controls'] # [N, 31, 6]
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        return self.X[idx], self.U[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--profile', action='store_true', help='Enable timing profiling')
    args = parser.parse_args()

    # encoding='utf-8' 옵션을 추가합니다.
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    train_path = os.path.join(cfg['data']['save_dir'], 'train_data.npz')
    val_path = os.path.join(cfg['data']['save_dir'], 'val_data.npz')
    
    if not os.path.exists(train_path):
        print(f"Data not found at {train_path}. Please run generate_data.py first.")
        return
        
    train_dataset = KoopmanDataset(train_path)
    val_dataset = KoopmanDataset(val_path)
    
    # Optimization: Use num_workers and pin_memory for faster data loading
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=False, 
        num_workers=2,  # Less workers for validation
        pin_memory=True,
        persistent_workers=True
    )
    
    # Init Model
    model = DeepBilinearKoopman(cfg).to(device)
    
    # Compute Normalization Stats
    print("Computing normalization statistics...")
    all_X = torch.from_numpy(train_dataset.X).float().view(-1, cfg['dims']['state'])
    all_U = torch.from_numpy(train_dataset.U).float().view(-1, cfg['dims']['control'])
    
    x_scaler = StandardScaler(device=device)
    u_scaler = StandardScaler(device=device)
    
    x_scaler.fit(all_X)
    u_scaler.fit(all_U)
    
    # Save scalers
    ckpt_dir = cfg['training']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    scaler_path = os.path.join(ckpt_dir, 'scalers.pth')
    torch.save({'x_mean': x_scaler.mean, 'x_std': x_scaler.std, 
                'u_mean': u_scaler.mean, 'u_std': u_scaler.std}, scaler_path)
    print(f"Saved normalization stats to {scaler_path}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    # Mixed Precision Scaler (Disabled for Stability per Paper Specs)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    # Logging
    writer = SummaryWriter(log_dir='runs/dbkmpc_experiment')
    
    # Loop
    epochs = cfg['training']['epochs']
    best_val_loss = float('inf')
    
    print("Starting training...")
    epoch_start = time.time()
    
    for epoch in range(epochs):
        iter_start = time.time()
        
        train_loss, train_logs = train_epoch(model, train_loader, optimizer, cfg, device, x_scaler, u_scaler, scaler=scaler)
        
        iter_time = time.time() - iter_start
        
        if epoch % 100 == 0:
            val_loss, val_logs = validate(model, val_loader, cfg, device, x_scaler, u_scaler)
            
            # Log
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            for k, v in train_logs.items():
                writer.add_scalar(f'Components_Train/{k}', v, epoch)
            
            if args.profile and epoch == 10:
                print(f"\n=== PERFORMANCE ANALYSIS ===")
                print(f"Epoch time: {iter_time:.2f}s")
                print(f"Batches: {len(train_loader)}")
                print(f"Time per batch: {iter_time/len(train_loader)*1000:.1f}ms")
                print(f"Estimated time for 1000 epochs: {iter_time*100:.1f}s ({iter_time*100/60:.1f}min)")
                print(f"===========================\n")
            
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f} ({iter_time:.2f}s)")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                
        if epoch % 1000 == 0:
             torch.save(model.state_dict(), os.path.join(ckpt_dir, f'ckpt_{epoch}.pth'))
             
    print("Training Complete.")
    writer.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("CRITICAL ERROR IN SCRIPT:")
        traceback.print_exc()
        sys.exit(1)
