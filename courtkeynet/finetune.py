"""
Fine-tuning Script for CourtKeyNet
Uses pretrained weights + clean dataset for precision improvement
"""
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import wandb
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import os

from models.courtkeynet import CourtKeyNet
from losses.geometric_loss import TotalLoss
from utils.dataloader import CourtKeypointDataset
from utils.metrics import compute_pck, compute_iou
from utils.visualization import plot_batch


# ==================== CONFIGURATION ====================
# Clean Dataset Location (hardcoded)
CLEAN_DATASET_PATH = r"C:\Research\Datasets\Badminton_court\7k+Clean_Dataset"

# Config file for fine-tuning
FINETUNE_CONFIG = "configs/finetune.yaml"
# =======================================================


def select_pretrained_weights():
    """GUI popup to select pretrained weights file"""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    # Set initial directory to runs folder if exists
    initial_dir = "runs/courtkeynet"
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()
    
    messagebox.showinfo(
        "CourtKeyNet Fine-tuning",
        "Please select the pretrained weights file (best.pt) from your training run.\n\n"
        "This should be from runs/courtkeynet/exp_YYYYMMDD_HHMMSS_*/best.pt"
    )
    
    filepath = filedialog.askopenfilename(
        title="Select Pretrained Weights",
        initialdir=initial_dir,
        filetypes=[
            ("PyTorch Weights", "*.pt"),
            ("All Files", "*.*")
        ]
    )
    
    root.destroy()
    
    if not filepath:
        print("No weights file selected. Exiting.")
        exit(1)
    
    return filepath


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, config, save_dir):
    model.train()
    losses = []
    loss_components = {'l_kpt': [], 'l_hm': [], 'l_edge': [], 'l_diag': [], 'l_angle': []}
    
    wandb_enabled = config.get('wandb', {}).get('enabled', False)
    log_freq = config.get('wandb', {}).get('log_freq', 50)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        imgs = batch['img'].to(device)
        targets = {'kpts': batch['kpts'].to(device)}
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda', enabled=scaler is not None):
            outputs = model(imgs)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total']
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        for key in loss_components:
            loss_components[key].append(loss_dict[key].item())
        
        pbar.set_postfix({'loss': np.mean(losses[-100:])})
        
        # Visualize first batch of first epoch
        if epoch == 0 and batch_idx == 0:
            plot_batch(imgs, targets['kpts'], outputs['kpts_refined'], save_dir / 'finetune_batch0.jpg')
            if wandb_enabled:
                wandb.log({"finetune_batch": wandb.Image(str(save_dir / 'finetune_batch0.jpg'))})
        
        # Log to wandb
        if wandb_enabled and batch_idx % log_freq == 0:
            wandb.log({
                'finetune/loss': loss.item(),
                'finetune/l_kpt': loss_dict['l_kpt'].item(),
                'finetune/l_hm': loss_dict['l_hm'].item(),
                'finetune/l_edge': loss_dict['l_edge'].item(),
                'finetune/l_diag': loss_dict['l_diag'].item(),
                'finetune/l_angle': loss_dict['l_angle'].item(),
                'finetune/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'batch': batch_idx
            })
    
    return np.mean(losses), {k: np.mean(v) for k, v in loss_components.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, save_dir, wandb_enabled=False):
    model.eval()
    losses = []
    pck_scores = []
    iou_scores = []
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Validation")):
        imgs = batch['img'].to(device)
        targets = {'kpts': batch['kpts'].to(device)}
        
        outputs = model(imgs)
        loss_dict = criterion(outputs, targets)
        
        losses.append(loss_dict['total'].item())
        pck_scores.append(compute_pck(outputs['kpts_refined'], targets['kpts']))
        iou_scores.append(compute_iou(outputs['kpts_refined'], targets['kpts']))
        
        # Visualize first batch
        if batch_idx == 0:
            plot_batch(imgs, targets['kpts'], outputs['kpts_refined'], 
                      save_dir / f'finetune_val_{epoch}.jpg')
            if wandb_enabled:
                try:
                    wandb.log({f"finetune_val": wandb.Image(str(save_dir / f'finetune_val_{epoch}.jpg'))})
                except Exception:
                    pass

    return {
        'loss': np.mean(losses),
        'pck': np.mean(pck_scores),
        'iou': np.mean(iou_scores)
    }


def main():
    print("=" * 60)
    print("  CourtKeyNet Fine-tuning Script")
    print("  Clean Dataset: 7k+ perfectly annotated images")
    print("=" * 60)
    
    # Select pretrained weights via GUI
    pretrained_path = select_pretrained_weights()
    print(f"\nSelected pretrained weights: {pretrained_path}")
    
    # Load config
    with open(FINETUNE_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check dataset exists
    if not os.path.exists(CLEAN_DATASET_PATH):
        print(f"ERROR: Clean dataset not found at {CLEAN_DATASET_PATH}")
        exit(1)
    
    print(f"Clean dataset: {CLEAN_DATASET_PATH}")
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Weights & Biases
    wandb_enabled = config.get('wandb', {}).get('enabled', False)
    if wandb_enabled:
        wandb_config = config.get('wandb', {})
        ts_display = datetime.datetime.now().strftime("%m/%d_%H:%M")
        run_display_name = f"Finetune_{ts_display}"
        
        wandb.init(
            project=wandb_config.get('project', 'courtkeynet'),
            entity=wandb_config.get('entity'),
            name=run_display_name,
            tags=wandb_config.get('tags', []),
            config=config
        )
        print(f"Weights & Biases initialized: {wandb.run.url}")
    
    # Create datasets with clean data (properly annotated - no skip needed)
    train_ds = CourtKeypointDataset(
        CLEAN_DATASET_PATH, 'train', 
        imgsz=config['train']['imgsz'],
        augment=True,
        config=config
    )
    val_ds = CourtKeypointDataset(
        CLEAN_DATASET_PATH, 'valid',
        imgsz=config['train']['imgsz'],
        augment=False
    )
    
    print(f"[finetune-train] Found {len(train_ds)} images")
    print(f"[finetune-valid] Found {len(val_ds)} images")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['val']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = CourtKeyNet(config).to(device)
    
    # Load pretrained weights
    print(f"\nLoading pretrained weights...")
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    pretrained_epoch = ckpt.get('epoch', 'unknown')
    pretrained_loss = ckpt.get('best_val_loss', 'unknown')
    print(f"  Loaded from epoch {pretrained_epoch}, best_val_loss={pretrained_loss}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    if wandb_enabled:
        wandb.log({'model/parameters': num_params})
        wandb.config.update({'pretrained_weights': pretrained_path})
    
    # Loss and optimizer with fine-tuning LR
    criterion = TotalLoss(config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr0'],
        weight_decay=config['train']['weight_decay']
    )
    print(f"Fine-tuning LR: {config['train']['lr0']}")
    
    # LR scheduler with warmup
    warmup_epochs = config['train'].get('warmup_epochs', 3)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (config['train']['epochs'] - warmup_epochs)
            return config['train']['lrf'] + 0.5 * (1 - config['train']['lrf']) * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = GradScaler('cuda') if config['train']['mixed_precision'] else None
    
    # Output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config['train']['name']
    
    if wandb_enabled and wandb.run:
        run_name = f"{run_name}_{timestamp}_{wandb.run.id}"
    else:
        run_name = f"{run_name}_{timestamp}"

    save_dir = Path(config['train']['project']) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")
    
    # Training state
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = config['train'].get('early_stopping_patience', 15)
    
    print("\n" + "=" * 60)
    print("  Starting Fine-tuning...")
    print("=" * 60 + "\n")
    
    # Training loop
    for epoch in range(config['train']['epochs']):
        # Update criterion epoch for geometric loss warmup
        criterion.set_epoch(epoch)
        
        train_loss, train_components = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config, save_dir
        )
        
        val_metrics = validate(model, val_loader, criterion, device, epoch + 1, save_dir, wandb_enabled)
        
        print(f"\nEpoch {epoch+1}/{config['train']['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val PCK: {val_metrics['pck']:.4f}")
        print(f"  Val IoU: {val_metrics['iou']:.4f}")
        
        # Log to wandb
        if wandb_enabled:
            wandb.log({
                'epoch': epoch + 1,
                'finetune/epoch_loss': train_loss,
                'finetune/val_loss': val_metrics['loss'],
                'finetune/val_pck': val_metrics['pck'],
                'finetune/val_iou': val_metrics['iou'],
                'lr': scheduler.get_last_lr()[0]
            })
        
        # Save best and check early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'pretrained_from': pretrained_path
            }
            torch.save(checkpoint, save_dir / 'best.pt')
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
            
            if wandb_enabled:
                wandb.run.summary['best_val_loss'] = best_val_loss
                wandb.run.summary['best_epoch'] = epoch + 1
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")
            
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚡ Early stopping triggered after {epoch+1} epochs!")
                break
        
        # Save periodic checkpoint
        if (epoch + 1) % config['train']['save_interval'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, save_dir / f'epoch_{epoch+1}.pt')
        
        scheduler.step()
    
    # Save final
    torch.save({
        'epoch': config['train']['epochs'],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'config': config
    }, save_dir / 'last.pt')
    
    if wandb_enabled:
        artifact = wandb.Artifact('courtkeynet-finetuned', type='model')
        artifact.add_file(str(save_dir / 'best.pt'))
        wandb.log_artifact(artifact)
        wandb.finish()
    
    print("\n" + "=" * 60)
    print(f"  Fine-tuning Complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Weights saved to: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
