"""
Training Script for CourtKeyNet with Weights & Biases Integration
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

from models.courtkeynet import CourtKeyNet
from losses.geometric_loss import TotalLoss
from utils.dataloader import CourtKeypointDataset
from utils.metrics import compute_pck, compute_iou
from utils.visualization import plot_batch


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, config, save_dir):
    model.train()
    losses = []
    loss_components = {'l_kpt': [], 'l_hm': [], 'l_edge': [], 'l_diag': [], 'l_angle': []}
    
    wandb_enabled = config.get('wandb', {}).get('enabled', False)
    log_freq = config.get('wandb', {}).get('log_freq', 100)
    
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
            plot_batch(imgs, targets['kpts'], None, save_dir / 'train_batch0.jpg')
            if wandb_enabled:
                wandb.log({"train_batch": wandb.Image(str(save_dir / 'train_batch0.jpg'))})
        
        # Log to wandb
        if wandb_enabled and batch_idx % log_freq == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/l_kpt': loss_dict['l_kpt'].item(),
                'train/l_hm': loss_dict['l_hm'].item(),
                'train/l_edge': loss_dict['l_edge'].item(),
                'train/l_diag': loss_dict['l_diag'].item(),
                'train/l_angle': loss_dict['l_angle'].item(),
                'train/lr': optimizer.param_groups[0]['lr'],
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
    
    # Iterate with checking for first batch
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
                      save_dir / f'val_batch{epoch}.jpg')
            if wandb_enabled:
                import wandb
                try:
                    wandb.log({f"val_predictions": wandb.Image(str(save_dir / f'val_batch{epoch}.jpg'))})
                except Exception:
                    pass

    return {
        'loss': np.mean(losses),
        'pck': np.mean(pck_scores),
        'iou': np.mean(iou_scores)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                       default=r'C:\Research\Datasets\Badminton_court\20k+Dataset',
                       help='Path to dataset root (containing train/val/test)')
    parser.add_argument('--cfg', type=str, default='configs/courtkeynet.yaml')
    parser.add_argument('--data_cfg', type=str, default='configs/dataset.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb_run_id', type=str, default=None, 
                       help='Resume wandb run with this ID')
    args = parser.parse_args()
    
    # Load configs
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.data_cfg, 'r') as f:
        data_config = yaml.safe_load(f)
    
    config['dataset'] = data_config  # Merge
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Weights & Biases
    wandb_enabled = config.get('wandb', {}).get('enabled', False)
    if wandb_enabled:
        wandb_config = config.get('wandb', {})
        
        # Generate readable timestamp for W&B Run Name
        # Format: Name_Month/Day_Hour:Minute (e.g. CourtKeyNet_01/21_12:30)
        ts_display = datetime.datetime.now().strftime("%m/%d_%H:%M")
        wb_name = wandb_config.get('name', 'CourtKeyNet')
        run_display_name = f"{wb_name}_{ts_display}"
        
        wandb.init(
            project=wandb_config.get('project', 'CourtKeyNet'),
            entity=wandb_config.get('entity'),
            name=run_display_name,
            tags=wandb_config.get('tags', []),
            config=config,
            resume='allow' if args.wandb_run_id else None,
            id=args.wandb_run_id
        )
        print(f"Weights & Biases initialized: {wandb.run.url}")
    
    # Create datasets
    train_ds = CourtKeypointDataset(
        args.data_root, 'train', 
        imgsz=config['train']['imgsz'],
        augment=True,
        config=config
    )
    val_ds = CourtKeypointDataset(
        args.data_root, 'valid',
        imgsz=config['train']['imgsz'],
        augment=False
    )
    
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
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    if wandb_enabled:
        wandb.log({'model/parameters': num_params})
    
    # Loss and optimizer
    criterion = TotalLoss(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr0'],
        weight_decay=config['train']['weight_decay']
    )
    
    # LR scheduler with warmup
    warmup_epochs = config['train'].get('warmup_epochs', 5)
    
    def lr_lambda(epoch):
        # Linear warmup for first N epochs, then cosine decay
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (config['train']['epochs'] - warmup_epochs)
            return config['train']['lrf'] + 0.5 * (1 - config['train']['lrf']) * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = GradScaler('cuda') if config['train']['mixed_precision'] else None
    
    # Output directory
    # Generate a unique run name with timestamp and wandb ID if available
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config['train']['name']
    
    if wandb_enabled and wandb.run:
        # Append wandb run ID for easy mapping
        run_name = f"{run_name}_{timestamp}_{wandb.run.id}"
        # Also update wandb run name to match if it's currently generic
        if wandb.run.name == config['wandb'].get('name'):
             wandb.run.name = run_name
             wandb.run.save()
    else:
        run_name = f"{run_name}_{timestamp}"

    save_dir = Path(config['train']['project']) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = config['train'].get('early_stopping_patience', 20)
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config['train']['epochs']):
        # Update criterion epoch for geometric loss warmup
        #criterion.set_epoch(epoch) # Uncomment if to use geometric loss - currently disabled
        
        train_loss, train_components = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config, save_dir
        )
        
        if (epoch + 1) % config['val']['interval'] == 0:
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
                    'train/epoch_loss': train_loss,
                    'val/loss': val_metrics['loss'],
                    'val/pck': val_metrics['pck'],
                    'val/iou': val_metrics['iou'],
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
                    'config': config
                }
                torch.save(checkpoint, save_dir / 'best.pt')
                print(f"  Saved best model (val_loss={best_val_loss:.4f})")
                
                if wandb_enabled:
                    wandb.run.summary['best_val_loss'] = best_val_loss
                    wandb.run.summary['best_epoch'] = epoch + 1
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")
                
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs!")
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
        # Log final model as artifact
        artifact = wandb.Artifact('courtkeynet-model', type='model')
        artifact.add_file(str(save_dir / 'best.pt'))
        wandb.log_artifact(artifact)
        wandb.finish()
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to: {save_dir}")


if __name__ == '__main__':
    main()
