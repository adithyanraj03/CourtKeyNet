import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from our package
from courtkeynet import build_courtkeynet, CourtKeyNetLoss
from courtkeynet.data import CourtKeypointDataset, collate_fn
from courtkeynet.transforms import get_transform
from courtkeynet.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train CourtKeyNet')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save-dir', type=str, default='./weights', help='Directory to save weights')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--device', type=str, default=None, help='Device to train on (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()

def train(args):
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.log_dir, "train")
    logger.info(f"Training with arguments: {args}")
    
    # Load data config
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get data paths
    data_dir = os.path.dirname(args.data)
    train_img_dir = os.path.join(data_dir, data_config['train'], 'images')
    train_label_dir = os.path.join(data_dir, data_config['train'], 'labels')
    val_img_dir = os.path.join(data_dir, data_config['val'], 'images')
    val_label_dir = os.path.join(data_dir, data_config['val'], 'labels')
    
    # Create transforms
    train_transform = get_transform(args.img_size, is_train=True)
    val_transform = get_transform(args.img_size, is_train=False)
    
    # Create datasets
    train_dataset = CourtKeypointDataset(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        img_size=args.img_size,
        transform=train_transform
    )
    
    val_dataset = CourtKeypointDataset(
        img_dir=val_img_dir,
        label_dir=val_label_dir,
        img_size=args.img_size,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Create model
    model = build_courtkeynet()
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function
    criterion = CourtKeyNetLoss(
        keypoint_weight=1.0,
        heatmap_weight=1.0,
        court_weight=0.5,
        geometric_weight=1.0
    )
    
    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint (epoch {start_epoch})")
        else:
            logger.warning(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger
        )
        
        # Validation phase
        val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            logger=logger
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pt'))
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, os.path.join(args.save_dir, 'last.pt'))
    
    logger.info(f"Training completed. Best val_loss: {best_val_loss:.4f}")
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1} [Train]") as pbar:
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)
            targets_on_device = []
            
            for target in targets:
                target_on_device = {
                    'box': target['box'].to(device),
                    'keypoints': target['keypoints'].to(device),
                    'class_id': target['class_id']
                }
                targets_on_device.append(target_on_device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            batch_targets = {
                'box': torch.stack([t['box'] for t in targets_on_device]),
                'keypoints': torch.stack([t['keypoints'] for t in targets_on_device])
            }
            
            loss_dict = criterion(outputs, batch_targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'court_loss': f"{loss_dict['court_loss'].item():.4f}",
                'keypoint_loss': f"{loss_dict['keypoint_loss'].item():.4f}",
                'geometric_loss': f"{loss_dict['geometric_loss'].item():.4f}"
            })
            pbar.update()
    
    # Calculate average epoch loss
    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
    
    return avg_loss

def validate(model, dataloader, criterion, device, epoch, logger):
    """Validate the model"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1} [Val]") as pbar:
            for batch_idx, (images, targets) in enumerate(dataloader):
                # Move data to device
                images = images.to(device)
                targets_on_device = []
                
                for target in targets:
                    target_on_device = {
                        'box': target['box'].to(device),
                        'keypoints': target['keypoints'].to(device),
                        'class_id': target['class_id']
                    }
                    targets_on_device.append(target_on_device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute loss
                batch_targets = {
                    'box': torch.stack([t['box'] for t in targets_on_device]),
                    'keypoints': torch.stack([t['keypoints'] for t in targets_on_device])
                }
                
                loss_dict = criterion(outputs, batch_targets)
                loss = loss_dict['total_loss']
                
                # Update metrics
                epoch_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                pbar.update()
    
    # Calculate average epoch loss
    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Epoch {epoch+1} Val Loss: {avg_loss:.4f}")
    
    return avg_loss

if __name__ == "__main__":
    args = parse_args()
    train(args)