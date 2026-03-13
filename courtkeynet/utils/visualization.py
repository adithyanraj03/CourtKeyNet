import cv2
import numpy as np
import torch
from pathlib import Path

def plot_batch(imgs, kpts_gt, kpts_pred=None, fpath='test.jpg', max_imgs=16):
    """
    Plot batch of images with keypoints
    imgs: (B, 3, H, W) tensor, normalized
    kpts_gt: (B, 4, 2) tensor, normalized [0, 1]
    kpts_pred: (B, 4, 2) tensor, normalized [0, 1]
    """
    # Move to CPU/Numpy
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().float().numpy()
    if isinstance(kpts_gt, torch.Tensor):
        kpts_gt = kpts_gt.detach().cpu().float().numpy()
    if kpts_pred is not None and isinstance(kpts_pred, torch.Tensor):
        kpts_pred = kpts_pred.detach().cpu().float().numpy()

    B, C, H, W = imgs.shape
    B = min(B, max_imgs)
    
    # Calculate grid size
    grid_cols = int(np.ceil(np.sqrt(B)))
    grid_rows = int(np.ceil(B / grid_cols))
    
    # Create canvas
    canvas = np.zeros((grid_rows * H, grid_cols * W, 3), dtype=np.uint8)
    
    for i in range(B):
        # Denormalize image
        img = imgs[i].transpose(1, 2, 0) # CHW -> HWC
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Draw GT keypoints (Green)
        for j, pt in enumerate(kpts_gt[i]):
            x = int(pt[0] * W)
            y = int(pt[1] * H)
            
            # Clamp to viewable area so we don't hide points on the edge
            x = min(max(x, 0), W-1)
            y = min(max(y, 0), H-1)
            
            # Draw point
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1) # Green filled
            # Add index text
            cv2.putText(img, str(j+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw lines between corners to form quadrilateral (0-1-2-3-0)
            next_pt = kpts_gt[i][(j + 1) % 4]
            nx, ny = int(next_pt[0] * W), int(next_pt[1] * H)
            
            # Clamp next point too
            nx = min(max(nx, 0), W-1)
            ny = min(max(ny, 0), H-1)
            
            cv2.line(img, (x, y), (nx, ny), (0, 255, 0), 2)
        
        # Draw Pred keypoints (Red)
        if kpts_pred is not None:
            for j, pt in enumerate(kpts_pred[i]):
                x = int(pt[0] * W)
                y = int(pt[1] * H)
                # Clamp for drawing safety
                x = min(max(x, 0), W-1)
                y = min(max(y, 0), H-1)
                
                cv2.circle(img, (x, y), 4, (0, 0, 255), -1) # Red filled
                cv2.circle(img, (x, y), 6, (255, 255, 255), 1) # White outline
                
                # Draw lines
                next_pt = kpts_pred[i][(j + 1) % 4]
                nx, ny = int(next_pt[0] * W), int(next_pt[1] * H)
                nx = min(max(nx, 0), W-1)
                ny = min(max(ny, 0), H-1)
                cv2.line(img, (x, y), (nx, ny), (0, 0, 255), 2)

        # Place in grid
        r, c = i // grid_cols, i % grid_cols
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = img
        
    Path(fpath).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(fpath), canvas)
