import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def draw_court(image, keypoints, original_size=None):
    """
    Draw court keypoints and lines on the image
    
    Args:
        image: Input image (numpy array, RGB format)
        keypoints: Keypoints as numpy array or torch tensor [K, 2]
        original_size: Original image size (width, height) for denormalization
    
    Returns:
        Image with drawn court keypoints and lines
    """
    # Convert image to RGB if it's not
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        # If normalized, denormalize
        if image.max() <= 1.0:
            image = image * 255.0
    
    # Ensure image is uint8
    image = image.astype(np.uint8).copy()
    h, w = image.shape[:2]
    
    # Handle torch tensor input
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()
    
    # Denormalize keypoints if needed
    if original_size is not None:
        orig_w, orig_h = original_size
        keypoints[:, 0] *= orig_w
        keypoints[:, 1] *= orig_h
    elif keypoints.max() <= 1.0:
        # Assume normalized [0,1] coordinates
        keypoints[:, 0] *= w
        keypoints[:, 1] *= h
    
    # Draw keypoints
    kp_colors = [
        (255, 0, 0),    
        (0, 255, 0),    
        (0, 0, 255),    
        (255, 255, 0),  
    ]
    
    # Connect keypoints to form court outline
    line_connections = [
        (0, 1),  # Top line
        (1, 2),  # Right line
        (2, 3),  # Bottom line
        (3, 0),  # Left line
        (0, 2),  # Diagonal 1
        (1, 3),  # Diagonal 2
    ]
    
    line_colors = [
        (255, 255, 255),  # Top line (white)
        (255, 255, 255),  # Right line (white)
        (255, 255, 255),  # Bottom line (white)
        (255, 255, 255),  # Left line (white)
        (0, 255, 255),    # Diagonal 1 (cyan)
        (255, 0, 255),    # Diagonal 2 (magenta)
    ]
    
    # Draw lines
    for i, (p1_idx, p2_idx) in enumerate(line_connections):
        if p1_idx < len(keypoints) and p2_idx < len(keypoints):
            p1 = (int(keypoints[p1_idx, 0]), int(keypoints[p1_idx, 1]))
            p2 = (int(keypoints[p2_idx, 0]), int(keypoints[p2_idx, 1]))
            cv2.line(image, p1, p2, line_colors[i], 2)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if i < len(kp_colors):
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), 5, kp_colors[i], -1)
            cv2.circle(image, (x, y), 8, kp_colors[i], 2)
    
    return image


def visualize_predictions(images, targets, predictions, max_samples=8):
    """
    Visualize model predictions vs ground truth
    
    Args:
        images: Batch of input images [B, C, H, W]
        targets: Ground truth keypoints
        predictions: Model predictions
        max_samples: Maximum number of samples to visualize
    
    Returns:
        Figure with visualizations
    """
    batch_size = min(len(images), max_samples)
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 3 * batch_size))
    
    # Ensure axes is 2D even for single sample
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Get image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Get ground truth keypoints
        gt_keypoints = targets[i]['keypoints'][:, :2].cpu().numpy()
        
        # Get predicted keypoints
        pred_keypoints = predictions['keypoints'][i].cpu().numpy()
        
        # Draw ground truth
        gt_img = draw_court(img.copy(), gt_keypoints)
        axes[i, 0].imshow(gt_img)
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].axis('off')
        
        # Draw predictions
        pred_img = draw_court(img.copy(), pred_keypoints)
        axes[i, 1].imshow(pred_img)
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_heatmaps(image, heatmaps, alpha=0.5):
    """
    Plot keypoint heatmaps overlaid on the image
    
    Args:
        image: Input image [C, H, W]
        heatmaps: Keypoint heatmaps [K, H, W]
        alpha: Transparency for the overlay
    
    Returns:
        Figure with heatmap visualization
    """
    # Convert image to numpy and denormalize if needed
    if isinstance(image, torch.Tensor):
        img = image.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
    else:
        img = image
    
    # Convert heatmaps to numpy
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    
    num_keypoints = heatmaps.shape[0]
    fig, axes = plt.subplots(1, num_keypoints + 1, figsize=(3 * (num_keypoints + 1), 3))
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot each heatmap
    for i in range(num_keypoints):
        axes[i + 1].imshow(img)
        heatmap = heatmaps[i]
        
        # Normalize heatmap if needed
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply colormap and overlay
        heatmap_colored = plt.cm.jet(heatmap)[..., :3]
        mask = heatmap > 0.1
        
        overlay = img.copy()
        overlay[mask] = overlay[mask] * (1 - alpha) + heatmap_colored[mask] * alpha
        
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f'Keypoint {i+1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    return fig