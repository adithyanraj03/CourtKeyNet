import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(img_size, is_train=True):
    """
    Get data transformations for training or validation/testing.
    
    Args:
        img_size: Target image size
        is_train: Whether to use training augmentations
    
    Returns:
        transform: Albumentations transformation
    """
    if is_train:
        # Training transformations with augmentations
        transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            ),
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ], p=0.3),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        # Validation/Testing transformations (no augmentations)
        transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    return transform

def generate_heatmaps(keypoints, img_shape, sigma=2):
    """
    Generate Gaussian heatmaps for keypoints.
    
    Args:
        keypoints: Keypoint coordinates normalized to [0, 1] - shape [batch_size, num_keypoints, 2]
        img_shape: Shape of the target heatmap (height, width)
        sigma: Standard deviation of the Gaussian
    
    Returns:
        heatmaps: Generated heatmaps - shape [batch_size, num_keypoints, height, width]
    """
    batch_size = keypoints.shape[0]
    num_keypoints = keypoints.shape[1]
    height, width = img_shape
    
    # Denormalize keypoints to pixel coordinates
    keypoints_px = keypoints.clone()
    keypoints_px[:, :, 0] *= width
    keypoints_px[:, :, 1] *= height
    
    # Create coordinate grids
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y, x, indexing='ij')
    
    # Generate heatmaps
    heatmaps = np.zeros((batch_size, num_keypoints, height, width), dtype=np.float32)
    
    for b in range(batch_size):
        for k in range(num_keypoints):
            x_k, y_k = keypoints_px[b, k].cpu().numpy()
            
            # Skip if keypoint is outside the image
            if x_k < 0 or y_k < 0 or x_k >= width or y_k >= height:
                continue
            
            # Gaussian formula: exp(-((x - x_k)^2 + (y - y_k)^2) / (2 * sigma^2))
            heatmap = np.exp(-((x_grid - x_k) ** 2 + (y_grid - y_k) ** 2) / (2 * sigma ** 2))
            
            # Normalize peak to 1
            heatmap = heatmap / np.max(heatmap)
            
            heatmaps[b, k] = heatmap
    
    return heatmaps