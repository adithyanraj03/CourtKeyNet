import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(img_size, is_train=True):
    """
    Returns appropriate transformation based on train/val mode
    
    Args:
        img_size: Target image size
        is_train: Whether to use training augmentations
    
    Returns:
        Albumentations transform
    """
    if is_train:
        return TrainTransform(img_size)
    else:
        return ValTransform(img_size)


class TrainTransform:
    """
    Transformations for training data
    """
    def __init__(self, img_size):
        self.img_size = img_size
        self.transform = A.Compose([
            # Geometric transforms
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.5
            ),
            A.RandomRotate90(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            
            # Color transforms
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.GaussianBlur(p=0.2),
            A.CLAHE(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            
            # Court-specific augmentations
            A.OneOf([
                A.RandomShadow(p=0.5),
                A.RandomFog(p=0.5),
                A.RandomSunFlare(p=0.5)
            ], p=0.3),
            
            # Normalization and conversion to tensor
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def __call__(self, image, keypoints=None, **kwargs):
        if keypoints is None:
            keypoints = []
        
        return self.transform(image=image, keypoints=keypoints, **kwargs)


class ValTransform:
    """
    Transformations for validation data
    """
    def __init__(self, img_size):
        self.img_size = img_size
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def __call__(self, image, keypoints=None, **kwargs):
        if keypoints is None:
            keypoints = []
        
        return self.transform(image=image, keypoints=keypoints, **kwargs)