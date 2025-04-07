import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import yaml
from pathlib import Path

class CourtKeypointDataset(data.Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        
        # Get all image files
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Verify corresponding label files exist
        self.label_files = []
        for img_file in self.img_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(label_dir, base_name + '.txt')
            
            if os.path.exists(label_file):
                self.label_files.append(label_file)
            else:
                print(f"Warning: No label file for {img_file}")
                self.img_files.remove(img_file)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load label
        label_path = self.label_files[idx]
        with open(label_path, 'r') as f:
            label_data = f.read().strip()
        
        # Parse our custom annotation format
        parts = label_data.split()
        class_id = int(parts[0])
        
        # Get bounding box (normalized coordinates)
        box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        
        # Get keypoints
        keypoints = []
        visibilities = []
        
        for i in range(5, len(parts), 3):
            if i + 2 < len(parts):
                kx = float(parts[i])
                ky = float(parts[i + 1])
                visibility = int(float(parts[i + 2]))
                
                keypoints.append([kx, ky])
                visibilities.append(visibility)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=img, keypoints=keypoints)
            img = transformed['image']
            keypoints = transformed['keypoints']
        
        # Prepare keypoints with visibility
        keypoints_with_vis = []
        for idx, (kx, ky) in enumerate(keypoints):
            keypoints_with_vis.append([kx, ky, visibilities[idx]])
        
        # Create target dict
        target = {
            'box': torch.tensor(box, dtype=torch.float32),
            'keypoints': torch.tensor(keypoints_with_vis, dtype=torch.float32),
            'class_id': class_id
        }
        
        # Convert image to tensor if not already done by transform
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        
        return img, target


def collate_fn(batch):
    """
    Custom collate function to handle variable keypoint counts
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, 0)
    
    return images, targets