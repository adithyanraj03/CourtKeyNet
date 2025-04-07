import os
import cv2
import torch
import numpy as np
import torch.utils.data as data
from ..utils.annotation import CourtAnnotationDecoder


class CourtKeypointDataset(data.Dataset):
    """
    Dataset for badminton court keypoint detection
    """
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
        
        # Parse annotation
        annotation = CourtAnnotationDecoder.decode_keypoint_format(
            label_data, img.shape[1], img.shape[0]
        )
        
        # Get keypoints and convert to absolute coordinates for transformations
        keypoints = annotation['keypoints'].numpy()
        abs_keypoints = []
        visibilities = []
        
        for kp in keypoints:
            x, y, v = kp
            abs_keypoints.append([x * img.shape[1], y * img.shape[0]])
            visibilities.append(int(v))
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=img, keypoints=abs_keypoints)
            img = transformed['image']
            keypoints = transformed['keypoints']
            
            # Convert back to normalized coordinates
            h, w = self.img_size, self.img_size
            keypoints_normalized = []
            
            for idx, (kx, ky) in enumerate(keypoints):
                kx_norm = kx / w
                ky_norm = ky / h
                keypoints_normalized.append([kx_norm, ky_norm, visibilities[idx]])
            
            keypoints = keypoints_normalized
        
        # Create target dict
        target = {
            'box': annotation['box'],
            'keypoints': torch.tensor(keypoints, dtype=torch.float32),
            'class_id': annotation['class_id']
        }
        
        return img, target