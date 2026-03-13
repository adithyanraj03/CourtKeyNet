"""
YOLO Pose Format Dataset Loader
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from tqdm import tqdm


class CourtKeypointDataset(Dataset):
    """
    Loads YOLO pose labels: class cx cy w h kx1 ky1 v1 ... kx4 ky4 v4
    """
    def __init__(self, root, split='train', imgsz=640, augment=False, config=None, 
                 validate_labels=False, skip_flipped=False):
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.augment = augment and split == 'train'
        self.skip_flipped = skip_flipped
        
        # Image and label paths
        self.img_dir = self.root / split / 'images'
        self.lbl_dir = self.root / split / 'labels'
        
        # Get image files
        all_images = sorted(list(self.img_dir.glob('*.jpg')) + 
                           list(self.img_dir.glob('*.png')) +
                           list(self.img_dir.glob('*.jpeg')))
        
        # Filter out flipped images if requested (they often have incorrect labels)
        if skip_flipped:
            orig_count = len(all_images)
            all_images = [p for p in all_images if '_flip' not in p.name.lower()]
            print(f"[{split}] Skipping flipped images: {orig_count} → {len(all_images)}")
        
        if validate_labels:
            print(f"[{split}] Scanning {len(all_images)} images for valid keypoint labels...")
            
            # Filter images with valid keypoint labels
            self.images = []
            invalid_count = 0
            for img_path in tqdm(all_images, desc=f"Validating {split} labels"):
                lbl_path = self.lbl_dir / (img_path.stem + '.txt')
                if lbl_path.exists():
                    try:
                        with open(lbl_path, 'r') as f:
                            label = f.read().strip().split()
                        # Need at least: class cx cy w h + 4 keypoints * 3 values = 17 values
                        if len(label) >= 17:
                            self.images.append(img_path)
                        else:
                            invalid_count += 1
                    except Exception:
                        invalid_count += 1
                else:
                    invalid_count += 1
            
            print(f"[{split}] Found {len(self.images)} valid images ({invalid_count} skipped)")
        else:
            # Fast mode: assume all labels are valid
            self.images = all_images
            print(f"[{split}] Found {len(self.images)} images")
        
        # Augmentation pipeline
        if self.augment and config:
            aug_cfg = config.get('dataset', config)
            self.transform = A.Compose([
                A.ColorJitter(
                    hue=aug_cfg.get('hsv_h', 0.015),
                    saturation=aug_cfg.get('hsv_s', 0.7),
                    brightness=aug_cfg.get('hsv_v', 0.4)
                ),
                A.Affine(
                    rotate=(-aug_cfg.get('degrees', 5.0), aug_cfg.get('degrees', 5.0)),
                    translate_percent=aug_cfg.get('translate', 0.1),
                    scale=(1-aug_cfg.get('scale', 0.3), 1+aug_cfg.get('scale', 0.3)),
                    shear=0
                ),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            # Return a random valid sample instead
            return self.__getitem__(np.random.randint(len(self.images)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        
        # Load label
        lbl_path = self.lbl_dir / (img_path.stem + '.txt')
        try:
            with open(lbl_path, 'r') as f:
                label = f.read().strip().split()
            
            # Parse: [class, cx, cy, w, h, kx1, ky1, v1, ..., kx4, ky4, v4]
            vals = list(map(float, label))
            if len(vals) < 17:  # Need at least 5 bbox + 12 keypoint values
                raise ValueError(f"Label has only {len(vals)} values, need 17+")
            kpts_raw = np.array(vals[5:17]).reshape(4, 3)  # (4, 3) - only take first 4 keypoints
        except Exception as e:
            # Return a random valid sample instead
            return self.__getitem__(np.random.randint(len(self.images)))
        
        # Convert normalized to pixel coords for augmentation
        kpts_px = kpts_raw.copy()
        kpts_px[:, 0] *= w0
        kpts_px[:, 1] *= h0
        
        # Apply augmentation
        if self.transform:
            transformed = self.transform(
                image=img,
                keypoints=kpts_px[:, :2]
            )
            img = transformed['image']
            kpts_px[:, :2] = np.array(transformed['keypoints'])
        
        # Resize image
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        
        # Normalize keypoints to [0, 1] based on original image size
        # Note: Albumentations Affine keeps image size constant, so w0/h0 is still correct
        kpts = kpts_px.copy()
        kpts[:, 0] /= w0
        kpts[:, 1] /= h0
        kpts = np.clip(kpts, 0, 1)
        
        # Convert to torch
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        kpts_xy = torch.from_numpy(kpts[:, :2]).float()
        vis = torch.from_numpy(kpts[:, 2:3]).float()
        
        return {
            'img': img,
            'kpts': kpts_xy,
            'vis': vis,
            'img_path': str(img_path)
        }
