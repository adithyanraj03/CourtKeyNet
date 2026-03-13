"""
Court Keypoint Dataset Augmentation
Handles keypoint transformations with GUI folder selection

Court corner order convention:
    0: Top-Left
    1: Top-Right  
    2: Bottom-Right
    3: Bottom-Left
"""

import os
import cv2
import albumentations as A
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


def read_yolo_keypoints(label_path, w, h):
    """
    Reads a YOLO format keypoint file.
    YOLO Keypoint Format: 
    <class> <cx> <cy> <bw> <bh> <kp1_x> <kp1_y> <kp1_vis> <kp2_x> ...

    Returns: 
        keypoints: list of (x_pixel, y_pixel) tuples
        meta: (class_id, [cx, cy, bw, bh], [vis1, vis2, ...])
    """
    if not os.path.exists(label_path):
        return None, None

    with open(label_path, 'r') as f:
        lines = f.readlines()

    if not lines or len(lines[0].strip()) == 0:
        return None, None

    parts = list(map(float, lines[0].strip().split()))

    if len(parts) < 17:  # class(1) + bbox(4) + 4 keypoints(12) = 17 minimum
        print(f"Warning: Invalid label format in {label_path}")
        return None, None

    class_id = int(parts[0])
    bbox = parts[1:5]  # cx, cy, bw, bh (normalized)

    # Extract keypoints: x, y, visibility for each
    raw_kpts = parts[5:]
    keypoints = []
    visibilities = []

    for i in range(0, len(raw_kpts), 3):
        if i + 2 >= len(raw_kpts):
            break
        kx, ky, kv = raw_kpts[i], raw_kpts[i+1], int(raw_kpts[i+2])
        # Convert normalized (0-1) to pixel coordinates
        keypoints.append((kx * w, ky * h)) 
        visibilities.append(kv)

    if len(keypoints) != 4:
        print(f"Warning: Expected 4 keypoints, got {len(keypoints)} in {label_path}")
        return None, None

    return keypoints, (class_id, bbox, visibilities)


def compute_bbox_from_keypoints(keypoints, w, h, padding=0.05):
    """
    Compute bounding box from keypoints with padding.
    Returns normalized [cx, cy, bw, bh]
    """
    xs = [kp[0] for kp in keypoints]
    ys = [kp[1] for kp in keypoints]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add padding
    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding

    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)

    # Convert to YOLO format (normalized cx, cy, bw, bh)
    cx = (x_min + x_max) / 2 / w
    cy = (y_min + y_max) / 2 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h

    return [cx, cy, bw, bh]


def validate_keypoints(keypoints, w, h, margin=0.01):
    """
    Check if all keypoints are within image bounds.
    Returns True if valid, False otherwise.
    """
    for (kx, ky) in keypoints:
        if kx < -margin * w or kx > w * (1 + margin):
            return False
        if ky < -margin * h or ky > h * (1 + margin):
            return False
    return True


def clamp_keypoints(keypoints, w, h):
    """
    Clamp keypoints to image bounds.
    """
    clamped = []
    for (kx, ky) in keypoints:
        kx = max(0, min(w - 1, kx))
        ky = max(0, min(h - 1, ky))
        clamped.append((kx, ky))
    return clamped


def save_yolo_keypoints(save_path, keypoints, meta, w, h):
    """
    Saves keypoints back to YOLO format.
    Recomputes bbox from keypoints for accuracy.
    """
    class_id, _, visibilities = meta

    # Recompute bbox from transformed keypoints
    bbox = compute_bbox_from_keypoints(keypoints, w, h)

    # Normalize keypoints back to 0-1
    parts = [class_id] + bbox

    for (kx, ky), kv in zip(keypoints, visibilities):
        parts.extend([kx / w, ky / h, kv])

    line_str = " ".join(map(str, parts))

    with open(save_path, 'w') as f:
        f.write(line_str + "\n")


def select_folder(title="Select Folder"):
    """
    Open a GUI dialog to select a folder.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path


def augment_dataset(
    input_dir, 
    label_dir,
    output_img_dir=None,
    output_label_dir=None,
    include_brightness=True,
    include_rotation=True,
    rotation_limit=10,
    skip_invalid=True
):
    """
    Augment dataset with proper keypoint handling.

    Args:
        input_dir: Path to input images
        label_dir: Path to input labels
        output_img_dir: Path to save augmented images (None = same as input)
        output_label_dir: Path to save augmented labels (None = same as input)
        include_brightness: Include brightness/contrast augmentation
        include_rotation: Include rotation augmentation
        rotation_limit: Max rotation angle in degrees
        skip_invalid: Skip samples where keypoints go out of bounds
    """

    if output_img_dir is None:
        output_img_dir = input_dir
    if output_label_dir is None:
        output_label_dir = label_dir

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Albumentations keypoint params
    kp_params = A.KeypointParams(
        format='xy', 
        remove_invisible=False,
        angle_in_degrees=True
    )

    # Build augmentation list
    augmentations = []

    if include_brightness:
        augmentations.append((
            A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                )
            ], keypoint_params=kp_params),
            "bright"
        ))

    if include_rotation:
        augmentations.append((
            A.Compose([
                A.Rotate(
                    limit=rotation_limit, 
                    p=1.0, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0,
                    crop_border=False
                )
            ], keypoint_params=kp_params),
            "rot"
        ))

    # Combined augmentation (brightness + rotation)
    if include_brightness and include_rotation:
        augmentations.append((
            A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.Rotate(
                    limit=rotation_limit, 
                    p=1.0, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0,
                    crop_border=False
                )
            ], keypoint_params=kp_params),
            "bright_rot"
        ))

    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    print(f"\nFound {len(image_files)} images")
    print(f"Augmentations: {[a[1] for a in augmentations]}")

    stats = {
        'processed': 0,
        'skipped_no_label': 0,
        'skipped_invalid': 0,
        'augmented': 0
    }

    for file_path in tqdm(image_files, desc="Augmenting"):
        # Read image
        image = cv2.imread(file_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read corresponding label
        name_no_ext = Path(file_path).stem
        label_path = os.path.join(label_dir, name_no_ext + ".txt")

        keypoints, meta = read_yolo_keypoints(label_path, w, h)
        if keypoints is None:
            stats['skipped_no_label'] += 1
            continue

        stats['processed'] += 1

        # Apply each augmentation
        for aug_pipeline, suffix in augmentations:
            try:
                augmented = aug_pipeline(image=image_rgb, keypoints=keypoints)
                aug_img = augmented['image']
                aug_kpts = list(augmented['keypoints'])

                # Validate keypoints are within bounds
                if skip_invalid and not validate_keypoints(aug_kpts, w, h):
                    stats['skipped_invalid'] += 1
                    continue

                # Clamp keypoints to image bounds
                aug_kpts = clamp_keypoints(aug_kpts, w, h)

                # Save augmented image
                new_img_name = f"{name_no_ext}_{suffix}.jpg"
                save_img_path = os.path.join(output_img_dir, new_img_name)
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_img_path, aug_img_bgr)

                # Save augmented label
                new_label_name = f"{name_no_ext}_{suffix}.txt"
                save_label_path = os.path.join(output_label_dir, new_label_name)
                save_yolo_keypoints(save_label_path, aug_kpts, meta, w, h)

                stats['augmented'] += 1

            except Exception as e:
                print(f"\nError augmenting {file_path} with {suffix}: {e}")
                continue

    print(f"\n{'='*50}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*50}")
    print(f"Processed:          {stats['processed']}")
    print(f"Skipped (no label): {stats['skipped_no_label']}")
    print(f"Skipped (invalid):  {stats['skipped_invalid']}")
    print(f"Augmented samples:  {stats['augmented']}")
    print(f"Total new files:    {stats['augmented']}")


def verify_augmentation(img_dir, label_dir, num_samples=5):
    """
    Visualize augmented samples to verify correctness.
    """
    import random

    image_files = glob.glob(os.path.join(img_dir, "*_bright.jpg"))
    if not image_files:
        image_files = glob.glob(os.path.join(img_dir, "*.jpg"))

    samples = random.sample(image_files, min(num_samples, len(image_files)))

    output_dir = Path("augmentation_verify")
    output_dir.mkdir(exist_ok=True)

    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # TL, TR, BR, BL
    labels = ["TL", "TR", "BR", "BL"]

    for img_path in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        name = Path(img_path).stem
        label_path = os.path.join(label_dir, name + ".txt")

        keypoints, meta = read_yolo_keypoints(label_path, w, h)
        if keypoints is None:
            continue

        # Draw keypoints
        pts = np.array(keypoints, dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        for i, (pt, color, label) in enumerate(zip(pts, colors, labels)):
            cv2.circle(img, tuple(pt), 8, color, -1)
            cv2.putText(img, f"{i}:{label}", (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imwrite(str(output_dir / f"verify_{name}.jpg"), img)

    print(f"\nSaved verification images to: {output_dir}")


if __name__ == "__main__":
    print("="*60)
    print("Court Keypoint Dataset Augmentation Tool")
    print("="*60)

    # Select images folder
    print("\nStep 1: Select IMAGES folder...")
    images_dir = select_folder("Select Images Folder")

    if not images_dir:
        print("No images folder selected. Exiting.")
        exit()

    print(f"Selected images folder: {images_dir}")

    # Select labels folder
    print("\nStep 2: Select LABELS folder...")
    labels_dir = select_folder("Select Labels Folder")

    if not labels_dir:
        print("No labels folder selected. Exiting.")
        exit()

    print(f"Selected labels folder: {labels_dir}")

    # Confirm
    print("\n" + "="*60)
    print("Configuration:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"  Augmentations: Brightness, Rotation, Brightness+Rotation")
    print(f"  Rotation Limit: ±10 degrees")
    print("="*60)

    proceed = input("\nProceed with augmentation? (y/n): ").strip().lower()

    if proceed != 'y':
        print("Augmentation cancelled.")
        exit()

    # Run augmentation
    augment_dataset(
        input_dir=images_dir,
        label_dir=labels_dir,
        output_img_dir=images_dir,
        output_label_dir=labels_dir,
        include_brightness=True,
        include_rotation=True,
        rotation_limit=10
    )

    # Ask for verification
    verify = input("\nVisualize results for verification? (y/n): ").strip().lower()

    if verify == 'y':
        verify_augmentation(images_dir, labels_dir)

    print("\nDone!")
