"""
Evaluation Metrics
"""
import torch
import numpy as np
from shapely.geometry import Polygon


def compute_pck(pred_kpts, gt_kpts, threshold=0.05):
    """
    Percentage of Correct Keypoints (PCK)
    threshold: normalized distance threshold (e.g., 5% of image diagonal)
    """
    dist = torch.norm(pred_kpts - gt_kpts, dim=-1)  # (B, K)
    diagonal = np.sqrt(2)  # Normalized image diagonal
    correct = (dist < threshold * diagonal).float()
    return correct.mean().item()


def compute_iou(pred_kpts, gt_kpts):
    """
    IoU of quadrilaterals using Shapely for accurate polygon intersection
    """
    pred_np = pred_kpts.cpu().numpy()
    gt_np = gt_kpts.cpu().numpy()
    
    ious = []
    for i in range(pred_np.shape[0]):
        try:
            pred_poly = Polygon(pred_np[i])
            gt_poly = Polygon(gt_np[i])
            
            if not pred_poly.is_valid or not gt_poly.is_valid:
                ious.append(0.0)
                continue
                
            inter = pred_poly.intersection(gt_poly).area
            union = pred_poly.union(gt_poly).area
            
            if union < 1e-8:
                ious.append(0.0)
            else:
                ious.append(inter / union)
        except Exception:
            ious.append(0.0)
    
    return np.mean(ious) if ious else 0.0
