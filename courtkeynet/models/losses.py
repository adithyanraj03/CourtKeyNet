import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GeometricConsistencyLoss(nn.Module):
    """
    Novel loss function that enforces proper quadrilateral properties
    during training, including edge length ratios, diagonal ratios,
    and angle consistency.
    """
    def __init__(self):
        super(GeometricConsistencyLoss, self).__init__()
    
    def compute_edge_lengths(self, points):
        """Compute lengths of the four edges of the quadrilateral"""
        batch_size = points.shape[0]
        num_points = points.shape[1]
        
        edge_lengths = []
        for i in range(num_points):
            next_idx = (i + 1) % num_points
            
            # Extract current and next point
            current = points[:, i, :]  # [batch_size, 2]
            next_point = points[:, next_idx, :]  # [batch_size, 2]
            
            # Compute Euclidean distance
            dist = torch.sqrt(
                torch.sum((next_point - current) ** 2, dim=1)
            )  # [batch_size]
            
            edge_lengths.append(dist)
        
        # Stack to [batch_size, 4]
        return torch.stack(edge_lengths, dim=1)
    
    def compute_diagonal_lengths(self, points):
        """Compute lengths of the two diagonals of the quadrilateral"""
        batch_size = points.shape[0]
        
        # Diagonal 1: points[0] to points[2]
        diag1 = torch.sqrt(
            torch.sum((points[:, 0, :] - points[:, 2, :]) ** 2, dim=1)
        )  # [batch_size]
        
        # Diagonal 2: points[1] to points[3]
        diag2 = torch.sqrt(
            torch.sum((points[:, 1, :] - points[:, 3, :]) ** 2, dim=1)
        )  # [batch_size]
        
        # Stack to [batch_size, 2]
        return torch.stack([diag1, diag2], dim=1)
    
    def compute_internal_angles(self, points):
        """Compute the four internal angles of the quadrilateral"""
        batch_size = points.shape[0]
        num_points = points.shape[1]
        
        angles = []
        for i in range(num_points):
            prev_idx = (i - 1) % num_points
            next_idx = (i + 1) % num_points
            
            # Extract vectors from current point to previous and next
            current = points[:, i, :]  # [batch_size, 2]
            prev_point = points[:, prev_idx, :]  # [batch_size, 2]
            next_point = points[:, next_idx, :]  # [batch_size, 2]
            
            # Compute vectors
            vec_to_prev = prev_point - current  # [batch_size, 2]
            vec_to_next = next_point - current  # [batch_size, 2]
            
            # Normalize vectors
            norm_prev = torch.sqrt(torch.sum(vec_to_prev ** 2, dim=1, keepdim=True))
            norm_next = torch.sqrt(torch.sum(vec_to_next ** 2, dim=1, keepdim=True))
            
            vec_to_prev_norm = vec_to_prev / (norm_prev + 1e-6)
            vec_to_next_norm = vec_to_next / (norm_next + 1e-6)
            
            # Compute cosine of the angle
            cos_angle = torch.sum(vec_to_prev_norm * vec_to_next_norm, dim=1)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            
            # Compute angle in radians and convert to degrees
            angle = torch.acos(cos_angle) * 180.0 / math.pi
            
            angles.append(angle)
        
        # Stack to [batch_size, 4]
        return torch.stack(angles, dim=1)
    
    def edge_length_consistency_loss(self, pred_points, gt_points):
        """Loss for edge length ratios"""
        # Compute edge lengths
        pred_edge_lengths = self.compute_edge_lengths(pred_points)
        gt_edge_lengths = self.compute_edge_lengths(gt_points)
        
        # Compute perimeters
        pred_perimeter = torch.sum(pred_edge_lengths, dim=1, keepdim=True)
        gt_perimeter = torch.sum(gt_edge_lengths, dim=1, keepdim=True)
        
        # Normalize edge lengths
        pred_edge_norm = pred_edge_lengths / (pred_perimeter + 1e-6)
        gt_edge_norm = gt_edge_lengths / (gt_perimeter + 1e-6)
        
        # Compute L1 loss on normalized edge lengths
        loss = torch.mean(torch.abs(pred_edge_norm - gt_edge_norm))
        
        return loss
    
    def diagonal_consistency_loss(self, pred_points, gt_points):
        """Loss for diagonal length ratios"""
        # Compute diagonal lengths
        pred_diagonals = self.compute_diagonal_lengths(pred_points)
        gt_diagonals = self.compute_diagonal_lengths(gt_points)
        
        # Compute diagonal sums
        pred_diag_sum = torch.sum(pred_diagonals, dim=1, keepdim=True)
        gt_diag_sum = torch.sum(gt_diagonals, dim=1, keepdim=True)
        
        # Normalize diagonal lengths
        pred_diag_norm = pred_diagonals / (pred_diag_sum + 1e-6)
        gt_diag_norm = gt_diagonals / (gt_diag_sum + 1e-6)
        
        # Compute L1 loss on normalized diagonal lengths
        loss = torch.mean(torch.abs(pred_diag_norm - gt_diag_norm))
        
        return loss
    
    def angle_consistency_loss(self, pred_points, gt_points):
        """Loss for internal angle consistency"""
        # Compute internal angles
        pred_angles = self.compute_internal_angles(pred_points)
        gt_angles = self.compute_internal_angles(gt_points)
        
        # Compute L1 loss on angles
        loss = torch.mean(torch.abs(pred_angles - gt_angles))
        
        # Normalize loss to make it on par with other losses (degrees to [0,1] range)
        loss = loss / 180.0
        
        return loss
    
    def forward(self, pred_points, gt_points):
        """
        Args:
            pred_points: Predicted keypoints [batch_size, 4, 2]
            gt_points: Ground truth keypoints [batch_size, 4, 2]
        
        Returns:
            loss: Geometric consistency loss
        """
        edge_loss = self.edge_length_consistency_loss(pred_points, gt_points)
        diag_loss = self.diagonal_consistency_loss(pred_points, gt_points)
        angle_loss = self.angle_consistency_loss(pred_points, gt_points)
        
        # Combine losses
        total_loss = edge_loss + diag_loss + angle_loss
        
        return total_loss, {
            'edge_loss': edge_loss,
            'diag_loss': diag_loss,
            'angle_loss': angle_loss
        }

class HeatmapLoss(nn.Module):
    """
    Loss function for keypoint heatmap prediction.
    Uses MSE loss between predicted and ground truth heatmaps.
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
    
    def forward(self, pred_heatmaps, gt_heatmaps):
        """
        Args:
            pred_heatmaps: Predicted heatmaps [batch_size, num_keypoints, height, width]
            gt_heatmaps: Ground truth heatmaps [batch_size, num_keypoints, height, width]
        
        Returns:
            loss: Heatmap loss
        """
        return self.criterion(pred_heatmaps, gt_heatmaps)

class KeypointLoss(nn.Module):
    """
    Loss function for keypoint regression.
    Uses L1 loss between predicted and ground truth keypoints.
    """
    def __init__(self):
        super(KeypointLoss, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')
    
    def forward(self, pred_keypoints, gt_keypoints):
        """
        Args:
            pred_keypoints: Predicted keypoints [batch_size, num_keypoints, 2]
            gt_keypoints: Ground truth keypoints [batch_size, num_keypoints, 2]
        
        Returns:
            loss: Keypoint regression loss
        """
        return self.criterion(pred_keypoints, gt_keypoints)

class CourtKeyNetLoss(nn.Module):
    """
    Combined loss function for CourtKeyNet training.
    Includes keypoint regression loss, heatmap loss, court segmentation loss,
    and geometric consistency loss.
    """
    def __init__(self, keypoint_weight=1.0, heatmap_weight=1.0, 
                 court_weight=0.5, geometric_weight=1.0):
        super(CourtKeyNetLoss, self).__init__()
        self.keypoint_weight = keypoint_weight
        self.heatmap_weight = heatmap_weight
        self.court_weight = court_weight
        self.geometric_weight = geometric_weight
        
        self.keypoint_loss = KeypointLoss()
        self.heatmap_loss = HeatmapLoss()
        self.court_loss = nn.BCELoss()
        self.geometric_loss = GeometricConsistencyLoss()
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dictionary containing model outputs
                - 'keypoints': Predicted keypoints [batch_size, num_keypoints, 2]
                - 'keypoint_heatmaps': Predicted heatmaps [batch_size, num_keypoints, height, width]
                - 'court_heatmap': Predicted court segmentation [batch_size, 1, height, width]
            
            targets: Dictionary containing targets
                - 'keypoints': Ground truth keypoints [batch_size, num_keypoints, 2]
                - 'keypoint_heatmaps': Ground truth heatmaps [batch_size, num_keypoints, height, width]
                - 'court_mask': Ground truth court segmentation [batch_size, 1, height, width]
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual loss components
        """
        # Compute individual losses
        kp_loss = self.keypoint_loss(outputs['keypoints'], targets['keypoints'])
        
        # Heatmap loss
        if 'keypoint_heatmaps' in outputs and 'keypoint_heatmaps' in targets:
            hm_loss = self.heatmap_loss(outputs['keypoint_heatmaps'], targets['keypoint_heatmaps'])
        else:
            hm_loss = torch.tensor(0.0, device=kp_loss.device)
        
        # Court segmentation loss
        if 'court_heatmap' in outputs and 'court_mask' in targets:
            court_loss = self.court_loss(outputs['court_heatmap'], targets['court_mask'])
        else:
            court_loss = torch.tensor(0.0, device=kp_loss.device)
        
        # Geometric consistency loss
        geo_loss, geo_components = self.geometric_loss(outputs['keypoints'], targets['keypoints'])
        
        # Combine losses
        total_loss = (
            self.keypoint_weight * kp_loss +
            self.heatmap_weight * hm_loss +
            self.court_weight * court_loss +
            self.geometric_weight * geo_loss
        )
        
        # Create loss dictionary
        loss_dict = {
            'total_loss': total_loss,
            'keypoint_loss': kp_loss,
            'heatmap_loss': hm_loss,
            'court_loss': court_loss,
            'geometric_loss': geo_loss,
            'edge_loss': geo_components['edge_loss'],
            'diag_loss': geo_components['diag_loss'],
            'angle_loss': geo_components['angle_loss']
        }
        
        return loss_dict