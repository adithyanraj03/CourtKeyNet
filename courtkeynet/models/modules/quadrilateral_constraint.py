import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QuadrilateralConstraintModule(nn.Module):
    """
    Enforces geometric consistency among detected keypoints,
    ensuring predictions form valid court boundaries.
    
    Explicitly calculates quadrilateral properties (edge lengths, diagonal lengths,
    internal angles) and integrates them into the feature space.
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super(QuadrilateralConstraintModule, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # MLP for encoding geometric constraints
        self.constraint_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim // 2),  # 10 = 4 edges + 2 diagonals + 4 angles
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Projection for combining features
        self.fusion = nn.Linear(feature_dim * 2, feature_dim)
    
    def compute_edge_lengths(self, keypoints):
        """Compute lengths of the four edges of the quadrilateral"""
        batch_size = keypoints.shape[0]
        num_keypoints = keypoints.shape[1]
        
        edge_lengths = []
        for i in range(num_keypoints):
            next_idx = (i + 1) % num_keypoints
            
            # Extract current and next keypoint
            current = keypoints[:, i, :]  # [batch_size, 2]
            next_point = keypoints[:, next_idx, :]  # [batch_size, 2]
            
            # Compute Euclidean distance
            dist = torch.sqrt(
                torch.sum((next_point - current) ** 2, dim=1)
            )  # [batch_size]
            
            edge_lengths.append(dist)
        
        # Stack to [batch_size, 4]
        return torch.stack(edge_lengths, dim=1)
    
    def compute_diagonal_lengths(self, keypoints):
        """Compute lengths of the two diagonals of the quadrilateral"""
        batch_size = keypoints.shape[0]
        
        # Diagonal 1: keypoints[0] to keypoints[2]
        diag1 = torch.sqrt(
            torch.sum((keypoints[:, 0, :] - keypoints[:, 2, :]) ** 2, dim=1)
        )  # [batch_size]
        
        # Diagonal 2: keypoints[1] to keypoints[3]
        diag2 = torch.sqrt(
            torch.sum((keypoints[:, 1, :] - keypoints[:, 3, :]) ** 2, dim=1)
        )  # [batch_size]
        
        # Stack to [batch_size, 2]
        return torch.stack([diag1, diag2], dim=1)
    
    def compute_internal_angles(self, keypoints):
        """Compute the four internal angles of the quadrilateral"""
        batch_size = keypoints.shape[0]
        num_keypoints = keypoints.shape[1]
        
        angles = []
        for i in range(num_keypoints):
            prev_idx = (i - 1) % num_keypoints
            next_idx = (i + 1) % num_keypoints
            
            # Extract vectors from current keypoint to previous and next
            current = keypoints[:, i, :]  # [batch_size, 2]
            prev_point = keypoints[:, prev_idx, :]  # [batch_size, 2]
            next_point = keypoints[:, next_idx, :]  # [batch_size, 2]
            
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
    
    def encode_constraints(self, keypoints):
        """Encode geometric constraints from keypoints"""
        # Compute geometric properties
        edge_lengths = self.compute_edge_lengths(keypoints)
        diagonal_lengths = self.compute_diagonal_lengths(keypoints)
        internal_angles = self.compute_internal_angles(keypoints)
        
        # Normalize edge lengths by perimeter
        perimeter = torch.sum(edge_lengths, dim=1, keepdim=True)
        edge_lengths_norm = edge_lengths / (perimeter + 1e-6)
        
        # Normalize diagonal lengths by their sum
        diag_sum = torch.sum(diagonal_lengths, dim=1, keepdim=True)
        diagonal_lengths_norm = diagonal_lengths / (diag_sum + 1e-6)
        
        # Concatenate all geometric properties
        geometric_features = torch.cat([
            edge_lengths_norm,
            diagonal_lengths_norm,
            internal_angles / 180.0  # Normalize angles to [0, 1]
        ], dim=1)  # [batch_size, 10]
        
        # Encode through MLP
        constraint_features = self.constraint_encoder(geometric_features)  # [batch_size, feature_dim]
        
        return constraint_features
    
    def forward(self, keypoint_features, keypoints):
        """
        Args:
            keypoint_features: Features associated with each keypoint 
                               [batch_size, num_keypoints, feature_dim]
            keypoints: Current keypoint coordinates [batch_size, num_keypoints, 2]
        
        Returns:
            enhanced_features: Keypoint features enhanced with geometric constraints
                               [batch_size, num_keypoints, feature_dim]
        """
        batch_size = keypoints.shape[0]
        num_keypoints = keypoints.shape[1]
        
        # Encode geometric constraints
        constraint_features = self.encode_constraints(keypoints)  # [batch_size, feature_dim]
        
        # Expand constraint features for each keypoint
        constraint_features_expanded = constraint_features.unsqueeze(1).expand(
            -1, num_keypoints, -1
        )  # [batch_size, num_keypoints, feature_dim]
        
        # Combine with original keypoint features
        combined_features = torch.cat([
            keypoint_features,
            constraint_features_expanded
        ], dim=2)  # [batch_size, num_keypoints, feature_dim*2]
        
        # Fuse features
        enhanced_features = self.fusion(combined_features)  # [batch_size, num_keypoints, feature_dim]
        
        return enhanced_features