import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerAttentionBlock(nn.Module):
    """Self-attention block for modeling keypoint relationships"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(TransformerAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: [num_keypoints, batch_size, dim]
        # Self-attention
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + self.dropout1(attn_output)
        
        # Feed forward
        norm_x = self.norm2(x)
        ff_output = self.ff(norm_x)
        x = x + ff_output
        
        return x

class KeypointLocalizationModule(nn.Module):
    """
    Module for precise court corner detection using a hybrid approach
    with heatmaps and regression refinement.
    
    Incorporates contextual reasoning with transformer blocks and 
    geometric constraints for consistent keypoint localization.
    """
    def __init__(self, feature_dim, num_keypoints=4, hidden_dim=256, num_transformer_layers=2):
        super(KeypointLocalizationModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        
        # Heatmap generation branch
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_keypoints, kernel_size=1)
        )
        
        # Feature extraction at keypoint locations
        self.keypoint_features = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Transformer blocks for context modeling
        self.transformer_blocks = nn.ModuleList([
            TransformerAttentionBlock(hidden_dim) 
            for _ in range(num_transformer_layers)
        ])
        
        # Offset regressor for keypoint refinement
        self.offset_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # (x, y) offset for each keypoint
        )
    
    def extract_features_at_points(self, features, points):
        """Extract features at specified keypoint locations using bilinear sampling"""
        batch_size = features.shape[0]
        channels = features.shape[1]
        height = features.shape[2]
        width = features.shape[3]
        
        # Normalize points to [-1, 1] for grid_sample
        points_normalized = points.clone()
        points_normalized[:, :, 0] = 2.0 * points[:, :, 0] - 1.0  # x: [0, 1] -> [-1, 1]
        points_normalized[:, :, 1] = 2.0 * points[:, :, 1] - 1.0  # y: [0, 1] -> [-1, 1]
        
        # Reshape for grid_sample
        grid = points_normalized.view(batch_size, self.num_keypoints, 1, 2)
        
        # Sample features at keypoint locations
        sampled_features = F.grid_sample(
            features, grid, 
            mode='bilinear', 
            padding_mode='reflection',
            align_corners=True
        )
        
        # Reshape: [batch_size, channels, num_keypoints, 1] -> [batch_size, num_keypoints, channels]
        sampled_features = sampled_features.squeeze(-1).transpose(1, 2)
        
        return sampled_features
    
    def forward(self, features):
        batch_size = features.shape[0]
        
        # Generate heatmaps
        heatmaps = self.heatmap_head(features)
        
        # Find keypoint coordinates from heatmaps
        # Method 1: Simple argmax approach
        keypoints_flat = heatmaps.reshape(batch_size, self.num_keypoints, -1).softmax(dim=2)
        indices = keypoints_flat.argmax(dim=2)
        
        height = features.shape[2]
        width = features.shape[3]
        
        # Convert indices to x,y coordinates
        keypoints_y = (indices // width).float() / height
        keypoints_x = (indices % width).float() / width
        
        # Combine into [batch_size, num_keypoints, 2] tensor
        keypoints_initial = torch.stack([keypoints_x, keypoints_y], dim=2)
        
        # Extract features at keypoint locations
        keypoint_features = self.keypoint_features(features)
        extracted_features = self.extract_features_at_points(keypoint_features, keypoints_initial)
        
        # Process through transformer layers
        # [batch_size, num_keypoints, channels] -> [num_keypoints, batch_size, channels]
        transformer_input = extracted_features.transpose(0, 1)
        
        for transformer_block in self.transformer_blocks:
            transformer_input = transformer_block(transformer_input)
        
        # [num_keypoints, batch_size, channels] -> [batch_size, num_keypoints, channels]
        transformer_output = transformer_input.transpose(0, 1)
        
        # Predict refinement offsets
        offsets = self.offset_regressor(transformer_output)
        
        # Apply offsets to initial keypoints
        keypoints_refined = keypoints_initial + offsets
        
        # Ensure keypoints stay in [0, 1] range
        keypoints_refined = torch.clamp(keypoints_refined, 0.0, 1.0)
        
        return {
            'keypoints': keypoints_refined,
            'heatmaps': heatmaps,
            'initial_keypoints': keypoints_initial,
            'offsets': offsets
        }