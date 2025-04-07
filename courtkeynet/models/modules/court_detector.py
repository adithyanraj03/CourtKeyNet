import torch
import torch.nn as nn
import torch.nn.functional as F

class CourtDetector(nn.Module):
    """
    Detects the court region as a binary segmentation mask.
    This helps focus the keypoint detection on the relevant area.
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super(CourtDetector, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Court detection head
        self.court_head = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        Args:
            features: Feature maps from the backbone [batch_size, feature_dim, H, W]
        
        Returns:
            court_map: Binary court segmentation mask [batch_size, 1, H, W]
        """
        court_map = self.court_head(features)
        return court_map