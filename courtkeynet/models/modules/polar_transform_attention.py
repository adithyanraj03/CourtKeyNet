import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ...cpp import courtkeynet_cpp

class PolarTransformAttention(nn.Module):
    """
    Novel attention mechanism that transforms features to polar space
    Better suited for court corners which often appear at the boundaries
    """
    def __init__(self, channels):
        super(PolarTransformAttention, self).__init__()
        self.channels = channels
        
        # Polar transform parameters
        self.angle_bins = 16
        self.radius_bins = 8
        
        # C++ accelerated implementation
        self.polar_transform = courtkeynet_cpp.PolarTransform(self.angle_bins, self.radius_bins)
        
        # Feature extraction in polar space
        self.polar_conv = nn.Conv2d(channels, channels, 
                                   kernel_size=3, padding=1, groups=4)
        
        # Attention generation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, c, h, w = x.shape
        
        # Convert to numpy for C++ processing
        x_numpy = x.detach().cpu().numpy()
        
        # Transform to polar space (using C++ implementation)
        polar_features_numpy = self.polar_transform.cartesian_to_polar(x_numpy)
        polar_features = torch.from_numpy(polar_features_numpy).to(x.device)
        
        # Process in polar space
        processed_polar = self.polar_conv(polar_features)
        
        # Convert back to numpy for C++ processing
        processed_polar_numpy = processed_polar.detach().cpu().numpy()
        
        # Transform back to cartesian space (using C++ implementation)
        processed_cartesian_numpy = self.polar_transform.polar_to_cartesian(
            processed_polar_numpy, h, w)
        processed_cartesian = torch.from_numpy(processed_cartesian_numpy).to(x.device)
        
        # Generate attention weights
        attention = self.attention_conv(processed_cartesian)
        
        # Apply attention
        return x * attention