import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfCalibrationBlock(nn.Module):
    """
    Self-calibration block that adaptively adjusts feature extraction
    based on input characteristics.
    
    Inspired by Self-Calibrated Convolutions with dual attention mechanisms
    for spatial and channel calibration.
    """
    def __init__(self, channels, reduction=8):
        super(SelfCalibrationBlock, self).__init__()
        self.channels = channels
        
        # Spatial calibration branch
        self.spatial_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.spatial_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=2, groups=channels)
        self.spatial_conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=3, groups=channels)
        self.spatial_combine = nn.Conv2d(channels * 3, channels, kernel_size=1)
        
        # Channel calibration branch
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_down = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.channel_act = nn.ReLU(inplace=True)
        self.channel_up = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        
        # Fusion branch
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        # Spatial calibration path
        spatial1 = self.spatial_conv1(x)
        spatial2 = self.spatial_conv2(x)
        spatial3 = self.spatial_conv3(x)
        spatial_concat = torch.cat([spatial1, spatial2, spatial3], dim=1)
        spatial_out = self.spatial_combine(spatial_concat)
        spatial_attention = torch.sigmoid(spatial_out)
        
        # Channel calibration path
        channel_pool = self.channel_pool(x)
        channel_down = self.channel_down(channel_pool)
        channel_act = self.channel_act(channel_down)
        channel_up = self.channel_up(channel_act)
        channel_attention = torch.sigmoid(channel_up)
        
        # Apply attentions
        spatial_enhanced = x * spatial_attention
        channel_enhanced = x * channel_attention
        
        # Combine features
        combined = torch.cat([spatial_enhanced, channel_enhanced], dim=1)
        out = self.fusion(combined)
        out = self.norm(out)
        out = self.act(out)
        
        # Residual connection
        return out + x