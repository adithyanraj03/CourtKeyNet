import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OrientedConv(nn.Module):
    """Convolution with oriented filters at specific angles"""
    def __init__(self, in_channels, out_channels, kernel_size=3, angle=0):
        super(OrientedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.angle = angle
        
        # Create standard convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        
        # Initialize with orientation bias
        self._initialize_oriented_kernel()
    
    def _initialize_oriented_kernel(self):
        # Initialize kernel with orientation bias
        center = self.kernel_size // 2
        with torch.no_grad():
            for i in range(self.out_channels):
                for j in range(self.in_channels):
                    # Create oriented filter based on angle
                    for y in range(self.kernel_size):
                        for x in range(self.kernel_size):
                            # Calculate direction from center
                            dy = y - center
                            dx = x - center
                            
                            # Skip center pixel
                            if dx == 0 and dy == 0:
                                continue
                            
                            # Calculate angle from horizontal
                            pixel_angle = math.atan2(dy, dx) * 180 / math.pi
                            
                            # Calculate angular distance
                            angular_diff = min(
                                abs(pixel_angle - self.angle),
                                abs(pixel_angle - self.angle + 360),
                                abs(pixel_angle - self.angle - 360)
                            )
                            
                            # Set weight based on alignment with target angle
                            if angular_diff < 30:
                                self.conv.weight.data[i, j, y, x] = 0.5
                            else:
                                self.conv.weight.data[i, j, y, x] = -0.1
    
    def forward(self, x):
        return self.conv(x)

class CourtFeatureAggregator(nn.Module):
    """
    Aggregates court features from multiple orientations.
    Uses multiple oriented filters to capture court boundaries 
    from various viewing angles.
    """
    def __init__(self, in_channels, out_channels, num_orientations=8):
        super(CourtFeatureAggregator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_orientations = num_orientations
        
        # Create convolutions for different orientations
        self.conv_layers = nn.ModuleList()
        for i in range(num_orientations):
            angle = i * (360 / num_orientations)
            self.conv_layers.append(
                OrientedConv(
                    in_channels,
                    out_channels // num_orientations,
                    kernel_size=5,
                    angle=angle
                )
            )
        
        # Add 1x1 convolution to combine features
        self.combine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        # Apply oriented convolutions
        oriented_features = []
        for conv in self.conv_layers:
            oriented_features.append(conv(x))
        
        # Combine features from all orientations
        combined = torch.cat(oriented_features, dim=1)
        
        # Final processing
        output = self.combine(combined)
        
        return output