import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CourtSpecificShapeKernel(nn.Module):
    """
    Applies specialized kernels designed to detect court-specific patterns.
    This includes horizontal lines, vertical lines, and corner shapes.
    """
    def __init__(self, in_channels, out_channels):
        super(CourtSpecificShapeKernel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Calculate output channels per pattern
        self.channels_per_pattern = out_channels // 6
        self.extra_channels = out_channels % 6
        
        # Define pattern-specific kernels
        self.horizontal_kernel = nn.Conv2d(
            in_channels, 
            self.channels_per_pattern, 
            kernel_size=(1, 5), 
            padding=(0, 2),
            bias=True
        )
        
        self.vertical_kernel = nn.Conv2d(
            in_channels, 
            self.channels_per_pattern, 
            kernel_size=(5, 1), 
            padding=(2, 0),
            bias=True
        )
        
        # Corner detectors - Top-Left, Top-Right, Bottom-Left, Bottom-Right
        self.corner_tl_kernel = nn.Conv2d(
            in_channels, 
            self.channels_per_pattern, 
            kernel_size=3, 
            padding=1,
            bias=True
        )
        
        self.corner_tr_kernel = nn.Conv2d(
            in_channels, 
            self.channels_per_pattern, 
            kernel_size=3, 
            padding=1,
            bias=True
        )
        
        self.corner_bl_kernel = nn.Conv2d(
            in_channels, 
            self.channels_per_pattern, 
            kernel_size=3, 
            padding=1,
            bias=True
        )
        
        self.corner_br_kernel = nn.Conv2d(
            in_channels, 
            self.channels_per_pattern + self.extra_channels, 
            kernel_size=3, 
            padding=1,
            bias=True
        )
        
        # Initialize kernels with specialized patterns
        self._initialize_kernels()
        
    def _initialize_kernels(self):
        # Initialize horizontal line detector
        nn.init.normal_(self.horizontal_kernel.weight, std=0.01)
        nn.init.zeros_(self.horizontal_kernel.bias)
        with torch.no_grad():
            for i in range(self.channels_per_pattern):
                for j in range(self.in_channels):
                    # Set center values to be positive
                    self.horizontal_kernel.weight[i, j, 0, 2] += 1.0
                    # Set surrounding values to be negative
                    for k in [0, 1, 3, 4]:
                        self.horizontal_kernel.weight[i, j, 0, k] -= 0.2
        
        # Initialize vertical line detector
        nn.init.normal_(self.vertical_kernel.weight, std=0.01)
        nn.init.zeros_(self.vertical_kernel.bias)
        with torch.no_grad():
            for i in range(self.channels_per_pattern):
                for j in range(self.in_channels):
                    # Set center values to be positive
                    self.vertical_kernel.weight[i, j, 2, 0] += 1.0
                    # Set surrounding values to be negative
                    for k in [0, 1, 3, 4]:
                        self.vertical_kernel.weight[i, j, k, 0] -= 0.2
        
        # Initialize top-left corner detector
        nn.init.normal_(self.corner_tl_kernel.weight, std=0.01)
        nn.init.zeros_(self.corner_tl_kernel.bias)
        with torch.no_grad():
            for i in range(self.channels_per_pattern):
                for j in range(self.in_channels):
                    # Create L-shape pattern
                    for k in range(3):
                        self.corner_tl_kernel.weight[i, j, 0, k] += 0.5  # Top row
                        self.corner_tl_kernel.weight[i, j, k, 0] += 0.5  # Left column
                    # Negative bias for background
                    for y in range(1, 3):
                        for x in range(1, 3):
                            self.corner_tl_kernel.weight[i, j, y, x] -= 0.2
        
        # Initialize top-right corner detector
        nn.init.normal_(self.corner_tr_kernel.weight, std=0.01)
        nn.init.zeros_(self.corner_tr_kernel.bias)
        with torch.no_grad():
            for i in range(self.channels_per_pattern):
                for j in range(self.in_channels):
                    # Create inverted L-shape pattern
                    for k in range(3):
                        self.corner_tr_kernel.weight[i, j, 0, k] += 0.5  # Top row
                        self.corner_tr_kernel.weight[i, j, k, 2] += 0.5  # Right column
                    # Negative bias for background
                    for y in range(1, 3):
                        for x in range(0, 2):
                            self.corner_tr_kernel.weight[i, j, y, x] -= 0.2
        
        # Initialize bottom-left corner detector
        nn.init.normal_(self.corner_bl_kernel.weight, std=0.01)
        nn.init.zeros_(self.corner_bl_kernel.bias)
        with torch.no_grad():
            for i in range(self.channels_per_pattern):
                for j in range(self.in_channels):
                    # Create inverted L-shape pattern
                    for k in range(3):
                        self.corner_bl_kernel.weight[i, j, 2, k] += 0.5  # Bottom row
                        self.corner_bl_kernel.weight[i, j, k, 0] += 0.5  # Left column
                    # Negative bias for background
                    for y in range(0, 2):
                        for x in range(1, 3):
                            self.corner_bl_kernel.weight[i, j, y, x] -= 0.2
        
        # Initialize bottom-right corner detector
        nn.init.normal_(self.corner_br_kernel.weight, std=0.01)
        nn.init.zeros_(self.corner_br_kernel.bias)
        with torch.no_grad():
            for i in range(self.channels_per_pattern + self.extra_channels):
                for j in range(self.in_channels):
                    # Create L-shape pattern
                    for k in range(3):
                        self.corner_br_kernel.weight[i, j, 2, k] += 0.5  # Bottom row
                        self.corner_br_kernel.weight[i, j, k, 2] += 0.5  # Right column
                    # Negative bias for background
                    for y in range(0, 2):
                        for x in range(0, 2):
                            self.corner_br_kernel.weight[i, j, y, x] -= 0.2
    
    def forward(self, x):
        # Apply different kernels
        h_features = self.horizontal_kernel(x)
        v_features = self.vertical_kernel(x)
        tl_features = self.corner_tl_kernel(x)
        tr_features = self.corner_tr_kernel(x)
        bl_features = self.corner_bl_kernel(x)
        br_features = self.corner_br_kernel(x)
        
        # Concatenate all features
        output = torch.cat([
            h_features, v_features, 
            tl_features, tr_features, 
            bl_features, br_features
        ], dim=1)
        
        return output