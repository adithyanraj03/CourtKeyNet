import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalSelfSimilarity(nn.Module):
    """
    Non-Local Self-Similarity module that captures long-range dependencies 
    and repeated patterns in feature maps.
    
    Similar to Non-Local Neural Networks, but specifically designed to detect 
    the self-similar structures present in courts.
    """
    def __init__(self, in_channels, reduction=8, use_scale=True):
        super(NonLocalSelfSimilarity, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        
        # Define query, key, value projections
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Normalization and activation
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        
        # Scaling factor for dot products
        self.scale = in_channels ** -0.5 if use_scale else 1.0
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Generate query, key and value tensors
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # Compute attention map
        attention = torch.bmm(query, key) * self.scale  # Dot product attention
        attention = F.softmax(attention, dim=2)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Output projection
        out = self.out_conv(out)
        out = self.norm(out)
        
        # Residual connection
        out = out + x
        out = self.act(out)
        
        return out