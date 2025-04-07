import torch
import torch.nn as nn
import torch.nn.functional as F

class FractalResidualUnit(nn.Module):
    """A basic unit in the Fractal Residual Block"""
    def __init__(self, channels, dropout=0.0):
        super(FractalResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.silu1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.silu2 = nn.SiLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += identity
        out = self.silu2(out)
        
        return out

class FractalResidualBlock(nn.Module):
    """
    Novel residual block with fractal-like recursive structure.
    Contains smaller versions of itself at deeper levels.
    
    Unlike standard residual blocks, this allows for multi-scale pattern recognition
    through a self-similar recursive structure.
    """
    def __init__(self, channels, levels=3, dropout=0.0):
        super(FractalResidualBlock, self).__init__()
        self.channels = channels
        self.levels = levels
        
        if levels == 1:
            # Base case: single residual unit
            self.block = FractalResidualUnit(channels, dropout)
        else:
            # Recursive case: combine smaller blocks
            self.proj_down = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
            self.inner_block = FractalResidualBlock(channels // 2, levels - 1, dropout)
            self.proj_up = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False)
            self.outer_block = FractalResidualUnit(channels, dropout)
    
    def forward(self, x):
        if self.levels == 1:
            # Base case
            return self.block(x)
        else:
            # Recursive case
            identity = x
            
            # Process through outer block
            out1 = self.outer_block(x)
            
            # Process through inner block (fractal path)
            inner_input = self.proj_down(x)
            inner_output = self.inner_block(inner_input)
            inner_output = self.proj_up(inner_output)
            
            # Combine outputs
            return out1 + inner_output