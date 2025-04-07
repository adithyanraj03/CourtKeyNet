import torch
import torch.nn as nn
import torch.nn.functional as F
from ...cpp import courtkeynet_cpp

# Import custom modules
from .court_specific_shape_kernel import CourtSpecificShapeKernel
from .fractal_residual_block import FractalResidualBlock
from .self_calibration_block import SelfCalibrationBlock
from .non_local_self_similarity import NonLocalSelfSimilarity
from .court_feature_aggregator import CourtFeatureAggregator
from .fourier_feature_encoder import FourierFeatureEncoder
from .polar_transform_attention import PolarTransformAttention

class OctaveFeatureExtractor(nn.Module):
    """
    Novel feature extractor that processes features at multiple frequency octaves
    Better captures both fine details and global structure of courts
    """
    def __init__(self):
        super(OctaveFeatureExtractor, self).__init__()
        
        # Initial processing
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # High frequency path (fine details)
        self.high_freq_path = nn.Sequential(
            CourtSpecificShapeKernel(32, 64),
            FractalResidualBlock(64, levels=3),
            SelfCalibrationBlock(64)
        )
        
        # Mid frequency path (mid-level structures)
        self.mid_freq_path = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            CourtSpecificShapeKernel(32, 64),
            FractalResidualBlock(64, levels=2),
            NonLocalSelfSimilarity(64)
        )
        
        # Low frequency path (global context)
        self.low_freq_path = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            CourtFeatureAggregator(32, 64),
            FourierFeatureEncoder(64, 64)
        )
        
        # Combine octaves
        self.octave_fusion = nn.Conv2d(64*3, 128, kernel_size=1)
        
        # Additional processing
        self.process = nn.Sequential(
            FractalResidualBlock(128, levels=3),
            PolarTransformAttention(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x):
        # Initial processing
        x = self.stem(x)
        
        # Process at different octaves
        high_freq = self.high_freq_path(x)
        mid_freq = self.mid_freq_path(x)
        low_freq = self.low_freq_path(x)
        
        # Upsample lower octaves to match high frequency
        mid_freq_up = F.interpolate(mid_freq, size=high_freq.shape[2:], mode='bilinear', align_corners=False)
        low_freq_up = F.interpolate(low_freq, size=high_freq.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine octaves
        combined = torch.cat([high_freq, mid_freq_up, low_freq_up], dim=1)
        fused = self.octave_fusion(combined)
        
        # Final processing
        features = self.process(fused)
        
        return features