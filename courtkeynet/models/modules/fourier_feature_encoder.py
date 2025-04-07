import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FourierFeatureEncoder(nn.Module):
    """
    Processes features in the frequency domain to better capture 
    periodic structures like court lines.
    
    Uses Fast Fourier Transform (FFT) to transform features, applies
    learned filtering in frequency space, and transforms back.
    """
    def __init__(self, in_channels, out_channels, use_phase=True):
        super(FourierFeatureEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_phase = use_phase
        
        # Learnable parameters for frequency domain filtering
        # One filter per channel for magnitude
        self.freq_filter_mag = nn.Parameter(torch.ones(in_channels, 1, 1))
        
        # Optional phase filter
        if use_phase:
            self.freq_filter_phase = nn.Parameter(torch.zeros(in_channels, 1, 1))
        
        # 1x1 convolutions for input and output projections
        self.in_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels*2 if use_phase else in_channels, 
                                  out_channels, kernel_size=1)
        
        # Normalization and activation
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
        # Initialize with frequency patterns
        self._initialize_frequency_filter()
    
    def _initialize_frequency_filter(self):
        # Initialize magnitude filters to prioritize mid-range frequencies
        freq_init = torch.zeros_like(self.freq_filter_mag.data)
        
        # Add band-pass like initialization - enhance mid-frequencies
        with torch.no_grad():
            # Different preferences for different channels
            for i in range(self.in_channels):
                # Create various frequency preferences (low, mid, high)
                if i % 3 == 0:
                    # Low-frequency pass
                    freq_init[i, 0, 0] = 1.5  # Boost low frequencies
                elif i % 3 == 1:
                    # Mid-frequency pass (court lines typically appear here)
                    freq_init[i, 0, 0] = 2.0  # Boost mid frequencies
                else:
                    # High-frequency pass
                    freq_init[i, 0, 0] = 1.0  # Neutral to high frequencies
        
        self.freq_filter_mag.data = freq_init
        
        # Initialize phase filter around zero (if used)
        if self.use_phase:
            self.freq_filter_phase.data.zero_()
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Input projection
        x = self.in_proj(x)
        
        # Apply 2D FFT - convert to frequency domain
        # Operate on each channel separately
        x_freq = torch.fft.rfft2(x, dim=(2, 3))
        
        # Split into magnitude and phase
        x_mag = torch.abs(x_freq)
        x_phase = torch.angle(x_freq)
        
        # Apply learnable frequency filtering on magnitude
        x_mag_filtered = x_mag * F.adaptive_avg_pool2d(
            self.freq_filter_mag.expand(batch_size, -1, -1, -1),
            output_size=x_mag.shape[2:])
        
        # Apply optional phase filtering
        if self.use_phase:
            x_phase_filtered = x_phase + F.adaptive_avg_pool2d(
                self.freq_filter_phase.expand(batch_size, -1, -1, -1),
                output_size=x_phase.shape[2:])
            
            # Reconstruct complex tensor
            x_freq_filtered = torch.complex(
                x_mag_filtered * torch.cos(x_phase_filtered),
                x_mag_filtered * torch.sin(x_phase_filtered)
            )
        else:
            # Keep original phase
            x_freq_filtered = torch.complex(
                x_mag_filtered * torch.cos(x_phase),
                x_mag_filtered * torch.sin(x_phase)
            )
        
        # Apply inverse FFT to get back to spatial domain
        x_filtered = torch.fft.irfft2(x_freq_filtered, s=(height, width), dim=(2, 3))
        
        # Combine with original input if using phase 
        if self.use_phase:
            output = torch.cat([x, x_filtered], dim=1)
        else:
            output = x_filtered
        
        # Final projection
        output = self.out_proj(output)
        output = self.norm(output)
        output = self.act(output)
        
        return output