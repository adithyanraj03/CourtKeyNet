#include "ops/court_specific_kernel.h"
#include <cmath>
#include <iostream>

namespace courtkeynet {
namespace ops {

CourtSpecificKernel::CourtSpecificKernel(int in_channels, int out_channels)
    : mInChannels(in_channels), mOutChannels(out_channels) {
    // Initialize kernels
    mHorizontalKernel = Tensor(out_channels / 6, in_channels, 1, 5);
    mVerticalKernel = Tensor(out_channels / 6, in_channels, 5, 1);
    mCornerTLKernel = Tensor(out_channels / 6, in_channels, 3, 3);
    mCornerTRKernel = Tensor(out_channels / 6, in_channels, 3, 3);
    mCornerBLKernel = Tensor(out_channels / 6, in_channels, 3, 3);
    mCornerBRKernel = Tensor(out_channels / 6, in_channels, 3, 3);
    
    initializeKernels();
}

CourtSpecificKernel::~CourtSpecificKernel() {
    // Nothing to do here, tensor destructor will handle cleanup
}

void CourtSpecificKernel::initializeKernels() {
    // Initialize the kernels with specific patterns for court detection
    
    // Horizontal line detection kernel
    for (int oc = 0; oc < mHorizontalKernel.shape[0]; ++oc) {
        for (int ic = 0; ic < mHorizontalKernel.shape[1]; ++ic) {
            for (int i = 0; i < mHorizontalKernel.shape[3]; ++i) {
                mHorizontalKernel.at(oc, ic, 0, i) = 1.0f;
            }
        }
    }
    
    // Vertical line detection kernel
    for (int oc = 0; oc < mVerticalKernel.shape[0]; ++oc) {
        for (int ic = 0; ic < mVerticalKernel.shape[1]; ++ic) {
            for (int i = 0; i < mVerticalKernel.shape[2]; ++i) {
                mVerticalKernel.at(oc, ic, i, 0) = 1.0f;
            }
        }
    }
    
    // Top-left corner detection kernel (L shape)
    for (int oc = 0; oc < mCornerTLKernel.shape[0]; ++oc) {
        for (int ic = 0; ic < mCornerTLKernel.shape[1]; ++ic) {
            // Vertical part of L
            for (int i = 0; i < 3; ++i) {
                mCornerTLKernel.at(oc, ic, i, 0) = 1.0f;
            }
            // Horizontal part of L
            for (int i = 0; i < 3; ++i) {
                mCornerTLKernel.at(oc, ic, 2, i) = 1.0f;
            }
        }
    }
    
    // Top-right corner detection kernel (backward L shape)
    for (int oc = 0; oc < mCornerTRKernel.shape[0]; ++oc) {
        for (int ic = 0; ic < mCornerTRKernel.shape[1]; ++ic) {
            // Vertical part of L
            for (int i = 0; i < 3; ++i) {
                mCornerTRKernel.at(oc, ic, i, 2) = 1.0f;
            }
            // Horizontal part of L
            for (int i = 0; i < 3; ++i) {
                mCornerTRKernel.at(oc, ic, 2, i) = 1.0f;
            }
        }
    }
    
    // Bottom-left corner detection kernel (backward L upside down)
    for (int oc = 0; oc < mCornerBLKernel.shape[0]; ++oc) {
        for (int ic = 0; ic < mCornerBLKernel.shape[1]; ++ic) {
            // Vertical part of L
            for (int i = 0; i < 3; ++i) {
                mCornerBLKernel.at(oc, ic, i, 0) = 1.0f;
            }
            // Horizontal part of L
            for (int i = 0; i < 3; ++i) {
                mCornerBLKernel.at(oc, ic, 0, i) = 1.0f;
            }
        }
    }
    
    // Bottom-right corner detection kernel (L upside down)
    for (int oc = 0; oc < mCornerBRKernel.shape[0]; ++oc) {
        for (int ic = 0; ic < mCornerBRKernel.shape[1]; ++ic) {
            // Vertical part of L
            for (int i = 0; i < 3; ++i) {
                mCornerBRKernel.at(oc, ic, i, 2) = 1.0f;
            }
            // Horizontal part of L
            for (int i = 0; i < 3; ++i) {
                mCornerBRKernel.at(oc, ic, 0, i) = 1.0f;
            }
        }
    }
}

Tensor CourtSpecificKernel::forward(const Tensor& input) {
    // Get input dimensions
    int batch_size = input.shape[0];
    int in_channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    
    // Create output tensor
    Tensor output(batch_size, mOutChannels, height, width);
    output.zero();
    
    // Apply each specialized kernel and concatenate results
    int output_channel_offset = 0;
    
    // Apply horizontal line kernel
    applyKernel(input, output, mHorizontalKernel, output_channel_offset);
    output_channel_offset += mHorizontalKernel.shape[0];
    
    // Apply vertical line kernel
    applyKernel(input, output, mVerticalKernel, output_channel_offset);
    output_channel_offset += mVerticalKernel.shape[0];
    
    // Apply top-left corner kernel
    applyKernel(input, output, mCornerTLKernel, output_channel_offset);
    output_channel_offset += mCornerTLKernel.shape[0];
    
    // Apply top-right corner kernel
    applyKernel(input, output, mCornerTRKernel, output_channel_offset);
    output_channel_offset += mCornerTRKernel.shape[0];
    
    // Apply bottom-left corner kernel
    applyKernel(input, output, mCornerBLKernel, output_channel_offset);
    output_channel_offset += mCornerBLKernel.shape[0];
    
    // Apply bottom-right corner kernel
    applyKernel(input, output, mCornerBRKernel, output_channel_offset);
    
    return output;
}

// Helper method to apply a kernel to the input and store in output
void CourtSpecificKernel::applyKernel(const Tensor& input, Tensor& output, 
                                       const Tensor& kernel, int output_channel_offset) {
    int batch_size = input.shape[0];
    int in_channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    
    int kernel_height = kernel.shape[2];
    int kernel_width = kernel.shape[3];
    int pad_h = kernel_height / 2;
    int pad_w = kernel_width / 2;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < kernel.shape[0]; ++oc) {
            int output_channel = output_channel_offset + oc;
            
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    float sum = 0.0f;
                    
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            int h_in = h - pad_h + kh;
                            int w_in = w - pad_w + kw;
                            
                            // Check bounds
                            if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                                for (int ic = 0; ic < in_channels; ++ic) {
                                    sum += input.at(b, ic, h_in, w_in) * kernel.at(oc, ic, kh, kw);
                                }
                            }
                        }
                    }
                    
                    output.at(b, output_channel, h, w) = sum;
                }
            }
        }
    }
}

}  // namespace ops
}  // namespace courtkeynet