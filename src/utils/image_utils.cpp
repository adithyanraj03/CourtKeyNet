#include "utils/image_utils.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace courtkeynet {
namespace utils {

Tensor ImageUtils::resize(const Tensor& input, int new_height, int new_width) {
    int batch_size = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    
    // Create output tensor
    Tensor output(batch_size, channels, new_height, new_width);
    
    // Calculate scaling factors
    float scale_y = static_cast<float>(height - 1) / (new_height - 1);
    float scale_x = static_cast<float>(width - 1) / (new_width - 1);
    
    // Perform bilinear interpolation
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < new_height; ++y) {
                for (int x = 0; x < new_width; ++x) {
                    float src_y = y * scale_y;
                    float src_x = x * scale_x;
                    
                    output.at(b, c, y, x) = bilinearInterpolate(input, b, c, src_y, src_x);
                }
            }
        }
    }
    
    return output;
}

float ImageUtils::bilinearInterpolate(const Tensor& input, int batch, int channel, 
                                     float y, float x) {
    int height = input.shape[2];
    int width = input.shape[3];
    
    // Get the four surrounding points
    int y0 = static_cast<int>(std::floor(y));
    int x0 = static_cast<int>(std::floor(x));
    int y1 = std::min(y0 + 1, height - 1);
    int x1 = std::min(x0 + 1, width - 1);
    
    // Calculate interpolation weights
    float wy = y - y0;
    float wx = x - x0;
    
    // Perform bilinear interpolation
    float val = (1 - wy) * (1 - wx) * input.at(batch, channel, y0, x0) +
                (1 - wy) * wx * input.at(batch, channel, y0, x1) +
                wy * (1 - wx) * input.at(batch, channel, y1, x0) +
                wy * wx * input.at(batch, channel, y1, x1);
    
    return val;
}

Tensor ImageUtils::gaussianBlur(const Tensor& input, int kernel_size, float sigma) {
    if (kernel_size % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd");
    }
    
    int batch_size = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    
    // Create output tensor
    Tensor output(batch_size, channels, height, width);
    
    // Generate Gaussian kernel
    std::vector<float> kernel = generateGaussianKernel(kernel_size, sigma);
    
    // Apply separable convolution (horizontal pass)
    Tensor temp(batch_size, channels, height, width);
    int radius = kernel_size / 2;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float sum = 0.0f;
                    
                    for (int i = -radius; i <= radius; ++i) {
                        int src_x = std::max(0, std::min(width - 1, x + i));
                        sum += input.at(b, c, y, src_x) * kernel[i + radius];
                    }
                    
                    temp.at(b, c, y, x) = sum;
                }
            }
        }
    }
    
    // Apply separable convolution (vertical pass)
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float sum = 0.0f;
                    
                    for (int i = -radius; i <= radius; ++i) {
                        int src_y = std::max(0, std::min(height - 1, y + i));
                        sum += temp.at(b, c, src_y, x) * kernel[i + radius];
                    }
                    
                    output.at(b, c, y, x) = sum;
                }
            }
        }
    }
    
    return output;
}

std::vector<float> ImageUtils::generateGaussianKernel(int size, float sigma) {
    std::vector<float> kernel(size);
    int radius = size / 2;
    float sum = 0.0f;
    
    for (int i = -radius; i <= radius; ++i) {
        float x = i;
        kernel[i + radius] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i + radius];
    }
    
    // Normalize the kernel
    for (int i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }
    
    return kernel;
}

Tensor ImageUtils::normalize(const Tensor& input) {
    Tensor output(input.shape);
    int batch_size = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            // Find min and max values
            float min_val = input.at(b, c, 0, 0);
            float max_val = min_val;
            
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float val = input.at(b, c, y, x);
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
            }
            
            // Normalize to [0, 1] range
            float range = max_val - min_val;
            if (range > 1e-6) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        output.at(b, c, y, x) = (input.at(b, c, y, x) - min_val) / range;
                    }
                }
            } else {
                // If the range is too small, just set all values to 0.5
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        output.at(b, c, y, x) = 0.5f;
                    }
                }
            }
        }
    }
    
    return output;
}

Tensor ImageUtils::rgbToGray(const Tensor& input) {
    if (input.shape[1] != 3) {
        throw std::invalid_argument("Input tensor must have 3 channels for RGB to grayscale conversion");
    }
    
    int batch_size = input.shape[0];
    int height = input.shape[2];
    int width = input.shape[3];
    
    // Create output tensor with single channel
    Tensor output(batch_size, 1, height, width);
    
    // Apply weighted conversion (ITU-R BT.601)
    const float r_weight = 0.299f;
    const float g_weight = 0.587f;
    const float b_weight = 0.114f;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float r = input.at(b, 0, y, x);
                float g = input.at(b, 1, y, x);
                float bl = input.at(b, 2, y, x);
                
                output.at(b, 0, y, x) = r_weight * r + g_weight * g + b_weight * bl;
            }
        }
    }
    
    return output;
}

Tensor ImageUtils::sobelFilter(const Tensor& input) {
    int batch_size = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    
    // Sobel kernels
    const float sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    const float sobel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };
    
    // Create output tensor
    Tensor output(batch_size, channels, height, width);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float gx = 0.0f;
                    float gy = 0.0f;
                    
                    // Apply Sobel kernels
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            int py = std::max(0, std::min(height - 1, y + ky));
                            int px = std::max(0, std::min(width - 1, x + kx));
                            
                            float val = input.at(b, c, py, px);
                            gx += val * sobel_x[ky + 1][kx + 1];
                            gy += val * sobel_y[ky + 1][kx + 1];
                        }
                    }
                    
                    // Calculate gradient magnitude
                    output.at(b, c, y, x) = std::sqrt(gx * gx + gy * gy);
                }
            }
        }
    }
    
    return output;
}

}  // namespace utils
}  // namespace courtkeynet