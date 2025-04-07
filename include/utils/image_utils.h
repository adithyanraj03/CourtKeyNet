#pragma once
#include <vector>
#include "core/tensor.h"

namespace courtkeynet {
namespace utils {

class ImageUtils {
public:
    // Resize a tensor (assuming it represents image-like data)
    static Tensor resize(const Tensor& input, int new_height, int new_width);
    
    // Perform bilinear interpolation for a single sample
    static float bilinearInterpolate(const Tensor& input, int batch, int channel, 
                                     float y, float x);
    
    // Apply Gaussian blur to a tensor
    static Tensor gaussianBlur(const Tensor& input, int kernel_size, float sigma);
    
    // Generate a Gaussian kernel
    static std::vector<float> generateGaussianKernel(int size, float sigma);
    
    // Normalize tensor values to [0, 1] range
    static Tensor normalize(const Tensor& input);
    
    // Convert RGB tensor to grayscale
    static Tensor rgbToGray(const Tensor& input);
    
    // Apply a sobel filter for edge detection
    static Tensor sobelFilter(const Tensor& input);
};

}  // namespace utils
}  // namespace courtkeynet