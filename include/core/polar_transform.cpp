#include "core/polar_transform.h"
#include <cmath>

namespace courtkeynet {
namespace core {

PolarTransform::PolarTransform(int angle_bins, int radius_bins)
    : mAngleBins(angle_bins), 
      mRadiusBins(radius_bins),
      mIndicesComputed(false) {}

PolarTransform::~PolarTransform() {}

void PolarTransform::precomputeIndices(int height, int width) {
    // Reserve space for indices
    mRadiusBin.resize(height * width);
    mThetaBin.resize(height * width);
    mWeights.resize(height * width);
    
    // Center coordinates
    float center_y = height / 2.0f;
    float center_x = width / 2.0f;
    
    // Maximum radius for normalization
    float max_radius = std::sqrt(center_y * center_y + center_x * center_x);
    
    // Compute indices for each pixel
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            
            // Center-relative coordinates
            float y_centered = y - center_y;
            float x_centered = x - center_x;
            
            // Convert to polar coordinates
            float radius = std::sqrt(y_centered * y_centered + x_centered * x_centered);
            float theta = std::atan2(y_centered, x_centered);
            
            // Normalize radius and theta
            float norm_radius = radius / max_radius;
            float norm_theta = (theta + M_PI) / (2 * M_PI);
            
            // Quantize to bins
            int r_bin = std::min(mRadiusBins - 1, static_cast<int>(norm_radius * mRadiusBins));
            int t_bin = std::min(mAngleBins - 1, static_cast<int>(norm_theta * mAngleBins));
            
            // Store indices
            mRadiusBin[idx] = r_bin;
            mThetaBin[idx] = t_bin;
            mWeights[idx] = 1.0f;  // Could implement bilinear interpolation here
        }
    }
    
    mIndicesComputed = true;
}

Tensor PolarTransform::cartesianToPolar(const Tensor& input) {
    int batch_size = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    
    // Precompute indices if not done yet
    if (!mIndicesComputed || mRadiusBin.size() != height * width) {
        precomputeIndices(height, width);
    }
    
    // Create output tensor
    Tensor output(batch_size, channels, mRadiusBins, mAngleBins);
    output.zero();
    
    // Count pixels mapping to each bin for averaging
    std::vector<int> bin_counts(mRadiusBins * mAngleBins, 0);
    
    // Transform each pixel
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            // Reset bin counts for each channel
            std::fill(bin_counts.begin(), bin_counts.end(), 0);
            
            // Map pixels to polar bins
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int idx = y * width + x;
                    int r_bin = mRadiusBin[idx];
                    int t_bin = mThetaBin[idx];
                    int bin_idx = r_bin * mAngleBins + t_bin;
                    
                    // Accumulate values
                    float value = input.at(b, c, y, x);
                    output.at(b, c, r_bin, t_bin) += value;
                    bin_counts[bin_idx]++;
                }
            }
            
            // Average accumulated values
            for (int r = 0; r < mRadiusBins; ++r) {
                for (int t = 0; t < mAngleBins; ++t) {
                    int bin_idx = r * mAngleBins + t;
                    if (bin_counts[bin_idx] > 0) {
                        output.at(b, c, r, t) /= bin_counts[bin_idx];
                    }
                }
            }
        }
    }
    
    return output;
}

Tensor PolarTransform::polarToCartesian(const Tensor& input, int height, int width) {
    int batch_size = input.shape[0];
    int channels = input.shape[1];
    
    // Precompute indices if not done yet
    if (!mIndicesComputed || mRadiusBin.size() != height * width) {
        precomputeIndices(height, width);
    }
    
    // Create output tensor
    Tensor output(batch_size, channels, height, width);
    
    // Map polar bins back to Cartesian pixels
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int idx = y * width + x;
                    int r_bin = mRadiusBin[idx];
                    int t_bin = mThetaBin[idx];
                    
                    // Sample from polar tensor
                    output.at(b, c, y, x) = input.at(b, c, r_bin, t_bin);
                }
            }
        }
    }
    
    return output;
}

}  // namespace core
}  // namespace courtkeynet