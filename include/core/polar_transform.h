#pragma once
#include <vector>
#include <cmath>
#include "tensor.h"

namespace courtkeynet {
namespace core {

class PolarTransform {
public:
    PolarTransform(int angle_bins, int radius_bins);
    ~PolarTransform();
    
    // Transform Cartesian tensor to polar coordinates
    Tensor cartesianToPolar(const Tensor& input);
    
    // Transform polar tensor back to Cartesian coordinates
    Tensor polarToCartesian(const Tensor& input, int height, int width);
    
    // Generate polar indices for faster transformation
    void precomputeIndices(int height, int width);

private:
    int mAngleBins;
    int mRadiusBins;
    bool mIndicesComputed;
    
    // Precomputed indices for efficient transformation
    std::vector<int> mRadiusBin;
    std::vector<int> mThetaBin;
    std::vector<float> mWeights;
};

}  // namespace core
}  // namespace courtkeynet