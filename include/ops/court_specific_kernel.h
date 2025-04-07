#pragma once
#include "tensor.h"

namespace courtkeynet {
namespace ops {

class CourtSpecificKernel {
public:
    CourtSpecificKernel(int in_channels, int out_channels);
    ~CourtSpecificKernel();
    
    // Initialize kernels with court-specific patterns
    void initializeKernels();
    
    // Forward pass
    Tensor forward(const Tensor& input);

private:
    int mInChannels;
    int mOutChannels;
    
    // Kernels for different features
    Tensor mHorizontalKernel;
    Tensor mVerticalKernel;
    Tensor mCornerTLKernel;
    Tensor mCornerTRKernel;
    Tensor mCornerBLKernel;
    Tensor mCornerBRKernel;
};

}  // namespace ops
}  // namespace courtkeynet