#pragma once
#include <vector>
#include "tensor.h"

namespace courtkeynet {
namespace core {

class GeometricUtils {
public:
    // Compute pairwise angles between points
    static Tensor computeAngles(const Tensor& points);
    
    // Compute edge lengths of quadrilateral
    static Tensor computeEdgeLengths(const Tensor& points);
    
    // Compute diagonal lengths of quadrilateral
    static Tensor computeDiagonals(const Tensor& points);
    
    // Compute internal angles of quadrilateral
    static Tensor computeInternalAngles(const Tensor& points);
    
    // Normalize tensor values to sum to 1 along a dimension
    static Tensor normalizeSum(const Tensor& input, int dim);
};

}  // namespace core
}  // namespace courtkeynet