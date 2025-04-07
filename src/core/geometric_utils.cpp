#include "core/geometric_utils.h"
#include <cmath>
#include <vector>

namespace courtkeynet {
namespace core {

Tensor GeometricUtils::computeAngles(const Tensor& points) {
    // points shape: [batch_size, num_points, 2]
    int batch_size = points.shape[0];
    int num_points = points.shape[1];
    
    // Output shape: [batch_size, num_points]
    Tensor angles(batch_size, num_points);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < num_points; ++i) {
            // Get vectors to previous and next points
            int prev_idx = (i - 1 + num_points) % num_points;
            int next_idx = (i + 1) % num_points;
            
            // Vector from current point to previous point
            float prev_x = points.at(b, prev_idx, 0) - points.at(b, i, 0);
            float prev_y = points.at(b, prev_idx, 1) - points.at(b, i, 1);
            
            // Vector from current point to next point
            float next_x = points.at(b, next_idx, 0) - points.at(b, i, 0);
            float next_y = points.at(b, next_idx, 1) - points.at(b, i, 1);
            
            // Compute dot product
            float dot = prev_x * next_x + prev_y * next_y;
            
            // Compute magnitudes
            float prev_mag = std::sqrt(prev_x * prev_x + prev_y * prev_y);
            float next_mag = std::sqrt(next_x * next_x + next_y * next_y);
            
            // Compute cosine
            float cos_theta = dot / (prev_mag * next_mag);
            
            // Clamp to valid range for acos
            cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
            
            // Compute angle in radians and convert to degrees
            float angle = std::acos(cos_theta) * 180.0f / M_PI;
            
            angles.at(b, i) = angle;
        }
    }
    
    return angles;
}

Tensor GeometricUtils::computeEdgeLengths(const Tensor& points) {
    // points shape: [batch_size, num_points, 2]
    int batch_size = points.shape[0];
    int num_points = points.shape[1];
    
    // Output shape: [batch_size, num_points]
    Tensor edge_lengths(batch_size, num_points);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < num_points; ++i) {
            int next_idx = (i + 1) % num_points;
            
            float dx = points.at(b, next_idx, 0) - points.at(b, i, 0);
            float dy = points.at(b, next_idx, 1) - points.at(b, i, 1);
            
            edge_lengths.at(b, i) = std::sqrt(dx * dx + dy * dy);
        }
    }
    
    return edge_lengths;
}

Tensor GeometricUtils::computeDiagonals(const Tensor& points) {
    // points shape: [batch_size, 4, 2] - Assuming quadrilateral
    int batch_size = points.shape[0];
    
    // Output shape: [batch_size, 2] - Two diagonals per quadrilateral
    Tensor diagonals(batch_size, 2);
    
    for (int b = 0; b < batch_size; ++b) {
        // Diagonal 1-3
        float dx1 = points.at(b, 0, 0) - points.at(b, 2, 0);
        float dy1 = points.at(b, 0, 1) - points.at(b, 2, 1);
        diagonals.at(b, 0) = std::sqrt(dx1 * dx1 + dy1 * dy1);
        
        // Diagonal 2-4
        float dx2 = points.at(b, 1, 0) - points.at(b, 3, 0);
        float dy2 = points.at(b, 1, 1) - points.at(b, 3, 1);
        diagonals.at(b, 1) = std::sqrt(dx2 * dx2 + dy2 * dy2);
    }
    
    return diagonals;
}

Tensor GeometricUtils::computeInternalAngles(const Tensor& points) {
    return computeAngles(points);
}

Tensor GeometricUtils::normalizeSum(const Tensor& input, int dim) {
    int batch_size = input.shape[0];
    int size_dim = input.shape[dim];
    
    Tensor output = input.clone();
    
    // Only supporting dim=1 for now
    if (dim != 1) {
        throw std::runtime_error("normalizeSum only supports dim=1 currently");
    }
    
    for (int b = 0; b < batch_size; ++b) {
        // Compute sum
        float sum = 0.0f;
        for (int i = 0; i < size_dim; ++i) {
            sum += input.at(b, i);
        }
        
        // Normalize
        if (sum > 0) {
            for (int i = 0; i < size_dim; ++i) {
                output.at(b, i) = input.at(b, i) / sum;
            }
        }
    }
    
    return output;
}

}  // namespace core
}  // namespace courtkeynet