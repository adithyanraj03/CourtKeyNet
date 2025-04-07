#include "utils/math_utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace courtkeynet {
namespace utils {

float MathUtils::fastExp(float x) {
    // Fast approximation of the exponential function
    // Based on Schraudolph's algorithm
    
    // Clamp input to avoid overflow/underflow
    x = std::max(-80.0f, std::min(80.0f, x));
    
    union {
        float f;
        int i;
    } u;
    
    const float a = 1064872507.1541044f;  // (1 << 23) / log(2)
    const float b = 12102203.161561485f;  // 127 * (1 << 23)
    
    u.i = static_cast<int>(a * x + b);
    
    return u.f;
}

Tensor MathUtils::softmax(const Tensor& input, int dim) {
    // Only support softmax along dimension 1 for now
    if (dim != 1) {
        throw std::invalid_argument("Softmax only supports dim=1 for now");
    }
    
    Tensor output(input.shape);
    
    int batch_size = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // Find max value for numerical stability
                float max_val = input.at(b, 0, h, w);
                for (int c = 1; c < channels; ++c) {
                    max_val = std::max(max_val, input.at(b, c, h, w));
                }
                
                // Compute exp(x - max_val) for each element
                std::vector<float> exp_values(channels);
                float sum_exp = 0.0f;
                
                for (int c = 0; c < channels; ++c) {
                    float exp_val = fastExp(input.at(b, c, h, w) - max_val);
                    exp_values[c] = exp_val;
                    sum_exp += exp_val;
                }
                
                // Normalize by sum of exponentials
                for (int c = 0; c < channels; ++c) {
                    output.at(b, c, h, w) = exp_values[c] / sum_exp;
                }
            }
        }
    }
    
    return output;
}

Tensor MathUtils::sigmoid(const Tensor& input) {
    Tensor output(input.shape);
    int size = input.size();
    
    for (int i = 0; i < size; ++i) {
        // Using direct indexing for efficiency
        float x = (*input.mData)[i];
        // Sigmoid function: 1 / (1 + exp(-x))
        (*output.mData)[i] = 1.0f / (1.0f + fastExp(-x));
    }
    
    return output;
}

Tensor MathUtils::tanh(const Tensor& input) {
    Tensor output(input.shape);
    int size = input.size();
    
    for (int i = 0; i < size; ++i) {
        // Using direct indexing for efficiency
        float x = (*input.mData)[i];
        // Tanh can be computed from sigmoid: 2 * sigmoid(2x) - 1
        (*output.mData)[i] = 2.0f * (1.0f / (1.0f + fastExp(-2.0f * x))) - 1.0f;
    }
    
    return output;
}

Tensor MathUtils::relu(const Tensor& input) {
    Tensor output(input.shape);
    int size = input.size();
    
    for (int i = 0; i < size; ++i) {
        // Using direct indexing for efficiency
        float x = (*input.mData)[i];
        // ReLU function: max(0, x)
        (*output.mData)[i] = std::max(0.0f, x);
    }
    
    return output;
}

Tensor MathUtils::leakyRelu(const Tensor& input, float alpha) {
    Tensor output(input.shape);
    int size = input.size();
    
    for (int i = 0; i < size; ++i) {
        // Using direct indexing for efficiency
        float x = (*input.mData)[i];
        // Leaky ReLU function: max(alpha * x, x)
        (*output.mData)[i] = x > 0.0f ? x : alpha * x;
    }
    
    return output;
}

Tensor MathUtils::multiply(const Tensor& a, const Tensor& b) {
    // Check if shapes are compatible
    if (a.shape != b.shape) {
        throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }
    
    Tensor output(a.shape);
    int size = a.size();
    
    for (int i = 0; i < size; ++i) {
        // Using direct indexing for efficiency
        (*output.mData)[i] = (*a.mData)[i] * (*b.mData)[i];
    }
    
    return output;
}

float MathUtils::dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }
    
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

float MathUtils::cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for cosine similarity");
    }
    
    float dot = dotProduct(a, b);
    
    float norm_a = std::sqrt(dotProduct(a, a));
    float norm_b = std::sqrt(dotProduct(b, b));
    
    // Handle zero vectors
    if (norm_a < 1e-6 || norm_b < 1e-6) {
        return 0.0f;
    }
    
    return dot / (norm_a * norm_b);
}

}  // namespace utils
}  // namespace courtkeynet