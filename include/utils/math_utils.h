#pragma once
#include <cmath>
#include <vector>
#include "core/tensor.h"

namespace courtkeynet {
namespace utils {

class MathUtils {
public:
    // Fast calculation of exponential function
    static float fastExp(float x);
    
    // Calculate softmax along a specified dimension
    static Tensor softmax(const Tensor& input, int dim);
    
    // Calculate sigmoid function elementwise
    static Tensor sigmoid(const Tensor& input);
    
    // Calculate tanh function elementwise
    static Tensor tanh(const Tensor& input);
    
    // Calculate ReLU function elementwise
    static Tensor relu(const Tensor& input);
    
    // Calculate Leaky ReLU function elementwise
    static Tensor leakyRelu(const Tensor& input, float alpha = 0.01f);
    
    // Calculate element-wise product of two tensors
    static Tensor multiply(const Tensor& a, const Tensor& b);
    
    // Calculate dot product of two vectors
    static float dotProduct(const std::vector<float>& a, const std::vector<float>& b);
    
    // Compute cosine similarity between two vectors
    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
};

}  // namespace utils
}  // namespace courtkeynet