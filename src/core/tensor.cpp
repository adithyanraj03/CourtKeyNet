#include "core/tensor.h"
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace courtkeynet {

Tensor::Tensor(int dim0, int dim1, int dim2, int dim3) 
    : shape({dim0, dim1, dim2, dim3}) {
    // Calculate total size
    int total_size = dim0 * dim1 * dim2 * dim3;
    
    // Allocate memory
    mData = std::make_shared<std::vector<float>>(total_size, 0.0f);
}

Tensor::Tensor(const std::array<int, 4>& shape_array) 
    : shape(shape_array) {
    // Calculate total size
    int total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    
    // Allocate memory
    mData = std::make_shared<std::vector<float>>(total_size, 0.0f);
}

Tensor::Tensor(const Tensor& other) 
    : shape(other.shape), mData(other.mData) {
    // Copy constructor uses shared_ptr's reference counting
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape(std::move(other.shape)), mData(std::move(other.mData)) {
    // Move constructor
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape = other.shape;
        mData = other.mData;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape = std::move(other.shape);
        mData = std::move(other.mData);
    }
    return *this;
}

Tensor::~Tensor() {
    // No explicit cleanup needed due to shared_ptr
}

float Tensor::at(int i0, int i1, int i2, int i3) const {
    validateIndices(i0, i1, i2, i3);
    return (*mData)[index(i0, i1, i2, i3)];
}

float& Tensor::at(int i0, int i1, int i2, int i3) {
    validateIndices(i0, i1, i2, i3);
    return (*mData)[index(i0, i1, i2, i3)];
}

void Tensor::zero() {
    std::fill(mData->begin(), mData->end(), 0.0f);
}

void Tensor::fill(float value) {
    std::fill(mData->begin(), mData->end(), value);
}

int Tensor::size() const {
    return mData->size();
}

void Tensor::reshape(int dim0, int dim1, int dim2, int dim3) {
    // Check if the total size is compatible
    int new_size = dim0 * dim1 * dim2 * dim3;
    if (new_size != mData->size()) {
        std::ostringstream msg;
        msg << "Cannot reshape tensor of size " << mData->size() 
            << " to shape [" << dim0 << ", " << dim1 << ", " 
            << dim2 << ", " << dim3 << "] with size " << new_size;
        throw std::invalid_argument(msg.str());
    }
    
    // Update shape
    shape = {dim0, dim1, dim2, dim3};
}

Tensor Tensor::clone() const {
    Tensor result(shape);
    
    // Deep copy of the data
    *result.mData = *mData;
    
    return result;
}

int Tensor::index(int i0, int i1, int i2, int i3) const {
    return ((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3;
}

void Tensor::validateIndices(int i0, int i1, int i2, int i3) const {
    if (i0 < 0 || i0 >= shape[0] ||
        i1 < 0 || i1 >= shape[1] ||
        i2 < 0 || i2 >= shape[2] ||
        i3 < 0 || i3 >= shape[3]) {
        std::ostringstream msg;
        msg << "Index [" << i0 << ", " << i1 << ", " << i2 << ", " << i3 
            << "] is out of bounds for tensor with shape [" 
            << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "]";
        throw std::out_of_range(msg.str());
    }
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(shape=[" << tensor.shape[0] << ", " << tensor.shape[1] << ", "
       << tensor.shape[2] << ", " << tensor.shape[3] << "], size=" << tensor.size() << ")";
    return os;
}

}  // namespace courtkeynet