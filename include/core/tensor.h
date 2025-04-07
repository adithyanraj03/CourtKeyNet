#pragma once
#include <vector>
#include <array>
#include <stdexcept>
#include <memory>
#include <string>
#include <ostream>

namespace courtkeynet {

class Tensor {
public:
    // Constructor with dimensions
    Tensor(int dim0, int dim1 = 1, int dim2 = 1, int dim3 = 1);
    
    // Constructor from shape array
    Tensor(const std::array<int, 4>& shape);
    
    // Copy constructor
    Tensor(const Tensor& other);
    
    // Move constructor
    Tensor(Tensor&& other) noexcept;
    
    // Assignment operators
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Destructor
    ~Tensor();
    
    // Access element (const version)
    float at(int i0, int i1 = 0, int i2 = 0, int i3 = 0) const;
    
    // Access element (non-const version)
    float& at(int i0, int i1 = 0, int i2 = 0, int i3 = 0);
    
    // Fill with zeros
    void zero();
    
    // Fill with a value
    void fill(float value);
    
    // Get total number of elements
    int size() const;
    
    // Reshape
    void reshape(int dim0, int dim1 = 1, int dim2 = 1, int dim3 = 1);
    
    // Clone
    Tensor clone() const;
    
    // Shape information
    std::array<int, 4> shape;
    
private:
    // Raw data storage
    std::shared_ptr<std::vector<float>> mData;
    
    // Calculate linear index from multi-dimensional indices
    int index(int i0, int i1, int i2, int i3) const;
};

// Stream operator for printing tensor info
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

}  // namespace courtkeynet