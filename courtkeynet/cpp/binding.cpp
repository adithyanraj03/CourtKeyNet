#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "core/polar_transform.h"
#include "core/geometric_utils.h"
#include "ops/court_specific_kernel.h"

namespace py = pybind11;
using namespace courtkeynet;

// Convert numpy array to our Tensor
Tensor numpy_to_tensor(py::array_t<float> array) {
    py::buffer_info buf = array.request();
    
    // Get dimensions
    int ndim = buf.ndim;
    std::array<int, 4> shape = {1, 1, 1, 1};
    
    for (int i = 0; i < ndim && i < 4; ++i) {
        shape[i] = buf.shape[i];
    }
    
    // Create tensor
    Tensor tensor(shape);
    
    // Copy data
    float* ptr = static_cast<float*>(buf.ptr);
    for (int i = 0; i < tensor.size(); ++i) {
        tensor.at(i / (shape[1] * shape[2] * shape[3]),
                 (i / (shape[2] * shape[3])) % shape[1],
                 (i / shape[3]) % shape[2],
                 i % shape[3]) = ptr[i];
    }
    
    return tensor;
}

// Convert our Tensor to numpy array
py::array_t<float> tensor_to_numpy(const Tensor& tensor) {
    auto shape = tensor.shape;
    
    // Remove trailing dimensions of size 1
    int ndim = 4;
    while (ndim > 1 && shape[ndim - 1] == 1) {
        ndim--;
    }
    
    // Create numpy shape
    std::vector<ssize_t> numpy_shape;
    for (int i = 0; i < ndim; ++i) {
        numpy_shape.push_back(shape[i]);
    }
    
    // Create numpy array
    py::array_t<float> array(numpy_shape);
    py::buffer_info buf = array.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    // Copy data
    for (int i = 0; i < tensor.size(); ++i) {
        ptr[i] = tensor.at(i / (shape[1] * shape[2] * shape[3]),
                          (i / (shape[2] * shape[3])) % shape[1],
                          (i / shape[3]) % shape[2],
                          i % shape[3]);
    }
    
    return array;
}

PYBIND11_MODULE(courtkeynet_cpp, m) {
    m.doc() = "C++ implementations for CourtKeyNet";
    
    // PolarTransform
    py::class_<core::PolarTransform>(m, "PolarTransform")
        .def(py::init<int, int>())
        .def("cartesian_to_polar", [](core::PolarTransform& self, py::array_t<float> input) {
            Tensor tensor = numpy_to_tensor(input);
            Tensor result = self.cartesianToPolar(tensor);
            return tensor_to_numpy(result);
        })
        .def("polar_to_cartesian", [](core::PolarTransform& self, py::array_t<float> input, int height, int width) {
            Tensor tensor = numpy_to_tensor(input);
            Tensor result = self.polarToCartesian(tensor, height, width);
            return tensor_to_numpy(result);
        })
        .def("precompute_indices", &core::PolarTransform::precomputeIndices);
    
    // GeometricUtils
    py::module_ geom = m.def_submodule("geometric", "Geometric utilities");
    
    geom.def("compute_angles", [](py::array_t<float> points) {
        Tensor tensor = numpy_to_tensor(points);
        Tensor result = core::GeometricUtils::computeAngles(tensor);
        return tensor_to_numpy(result);
    });
    
    geom.def("compute_edge_lengths", [](py::array_t<float> points) {
        Tensor tensor = numpy_to_tensor(points);
        Tensor result = core::GeometricUtils::computeEdgeLengths(tensor);
        return tensor_to_numpy(result);
    });
    
    geom.def("compute_diagonals", [](py::array_t<float> points) {
        Tensor tensor = numpy_to_tensor(points);
        Tensor result = core::GeometricUtils::computeDiagonals(tensor);
        return tensor_to_numpy(result);
    });
    
    // CourtSpecificKernel
    py::class_<ops::CourtSpecificKernel>(m, "CourtSpecificKernel")
        .def(py::init<int, int>())
        .def("forward", [](ops::CourtSpecificKernel& self, py::array_t<float> input) {
            Tensor tensor = numpy_to_tensor(input);
            Tensor result = self.forward(tensor);
            return tensor_to_numpy(result);
        })
        .def("initialize_kernels", &ops::CourtSpecificKernel::initializeKernels);
}