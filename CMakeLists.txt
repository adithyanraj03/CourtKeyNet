cmake_minimum_required(VERSION 3.10)
project(courtkeynet)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add source files
file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp"
)

# Add pybind11 module
pybind11_add_module(courtkeynet_cpp
    ${SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/courtkeynet/cpp/binding.cpp
)

# Set target properties
set_target_properties(courtkeynet_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/courtkeynet/cpp
)

# Enable optimizations for Release builds
target_compile_options(courtkeynet_cpp PRIVATE
    $<$<CONFIG:Release>:-O3>
)

# Link libraries
target_link_libraries(courtkeynet_cpp PRIVATE
    pybind11::module
)