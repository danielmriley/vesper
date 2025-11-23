#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

void test_tensor_creation() {
#if USE_HIP_BACKEND
    {
    std::cout << "Testing HIP Tensor creation..." << std::endl;
    
    const std::vector<int64_t> shape = {2, 3, 4};
    const vesper::DType dtype = vesper::DType::Float32;
    const vesper::Device device = vesper::Device::HIP;

    vesper::Tensor tensor = vesper::empty(shape, dtype, device);

    assert(tensor.shape() == shape);
    assert(tensor.dtype() == dtype);
    assert(tensor.device() == device);
    assert(tensor.numel() == 24);
    assert(tensor.offset() == 0);
    assert(tensor.is_contiguous());

    // Check strides: (3*4, 4, 1) -> (12, 4, 1)
    const auto& strides = tensor.strides();
    assert(strides[0] == 12);
    assert(strides[1] == 4);
    assert(strides[2] == 1);

    // Test data pointer (just check it's not null)
    float* data = tensor.data_ptr<float>();
    assert(data != nullptr);

    std::cout << "HIP Tensor creation test passed!" << std::endl;
    }
#else
    std::cout << "Skipping HIP Tensor creation test (HIP backend disabled)." << std::endl;
#endif

#if USE_CUDA_BACKEND
    {
    std::cout << "Testing CUDA Tensor creation..." << std::endl;
    
    const std::vector<int64_t> shape = {2, 3, 4};
    const vesper::DType dtype = vesper::DType::Float32;
    const vesper::Device device = vesper::Device::CUDA;

    vesper::Tensor tensor = vesper::empty(shape, dtype, device);

    assert(tensor.shape() == shape);
    assert(tensor.dtype() == dtype);
    assert(tensor.device() == device);
    assert(tensor.numel() == 24);
    assert(tensor.offset() == 0);
    assert(tensor.is_contiguous());

    // Check strides: (3*4, 4, 1) -> (12, 4, 1)
    const auto& strides = tensor.strides();
    assert(strides[0] == 12);
    assert(strides[1] == 4);
    assert(strides[2] == 1);

    // Test data pointer (just check it's not null)
    float* data = tensor.data_ptr<float>();
    assert(data != nullptr);

    std::cout << "CUDA Tensor creation test passed!" << std::endl;
    }
#else
    std::cout << "Skipping CUDA Tensor creation test (CUDA backend disabled)." << std::endl;
#endif

#if USE_CPU_BACKEND
    {
    std::cout << "Testing CPU Tensor creation..." << std::endl;
    
    const std::vector<int64_t> shape = {2, 3, 4};
    const vesper::DType dtype = vesper::DType::Float32;
    const vesper::Device device = vesper::Device::CPU;

    vesper::Tensor tensor = vesper::empty(shape, dtype, device);

    assert(tensor.shape() == shape);
    assert(tensor.dtype() == dtype);
    assert(tensor.device() == device);
    assert(tensor.numel() == 24);
    assert(tensor.offset() == 0);
    assert(tensor.is_contiguous());

    // Check strides: (3*4, 4, 1) -> (12, 4, 1)
    const auto& strides = tensor.strides();
    assert(strides[0] == 12);
    assert(strides[1] == 4);
    assert(strides[2] == 1);

    // Test data pointer (just check it's not null)
    float* data = tensor.data_ptr<float>();
    assert(data != nullptr);

    std::cout << "CPU Tensor creation test passed!" << std::endl;
    }
#else
    std::cout << "Skipping CPU Tensor creation test (CPU backend disabled)." << std::endl;
#endif
}


int main() {
    test_tensor_creation();
    return 0;
}
