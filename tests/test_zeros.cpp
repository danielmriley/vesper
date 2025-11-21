#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>

#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

void test_zeros_float32() {
    std::cout << "Testing zeros Float32..." << std::endl;
#if USE_HIP_BACKEND
    {
        std::vector<int64_t> shape = {10};
        vesper::Tensor t = vesper::zeros(shape, vesper::DType::Float32, vesper::Device::HIP);
        
        std::vector<float> host_data(10, 1.0f); // Initialize with non-zeros
        hipError_t err;
        err = hipMemcpy(host_data.data(), t.data_ptr<float>(), 10 * sizeof(float), hipMemcpyDeviceToHost);
        assert(err == hipSuccess);
        
        for (float val : host_data) {
            assert(val == 0.0f);
        }
    }
#endif
}

void test_zeros_int32() {
    std::cout << "Testing zeros Int32..." << std::endl;
#if USE_HIP_BACKEND
    {
        std::vector<int64_t> shape = {5, 5};
        vesper::Tensor t = vesper::zeros(shape, vesper::DType::Int32, vesper::Device::HIP);
        
        std::vector<int> host_data(25, 1);
        hipError_t err;
        err = hipMemcpy(host_data.data(), t.data_ptr<int>(), 25 * sizeof(int), hipMemcpyDeviceToHost);
        assert(err == hipSuccess);
        
        for (int val : host_data) {
            assert(val == 0);
        }
    }
#endif
}

void test_zeros_large() {
    std::cout << "Testing zeros large tensor..." << std::endl;
#if USE_HIP_BACKEND
    {
        // 1 million elements
        std::vector<int64_t> shape = {1000, 1000}; 
        vesper::Tensor t = vesper::zeros(shape, vesper::DType::Float32, vesper::Device::HIP);
        
        // Check first and last element only to save time, or copy a chunk
        float first = 1.0f;
        float last = 1.0f;
        
        hipError_t err;
        err = hipMemcpy(&first, t.data_ptr<float>(), sizeof(float), hipMemcpyDeviceToHost);
        assert(err == hipSuccess);
        err = hipMemcpy(&last, t.data_ptr<float>() + 999999, sizeof(float), hipMemcpyDeviceToHost);
        assert(err == hipSuccess);
        
        assert(first == 0.0f);
        assert(last == 0.0f);
    }
#endif
}

int main() {
    test_zeros_float32();
    test_zeros_int32();
    test_zeros_large();
    std::cout << "Zeros factory tests passed!" << std::endl;
    return 0;
}
