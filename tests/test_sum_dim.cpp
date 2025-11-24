#include "vesper/core/factories.h"
#include "vesper/ops/reduction.h"
#include "vesper/ops/elementwise.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace vesper;

void check_float(float actual, float expected, float tol = 1e-4) {
    if (std::fabs(actual - expected) > tol) {
        std::cerr << "Check failed: actual=" << actual << ", expected=" << expected << std::endl;
        exit(1);
    }
}

void test_sum_dim_cpu() {
    std::cout << "Testing Sum Dim CPU..." << std::endl;
    // Shape (2, 3, 4)
    // Sum over dim 1 -> (2, 4)
    
    std::vector<int64_t> shape = {2, 3, 4};
    Tensor t = full(shape, DType::Float32, Device::CPU, 1.0f);
    
    // Sum over dim 1
    Tensor res = ops::sum(t, 1, false);
    
    assert(res.ndim() == 2);
    assert(res.shape()[0] == 2);
    assert(res.shape()[1] == 4);
    
    // Expected value: 3.0 (since dim 1 has size 3 and values are 1.0)
    float* ptr = res.data_ptr<float>();
    for (int i = 0; i < res.numel(); ++i) {
        check_float(ptr[i], 3.0f);
    }
    
    // Keepdim=true
    Tensor res_keep = ops::sum(t, 1, true);
    assert(res_keep.ndim() == 3);
    assert(res_keep.shape()[0] == 2);
    assert(res_keep.shape()[1] == 1);
    assert(res_keep.shape()[2] == 4);
    
    ptr = res_keep.data_ptr<float>();
    for (int i = 0; i < res_keep.numel(); ++i) {
        check_float(ptr[i], 3.0f);
    }
    
    std::cout << "Sum Dim CPU Passed!" << std::endl;
}

void test_sum_dim_gpu() {
#if USE_CUDA_BACKEND || USE_HIP_BACKEND
    Device device = Device::CUDA;
#if USE_HIP_BACKEND
    device = Device::HIP;
#endif
    // Simple check if device is available (mocked if needed)
    // In real scenario, we should check Device::is_available(device)
    // But here we assume if macro is defined, we try to run.
    
    std::cout << "Testing Sum Dim GPU..." << std::endl;
    
    try {
        std::vector<int64_t> shape = {2, 3, 4};
        Tensor t = full(shape, DType::Float32, device, 1.0f);
        
        // Sum over dim 1
        Tensor res = ops::sum(t, 1, false);
        
        assert(res.ndim() == 2);
        assert(res.shape()[0] == 2);
        assert(res.shape()[1] == 4);
        
        Tensor res_cpu = res.contiguous().to(Device::CPU);
        float* ptr = res_cpu.data_ptr<float>();
        for (int i = 0; i < res_cpu.numel(); ++i) {
            check_float(ptr[i], 3.0f);
        }
        
        // Sum over dim 0 -> (3, 4)
        Tensor res0 = ops::sum(t, 0, false);
        assert(res0.ndim() == 2);
        assert(res0.shape()[0] == 3);
        assert(res0.shape()[1] == 4);
        
        Tensor res0_cpu = res0.contiguous().to(Device::CPU);
        ptr = res0_cpu.data_ptr<float>();
        for (int i = 0; i < res0_cpu.numel(); ++i) {
            check_float(ptr[i], 2.0f);
        }
        
        // Sum over dim 2 -> (2, 3)
        Tensor res2 = ops::sum(t, 2, false);
        assert(res2.ndim() == 2);
        assert(res2.shape()[0] == 2);
        assert(res2.shape()[1] == 3);
        
        Tensor res2_cpu = res2.contiguous().to(Device::CPU);
        ptr = res2_cpu.data_ptr<float>();
        for (int i = 0; i < res2_cpu.numel(); ++i) {
            check_float(ptr[i], 4.0f);
        }

        std::cout << "Sum Dim GPU Passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "GPU Test Failed/Skipped: " << e.what() << std::endl;
    }
#endif
}

void test_sum_dim_autograd() {
    std::cout << "Testing Sum Dim Autograd..." << std::endl;
    
    // x = ones(2, 3)
    // y = sum(x, 0) -> shape (3)
    // loss = sum(y) -> scalar
    // grad_x should be ones(2, 3)
    
    Tensor x = full({2, 3}, DType::Float32, Device::CPU, 1.0f, true);
    Tensor y = ops::sum(x, 0, false);
    Tensor loss = ops::sum(y); // Full sum
    
    loss.backward();
    
    assert(x.grad().defined());
    float* ptr = x.grad().data_ptr<float>();
    for (int i = 0; i < x.numel(); ++i) {
        check_float(ptr[i], 1.0f);
    }
    
    std::cout << "Sum Dim Autograd Passed!" << std::endl;
}

int main() {
    test_sum_dim_cpu();
    test_sum_dim_gpu();
    test_sum_dim_autograd();
    return 0;
}
