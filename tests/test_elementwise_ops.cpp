#include <vesper/ops/elementwise.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

void test_add_op() {
#if USE_HIP_BACKEND
    std::cout << "Testing element-wise add operation..." << std::endl;

    int deviceId;
    if (hipGetDevice(&deviceId) != hipSuccess) {
        std::cerr << "Failed to get device ID" << std::endl;
        return;
    }
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, deviceId) != hipSuccess) {
        std::cerr << "Failed to get device properties" << std::endl;
        return;
    }
    std::cout << "Running on device: " << props.name << std::endl;

    const std::vector<int64_t> shape = {2, 2};
    const vesper::DType dtype = vesper::DType::Float32;
    const vesper::Device device = vesper::Device::HIP;

    // 1. Prepare host data
    std::vector<float> a_host = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_host = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> result_host(4);

    // 2. Create tensors and copy data to device
    vesper::Tensor a = vesper::empty(shape, dtype, device);
    vesper::Tensor b = vesper::empty(shape, dtype, device);
    a.copy_from_host(a_host.data());
    b.copy_from_host(b_host.data());

    // 3. Perform the operation
    vesper::Tensor c = vesper::ops::add(a, b);

    // 4. Copy result back to host
    c.copy_to_host(result_host.data());

    // 5. Verify the result
    for (size_t i = 0; i < a_host.size(); ++i) {
        const float expected = a_host[i] + b_host[i];
        assert(std::fabs(result_host[i] - expected) < 1e-6);
    }

    std::cout << "Element-wise add test passed!" << std::endl;
#else
    std::cout << "Skipping element-wise add test (HIP backend disabled)." << std::endl;
#endif
}

void test_sub_op() {
#if USE_HIP_BACKEND
    std::cout << "Testing element-wise sub operation..." << std::endl;

    const std::vector<int64_t> shape = {2, 2};
    const vesper::DType dtype = vesper::DType::Float32;
    const vesper::Device device = vesper::Device::HIP;

    std::vector<float> a_host = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b_host = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> result_host(4);

    vesper::Tensor a = vesper::empty(shape, dtype, device);
    vesper::Tensor b = vesper::empty(shape, dtype, device);
    a.copy_from_host(a_host.data());
    b.copy_from_host(b_host.data());

    vesper::Tensor c = vesper::ops::sub(a, b);

    c.copy_to_host(result_host.data());

    for (size_t i = 0; i < a_host.size(); ++i) {
        const float expected = a_host[i] - b_host[i];
        assert(std::fabs(result_host[i] - expected) < 1e-6);
    }

    std::cout << "Element-wise sub test passed!" << std::endl;
#endif
}

void test_mul_op() {
#if USE_HIP_BACKEND
    std::cout << "Testing element-wise mul operation..." << std::endl;

    const std::vector<int64_t> shape = {2, 2};
    const vesper::DType dtype = vesper::DType::Float32;
    const vesper::Device device = vesper::Device::HIP;

    std::vector<float> a_host = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b_host = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> result_host(4);

    vesper::Tensor a = vesper::empty(shape, dtype, device);
    vesper::Tensor b = vesper::empty(shape, dtype, device);
    a.copy_from_host(a_host.data());
    b.copy_from_host(b_host.data());

    vesper::Tensor c = vesper::ops::mul(a, b);

    c.copy_to_host(result_host.data());

    for (size_t i = 0; i < a_host.size(); ++i) {
        const float expected = a_host[i] * b_host[i];
        assert(std::fabs(result_host[i] - expected) < 1e-6);
    }

    std::cout << "Element-wise mul test passed!" << std::endl;
#endif
}

void test_mul_scalar_op() {
#if USE_HIP_BACKEND
    std::cout << "Testing element-wise mul (scalar) operation..." << std::endl;

    const std::vector<int64_t> shape = {2, 2};
    const vesper::DType dtype = vesper::DType::Float32;
    const vesper::Device device = vesper::Device::HIP;

    std::vector<float> a_host = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar = 2.5f;
    std::vector<float> result_host(4);

    vesper::Tensor a = vesper::empty(shape, dtype, device);
    a.copy_from_host(a_host.data());

    vesper::Tensor c = vesper::ops::mul(a, scalar);

    c.copy_to_host(result_host.data());

    for (size_t i = 0; i < a_host.size(); ++i) {
        const float expected = a_host[i] * scalar;
        assert(std::fabs(result_host[i] - expected) < 1e-6);
    }

    std::cout << "Element-wise mul (scalar) test passed!" << std::endl;
#endif
}

void test_large_input_op() {
#if USE_HIP_BACKEND
    std::cout << "Testing large input element-wise add operation..." << std::endl;

    // 1 million elements
    const int64_t N = 1000000;
    const std::vector<int64_t> shape = {N};
    const vesper::DType dtype = vesper::DType::Float32;
    const vesper::Device device = vesper::Device::HIP;

    std::vector<float> a_host(N, 1.0f);
    std::vector<float> b_host(N, 2.0f);
    std::vector<float> result_host(N);

    vesper::Tensor a = vesper::empty(shape, dtype, device);
    vesper::Tensor b = vesper::empty(shape, dtype, device);
    
    a.copy_from_host(a_host.data());
    b.copy_from_host(b_host.data());

    vesper::Tensor c = vesper::ops::add(a, b);

    c.copy_to_host(result_host.data());

    // Verify start, middle, end
    assert(std::fabs(result_host[0] - 3.0f) < 1e-6);
    assert(std::fabs(result_host[N/2] - 3.0f) < 1e-6);
    assert(std::fabs(result_host[N-1] - 3.0f) < 1e-6);

    std::cout << "Large input test passed!" << std::endl;
#endif
}

int main() {
    test_add_op();
    test_sub_op();
    test_mul_op();
    test_mul_scalar_op();
    test_large_input_op();
    return 0;
}
