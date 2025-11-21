#include <vesper/ops/gemm.h>
#include <vesper/core/factories.h>
#include <vesper/core/reference_ops.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

void test_matmul_public_api() {
#if USE_HIP_BACKEND
    std::cout << "Testing public matmul API..." << std::endl;

    int M = 64, K = 32, N = 48;

    // 1. Prepare host data
    std::vector<float> h_A(M * K), h_B(K * N), h_C_gpu(M * N);
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& val : h_A) val = dist(rng);
    for (float& val : h_B) val = dist(rng);

    // 2. Compute ground truth on CPU using reference ops
    vesper::Tensor ref_A = vesper::empty({M, K}, vesper::DType::Float32, vesper::Device::CPU);
    vesper::Tensor ref_B = vesper::empty({K, N}, vesper::DType::Float32, vesper::Device::CPU);
    vesper::Tensor ref_C = vesper::empty({M, N}, vesper::DType::Float32, vesper::Device::CPU);
    
    ref_A.copy_from_host(h_A.data());
    ref_B.copy_from_host(h_B.data());
    
    vesper::reference::gemm(ref_A, ref_B, ref_C, false, false);

    // 3. Prepare device tensors
    vesper::Tensor d_A = vesper::empty({M, K}, vesper::DType::Float32, vesper::Device::HIP);
    vesper::Tensor d_B = vesper::empty({K, N}, vesper::DType::Float32, vesper::Device::HIP);
    d_A.copy_from_host(h_A.data());
    d_B.copy_from_host(h_B.data());

    // 4. Call the public matmul function
    vesper::Tensor d_C = vesper::ops::matmul(d_A, d_B);

    // 5. Copy result back and verify
    d_C.copy_to_host(h_C_gpu.data());

    const float* ref_ptr = ref_C.data_ptr<float>();
    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(ref_ptr[i] - h_C_gpu[i]) > 1e-3) {
            errors++;
        }
    }
    assert(errors == 0);
    std::cout << "Public matmul API test passed!" << std::endl;
#else
    std::cout << "Skipping public matmul API test (HIP backend disabled)." << std::endl;
#endif
}

int main() {
    test_matmul_public_api();
    return 0;
}
