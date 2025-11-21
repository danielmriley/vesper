#include <vesper/core/factories.h>
#include <vesper/core/reference_ops.h>
#include <vesper/ops/gemm.h> // For the dispatch function
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

void test_gemm() {
#if USE_HIP_BACKEND
    std::cout << "Testing GEMM kernel..." << std::endl;

    int M = 32, K = 48, N = 64; // Non-square, non-multiple-of-16 dimensions

    // 1. Prepare host data with random values
    std::vector<float> h_A(M * K), h_B(K * N), h_C_gpu(M * N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& val : h_A) val = dist(rng);
    for (float& val : h_B) val = dist(rng);

    // 2. Prepare device tensors and copy data
    vesper::Tensor d_A = vesper::empty({M, K}, vesper::DType::Float32, vesper::Device::HIP);
    vesper::Tensor d_B = vesper::empty({K, N}, vesper::DType::Float32, vesper::Device::HIP);
    vesper::Tensor d_C = vesper::empty({M, N}, vesper::DType::Float32, vesper::Device::HIP);
    d_A.copy_from_host(h_A.data());
    d_B.copy_from_host(h_B.data());

    // 3. Compute ground truth on CPU using reference ops
    // We need CPU tensors for the reference implementation
    vesper::Tensor ref_A = vesper::empty({M, K}, vesper::DType::Float32, vesper::Device::CPU);
    vesper::Tensor ref_B = vesper::empty({K, N}, vesper::DType::Float32, vesper::Device::CPU);
    vesper::Tensor ref_C = vesper::empty({M, N}, vesper::DType::Float32, vesper::Device::CPU);
    
    ref_A.copy_from_host(h_A.data());
    ref_B.copy_from_host(h_B.data());
    
    vesper::reference::gemm(ref_A, ref_B, ref_C, false, false);

    // 4. Launch the kernel via the dispatch function
    vesper::ops::gemm_hip_dispatch(d_A, d_B, d_C, false, false);
    
    // 5. Copy result back to host
    d_C.copy_to_host(h_C_gpu.data());

    // 6. Verify GPU result against CPU ground truth
    const float* ref_ptr = ref_C.data_ptr<float>();
    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(ref_ptr[i] - h_C_gpu[i]) > 1e-3) { // Use a tolerance for FP math
            errors++;
        }
    }
    assert(errors == 0);
    if (errors > 0) {
        std::cerr << "GEMM test failed with " << errors << " errors." << std::endl;
    } else {
        std::cout << "GEMM kernel test passed!" << std::endl;
    }
#else
    std::cout << "Skipping GEMM kernel test (HIP backend disabled)." << std::endl;
#endif
}

int main() {
    test_gemm();
    return 0;
}
