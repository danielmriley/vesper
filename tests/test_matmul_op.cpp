#include <vesper/ops/gemm.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

// Re-use the naive CPU GEMM for verification
void naive_gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void test_matmul_public_api() {
#if USE_HIP_BACKEND
    std::cout << "Testing public matmul API..." << std::endl;

    int M = 64, K = 32, N = 48;

    // 1. Prepare host data and compute CPU ground truth
    std::vector<float> h_A(M * K), h_B(K * N), h_C_cpu(M * N), h_C_gpu(M * N);
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& val : h_A) val = dist(rng);
    for (float& val : h_B) val = dist(rng);
    naive_gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    // 2. Prepare device tensors
    vesper::Tensor d_A = vesper::empty({M, K}, vesper::DType::Float32, vesper::Device::HIP);
    vesper::Tensor d_B = vesper::empty({K, N}, vesper::DType::Float32, vesper::Device::HIP);
    d_A.copy_from_host(h_A.data());
    d_B.copy_from_host(h_B.data());

    // 3. Call the public matmul function
    vesper::Tensor d_C = vesper::ops::matmul(d_A, d_B);

    // 4. Copy result back and verify
    d_C.copy_to_host(h_C_gpu.data());

    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3) {
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
