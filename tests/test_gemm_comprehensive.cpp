#include <vesper/ops/gemm.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <stdexcept>

// Helper for CPU GEMM
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
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

// Helper to run a test case
void run_gemm_test(int M, int K, int N, const std::string& test_name) {
#if USE_HIP_BACKEND
    std::cout << "Running " << test_name << " (" << M << "x" << K << " * " << K << "x" << N << ")... ";

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_gpu(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (auto& x : h_A) x = dist(rng);
    for (auto& x : h_B) x = dist(rng);

    cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    vesper::Tensor d_A = vesper::empty({M, K}, vesper::DType::Float32, vesper::Device::HIP);
    vesper::Tensor d_B = vesper::empty({K, N}, vesper::DType::Float32, vesper::Device::HIP);
    
    d_A.copy_from_host(h_A.data());
    d_B.copy_from_host(h_B.data());

    vesper::Tensor d_C = vesper::ops::matmul(d_A, d_B);

    d_C.copy_to_host(h_C_gpu.data());

    // Verification
    int errors = 0;
    for (size_t i = 0; i < h_C_cpu.size(); ++i) {
        // Use a slightly looser tolerance for larger accumulations
        if (std::fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3) {
            if (errors < 5) {
                std::cerr << "\nError at index " << i << ": CPU=" << h_C_cpu[i] << ", GPU=" << h_C_gpu[i];
            }
            errors++;
        }
    }

    if (errors > 0) {
        std::cout << "FAILED with " << errors << " errors." << std::endl;
        exit(1);
    } else {
        std::cout << "PASSED." << std::endl;
    }
#else
    std::cout << "Skipping " << test_name << " (HIP disabled)." << std::endl;
#endif
}

void test_prime_dimensions() {
    // Primes ensure we test boundary conditions (not multiples of TILE_WIDTH=16)
    run_gemm_test(17, 13, 19, "Prime Dimensions");
    run_gemm_test(31, 31, 31, "Prime Dimensions (31)"); // Just under 32
    run_gemm_test(33, 33, 33, "Prime Dimensions (33)"); // Just over 32
}

void test_small_matrices() {
    run_gemm_test(1, 1, 1, "1x1 Matrix");
    run_gemm_test(1, 10, 1, "Dot Product (1x10 * 10x1)");
    run_gemm_test(10, 1, 10, "Outer Product (10x1 * 1x10)");
}

void test_skinny_matrices() {
    run_gemm_test(100, 2, 50, "Skinny Middle");
    run_gemm_test(2, 100, 2, "Fat Middle");
}

void test_large_matrices() {
    // 512 is a multiple of 16, but large enough to stress the grid
    run_gemm_test(256, 256, 256, "256x256 Matrix");
}

void test_error_handling() {
#if USE_HIP_BACKEND
    std::cout << "Running Error Handling Tests... ";
    
    auto A = vesper::empty({10, 20}, vesper::DType::Float32, vesper::Device::HIP);
    auto B_wrong_shape = vesper::empty({21, 10}, vesper::DType::Float32, vesper::Device::HIP);
    auto B_wrong_rank = vesper::empty({20, 10, 5}, vesper::DType::Float32, vesper::Device::HIP);

    bool caught_shape = false;
    try {
        vesper::ops::matmul(A, B_wrong_shape);
    } catch (const std::runtime_error& e) {
        caught_shape = true;
    }

    bool caught_rank = false;
    try {
        vesper::ops::matmul(A, B_wrong_rank);
    } catch (const std::runtime_error& e) {
        caught_rank = true;
    }

    if (caught_shape && caught_rank) {
        std::cout << "PASSED." << std::endl;
    } else {
        std::cout << "FAILED (Exceptions not thrown)." << std::endl;
        exit(1);
    }
#endif
}

int main() {
    test_prime_dimensions();
    test_small_matrices();
    test_skinny_matrices();
    test_large_matrices();
    test_error_handling();
    return 0;
}
