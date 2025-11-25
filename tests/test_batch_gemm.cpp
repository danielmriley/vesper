#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/random.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/comparison.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstring>

using namespace vesper;

// Helper to check if tensors are close
void assert_tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-3, float atol = 1e-3) {
    // Move to CPU for checking
    Tensor a_cpu = a.to(Device::CPU);
    Tensor b_cpu = b.to(Device::CPU);

    const float* a_data = a_cpu.data_ptr<float>();
    const float* b_data = b_cpu.data_ptr<float>();
    size_t size = a.numel();

    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float tol = atol + rtol * std::abs(b_data[i]);
        if (diff > tol) {
            std::cerr << "Mismatch at index " << i << ": " << a_data[i] << " vs " << b_data[i] << std::endl;
            assert(false);
        }
    }
}

void test_batch_gemm_correctness() {
    std::cout << "Testing Batch GEMM Correctness..." << std::endl;
    
    int B = 4;
    int M = 32;
    int K = 16;
    int N = 32;
    
    // 1. Create data on CPU
    Tensor A_cpu = vesper::empty({B, M, K}, DType::Float32, Device::CPU);
    ops::normal_(A_cpu, 0.0f, 1.0f);
    
    Tensor B_cpu = vesper::empty({B, K, N}, DType::Float32, Device::CPU);
    ops::normal_(B_cpu, 0.0f, 1.0f);
    
    // 2. Compute Reference (Loop of GEMMs)
    std::vector<float> c_ref_data(B * M * N);
    
    for (int i = 0; i < B; ++i) {
        const float* a_ptr = A_cpu.data_ptr<float>() + i * M * K;
        const float* b_ptr = B_cpu.data_ptr<float>() + i * K * N;
        float* c_ptr = c_ref_data.data() + i * M * N;
        
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += a_ptr[m * K + k] * b_ptr[k * N + n];
                }
                c_ptr[m * N + n] = sum;
            }
        }
    }
    
    Tensor C_ref = vesper::empty({B, M, N}, DType::Float32, Device::CPU);
    C_ref.copy_from_host(c_ref_data.data());
    
    // 3. Compute on GPU
#if USE_CUDA_BACKEND
    if (Device::CUDA == Device::CUDA) { 
        Tensor A_gpu = A_cpu.to(Device::CUDA);
        Tensor B_gpu = B_cpu.to(Device::CUDA);
        Tensor C_gpu = ops::matmul(A_gpu, B_gpu);
        Tensor C_gpu_cpu = C_gpu.to(Device::CPU);
        
        assert_tensors_close(C_ref, C_gpu_cpu, 1e-3, 1e-3);
        std::cout << "Batch GEMM (CUDA) Passed!" << std::endl;
    }
#endif

#if USE_HIP_BACKEND
    // Similar for HIP
#endif
}

void test_broadcasting() {
    std::cout << "Testing Batch GEMM Broadcasting..." << std::endl;
    
    int B = 4;
    int M = 32;
    int K = 16;
    int N = 32;
    
    // (1, M, K) @ (B, K, N) -> (B, M, N)
    Tensor A_cpu = vesper::empty({1, M, K}, DType::Float32, Device::CPU);
    ops::normal_(A_cpu, 0.0f, 1.0f);
    
    Tensor B_cpu = vesper::empty({B, K, N}, DType::Float32, Device::CPU);
    ops::normal_(B_cpu, 0.0f, 1.0f);
    
    // Reference
    std::vector<float> c_ref_data(B * M * N);
    const float* a_ptr = A_cpu.data_ptr<float>(); // Always same A
    
    for (int i = 0; i < B; ++i) {
        const float* b_ptr = B_cpu.data_ptr<float>() + i * K * N;
        float* c_ptr = c_ref_data.data() + i * M * N;
        
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += a_ptr[m * K + k] * b_ptr[k * N + n];
                }
                c_ptr[m * N + n] = sum;
            }
        }
    }
    Tensor C_ref = vesper::empty({B, M, N}, DType::Float32, Device::CPU);
    C_ref.copy_from_host(c_ref_data.data());

#if USE_CUDA_BACKEND
    Tensor A_gpu = A_cpu.to(Device::CUDA);
    Tensor B_gpu = B_cpu.to(Device::CUDA);
    Tensor C_gpu = ops::matmul(A_gpu, B_gpu);
    Tensor C_gpu_cpu = C_gpu.to(Device::CPU);
    
    assert_tensors_close(C_ref, C_gpu_cpu, 1e-3, 1e-3);
    std::cout << "Broadcasting (CUDA) Passed!" << std::endl;
#endif
}

void test_broadcasting_mixed() {
    std::cout << "Testing Mixed Broadcasting..." << std::endl;
    
    int B = 4;
    int M = 16;
    int K = 16;
    int N = 16;
    
    // (B, M, K) @ (1, K, N) -> (B, M, N)
    Tensor A = vesper::empty({B, M, K}, DType::Float32, Device::CPU);
    ops::normal_(A, 0.0f, 1.0f);
    
    Tensor B_t = vesper::empty({1, K, N}, DType::Float32, Device::CPU);
    ops::normal_(B_t, 0.0f, 1.0f);
    
    // Reference
    // We can just replicate B_t manually to check
    Tensor B_expanded = vesper::empty({B, K, N}, DType::Float32, Device::CPU);
    for(int i=0; i<B; ++i) {
        // Copy B_t to slice i
        // This is a bit hacky without slice op, but we can use data ptrs
        float* dst = B_expanded.data_ptr<float>() + i * K * N;
        const float* src = B_t.data_ptr<float>();
        std::memcpy(dst, src, K * N * sizeof(float));
    }
    
#if USE_CUDA_BACKEND
    Tensor A_gpu = A.to(Device::CUDA);
    Tensor B_gpu = B_t.to(Device::CUDA);
    Tensor C_gpu = ops::matmul(A_gpu, B_gpu);
    
    Tensor B_exp_gpu = B_expanded.to(Device::CUDA);
    Tensor C_ref_gpu = ops::matmul(A_gpu, B_exp_gpu);
    
    assert_tensors_close(C_gpu, C_ref_gpu, 1e-3, 1e-3);
    std::cout << "Mixed Broadcasting (CUDA) Passed!" << std::endl;
#endif
}

void test_4d_tensors() {
    std::cout << "Testing 4D Batch GEMM..." << std::endl;
    
    int B = 2;
    int H = 4;
    int M = 16;
    int K = 16;
    int N = 16;
    
    Tensor A_cpu = vesper::empty({B, H, M, K}, DType::Float32, Device::CPU);
    ops::normal_(A_cpu, 0.0f, 1.0f);
    
    Tensor B_cpu = vesper::empty({B, H, K, N}, DType::Float32, Device::CPU);
    ops::normal_(B_cpu, 0.0f, 1.0f);
    
    // Reference
    std::vector<float> c_ref_data(B * H * M * N);
    
    for (int i = 0; i < B * H; ++i) {
        const float* a_ptr = A_cpu.data_ptr<float>() + i * M * K;
        const float* b_ptr = B_cpu.data_ptr<float>() + i * K * N;
        float* c_ptr = c_ref_data.data() + i * M * N;
        
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += a_ptr[m * K + k] * b_ptr[k * N + n];
                }
                c_ptr[m * N + n] = sum;
            }
        }
    }
    Tensor C_ref = vesper::empty({B, H, M, N}, DType::Float32, Device::CPU);
    C_ref.copy_from_host(c_ref_data.data());

#if USE_CUDA_BACKEND
    Tensor A_gpu = A_cpu.to(Device::CUDA);
    Tensor B_gpu = B_cpu.to(Device::CUDA);
    Tensor C_gpu = ops::matmul(A_gpu, B_gpu);
    Tensor C_gpu_cpu = C_gpu.to(Device::CPU);
    
    assert_tensors_close(C_ref, C_gpu_cpu, 1e-3, 1e-3);
    std::cout << "4D Batch GEMM (CUDA) Passed!" << std::endl;
#endif
}

void test_batch_gemm_backward() {
    std::cout << "Testing Batch GEMM Backward..." << std::endl;

#if USE_CUDA_BACKEND
    // A: (2, 32, 64), B: (2, 64, 32) -> C: (2, 32, 32)
    Tensor A = vesper::empty({2, 32, 64}, DType::Float32, Device::CUDA, true);
    ops::normal_(A, 0.0f, 1.0f);
    
    Tensor B = vesper::empty({2, 64, 32}, DType::Float32, Device::CUDA, true);
    ops::normal_(B, 0.0f, 1.0f);
    
    Tensor C = ops::matmul(A, B);
    
    // Fake gradient for C: (2, 32, 32)
    Tensor grad_C = vesper::empty({2, 32, 32}, DType::Float32, Device::CUDA);
    ops::normal_(grad_C, 0.0f, 1.0f);
    
    C.backward(grad_C);
    
    // Check that gradients are populated and have correct shape
    assert(A.grad().shape() == A.shape());
    assert(B.grad().shape() == B.shape());
    
    // Move to CPU to check values are not all zero
    Tensor grad_A_cpu = A.grad().to(Device::CPU);
    const float* ptr = grad_A_cpu.data_ptr<float>();
    bool non_zero = false;
    for(size_t i=0; i<grad_A_cpu.numel(); ++i) {
        if (ptr[i] != 0.0f) {
            non_zero = true;
            break;
        }
    }
    assert(non_zero);
    
    std::cout << "Batch GEMM Backward (CUDA) Passed!" << std::endl;
#else
    std::cout << "CUDA not available, skipping backward test." << std::endl;
#endif
}

int main() {
    try {
        test_batch_gemm_correctness();
        test_broadcasting();
        test_broadcasting_mixed();
        test_4d_tensors();
        test_batch_gemm_backward();
        std::cout << "All Batch GEMM tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
