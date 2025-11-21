#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <cmath>

using namespace vesper;

void test_large_vector_ops() {
    std::cout << "Testing large vector operations (N=1,000,000)..." << std::endl;
    int64_t N = 1000000;
    auto a = full({N}, DType::Float32, Device::CPU, 1.0f, true);
    auto b = full({N}, DType::Float32, Device::CPU, 2.0f, true);

    auto start = std::chrono::high_resolution_clock::now();
    
    auto c = ops::add(a, b); // 3.0
    auto d = ops::mul(c, a); // 3.0 * 1.0 = 3.0
    auto loss = ops::sum(d); // 3.0 * N
    
    loss.backward();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "  Time taken: " << diff.count() << " s" << std::endl;

    // Check gradients
    // d = (a+b)*a = a^2 + ab
    // d(loss)/da = 2a + b = 2(1) + 2 = 4
    // d(loss)/db = a = 1
    
    std::vector<float> grad_a(N);
    a.grad().copy_to_host(grad_a.data());
    
    // Check a few random indices to save time
    assert(grad_a[0] == 4.0f);
    assert(grad_a[N/2] == 4.0f);
    assert(grad_a[N-1] == 4.0f);

    std::cout << "Large vector ops passed!" << std::endl;
}

void test_large_matmul() {
    std::cout << "Testing large matmul (512x512)..." << std::endl;
    int64_t N = 512;
    auto a = full({N, N}, DType::Float32, Device::CPU, 1.0f, true);
    auto b = full({N, N}, DType::Float32, Device::CPU, 0.5f, true);

    auto start = std::chrono::high_resolution_clock::now();

    auto c = ops::matmul(a, b); // 1.0 * 0.5 * N = 0.5 * 512 = 256.0
    auto loss = ops::sum(c);
    loss.backward();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "  Time taken: " << diff.count() << " s" << std::endl;

    // Check value
    std::vector<float> c_data(N*N);
    c.copy_to_host(c_data.data());
    assert(c_data[0] == 256.0f);

    // Check gradients
    // C = A * B
    // L = sum(C)
    // dL/dA = dL/dC * B^T = ones * B^T
    // (ones * B^T)_ij = sum_k (1 * B_jk^T) = sum_k B_kj
    // B is constant 0.5. Sum of row/col is 0.5 * N = 256.
    
    std::vector<float> grad_a(N*N);
    a.grad().copy_to_host(grad_a.data());
    assert(grad_a[0] == 256.0f);

    std::cout << "Large matmul passed!" << std::endl;
}

void test_deep_graph() {
    std::cout << "Testing deep computation graph (depth=1000)..." << std::endl;
    auto x = full({1}, DType::Float32, Device::CPU, 1.0f, true);
    auto y = x;
    
    int depth = 1000;
    for (int i = 0; i < depth; ++i) {
        y = ops::add(y, x); 
    }
    // y = x + x + ... + x (1001 times x) -> y = 1001 * x
    
    y.backward();
    
    std::vector<float> grad_x(1);
    x.grad().copy_to_host(grad_x.data());
    
    // Gradient should be 1001
    // Initial x is used once.
    // Loop runs 1000 times.
    // i=0: y = x + x = 2x
    // i=1: y = 2x + x = 3x
    // ...
    // i=999: y = 1000x + x = 1001x
    
    assert(std::abs(grad_x[0] - (float)(depth + 1)) < 1e-5);
    std::cout << "Deep graph passed!" << std::endl;
}

void test_broadcasting_large() {
    std::cout << "Testing large broadcasting..." << std::endl;
    // [1000, 1000] + [1000]
    int64_t N = 1000;
    auto a = full({N, N}, DType::Float32, Device::CPU, 1.0f, true);
    auto b = full({N}, DType::Float32, Device::CPU, 2.0f, true); // 1D tensor
    
    // Note: Our current add implementation supports [B, N] + [N] if b matches last dim of a
    auto c = ops::add(a, b);
    
    // Check forward
    std::vector<float> c_data(N*N);
    c.copy_to_host(c_data.data());
    assert(c_data[0] == 3.0f);
    assert(c_data[N*N-1] == 3.0f);
    
    // Check backward
    auto loss = ops::sum(c);
    loss.backward();
    
    // dL/db
    // c_ij = a_ij + b_j
    // L = sum(c_ij)
    // dL/db_k = sum_{ij} dL/dc_ij * dc_ij/db_k
    // dc_ij/db_k = 1 if j=k else 0
    // dL/db_k = sum_i (1 * 1) = N
    
    std::vector<float> grad_b(N);
    b.grad().copy_to_host(grad_b.data());
    
    assert(grad_b[0] == (float)N);
    assert(grad_b[N-1] == (float)N);
    
    std::cout << "Large broadcasting passed!" << std::endl;
}

int main() {
    test_large_vector_ops();
    test_large_matmul();
    test_deep_graph();
    test_broadcasting_large();
    return 0;
}
