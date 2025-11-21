#include <vesper/ops/gemm.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/elementwise.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_matmul_transposed_input_grad() {
    std::cout << "Testing matmul with transposed input gradient..." << std::endl;
    // C = A^T * B
    // A: [3, 2], B: [3, 4], C: [2, 4]
    
    auto a = vesper::full({3, 2}, vesper::DType::Float32, vesper::Device::CPU, 1.0f);
    a.set_requires_grad(true);
    auto b = vesper::full({3, 4}, vesper::DType::Float32, vesper::Device::CPU, 1.0f);
    b.set_requires_grad(true);

    auto a_T = a.transpose(0, 1); // [2, 3]
    auto c = vesper::ops::matmul(a_T, b); // [2, 3] * [3, 4] -> [2, 4]
    auto loss = vesper::ops::sum(c);
    loss.backward();

    // Analytical check
    // C = A^T * B
    // dL/dC = ones(2, 4)
    // dL/dA = B * dL/dC^T ? No.
    // C_ij = sum_k (A^T)_ik * B_kj = sum_k A_ki * B_kj
    // dC_ij / dA_xy. Only non-zero if x=k, y=i.
    // dL/dA = B * (dL/dC)^T ?
    // Shape check: B [3, 4], dL/dC [2, 4]. B * dL/dC^T -> [3, 4] * [4, 2] -> [3, 2]. Matches A.
    // dL/dA = B * ones(4, 2).
    // Each element of dL/dA should be sum of corresponding row of B * sum of col of ones?
    // (B * ones(4, 2))_xy = sum_k B_xk * 1.
    // So each element (x, y) is sum of x-th row of B.
    // B is all ones. So sum of row is 4.
    // So grad(A) should be all 4s.

    std::vector<float> grad_a(a.numel());
    a.grad().copy_to_host(grad_a.data());
    
    for (float v : grad_a) {
        assert(std::abs(v - 4.0f) < 1e-5);
    }
    std::cout << "Matmul transposed input test passed!" << std::endl;
}

void test_grad_accumulation() {
    std::cout << "Testing gradient accumulation..." << std::endl;
    auto x = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f);
    x.set_requires_grad(true);

    // y1 = x * x (elementwise mul) -> grad = 2x = 2
    // y2 = x + x -> grad = 2
    // Total grad should be 4
    
    // Pass 1
    auto y1 = vesper::ops::mul(x, x);
    auto loss1 = vesper::ops::sum(y1);
    loss1.backward();
    
    // Check intermediate
    std::vector<float> g1(1);
    x.grad().copy_to_host(g1.data());
    assert(g1[0] == 2.0f);

    // Pass 2 (should accumulate)
    // Note: We need to clear grad if we didn't want accumulation, but here we DO want it.
    // But wait, usually we zero_grad between steps.
    // Here we simulate a graph where x is used twice.
    // Let's do a single graph with x used twice.
    
    x.grad() = vesper::zeros({1}, vesper::DType::Float32, vesper::Device::CPU); // Reset
    
    auto z1 = vesper::ops::mul(x, x);
    auto z2 = vesper::ops::add(x, x);
    auto total = vesper::ops::add(z1, z2);
    auto loss2 = vesper::ops::sum(total);
    loss2.backward();

    std::vector<float> g2(1);
    x.grad().copy_to_host(g2.data());
    // d(x^2 + 2x)/dx = 2x + 2. At x=1, grad=4.
    assert(g2[0] == 4.0f);

    std::cout << "Gradient accumulation test passed!" << std::endl;
}

int main() {
    test_matmul_transposed_input_grad();
    test_grad_accumulation();
    return 0;
}
