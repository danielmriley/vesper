#include <vesper/ops/gemm.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <cmath>

void test_matmul_backward() {
    std::cout << "Testing matmul backward pass..." << std::endl;

    auto device = vesper::Device::CPU;
    
    // 1. Create input tensors
    // A: [2, 3]
    auto a = vesper::empty({2, 3}, vesper::DType::Float32, device, true);
    // B: [3, 4]
    auto b = vesper::empty({3, 4}, vesper::DType::Float32, device, true);
    
    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    
    a.copy_from_host(a_data.data());
    b.copy_from_host(b_data.data());

    // 2. Forward pass and a dummy loss
    auto c = vesper::ops::matmul(a, b);
    auto loss = vesper::ops::sum(c); // loss = sum(A * B)

    // 3. Backward pass
    loss.backward();

    // 4. Verification
    // d(loss)/d(C_ij) = 1 for all i,j because the loss is sum. So C.grad is all 1s.
    // d(loss)/dA = C.grad * B^T = ones(2,4) * B^T
    // Shape: [2, 4] * [4, 3] -> [2, 3]
    // grad(A)_ij = sum_k (1 * B^T_kj) = sum_k B_jk
    // So grad(A)_ij is the sum of the j-th row of B.
    
    // B rows:
    // Row 0: 1, 2, 3, 4 -> Sum = 10
    // Row 1: 5, 6, 7, 8 -> Sum = 26
    // Row 2: 9, 10, 11, 12 -> Sum = 42
    
    // Wait, let's re-calculate carefully.
    // C = A * B.
    // dL/dA = dL/dC * B^T.
    // dL/dC is ones(2, 4).
    // B is (3, 4). B^T is (4, 3).
    // dL/dA = ones(2, 4) * B^T.
    // (dL/dA)_ij = sum_k (ones_ik * (B^T)_kj) = sum_k (1 * B_jk)
    // So (dL/dA)_ij = sum over k of B_jk.
    // This means for a fixed j (column index of A), the gradient is sum of k-th column of B^T, which is k-th row of B.
    // Wait.
    // (dL/dA)_ij corresponds to A_ij.
    // The formula is sum_k (dL/dC)_ik * B_jk.
    // Here k is the index of the inner dimension of the multiplication dL/dC * B^T.
    // dL/dC is [2, 4]. B^T is [4, 3].
    // So k goes from 0 to 3.
    // (dL/dA)_ij = sum_{k=0..3} (dL/dC)_ik * (B^T)_kj
    //            = sum_{k=0..3} 1 * B_jk
    //            = sum_{k=0..3} B_jk
    // This is the sum of the j-th ROW of B? No.
    // B is [3, 4]. Indices are B_row,col.
    // B_jk means row j, col k.
    // So we are summing B_j0 + B_j1 + B_j2 + B_j3.
    // This is indeed the sum of the j-th row of B.
    // But wait, j is the column index of A.
    // A is [2, 3]. So j goes 0..2.
    // So for each element A_ij, the gradient depends only on j.
    // So all rows of grad(A) should be identical.
    // And the value for column j is the sum of the j-th row of B.
    
    // Row 0 of B (j=0): 1+2+3+4 = 10.
    // Row 1 of B (j=1): 5+6+7+8 = 26.
    // Row 2 of B (j=2): 9+10+11+12 = 42.
    
    // So grad(A) should be:
    // [10, 26, 42]
    // [10, 26, 42]
    
    std::vector<float> a_grad_data(a.numel());
    a.grad().copy_to_host(a_grad_data.data());

    // Row 0
    assert(fabs(a_grad_data[0] - 10) < 1e-4);
    assert(fabs(a_grad_data[1] - 26) < 1e-4);
    assert(fabs(a_grad_data[2] - 42) < 1e-4);
    // Row 1
    assert(fabs(a_grad_data[3] - 10) < 1e-4);
    assert(fabs(a_grad_data[4] - 26) < 1e-4);
    assert(fabs(a_grad_data[5] - 42) < 1e-4);
    
    std::cout << "Matmul backward pass test passed!" << std::endl;
}

int main() {
    test_matmul_backward();
    return 0;
}
