#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <vesper/ops/elementwise.h>
#include <stdexcept>

namespace vesper::ops {

void gemm_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);
void gemm_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);

Tensor gemm(const Tensor& a, const Tensor& b, bool transA, bool transB) {
    // --- 1. Pre-condition Checks ---
    if (a.device() != b.device()) {
        throw std::runtime_error("GEMM requires tensors to be on the same device.");
    }
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("GEMM currently only supports 2D tensors.");
    }
    
    // Ensure contiguous tensors
    Tensor a_contig = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contig = b.is_contiguous() ? b : b.contiguous();

    // Determine logical dimensions
    // A: [M, K] if !transA, [K, M] if transA
    // B: [K, N] if !transB, [N, K] if transB
    
    int64_t M = transA ? a_contig.shape()[1] : a_contig.shape()[0];
    int64_t K_a = transA ? a_contig.shape()[0] : a_contig.shape()[1];
    
    int64_t K_b = transB ? b_contig.shape()[1] : b_contig.shape()[0];
    int64_t N = transB ? b_contig.shape()[0] : b_contig.shape()[1];

    if (K_a != K_b) {
        throw std::runtime_error("Inner dimensions of matrices do not match for GEMM.");
    }
    int64_t K = K_a;

    // --- 2. Prepare Output Tensor ---
    bool requires_grad = a_contig.requires_grad() || b_contig.requires_grad();
    Tensor c = empty({M, N}, a_contig.dtype(), a_contig.device(), requires_grad);

    // --- 3. Dispatch to Backend-Specific Implementation ---
    switch (a_contig.device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            gemm_hip_dispatch(a_contig, b_contig, c, transA, transB);
#else
            throw std::runtime_error("HIP backend not enabled during build, but required for GEMM.");
#endif
            break;
        
        case Device::CPU:
            gemm_cpu_dispatch(a_contig, b_contig, c, transA, transB);
            break;

        case Device::CUDA:
#if USE_CUDA_BACKEND
            gemm_cuda_dispatch(a_contig, b_contig, c, transA, transB);
#else
            throw std::runtime_error("CUDA backend not enabled during build.");
#endif
            break;

        default:
            throw std::runtime_error("Unsupported device for GEMM.");
    }

    // --- 4. Autograd ---
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        
        if (a_contig.requires_grad() && a_contig.grad_node) {
            node->next_edges.push_back({a_contig.grad_node});
        }
        if (b_contig.requires_grad() && b_contig.grad_node) {
            node->next_edges.push_back({b_contig.grad_node});
        }

        node->backward_fn = [a=a_contig, b=b_contig, c, transA, transB]() mutable {
            Tensor& grad_output = c.grad();
            
            // C = op(A) * op(B)
            // grad_A_contrib = grad_C * op(B)^T
            // grad_B_contrib = op(A)^T * grad_C
            
            // Case 1: C = A * B (transA=F, transB=F)
            // grad_A = grad_C * B^T
            // grad_B = A^T * grad_C
            
            // Case 2: C = A^T * B (transA=T, transB=F)
            // grad_A = (grad_C * B^T)^T = B * grad_C^T
            // grad_B = (A^T)^T * grad_C = A * grad_C
            // Wait, let's derive carefully.
            
            // General rule:
            // dL/dA = dL/dC * (dC/dA)
            
            if (a.requires_grad()) {
                Tensor grad_a_contrib = empty({0}, a.dtype(), a.device()); // Dummy init
                if (!transA && !transB) {
                    // C = A * B
                    // grad_A = grad_C * B^T
                    grad_a_contrib = gemm(grad_output, b, false, true);
                } else if (transA && !transB) {
                    // C = A^T * B
                    // grad_A = B * grad_C^T
                    grad_a_contrib = gemm(b, grad_output, false, true);
                } else if (!transA && transB) {
                    // C = A * B^T
                    // grad_A = grad_C * B
                    grad_a_contrib = gemm(grad_output, b, false, false);
                } else {
                    // C = A^T * B^T
                    // grad_A = B^T * grad_C^T
                    grad_a_contrib = gemm(b, grad_output, true, true);
                }
                // We need to cast away constness if a is const, but here a is a copy.
                // The error might be because 'a' is captured from a const reference?
                // Let's try to force it to be non-const.
                Tensor& a_ref = const_cast<Tensor&>(a);
                a_ref.grad() = ops::add(a_ref.grad(), grad_a_contrib);
            }

            if (b.requires_grad()) {
                Tensor grad_b_contrib = empty({0}, b.dtype(), b.device()); // Dummy init
                if (!transA && !transB) {
                    // C = A * B
                    // grad_B = A^T * grad_C
                    grad_b_contrib = gemm(a, grad_output, true, false);
                } else if (transA && !transB) {
                    // C = A^T * B
                    // grad_B = A * grad_C
                    grad_b_contrib = gemm(a, grad_output, false, false);
                } else if (!transA && transB) {
                    // C = A * B^T
                    // grad_B = grad_C^T * A
                    grad_b_contrib = gemm(grad_output, a, true, false);
                } else {
                    // C = A^T * B^T
                    // grad_B = grad_C^T * A^T
                    grad_b_contrib = gemm(grad_output, a, true, true);
                }
                Tensor& b_ref = const_cast<Tensor&>(b);
                b_ref.grad() = ops::add(b_ref.grad(), grad_b_contrib);
            }
        };
        
        c.grad_node = node;
    }

    return c;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    return gemm(a, b, false, false);
}

} // namespace vesper::ops
