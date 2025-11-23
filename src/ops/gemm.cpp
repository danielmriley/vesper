#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <vesper/ops/elementwise.h>
#include <stdexcept>

namespace vesper::ops {

void gemm_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);
void gemm_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);
void gemm_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);

Tensor gemm(const Tensor& a, const Tensor& b, bool transA, bool transB) {
    // --- 1. Pre-condition Checks ---
    if (a.device() != b.device()) {
        throw std::runtime_error("GEMM requires tensors to be on the same device.");
    }
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("GEMM currently only supports 2D tensors.");
    }
    
    // Support non-contiguous tensors via strided kernels!
    
    int64_t M = transA ? a.shape()[1] : a.shape()[0];
    int64_t K_a = transA ? a.shape()[0] : a.shape()[1];
    
    int64_t K_b = transB ? b.shape()[1] : b.shape()[0];
    int64_t N = transB ? b.shape()[0] : b.shape()[1];

    if (K_a != K_b) {
        throw std::runtime_error("Inner dimensions of matrices do not match for GEMM.");
    }
    int64_t K = K_a;

    // --- 2. Prepare Output Tensor ---
    bool requires_grad = a.requires_grad() || b.requires_grad();
    Tensor c = empty({M, N}, a.dtype(), a.device(), requires_grad);

    // --- 3. Dispatch to Backend-Specific Implementation ---
    switch (a.device()) {
        case Device::HIP:
#if USE_HIP_BACKEND
            gemm_hip_dispatch(a, b, c, transA, transB);
#else
            throw std::runtime_error("HIP backend not enabled during build, but required for GEMM.");
#endif
            break;
        
        case Device::CPU:
            gemm_cpu_dispatch(a, b, c, transA, transB);
            break;

        case Device::CUDA:
#if USE_CUDA_BACKEND
            gemm_cuda_dispatch(a, b, c, transA, transB);
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
        
        if (a.requires_grad() && a.grad_node) {
            node->next_edges.push_back({a.grad_node});
        }
        if (b.requires_grad() && b.grad_node) {
            node->next_edges.push_back({b.grad_node});
        }

        node->backward_fn = [a, b, c, transA, transB]() mutable {
            Tensor& grad_output = c.grad();
            
            if (a.requires_grad()) {
                Tensor grad_a_contrib = empty({0}, a.dtype(), a.device()); // Dummy init
                if (!transA && !transB) {
                    // C = A * B -> dA = dC * B^T
                    grad_a_contrib = gemm(grad_output, b, false, true);
                } else if (transA && !transB) {
                    // C = A^T * B -> dA = B * dC^T (transpose of dC*B^T ?)
                    // d(A^T)/dA is transpose.
                    // dL/dA = (dL/dC * B^T)^T = B * dL/dC^T
                    grad_a_contrib = gemm(b, grad_output, false, true);
                } else if (!transA && transB) {
                    // C = A * B^T -> dA = dC * B
                    grad_a_contrib = gemm(grad_output, b, false, false);
                } else {
                    // C = A^T * B^T -> dA = (B^T)^T * dC^T = B * dC^T
                    grad_a_contrib = gemm(b, grad_output, true, true);
                }
                Tensor& a_ref = const_cast<Tensor&>(a);
                a_ref.accumulate_grad(grad_a_contrib);
            }

            if (b.requires_grad()) {
                Tensor grad_b_contrib = empty({0}, b.dtype(), b.device());
                if (!transA && !transB) {
                    // C = A * B -> dB = A^T * dC
                    grad_b_contrib = gemm(a, grad_output, true, false);
                } else if (transA && !transB) {
                    // C = A^T * B -> dB = A * dC
                    grad_b_contrib = gemm(a, grad_output, false, false);
                } else if (!transA && transB) {
                    // C = A * B^T -> dB = (A^T * dC)^T = dC^T * A
                    grad_b_contrib = gemm(grad_output, a, true, false);
                } else {
                    // C = A^T * B^T -> dB = dC^T * A^T
                    grad_b_contrib = gemm(grad_output, a, true, true);
                }
                Tensor& b_ref = const_cast<Tensor&>(b);
                b_ref.accumulate_grad(grad_b_contrib);
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
