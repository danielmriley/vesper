#include <vesper/ops/gemm.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <vesper/autograd/guard.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <stdexcept>

namespace vesper::ops {

void gemm_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);
void gemm_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);
void gemm_cpu_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB);

void gemm_batch_hip_dispatch(const Tensor& a, const Tensor& b, Tensor& c, 
                             int64_t batch_count, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                             bool transA, bool transB);
void gemm_batch_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& c, 
                              int64_t batch_count, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                              bool transA, bool transB);

Tensor gemm(const Tensor& a, const Tensor& b, bool transA, bool transB) {
    // --- 1. Pre-condition Checks ---
    if (a.device() != b.device()) {
        throw std::runtime_error("GEMM requires tensors to be on the same device.");
    }
    
    int a_rank = a.ndim();
    int b_rank = b.ndim();

    if (a_rank < 2 || b_rank < 2) {
        throw std::runtime_error("GEMM requires tensors to be at least 2D.");
    }

    // --- 2. Handle Batch Dimensions ---
    bool is_batched = (a_rank > 2 || b_rank > 2);
    
    int64_t M = transA ? a.shape()[a_rank-1] : a.shape()[a_rank-2];
    int64_t K_a = transA ? a.shape()[a_rank-2] : a.shape()[a_rank-1];
    
    int64_t K_b = transB ? b.shape()[b_rank-1] : b.shape()[b_rank-2];
    int64_t N = transB ? b.shape()[b_rank-2] : b.shape()[b_rank-1];

    if (K_a != K_b) {
        throw std::runtime_error("Inner dimensions of matrices do not match for GEMM.");
    }
    int64_t K = K_a;

    // --- Fold batch into M for activation x weight linears (broadcast-B) ---
    // When A is N-D (rank >= 3), B is a plain 2D weight broadcast across the batch,
    // and A is not transposed, the leading batch dims of A fold into the M dimension.
    // This routes Linear/QKV/MLP/LM-head through the fast vectorized 2D GEMM instead
    // of the single-buffered batched kernel. It stays autograd-correct because the
    // reshapes and the 2D gemm each build their own backward: grad flows
    //   dC[.., M, N] -> reshape -> dC2[BM, N] -> gemm2d -> {dA2[BM, K], dB[K, N]}
    //   -> reshape -> dA[.., M, K]. dB already sums over all BM rows (== the batched
    // path's post-hoc batch reduction), so gradients match. transA must be false:
    // with transA the last two dims are [K, M] and the batch cannot merge with M via
    // a contiguous reshape, so those stay on the batched path below.
    if (a_rank > 2 && b_rank == 2 && !transA) {
        int64_t fold = 1;
        for (int i = 0; i < a_rank - 2; ++i) {
            fold *= a.shape()[i];
        }
        int64_t M2 = fold * M;

        Tensor a2 = a.reshape({M2, K});          // autograd-aware view (copy if non-contiguous)
        Tensor c2 = gemm(a2, b, false, transB);  // fast 2D path, builds its own autograd

        std::vector<int64_t> out_shape;
        out_shape.reserve(a_rank);
        for (int i = 0; i < a_rank - 2; ++i) {
            out_shape.push_back(a.shape()[i]);
        }
        out_shape.push_back(M);
        out_shape.push_back(N);
        return c2.reshape(out_shape);            // autograd-aware
    }

    // Calculate Batch Count and Output Shape
    std::vector<int64_t> c_shape;
    int64_t batch_count = 1;
    int64_t stride_a_batch = 0;
    int64_t stride_b_batch = 0;
    int64_t stride_c_batch = 0;

    if (is_batched) {
        if (a_rank == b_rank) {
            // Check batch dims match and compute batch strides for broadcasting
            // For 4D: (B_a, H_a, M, K) @ (B_b, H_b, K, N) -> (max(B_a,B_b), max(H_a,H_b), M, N)
            // We flatten batch dims: batch_count = B * H
            // For A with shape (1, H), iterating over b from 0..B*H:
            //   - b % H gives the H index (varies)
            //   - b / H gives the B index (should be 0 always due to broadcasting)
            // This requires modular arithmetic, which is complex with a single stride.
            // 
            // Simpler approach for 4D: handle separately with nested loops in CPU code.
            // For GPU dispatch, we compute proper strides for the kernel.
            
            bool has_broadcast_a = false;
            bool has_broadcast_b = false;
            
            for (int i = 0; i < a_rank - 2; ++i) {
                int64_t dim_a = a.shape()[i];
                int64_t dim_b = b.shape()[i];

                if (dim_a != dim_b) {
                    if (dim_a == 1 || dim_b == 1) {
                        // Broadcasting allowed
                        if (dim_a == 1) has_broadcast_a = true;
                        if (dim_b == 1) has_broadcast_b = true;
                    } else {
                         throw std::runtime_error("Batch dimensions must match (or be 1 for broadcasting).");
                    }
                }
                c_shape.push_back(std::max(dim_a, dim_b));
                batch_count *= std::max(dim_a, dim_b);
            }
            
            // Compute strides for batch dimensions
            if (a_rank == 3) {
                // For 3D: if dim is 1, use stride 0 for broadcasting
                stride_a_batch = (a.shape()[0] == 1) ? 0 : a.strides()[0];
                stride_b_batch = (b.shape()[0] == 1) ? 0 : b.strides()[0];
            } else {
                // Rank > 3 (typically 4D)
                // For 4D tensors with broadcasting, we need to be careful.
                // If A is (1, H, M, K) and B is (B, H, K, N), we can't use a single stride.
                // The GPU kernels handle this via blockIdx.z, but for CPU we need special handling.
                // 
                // Compute a "combined batch stride" that works when we can flatten.
                // This only works correctly when one or both are NOT broadcasting.
                
                // Compute product of batch dims for A and B
                int64_t a_batch_total = 1;
                int64_t b_batch_total = 1;
                for (int i = 0; i < a_rank - 2; ++i) {
                    a_batch_total *= a.shape()[i];
                    b_batch_total *= b.shape()[i];
                }
                
                // Simple stride: stride for the last batch dimension (before matrix dims)
                // This works for contiguous tensors where batch dims can be flattened
                if (!has_broadcast_a) {
                    stride_a_batch = a.strides()[a_rank-3];
                } else {
                    // Broadcasting: if first dim is 1, we wrap within the second dim only
                    // For (1, H, M, K): stride should be M*K (stride for H), but we also need
                    // to mod by H when computing the offset. This is handled in the CPU loop.
                    stride_a_batch = (a.shape()[a_rank-3] == 1) ? 0 : a.strides()[a_rank-3];
                }
                
                if (!has_broadcast_b) {
                    stride_b_batch = b.strides()[b_rank-3];
                } else {
                    stride_b_batch = (b.shape()[b_rank-3] == 1) ? 0 : b.strides()[b_rank-3];
                }
            }
            
        } else if (a_rank > b_rank && b_rank == 2) {
            // Broadcast B
            for (int i = 0; i < a_rank - 2; ++i) {
                c_shape.push_back(a.shape()[i]);
                batch_count *= a.shape()[i];
            }
            stride_a_batch = a.strides()[a_rank-3];
            stride_b_batch = 0; // Broadcast B
            stride_c_batch = M * N;
        } else if (b_rank > a_rank && a_rank == 2) {
            // Broadcast A
            for (int i = 0; i < b_rank - 2; ++i) {
                c_shape.push_back(b.shape()[i]);
                batch_count *= b.shape()[i];
            }
            stride_a_batch = 0; // Broadcast A
            stride_b_batch = b.strides()[b_rank-3];
            stride_c_batch = M * N;
        } else {
             throw std::runtime_error("Unsupported rank combination for Batch GEMM.");
        }
    }

    c_shape.push_back(M);
    c_shape.push_back(N);

    // --- 3. Prepare Output Tensor ---
    bool requires_grad = (a.requires_grad() || b.requires_grad()) && autograd::grad_mode_enabled;
    Tensor c = empty(c_shape, a.dtype(), a.device(), requires_grad);
    
    // Fix stride_c_batch if we just created it
    if (is_batched) {
        stride_c_batch = c.strides()[c.ndim()-3];
    }

    // --- 4. Dispatch ---
    if (!is_batched) {
        // Standard 2D GEMM
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
    } else {
        // Batch GEMM
        switch (a.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                gemm_batch_hip_dispatch(a, b, c, batch_count, stride_a_batch, stride_b_batch, stride_c_batch, transA, transB);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                // Fallback to loop for CPU
                {
                    // For 4D tensors with broadcasting, we need careful offset calculation
                    // Example: A=(1,H,M,K), B=(B,H,K,N) -> C=(B,H,M,N)
                    // batch_count = B*H, and we iterate i from 0 to B*H-1
                    // For A: offset = (i % a_inner_batch) * stride_a_inner
                    // For B: offset = i * stride_b_batch (or similar for B broadcasting)
                    
                    int64_t stride_a_0 = a.strides()[a_rank-2];
                    int64_t stride_a_1 = a.strides()[a_rank-1];
                    int64_t stride_b_0 = b.strides()[b_rank-2];
                    int64_t stride_b_1 = b.strides()[b_rank-1];
                    int64_t stride_c_0 = c.strides()[c.ndim()-2];
                    int64_t stride_c_1 = c.strides()[c.ndim()-1];

                    // Compute batch sizes for proper indexing with broadcasting
                    // For 4D: a = (B_a, H_a, M, K), b = (B_b, H_b, K, N)
                    // output = (max(B_a,B_b), max(H_a,H_b), M, N)
                    
                    // Compute the number of "inner" batches (product of dims after first batch dim)
                    int64_t a_batch_outer = (a_rank > 3) ? a.shape()[0] : 1;
                    int64_t a_batch_inner = (a_rank > 3) ? a.shape()[a_rank-3] : ((a_rank == 3) ? a.shape()[0] : 1);
                    int64_t b_batch_outer = (b_rank > 3) ? b.shape()[0] : 1;
                    int64_t b_batch_inner = (b_rank > 3) ? b.shape()[b_rank-3] : ((b_rank == 3) ? b.shape()[0] : 1);
                    
                    int64_t c_batch_outer = (c.ndim() > 3) ? c.shape()[0] : 1;
                    int64_t c_batch_inner = (c.ndim() > 3) ? c.shape()[c.ndim()-3] : ((c.ndim() == 3) ? c.shape()[0] : 1);
                    
                    // Strides for outer and inner batch dimensions
                    int64_t stride_a_outer = (a_rank > 3) ? a.strides()[0] : 0;
                    int64_t stride_a_inner = (a_rank > 3) ? a.strides()[a_rank-3] : ((a_rank == 3) ? a.strides()[0] : 0);
                    int64_t stride_b_outer = (b_rank > 3) ? b.strides()[0] : 0;
                    int64_t stride_b_inner = (b_rank > 3) ? b.strides()[b_rank-3] : ((b_rank == 3) ? b.strides()[0] : 0);
                    int64_t stride_c_outer = (c.ndim() > 3) ? c.strides()[0] : 0;
                    int64_t stride_c_inner = (c.ndim() > 3) ? c.strides()[c.ndim()-3] : ((c.ndim() == 3) ? c.strides()[0] : 0);

                    for (int64_t i = 0; i < batch_count; ++i) {
                        // Compute outer and inner batch indices
                        int64_t outer_idx = i / c_batch_inner;
                        int64_t inner_idx = i % c_batch_inner;
                        
                        // Apply broadcasting: if a dimension is 1, clamp index to 0
                        int64_t a_outer = (a_batch_outer == 1) ? 0 : outer_idx;
                        int64_t a_inner = (a_batch_inner == 1) ? 0 : inner_idx;
                        int64_t b_outer = (b_batch_outer == 1) ? 0 : outer_idx;
                        int64_t b_inner = (b_batch_inner == 1) ? 0 : inner_idx;
                        
                        const float* ptr_a = a.data_ptr<float>() + a_outer * stride_a_outer + a_inner * stride_a_inner;
                        const float* ptr_b = b.data_ptr<float>() + b_outer * stride_b_outer + b_inner * stride_b_inner;
                        float* ptr_c = c.data_ptr<float>() + outer_idx * stride_c_outer + inner_idx * stride_c_inner;
                        
                        for (int64_t m = 0; m < M; ++m) {
                            for (int64_t n = 0; n < N; ++n) {
                                float sum = 0.0f;
                                for (int64_t k = 0; k < K; ++k) {
                                    float val_a;
                                    if (transA) {
                                        // A is (K, M) logically, so we access A[k, m]
                                        val_a = ptr_a[k * stride_a_0 + m * stride_a_1];
                                    } else {
                                        // A is (M, K) logically, so we access A[m, k]
                                        val_a = ptr_a[m * stride_a_0 + k * stride_a_1];
                                    }
                                    
                                    float val_b;
                                    if (transB) {
                                        // B is (N, K) logically, so we access B[n, k]
                                        val_b = ptr_b[n * stride_b_0 + k * stride_b_1];
                                    } else {
                                        // B is (K, N) logically, so we access B[k, n]
                                        val_b = ptr_b[k * stride_b_0 + n * stride_b_1];
                                    }
                                    
                                    sum += val_a * val_b;
                                }
                                ptr_c[m * stride_c_0 + n * stride_c_1] = sum;
                            }
                        }
                    }
                }
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                gemm_batch_cuda_dispatch(a, b, c, batch_count, stride_a_batch, stride_b_batch, stride_c_batch, transA, transB);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default: throw std::runtime_error("Unsupported device.");
        }
    }

    // --- 5. Autograd ---
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        
        if (a.requires_grad() && a.grad_node) {
            node->next_edges.push_back({a.grad_node});
        }
        if (b.requires_grad() && b.grad_node) {
            node->next_edges.push_back({b.grad_node});
        }

        node->backward_fn = [a, b, weak_c=c.weak(), transA, transB]() mutable {
            auto c_ptr = weak_c.lock();
            if (!c_ptr) return;
            Tensor& grad_output = c_ptr->grad();
            
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
                
                if (grad_a_contrib.ndim() > a.ndim()) {
                     int diff = grad_a_contrib.ndim() - a.ndim();
                     for (int i = 0; i < diff; ++i) {
                         grad_a_contrib = ops::sum(grad_a_contrib, 0, false); 
                     }
                }
                
                // Handle case where ranks match but dimensions are 1 (broadcasting)
                if (grad_a_contrib.ndim() == a.ndim()) {
                    for (int i = 0; i < a.ndim(); ++i) {
                        if (a.shape()[i] == 1 && grad_a_contrib.shape()[i] != 1) {
                            grad_a_contrib = ops::sum(grad_a_contrib, i, true);
                        }
                    }
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

                if (grad_b_contrib.ndim() > b.ndim()) {
                     int diff = grad_b_contrib.ndim() - b.ndim();
                     for (int i = 0; i < diff; ++i) {
                         grad_b_contrib = ops::sum(grad_b_contrib, 0, false);
                     }
                }

                // Handle case where ranks match but dimensions are 1 (broadcasting)
                if (grad_b_contrib.ndim() == b.ndim()) {
                    for (int i = 0; i < b.ndim(); ++i) {
                        if (b.shape()[i] == 1 && grad_b_contrib.shape()[i] != 1) {
                            grad_b_contrib = ops::sum(grad_b_contrib, i, true);
                        }
                    }
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
