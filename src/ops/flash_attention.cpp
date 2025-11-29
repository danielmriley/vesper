#include <vesper/ops/flash_attention.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <vesper/autograd/guard.h>
#include <vesper/ops/elementwise.h>
#include <stdexcept>
#include <cmath>

namespace vesper::ops {

Tensor flash_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                       float scale, bool is_causal) {
    // Validate inputs
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
        throw std::runtime_error("flash_attention: inputs must be 4D [B, H, N, D]");
    }
    
    if (q.shape() != k.shape() || k.shape() != v.shape()) {
        throw std::runtime_error("flash_attention: Q, K, V must have same shape");
    }
    
    if (q.dtype() != k.dtype() || k.dtype() != v.dtype()) {
        throw std::runtime_error("flash_attention: Q, K, V must have same dtype");
    }
    
    int64_t B = q.shape()[0];
    int64_t H = q.shape()[1];
    int64_t N = q.shape()[2];
    int64_t D = q.shape()[3];
    
    Device device = q.device();
    DType dtype = q.dtype();
    
    bool requires_grad = (q.requires_grad() || k.requires_grad() || v.requires_grad()) 
                         && autograd::grad_mode_enabled;
    
    // Allocate output and log-sum-exp buffer
    Tensor output = vesper::empty({B, H, N, D}, dtype, device, requires_grad);
    Tensor lse = vesper::empty({B, H, N}, DType::Float32, device, false);  // LSE always FP32
    
    // Dispatch to backend
    switch (device) {
        case Device::CPU:
            flash_attention_cpu_dispatch(q, k, v, output, lse, scale, is_causal);
            break;
        case Device::HIP:
#ifdef USE_HIP_BACKEND
            flash_attention_hip_dispatch(q, k, v, output, lse, scale, is_causal);
#else
            throw std::runtime_error("flash_attention: HIP backend not enabled");
#endif
            break;
        case Device::CUDA:
#ifdef USE_CUDA_BACKEND
            // For now, fall back to CPU
            flash_attention_cpu_dispatch(q.to(Device::CPU), k.to(Device::CPU), 
                                        v.to(Device::CPU), output, lse, scale, is_causal);
#else
            throw std::runtime_error("flash_attention: CUDA backend not enabled");
#endif
            break;
        default:
            throw std::runtime_error("flash_attention: unsupported device");
    }
    
    // Setup autograd
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        
        // Track dependencies
        if (q.grad_node) node->next_edges.push_back({q.grad_node});
        if (k.grad_node) node->next_edges.push_back({k.grad_node});
        if (v.grad_node) node->next_edges.push_back({v.grad_node});
        
        // Capture tensors for backward
        node->backward_fn = [q=q, k=k, v=v, output_saved=output, lse_saved=lse, 
                            scale, is_causal, 
                            result_weak=output.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            
            Tensor grad_output = result->grad();
            
            auto [grad_q, grad_k, grad_v] = flash_attention_backward(
                grad_output, q, k, v, output_saved, lse_saved, scale, is_causal);
            
            if (q.requires_grad()) {
                q.accumulate_grad(grad_q);
            }
            if (k.requires_grad()) {
                k.accumulate_grad(grad_k);
            }
            if (v.requires_grad()) {
                v.accumulate_grad(grad_v);
            }
        };
        
        output.grad_node = node;
    }
    
    return output;
}

std::tuple<Tensor, Tensor, Tensor> flash_attention_backward(
    const Tensor& grad_output, const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& output, const Tensor& lse, float scale, bool is_causal) {
    
    int64_t B = q.shape()[0];
    int64_t H = q.shape()[1];
    int64_t N = q.shape()[2];
    int64_t D = q.shape()[3];
    
    Device device = q.device();
    DType dtype = q.dtype();
    
    // Allocate gradient tensors
    Tensor grad_q = vesper::zeros({B, H, N, D}, dtype, device);
    Tensor grad_k = vesper::zeros({B, H, N, D}, dtype, device);
    Tensor grad_v = vesper::zeros({B, H, N, D}, dtype, device);
    
    // Dispatch to backend
    switch (device) {
        case Device::CPU:
            flash_attention_backward_cpu_dispatch(
                grad_output, q, k, v, output, lse,
                grad_q, grad_k, grad_v, scale, is_causal);
            break;
        case Device::HIP:
#ifdef USE_HIP_BACKEND
            flash_attention_backward_hip_dispatch(
                grad_output, q, k, v, output, lse,
                grad_q, grad_k, grad_v, scale, is_causal);
#else
            throw std::runtime_error("flash_attention_backward: HIP backend not enabled");
#endif
            break;
        default:
            throw std::runtime_error("flash_attention_backward: unsupported device");
    }
    
    return {grad_q, grad_k, grad_v};
}

// CPU reference implementation (for testing and fallback)
void flash_attention_cpu_dispatch(
    const Tensor& q, const Tensor& k, const Tensor& v,
    Tensor& output, Tensor& lse,
    float scale, bool is_causal) {
    
    // Ensure contiguous
    Tensor q_c = q.is_contiguous() ? q : q.contiguous();
    Tensor k_c = k.is_contiguous() ? k : k.contiguous();
    Tensor v_c = v.is_contiguous() ? v : v.contiguous();
    
    const float* Q = q_c.data_ptr<float>();
    const float* K = k_c.data_ptr<float>();
    const float* V = v_c.data_ptr<float>();
    float* O = output.data_ptr<float>();
    float* L = lse.data_ptr<float>();
    
    int64_t B = q.shape()[0];
    int64_t H = q.shape()[1];
    int64_t N = q.shape()[2];
    int64_t D = q.shape()[3];
    
    // Tile sizes for better cache utilization
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    
    // Process each batch and head
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            const float* q_ptr = Q + (b * H + h) * N * D;
            const float* k_ptr = K + (b * H + h) * N * D;
            const float* v_ptr = V + (b * H + h) * N * D;
            float* o_ptr = O + (b * H + h) * N * D;
            float* l_ptr = L + (b * H + h) * N;
            
            // For each query position
            for (int64_t q_start = 0; q_start < N; q_start += BLOCK_M) {
                int64_t q_end = std::min(q_start + BLOCK_M, N);
                
                // Initialize output accumulator and running stats
                std::vector<float> acc(BLOCK_M * D, 0.0f);
                std::vector<float> m_prev(BLOCK_M, -std::numeric_limits<float>::infinity());
                std::vector<float> l_prev(BLOCK_M, 0.0f);
                
                // Determine K/V range based on causality
                int64_t k_end_limit = is_causal ? std::min(q_end, N) : N;
                
                // Iterate over K/V blocks
                for (int64_t k_start = 0; k_start < k_end_limit; k_start += BLOCK_N) {
                    int64_t k_end = std::min(k_start + BLOCK_N, k_end_limit);
                    
                    // Compute attention scores for this block
                    for (int64_t qi = q_start; qi < q_end; ++qi) {
                        int qi_local = qi - q_start;
                        
                        // Compute scores and find max
                        std::vector<float> scores(k_end - k_start);
                        float m_block = -std::numeric_limits<float>::infinity();
                        
                        for (int64_t ki = k_start; ki < k_end; ++ki) {
                            // Causal mask
                            if (is_causal && ki > qi) {
                                scores[ki - k_start] = -std::numeric_limits<float>::infinity();
                                continue;
                            }
                            
                            // Dot product Q[qi] · K[ki]
                            float score = 0.0f;
                            for (int64_t d = 0; d < D; ++d) {
                                score += q_ptr[qi * D + d] * k_ptr[ki * D + d];
                            }
                            score *= scale;
                            scores[ki - k_start] = score;
                            m_block = std::max(m_block, score);
                        }
                        
                        // Online softmax update
                        float m_new = std::max(m_prev[qi_local], m_block);
                        
                        // Compute softmax weights and sum
                        float l_block = 0.0f;
                        for (size_t i = 0; i < scores.size(); ++i) {
                            scores[i] = std::exp(scores[i] - m_new);
                            l_block += scores[i];
                        }
                        
                        // Rescale previous accumulator
                        float scale_prev = std::exp(m_prev[qi_local] - m_new);
                        l_prev[qi_local] = l_prev[qi_local] * scale_prev + l_block;
                        
                        // Update accumulator
                        for (int64_t d = 0; d < D; ++d) {
                            acc[qi_local * D + d] *= scale_prev;
                            for (int64_t ki = k_start; ki < k_end; ++ki) {
                                acc[qi_local * D + d] += scores[ki - k_start] * v_ptr[ki * D + d];
                            }
                        }
                        
                        m_prev[qi_local] = m_new;
                    }
                }
                
                // Write output and LSE
                for (int64_t qi = q_start; qi < q_end; ++qi) {
                    int qi_local = qi - q_start;
                    float inv_l = 1.0f / l_prev[qi_local];
                    
                    for (int64_t d = 0; d < D; ++d) {
                        o_ptr[qi * D + d] = acc[qi_local * D + d] * inv_l;
                    }
                    
                    // Log-sum-exp for backward
                    l_ptr[qi] = m_prev[qi_local] + std::log(l_prev[qi_local]);
                }
            }
        }
    }
}

void flash_attention_backward_cpu_dispatch(
    const Tensor& grad_output, const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& output, const Tensor& lse,
    Tensor& grad_q, Tensor& grad_k, Tensor& grad_v,
    float scale, bool is_causal) {
    
    // Ensure contiguous
    Tensor dO_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor q_c = q.is_contiguous() ? q : q.contiguous();
    Tensor k_c = k.is_contiguous() ? k : k.contiguous();
    Tensor v_c = v.is_contiguous() ? v : v.contiguous();
    Tensor o_c = output.is_contiguous() ? output : output.contiguous();
    Tensor l_c = lse.is_contiguous() ? lse : lse.contiguous();
    
    const float* dO = dO_c.data_ptr<float>();
    const float* Q = q_c.data_ptr<float>();
    const float* K = k_c.data_ptr<float>();
    const float* V = v_c.data_ptr<float>();
    const float* O = o_c.data_ptr<float>();
    const float* L = l_c.data_ptr<float>();
    
    float* dQ = grad_q.data_ptr<float>();
    float* dK = grad_k.data_ptr<float>();
    float* dV = grad_v.data_ptr<float>();
    
    int64_t B = q.shape()[0];
    int64_t H = q.shape()[1];
    int64_t N = q.shape()[2];
    int64_t D = q.shape()[3];
    
    // Process each batch and head
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            size_t bh_offset = (b * H + h) * N * D;
            size_t bh_l_offset = (b * H + h) * N;
            
            const float* q_ptr = Q + bh_offset;
            const float* k_ptr = K + bh_offset;
            const float* v_ptr = V + bh_offset;
            const float* o_ptr = O + bh_offset;
            const float* l_ptr = L + bh_l_offset;
            const float* do_ptr = dO + bh_offset;
            
            float* dq_ptr = dQ + bh_offset;
            float* dk_ptr = dK + bh_offset;
            float* dv_ptr = dV + bh_offset;
            
            // Compute D = rowsum(dO * O) for softmax backward
            std::vector<float> Di(N);
            for (int64_t i = 0; i < N; ++i) {
                float sum = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    sum += do_ptr[i * D + d] * o_ptr[i * D + d];
                }
                Di[i] = sum;
            }
            
            // For each query position
            for (int64_t qi = 0; qi < N; ++qi) {
                float li = l_ptr[qi];
                
                int64_t k_end = is_causal ? qi + 1 : N;
                
                for (int64_t ki = 0; ki < k_end; ++ki) {
                    // Recompute attention score
                    float score = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        score += q_ptr[qi * D + d] * k_ptr[ki * D + d];
                    }
                    score *= scale;
                    
                    // Attention weight
                    float p = std::exp(score - li);
                    
                    // dV += P^T @ dO
                    for (int64_t d = 0; d < D; ++d) {
                        dv_ptr[ki * D + d] += p * do_ptr[qi * D + d];
                    }
                    
                    // dP = dO @ V^T
                    float dp = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        dp += do_ptr[qi * D + d] * v_ptr[ki * D + d];
                    }
                    
                    // dS = P * (dP - Di) for softmax backward
                    float ds = p * (dp - Di[qi]);
                    
                    // dQ += scale * dS @ K
                    // dK += scale * dS @ Q
                    for (int64_t d = 0; d < D; ++d) {
                        dq_ptr[qi * D + d] += scale * ds * k_ptr[ki * D + d];
                        dk_ptr[ki * D + d] += scale * ds * q_ptr[qi * D + d];
                    }
                }
            }
        }
    }
}

} // namespace vesper::ops
