#pragma once

#include <vesper/nn/module.h>
#include <tuple>

namespace vesper::nn {

class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool use_bias = true, Device device = Device::CPU);

    Tensor forward(const Tensor& input) override;

    // The parameters of the layer, owned by the class
    Tensor weight;
    Tensor bias;
private:
    bool use_bias_;
};

/**
 * @brief Fused QKV Linear projection for efficient attention
 * 
 * Instead of 3 separate Linear layers for Q, K, V, this fuses them into
 * a single GEMM: input @ [in_features, q_features + k_features + v_features]
 * 
 * This improves GEMM efficiency by having larger output dimensions.
 * For example: [1024, 1024] @ [1024, 3072] instead of 3x [1024, 1024] @ [1024, 1024]
 */
class FusedQKVLinear : public Module {
public:
    /**
     * @param in_features Input dimension (embed_dim)
     * @param q_features Q output dimension (num_heads * head_dim)
     * @param kv_features K/V output dimension (num_kv_heads * head_dim)
     * @param use_bias Whether to use bias
     * @param device Device to allocate on
     */
    FusedQKVLinear(int64_t in_features, int64_t q_features, int64_t kv_features, 
                   bool use_bias = false, Device device = Device::CPU);
    
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Forward pass returning separate Q, K, V tensors
     * @param input Input tensor [B, T, in_features]
     * @return Tuple of (Q, K, V) tensors
     */
    std::tuple<Tensor, Tensor, Tensor> forward_split(const Tensor& input);
    
    // Fused weight: [in_features, q_features + 2*kv_features]
    Tensor weight;
    Tensor bias;
    
private:
    int64_t in_features_;
    int64_t q_features_;
    int64_t kv_features_;
    bool use_bias_;
};

} // namespace vesper::nn
