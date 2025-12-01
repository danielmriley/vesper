#include <vesper/nn/linear.h>
#include <vesper/core/factories.h>
#include <vesper/core/slice.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h> // For bias addition
#include <vesper/nn/init.h> // New include
#include <cmath>

namespace vesper::nn {

// If the header has default arguments, we only implement the full constructor.
// Assuming header has: Linear(..., bool use_bias = true, Device device = Device::CPU);

Linear::Linear(int64_t in_features, int64_t out_features, bool use_bias, Device device)
    : use_bias_(use_bias) {
    
    // 1. Initialize weight tensor with Kaiming initialization
    // requires_grad = true
    weight = empty({out_features, in_features}, DType::Float32, device, true);
    init::kaiming_uniform_(weight, std::sqrt(5.0f)); // a=sqrt(5) is PyTorch default for Linear
    
    // Type-safe member-pointer registration
    register_parameter<Linear>("weight", &Linear::weight);

    if (use_bias_) {
        // 2. Initialize bias tensor to zeros
        // PyTorch initializes bias with uniform(-bound, bound) where bound = 1/sqrt(fan_in)
        // requires_grad = true
        bias = empty({out_features}, DType::Float32, device, true);
        
        // Fan-in calculation
        int64_t fan_in = in_features;
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        init::uniform_(bias, -bound, bound);
        
        // Type-safe member-pointer registration
        register_parameter<Linear>("bias", &Linear::bias);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // ... (rest unchanged) ...
    auto output = ops::matmul(input, weight.transpose(0, 1));

    if (use_bias_) {
        output = ops::add(output, bias);
    }
    return output;
}

// =============================================================================
// Fused QKV Linear Implementation
// =============================================================================

FusedQKVLinear::FusedQKVLinear(int64_t in_features, int64_t q_features, int64_t kv_features, 
                               bool use_bias, Device device)
    : in_features_(in_features), q_features_(q_features), kv_features_(kv_features), use_bias_(use_bias) {
    
    // Total output features = Q + K + V
    int64_t total_features = q_features + 2 * kv_features;
    
    // Initialize weight: [total_features, in_features] (transposed storage for matmul)
    weight = empty({total_features, in_features}, DType::Float32, device, true);
    init::kaiming_uniform_(weight, std::sqrt(5.0f));
    
    register_parameter<FusedQKVLinear>("weight", &FusedQKVLinear::weight);
    
    if (use_bias_) {
        bias = empty({total_features}, DType::Float32, device, true);
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
        init::uniform_(bias, -bound, bound);
        register_parameter<FusedQKVLinear>("bias", &FusedQKVLinear::bias);
    }
}

Tensor FusedQKVLinear::forward(const Tensor& input) {
    // input: [*, in_features] -> output: [*, total_features]
    auto output = ops::matmul(input, weight.transpose(0, 1));
    
    if (use_bias_) {
        output = ops::add(output, bias);
    }
    return output;
}

std::tuple<Tensor, Tensor, Tensor> FusedQKVLinear::forward_split(const Tensor& input) {
    // Compute fused QKV
    Tensor qkv = forward(input);  // [B, T, q_features + 2*kv_features]
    
    // Split into Q, K, V using narrow on last dimension
    // Q: [0, q_features)
    // K: [q_features, q_features + kv_features)
    // V: [q_features + kv_features, q_features + 2*kv_features)
    
    int64_t last_dim = qkv.ndim() - 1;
    
    // narrow(dim, start, length) - slices on specific dimension
    Tensor q = qkv.narrow(last_dim, 0, q_features_);
    Tensor k = qkv.narrow(last_dim, q_features_, kv_features_);
    Tensor v = qkv.narrow(last_dim, q_features_ + kv_features_, kv_features_);
    
    // Make contiguous for subsequent operations
    return std::make_tuple(q.contiguous(), k.contiguous(), v.contiguous());
}

} // namespace vesper::nn
