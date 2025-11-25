```markdown
# Chapter 45: INT4 and GPTQ Quantization

## 1. Introduction

While INT8 provides 4x compression, modern LLM serving often requires even more aggressive quantization. **INT4 quantization** achieves:

- **8x memory reduction** (4 bits vs 32 bits)
- **Fits larger models in VRAM** (70B parameters in 24GB)
- **Faster memory-bound inference**

This chapter covers:
1. **Weight-only INT4**: Quantize weights, keep activations in FP16
2. **GPTQ**: Post-training quantization with optimal rounding
3. **AWQ**: Activation-aware weight quantization
4. **INT4 GEMM kernels**: Efficient dequantization during matmul

## 2. Weight-Only Quantization

For LLMs, weights dominate memory. Activations are small in comparison (batch_size × seq_len × dim). Strategy: **INT4 weights, FP16 activations**.

### 2.1 INT4 Representation

```cpp
// include/vesper/quant/int4.h

namespace vesper::quant {

// INT4 is packed: 2 values per byte
// Layout: [low_nibble | high_nibble]
struct Int4Packed {
    uint8_t data;
    
    int8_t low() const { return (data & 0x0F) - 8; }   // Range [-8, 7]
    int8_t high() const { return (data >> 4) - 8; }
    
    void set_low(int8_t v) { data = (data & 0xF0) | ((v + 8) & 0x0F); }
    void set_high(int8_t v) { data = (data & 0x0F) | (((v + 8) & 0x0F) << 4); }
};

struct Int4TensorInfo {
    std::vector<int64_t> shape;      // Logical shape
    std::vector<int64_t> packed_shape;  // Actual storage shape (last dim / 2)
    int group_size;                  // For group-wise quantization
    bool symmetric;
};

class Int4Tensor {
public:
    Int4Tensor(Tensor packed_data, Tensor scales, 
               std::optional<Tensor> zeros, Int4TensorInfo info);
    
    // Dequantize to FP16
    Tensor dequantize() const;
    
    // Access packed data
    const Tensor& packed_data() const { return packed_data_; }
    const Tensor& scales() const { return scales_; }
    const std::optional<Tensor>& zeros() const { return zeros_; }
    
    // Logical shape
    std::vector<int64_t> shape() const { return info_.shape; }
    
private:
    Tensor packed_data_;   // uint8, shape [..., N/2]
    Tensor scales_;        // FP16, shape [..., N/group_size]
    std::optional<Tensor> zeros_;  // INT4 packed, same shape as scales
    Int4TensorInfo info_;
};

} // namespace vesper::quant
```

### 2.2 Group-wise Quantization

For better accuracy, use separate scales for each group of 128 weights:

```
Weights:     [w0, w1, ..., w127, w128, w129, ..., w255, ...]
              |--- group 0 ---|  |---- group 1 ----|
              scale_0            scale_1
```

```cpp
// Quantize with group size
Int4Tensor quantize_int4(const Tensor& weights, int group_size = 128) {
    VESPER_CHECK(weights.size(-1) % group_size == 0, 
                "Last dimension must be divisible by group_size");
    
    auto shape = weights.shape();
    int64_t K = shape.back();
    int64_t num_groups = K / group_size;
    
    // Reshape for group-wise processing
    // [out_features, in_features] -> [out_features, num_groups, group_size]
    std::vector<int64_t> grouped_shape = shape;
    grouped_shape.pop_back();
    grouped_shape.push_back(num_groups);
    grouped_shape.push_back(group_size);
    
    Tensor grouped = weights.view(grouped_shape);
    
    // Compute scales per group (symmetric quantization)
    Tensor amax = grouped.abs().amax(-1, /*keepdim=*/true);  // Max per group
    Tensor scales = amax / 7.0f;  // INT4 symmetric range: [-8, 7]
    scales = scales.squeeze(-1);
    
    // Quantize
    Tensor scaled = grouped / scales.unsqueeze(-1);
    Tensor quantized = scaled.round().clamp(-8, 7).to(DType::Int8);
    
    // Pack pairs into bytes
    quantized = quantized.view(shape);
    Tensor packed = pack_int4(quantized);
    
    Int4TensorInfo info{shape, packed.shape(), group_size, true};
    return Int4Tensor(packed, scales.to(DType::Float16), std::nullopt, info);
}
```

## 3. GPTQ: Optimal Brain Quantization

GPTQ uses second-order information (Hessian) to find optimal rounding decisions.

### 3.1 The Algorithm

For each column of weights:
1. Compute **Hessian** H = X^T X (where X is calibration data)
2. For each weight w_ij:
   - Quantize: q_ij = round(w_ij / scale)
   - Compute error: δ = w_ij - dequantize(q_ij)
   - **Compensate** remaining weights: W[:, j+1:] -= δ * H^{-1}[j, j+1:] / H^{-1}[j,j]

This redistributes quantization error optimally.

### 3.2 Implementation

```cpp
// include/vesper/quant/gptq.h

namespace vesper::quant {

struct GPTQConfig {
    int group_size = 128;
    int block_size = 128;      // Process columns in blocks
    float dampening = 0.01f;   // Hessian regularization
    bool symmetric = true;
    int n_bits = 4;
    Device device = Device::HIP;
};

class GPTQQuantizer {
public:
    GPTQQuantizer(const GPTQConfig& config) : config_(config) {}
    
    // Quantize a linear layer given calibration inputs
    Int4Tensor quantize_layer(
        const Tensor& weight,           // [out_features, in_features]
        const std::vector<Tensor>& calibration_inputs  // List of [batch, in_features]
    );
    
private:
    GPTQConfig config_;
    
    // Compute Hessian from calibration data
    Tensor compute_hessian(const std::vector<Tensor>& inputs);
    
    // GPTQ column-wise quantization
    void quantize_block(
        Tensor& W,           // Weights (modified in-place)
        Tensor& Q,           // Quantized output
        const Tensor& H,     // Hessian
        int start_col,
        int end_col);
};

} // namespace vesper::quant
```

```cpp
// src/quant/gptq.cpp

Int4Tensor GPTQQuantizer::quantize_layer(
    const Tensor& weight,
    const std::vector<Tensor>& calibration_inputs) 
{
    int64_t out_features = weight.size(0);
    int64_t in_features = weight.size(1);
    
    // Clone weight for modification
    Tensor W = weight.clone().to(config_.device);
    
    // Compute Hessian: H = sum(X^T X) for all calibration batches
    Tensor H = compute_hessian(calibration_inputs);
    
    // Add damping for numerical stability
    float damp = config_.dampening * H.diag().mean().item<float>();
    H = H + damp * eye(in_features, H.dtype(), H.device());
    
    // Cholesky decomposition for efficient H^{-1} computation
    Tensor L = linalg_cholesky(H);
    Tensor Hinv = cholesky_inverse(L);
    
    // Prepare output
    Tensor Q = zeros({out_features, in_features}, DType::Int8, config_.device);
    Tensor scales = zeros({out_features, in_features / config_.group_size}, 
                          DType::Float16, config_.device);
    
    // Process in blocks
    for (int col = 0; col < in_features; col += config_.block_size) {
        int end_col = std::min(col + config_.block_size, (int)in_features);
        quantize_block(W, Q, scales, Hinv, col, end_col);
    }
    
    // Pack to INT4
    Tensor packed = pack_int4(Q);
    
    Int4TensorInfo info{weight.shape(), packed.shape(), config_.group_size, config_.symmetric};
    return Int4Tensor(packed, scales, std::nullopt, info);
}

void GPTQQuantizer::quantize_block(
    Tensor& W, Tensor& Q, Tensor& scales, const Tensor& Hinv,
    int start_col, int end_col) 
{
    int64_t out_features = W.size(0);
    
    for (int col = start_col; col < end_col; ++col) {
        // Get group index for this column
        int group_idx = col / config_.group_size;
        
        // Compute scale for this group (if at group boundary)
        if (col % config_.group_size == 0) {
            int group_end = std::min(col + config_.group_size, (int)W.size(1));
            Tensor group_weights = W.slice(1, col, group_end);
            Tensor amax = group_weights.abs().amax();
            scales.slice(1, group_idx, group_idx + 1) = amax / 7.0f;
        }
        
        Tensor scale = scales.select(1, group_idx);
        
        // Quantize column
        Tensor w_col = W.select(1, col);
        Tensor q_col = (w_col / scale).round().clamp(-8, 7);
        Q.select(1, col).copy_(q_col);
        
        // Compute quantization error
        Tensor error = w_col - q_col * scale;
        
        // Compensate remaining columns: W[:, col+1:] -= error * H^{-1}[col, col+1:] / H^{-1}[col, col]
        if (col + 1 < end_col) {
            Tensor h_diag = Hinv[col][col];
            Tensor h_row = Hinv.select(0, col).slice(0, col + 1, end_col);
            
            Tensor compensation = error.unsqueeze(1) * h_row.unsqueeze(0) / h_diag;
            W.slice(1, col + 1, end_col) -= compensation;
        }
    }
}

Tensor GPTQQuantizer::compute_hessian(const std::vector<Tensor>& inputs) {
    int64_t in_features = inputs[0].size(-1);
    Tensor H = zeros({in_features, in_features}, DType::Float32, config_.device);
    
    for (const auto& X : inputs) {
        // X: [batch, seq_len, in_features] -> reshape to [batch*seq, in_features]
        Tensor X_2d = X.view({-1, in_features}).to(config_.device);
        // H += X^T @ X
        H += matmul(X_2d.t(), X_2d);
    }
    
    // Normalize by number of samples
    int64_t n_samples = 0;
    for (const auto& X : inputs) {
        n_samples += X.size(0) * (X.dim() > 2 ? X.size(1) : 1);
    }
    H /= n_samples;
    
    return H;
}
```

## 4. AWQ: Activation-Aware Weight Quantization

AWQ protects "salient" weights (those multiplied by large activations).

### 4.1 Key Insight

If activation channel i has large values, weight column i is important. Scale it up before quantization, then scale down during inference.

```cpp
// include/vesper/quant/awq.h

namespace vesper::quant {

struct AWQConfig {
    int group_size = 128;
    int n_bits = 4;
    float alpha = 0.5f;  // Balance between activation and weight scaling
    Device device = Device::HIP;
};

class AWQQuantizer {
public:
    AWQQuantizer(const AWQConfig& config) : config_(config) {}
    
    // Quantize with activation awareness
    std::pair<Int4Tensor, Tensor> quantize_layer(
        const Tensor& weight,
        const std::vector<Tensor>& calibration_inputs
    );
    
private:
    AWQConfig config_;
    
    // Compute per-channel importance
    Tensor compute_channel_importance(const std::vector<Tensor>& inputs);
    
    // Search for optimal scaling factors
    Tensor search_scales(const Tensor& weight, const Tensor& importance);
};

} // namespace vesper::quant
```

```cpp
// src/quant/awq.cpp

Tensor AWQQuantizer::compute_channel_importance(const std::vector<Tensor>& inputs) {
    int64_t in_features = inputs[0].size(-1);
    Tensor importance = zeros({in_features}, DType::Float32, config_.device);
    
    for (const auto& X : inputs) {
        Tensor X_flat = X.view({-1, in_features}).abs();
        importance += X_flat.mean(0);
    }
    
    importance /= inputs.size();
    return importance;
}

Tensor AWQQuantizer::search_scales(const Tensor& weight, const Tensor& importance) {
    // AWQ uses importance to find optimal per-channel scales
    // Higher importance channels get larger scales (protected from quantization)
    
    Tensor w_max = weight.abs().amax(0);  // Max per input channel
    
    // Scale = (importance / importance.max()) ^ alpha
    Tensor norm_importance = importance / importance.max();
    Tensor scales = norm_importance.pow(config_.alpha);
    
    // Clamp scales to prevent extreme values
    scales = scales.clamp(1e-4f, 1.0f);
    
    return scales;
}

std::pair<Int4Tensor, Tensor> AWQQuantizer::quantize_layer(
    const Tensor& weight,
    const std::vector<Tensor>& calibration_inputs) 
{
    // Compute channel importance from activations
    Tensor importance = compute_channel_importance(calibration_inputs);
    
    // Search for optimal scales
    Tensor channel_scales = search_scales(weight, importance);
    
    // Scale weights: W_scaled = W * scales (per input channel)
    Tensor W_scaled = weight * channel_scales.unsqueeze(0);
    
    // Quantize scaled weights
    Int4Tensor qweight = quantize_int4(W_scaled, config_.group_size);
    
    // Return quantized weights and scales (for dequant during inference)
    return {qweight, channel_scales};
}
```

## 5. INT4 GEMM Kernels

The challenge: Dequantize INT4 weights during GEMM efficiently.

### 5.1 Kernel Design

```cpp
// src/ops/hip/gemm_int4.hip

// Dequantize-fused GEMM
// C = A (FP16) @ dequant(B_int4) (FP16)

__global__ void gemm_int4_dequant_kernel(
    const half* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ B_packed, // [K/2, N] (INT4 packed)
    const half* __restrict__ scales,      // [K/group_size, N]
    half* __restrict__ C,                 // [M, N]
    int M, int K, int N,
    int group_size)
{
    // Each block computes a TILE_M x TILE_N tile of C
    const int TILE_M = 64;
    const int TILE_N = 64;
    const int TILE_K = 32;
    
    __shared__ half As[TILE_M][TILE_K];
    __shared__ half Bs[TILE_K][TILE_N];
    
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;
    
    float acc = 0.0f;
    
    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        // Load A tile
        if (row < M && k_start + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + k_start + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        
        // Load and dequantize B tile
        int k = k_start + threadIdx.y;
        if (k < K && col < N) {
            int packed_k = k / 2;
            uint8_t packed = B_packed[packed_k * N + col];
            
            // Extract INT4 value
            int4_t value;
            if (k % 2 == 0) {
                value = (packed & 0x0F) - 8;  // Low nibble
            } else {
                value = (packed >> 4) - 8;    // High nibble
            }
            
            // Get scale for this group
            int group_idx = k / group_size;
            half scale = scales[group_idx * N + col];
            
            // Dequantize
            Bs[threadIdx.y][threadIdx.x] = __float2half(
                static_cast<float>(value) * __half2float(scale));
        } else {
            Bs[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += __half2float(As[threadIdx.y][k]) * 
                   __half2float(Bs[k][threadIdx.x]);
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}
```

### 5.2 Vectorized Dequantization

Load 8 INT4 values (32 bits) at once:

```cpp
__device__ __forceinline__ void dequant_int4x8(
    uint32_t packed,    // 8 INT4 values in 32 bits
    half scale,
    half* out)          // Output 8 FP16 values
{
    float s = __half2float(scale);
    
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int4_t val = ((packed >> (i * 4)) & 0xF) - 8;
        out[i] = __float2half(val * s);
    }
}

// Vectorized kernel
__global__ void gemm_int4_vec_kernel(
    const half* __restrict__ A,
    const uint32_t* __restrict__ B_packed,  // 8 INT4 per uint32
    const half* __restrict__ scales,
    half* __restrict__ C,
    int M, int K, int N, int group_size)
{
    // Load 8 INT4 values at a time
    // ...
}
```

## 6. Using Quantized Models

```cpp
// include/vesper/quant/quantized_model.h

namespace vesper::quant {

class QuantizedLinear : public nn::Module {
public:
    QuantizedLinear(Int4Tensor weight, std::optional<Tensor> bias,
                    std::optional<Tensor> awq_scales = std::nullopt)
        : weight_(std::move(weight)), 
          bias_(std::move(bias)),
          awq_scales_(std::move(awq_scales)) {}
    
    Tensor forward(const Tensor& x) override {
        // Dequantize-fused GEMM
        Tensor out = matmul_int4(x, weight_.packed_data(), 
                                  weight_.scales(), weight_.info().group_size);
        
        // Apply AWQ scales if present
        if (awq_scales_) {
            out = out / *awq_scales_;
        }
        
        if (bias_) {
            out = out + *bias_;
        }
        
        return out;
    }
    
    static QuantizedLinear from_float(
        const nn::Linear& linear,
        const GPTQConfig& config,
        const std::vector<Tensor>& calibration_data);
    
private:
    Int4Tensor weight_;
    std::optional<Tensor> bias_;
    std::optional<Tensor> awq_scales_;
};

// Quantize entire model
void quantize_model_gptq(
    models::Transformer& model,
    DataLoader& calibration_loader,
    const GPTQConfig& config);

void quantize_model_awq(
    models::Transformer& model,
    DataLoader& calibration_loader,
    const AWQConfig& config);

} // namespace vesper::quant
```

### 6.1 Full Model Quantization

```cpp
void quantize_model_gptq(
    models::Transformer& model,
    DataLoader& calibration_loader,
    const GPTQConfig& config) 
{
    GPTQQuantizer quantizer(config);
    
    // Collect calibration data
    std::vector<Tensor> calib_inputs;
    for (auto& [x, y] : calibration_loader) {
        if (calib_inputs.size() >= 128) break;  // Typically 128 samples
        calib_inputs.push_back(x);
    }
    
    // Quantize each linear layer
    for (auto& [name, module] : model.named_modules()) {
        if (auto* linear = dynamic_cast<nn::Linear*>(module)) {
            std::cout << "Quantizing " << name << "..." << std::endl;
            
            // Get activations for this layer
            std::vector<Tensor> layer_inputs = capture_activations(
                model, name, calib_inputs);
            
            // Quantize
            Int4Tensor qweight = quantizer.quantize_layer(
                linear->weight(), layer_inputs);
            
            // Replace with quantized version
            auto qlinear = std::make_shared<QuantizedLinear>(
                std::move(qweight), linear->bias());
            
            model.replace_module(name, qlinear);
        }
    }
}
```

## 7. Testing Strategy

### 7.1 Unit Tests

```cpp
// tests/quant/test_int4.cpp

TEST(Int4, PackUnpack) {
    // Test packing 2 INT4 values per byte
    Tensor vals = tensor({-8, 7, -1, 0, 3, -5, 6, 2}, DType::Int8);
    Tensor packed = pack_int4(vals);
    
    EXPECT_EQ(packed.numel(), 4);  // 8 values -> 4 bytes
    
    Tensor unpacked = unpack_int4(packed);
    EXPECT_TRUE(allclose(vals, unpacked));
}

TEST(Int4, Quantize) {
    Tensor weights = randn({64, 128});
    
    Int4Tensor qw = quantize_int4(weights, /*group_size=*/32);
    
    EXPECT_EQ(qw.shape(), std::vector<int64_t>({64, 128}));
    EXPECT_EQ(qw.packed_data().size(-1), 64);  // 128/2
    EXPECT_EQ(qw.scales().size(-1), 4);  // 128/32 groups
}

TEST(Int4, Dequantize) {
    Tensor weights = randn({64, 128});
    
    Int4Tensor qw = quantize_int4(weights, 32);
    Tensor recon = qw.dequantize();
    
    EXPECT_EQ(recon.shape(), weights.shape());
    
    // Check reconstruction error (INT4 has larger error than INT8)
    float mse = ((weights - recon).pow(2)).mean().item<float>();
    float rmse = std::sqrt(mse);
    EXPECT_LT(rmse, 0.1f);  // Allow larger error for INT4
}

TEST(Int4, GemmInt4) {
    int M = 32, K = 128, N = 64;
    
    Tensor A = randn({M, K}, DType::Float16, Device::HIP);
    Tensor B = randn({K, N}, DType::Float32);
    
    // Reference
    Tensor C_ref = matmul(A.to(DType::Float32), B).to(DType::Float16);
    
    // Quantize B to INT4
    Int4Tensor qB = quantize_int4(B, 32);
    qB = Int4Tensor(qB.packed_data().to(Device::HIP),
                    qB.scales().to(Device::HIP),
                    std::nullopt, qB.info());
    
    // INT4 GEMM
    Tensor C_int4 = matmul_int4(A, qB.packed_data(), qB.scales(), 32);
    
    // Check error
    float rel_error = (C_ref - C_int4).abs().mean().item<float>() /
                      C_ref.abs().mean().item<float>();
    
    EXPECT_LT(rel_error, 0.1f);  // <10% relative error
}
```

### 7.2 GPTQ Tests

```cpp
TEST(GPTQ, BasicQuantization) {
    // Small layer for testing
    Tensor weight = randn({128, 256});
    
    // Generate calibration data
    std::vector<Tensor> calib;
    for (int i = 0; i < 32; ++i) {
        calib.push_back(randn({8, 256}));
    }
    
    GPTQConfig config;
    config.group_size = 64;
    config.block_size = 64;
    config.device = Device::CPU;
    
    GPTQQuantizer quantizer(config);
    Int4Tensor qweight = quantizer.quantize_layer(weight, calib);
    
    EXPECT_EQ(qweight.shape(), weight.shape());
}

TEST(GPTQ, AccuracyVsBasicInt4) {
    Tensor weight = randn({256, 512});
    std::vector<Tensor> calib;
    for (int i = 0; i < 64; ++i) {
        calib.push_back(randn({16, 512}));
    }
    
    // Basic INT4 quantization
    Int4Tensor basic_qw = quantize_int4(weight, 128);
    Tensor basic_recon = basic_qw.dequantize();
    float basic_mse = ((weight - basic_recon).pow(2)).mean().item<float>();
    
    // GPTQ quantization
    GPTQConfig config;
    GPTQQuantizer quantizer(config);
    Int4Tensor gptq_qw = quantizer.quantize_layer(weight, calib);
    Tensor gptq_recon = gptq_qw.dequantize();
    float gptq_mse = ((weight - gptq_recon).pow(2)).mean().item<float>();
    
    std::cout << "Basic INT4 MSE: " << basic_mse << std::endl;
    std::cout << "GPTQ INT4 MSE: " << gptq_mse << std::endl;
    
    // GPTQ should have lower error
    EXPECT_LT(gptq_mse, basic_mse);
}
```

### 7.3 AWQ Tests

```cpp
TEST(AWQ, ChannelScaling) {
    Tensor weight = randn({128, 256});
    
    // Create inputs with varying channel importance
    std::vector<Tensor> calib;
    for (int i = 0; i < 32; ++i) {
        Tensor x = randn({8, 256});
        // Make some channels more important
        x.select(1, 0) *= 10.0f;  // Channel 0 is 10x more active
        calib.push_back(x);
    }
    
    AWQConfig config;
    AWQQuantizer quantizer(config);
    
    auto [qweight, scales] = quantizer.quantize_layer(weight, calib);
    
    // Channel 0 should have larger scale (more protected)
    EXPECT_GT(scales[0].item<float>(), scales[128].item<float>());
}
```

### 7.4 Stress Tests

```cpp
TEST(Int4, StressTest_LargeGEMM) {
    int M = 1024, K = 4096, N = 4096;
    
    Tensor A = randn({M, K}, DType::Float16, Device::HIP);
    Tensor B = randn({K, N}, DType::Float32);
    
    Int4Tensor qB = quantize_int4(B, 128);
    qB = Int4Tensor(qB.packed_data().to(Device::HIP),
                    qB.scales().to(Device::HIP),
                    std::nullopt, qB.info());
    
    // Warmup
    Tensor C = matmul_int4(A, qB.packed_data(), qB.scales(), 128);
    hip_sync();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        C = matmul_int4(A, qB.packed_data(), qB.scales(), 128);
    }
    hip_sync();
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double tflops = 2.0 * M * K * N * 100 / (ms * 1e9);
    
    std::cout << "INT4 GEMM: " << tflops << " TFLOPS" << std::endl;
    std::cout << "Time per GEMM: " << (ms / 100) << " ms" << std::endl;
}

TEST(Int4, StressTest_MemoryReduction) {
    // Llama2-70B has roughly:
    // - 70B params = 280GB in FP32, 140GB in FP16, 35GB in INT4
    
    int64_t llama70b_params = 70'000'000'000LL;
    
    size_t fp32_bytes = llama70b_params * 4;
    size_t fp16_bytes = llama70b_params * 2;
    size_t int4_bytes = llama70b_params / 2;  // 4 bits = 0.5 bytes
    int4_bytes += (llama70b_params / 128) * 2;  // Scales (FP16, one per 128)
    
    std::cout << "Llama2-70B memory:" << std::endl;
    std::cout << "  FP32: " << (fp32_bytes / 1e9) << " GB" << std::endl;
    std::cout << "  FP16: " << (fp16_bytes / 1e9) << " GB" << std::endl;
    std::cout << "  INT4: " << (int4_bytes / 1e9) << " GB" << std::endl;
    std::cout << "  Reduction: " << (float(fp16_bytes) / int4_bytes) << "x vs FP16" << std::endl;
    
    // Verify we can fit in 24GB VRAM
    EXPECT_LT(int4_bytes, 24'000'000'000LL);
}

TEST(GPTQ, StressTest_FullLayerQuantization) {
    // Simulate Llama layer sizes
    int hidden_size = 4096;
    int ffn_size = 11008;
    
    // Generate calibration data
    std::vector<Tensor> calib;
    for (int i = 0; i < 128; ++i) {
        calib.push_back(randn({1, 512, hidden_size}));
    }
    
    GPTQConfig config;
    config.device = Device::HIP;
    GPTQQuantizer quantizer(config);
    
    // Q, K, V, O projections
    for (const auto& [name, shape] : {
        std::make_pair("q_proj", std::vector<int64_t>{hidden_size, hidden_size}),
        std::make_pair("k_proj", std::vector<int64_t>{hidden_size, hidden_size}),
        std::make_pair("v_proj", std::vector<int64_t>{hidden_size, hidden_size}),
        std::make_pair("o_proj", std::vector<int64_t>{hidden_size, hidden_size}),
        std::make_pair("gate_proj", std::vector<int64_t>{ffn_size, hidden_size}),
        std::make_pair("up_proj", std::vector<int64_t>{ffn_size, hidden_size}),
        std::make_pair("down_proj", std::vector<int64_t>{hidden_size, ffn_size})
    }) {
        Tensor weight = randn(shape);
        
        auto start = std::chrono::high_resolution_clock::now();
        Int4Tensor qw = quantizer.quantize_layer(weight, calib);
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << name << ": " << ms << " ms" << std::endl;
    }
}
```

## 8. Perplexity Evaluation

Critical: Verify quantized model accuracy on held-out text.

```cpp
float evaluate_perplexity(
    QuantizedModel& model,
    DataLoader& eval_loader,
    int max_samples = 128) 
{
    model.eval();
    
    float total_loss = 0;
    int64_t total_tokens = 0;
    
    with_no_grad([&]() {
        int samples = 0;
        for (auto& [input_ids, labels] : eval_loader) {
            if (samples >= max_samples) break;
            
            Tensor logits = model.forward(input_ids);
            
            // Compute cross-entropy loss
            Tensor loss = cross_entropy_loss(
                logits.view({-1, logits.size(-1)}),
                labels.view({-1}),
                /*reduction=*/"sum");
            
            total_loss += loss.item<float>();
            total_tokens += labels.numel();
            samples++;
        }
    });
    
    float avg_loss = total_loss / total_tokens;
    float perplexity = std::exp(avg_loss);
    
    return perplexity;
}

TEST(Int4, AccuracyVsFloat) {
    // Load a small test model
    auto config = models::TransformerConfig::gpt2_small();
    auto float_model = std::make_unique<models::Transformer>(config);
    
    // Generate synthetic eval data
    auto eval_loader = create_synthetic_loader(100, 128);
    
    float float_ppl = evaluate_perplexity(*float_model, eval_loader);
    
    // Quantize
    auto quant_model = quantize_model_gptq(*float_model, eval_loader, GPTQConfig());
    float quant_ppl = evaluate_perplexity(quant_model, eval_loader);
    
    std::cout << "Float PPL: " << float_ppl << std::endl;
    std::cout << "INT4 PPL: " << quant_ppl << std::endl;
    std::cout << "Degradation: " << ((quant_ppl / float_ppl - 1) * 100) << "%" << std::endl;
    
    // Perplexity increase should be less than 5%
    EXPECT_LT(quant_ppl / float_ppl, 1.05f);
}
```

## 9. Summary

This chapter covered:

1. **INT4 representation**: Packed storage, group-wise scales
2. **GPTQ**: Optimal rounding using Hessian information
3. **AWQ**: Protecting important weight channels
4. **INT4 GEMM kernels**: Fused dequantization during matmul
5. **Model quantization**: End-to-end workflow

Key takeaways:
- **8x compression** with INT4 vs FP32
- **GPTQ/AWQ** minimize accuracy loss
- **Group-wise quantization** (128 weights per scale) balances accuracy and overhead
- **Weight-only quantization** works well for LLMs (memory-bound inference)

With INT4 quantization, Vesper can run 70B+ parameter models on consumer GPUs.
```
