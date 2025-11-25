```markdown
# Chapter 44: INT8 Quantization

## 1. Introduction

Large language models require substantial memory and compute. **Quantization** reduces these requirements by using lower-precision data types. INT8 quantization stores weights and activations as 8-bit integers instead of 32-bit floats, achieving:

- **4x memory reduction** (8 bits vs 32 bits)
- **2-4x speedup** on hardware with INT8 tensor cores
- **Minimal accuracy loss** when done correctly

This chapter covers:
1. **Quantization fundamentals**: Scale, zero-point, and symmetric vs asymmetric
2. **Dynamic quantization**: Quantize on-the-fly during inference
3. **Static quantization**: Calibrate with representative data
4. **INT8 GEMM kernels**: Optimized matrix multiply for INT8

## 2. Quantization Fundamentals

### 2.1 Linear Quantization

Converting a float value to INT8:

$$q = \text{round}\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}$$

Converting back:

$$x' = \text{scale} \times (q - \text{zero\_point})$$

### 2.2 Symmetric vs Asymmetric

**Symmetric Quantization** (zero_point = 0):
- Range: [-127, 127]
- Simpler computation
- Works well for weights (usually centered around 0)

**Asymmetric Quantization**:
- Range: [0, 255]
- Full range utilization
- Better for activations (ReLU output is non-negative)

### 2.3 Per-Tensor vs Per-Channel

**Per-Tensor**: One scale for entire tensor
- Simpler
- Less accurate

**Per-Channel**: One scale per output channel
- More complex
- Better accuracy (4x less quantization error)

## 3. Quantization Data Types

```cpp
// include/vesper/quant/types.h

namespace vesper::quant {

enum class QuantScheme {
    Symmetric,      // [-127, 127], zero_point = 0
    Asymmetric,     // [0, 255], arbitrary zero_point
    PerTensor,      // One scale for entire tensor
    PerChannel,     // One scale per channel
};

struct QuantParams {
    float scale;
    int32_t zero_point = 0;
    QuantScheme scheme = QuantScheme::Symmetric;
    int axis = -1;  // For per-channel: which axis
    
    // For per-channel quantization
    std::vector<float> scales;
    std::vector<int32_t> zero_points;
};

// Quantized tensor: stores INT8 data + quantization params
class QuantizedTensor {
public:
    QuantizedTensor(Tensor int8_data, QuantParams params);
    
    // Dequantize to float
    Tensor dequantize() const;
    
    // Access raw INT8 data
    const Tensor& data() const { return data_; }
    const QuantParams& params() const { return params_; }
    
    // Shape and device
    std::vector<int64_t> shape() const { return data_.shape(); }
    Device device() const { return data_.device(); }
    
private:
    Tensor data_;  // INT8 tensor
    QuantParams params_;
};

} // namespace vesper::quant
```

## 4. Quantization Functions

### 4.1 CPU Implementation

```cpp
// src/quant/quantize.cpp

namespace vesper::quant {

// Find optimal scale for symmetric quantization
float compute_scale_symmetric(const Tensor& x) {
    float max_val = std::abs(x.max().item<float>());
    float min_val = std::abs(x.min().item<float>());
    float amax = std::max(max_val, min_val);
    return amax / 127.0f;
}

// Find optimal scale and zero_point for asymmetric quantization
std::pair<float, int32_t> compute_scale_asymmetric(const Tensor& x) {
    float max_val = x.max().item<float>();
    float min_val = x.min().item<float>();
    
    float scale = (max_val - min_val) / 255.0f;
    int32_t zero_point = static_cast<int32_t>(std::round(-min_val / scale));
    zero_point = std::clamp(zero_point, 0, 255);
    
    return {scale, zero_point};
}

// Quantize float tensor to INT8
QuantizedTensor quantize(const Tensor& x, QuantScheme scheme) {
    QuantParams params;
    params.scheme = scheme;
    
    Tensor q;
    
    if (scheme == QuantScheme::Symmetric) {
        params.scale = compute_scale_symmetric(x);
        params.zero_point = 0;
        
        // q = round(x / scale), clamped to [-127, 127]
        q = (x / params.scale).round().clamp(-127, 127).to(DType::Int8);
    } else {
        auto [scale, zp] = compute_scale_asymmetric(x);
        params.scale = scale;
        params.zero_point = zp;
        
        // q = round(x / scale) + zero_point, clamped to [0, 255]
        q = ((x / scale) + zp).round().clamp(0, 255).to(DType::UInt8);
    }
    
    return QuantizedTensor(q, params);
}

// Dequantize INT8 tensor to float
Tensor dequantize(const QuantizedTensor& qx) {
    Tensor x = qx.data().to(DType::Float32);
    
    if (qx.params().scheme == QuantScheme::Symmetric) {
        return x * qx.params().scale;
    } else {
        return (x - qx.params().zero_point) * qx.params().scale;
    }
}

// Per-channel quantization for weights
QuantizedTensor quantize_per_channel(const Tensor& weights, int axis) {
    VESPER_CHECK(axis >= 0 && axis < weights.dim(), 
                "Invalid axis for per-channel quantization");
    
    int64_t n_channels = weights.size(axis);
    
    QuantParams params;
    params.scheme = QuantScheme::PerChannel;
    params.axis = axis;
    params.scales.resize(n_channels);
    params.zero_points.resize(n_channels);
    
    // Compute scale per channel
    for (int64_t c = 0; c < n_channels; ++c) {
        Tensor channel = weights.select(axis, c);
        float amax = channel.abs().max().item<float>();
        params.scales[c] = amax / 127.0f;
        params.zero_points[c] = 0;  // Symmetric
    }
    
    // Quantize
    Tensor q = empty_like(weights, DType::Int8);
    
    for (int64_t c = 0; c < n_channels; ++c) {
        Tensor src = weights.select(axis, c);
        Tensor dst = q.select(axis, c);
        dst.copy_((src / params.scales[c]).round().clamp(-127, 127));
    }
    
    return QuantizedTensor(q, params);
}

} // namespace vesper::quant
```

### 4.2 GPU Kernels

```cpp
// src/ops/hip/quantize_kernels.hip

__global__ void quantize_symmetric_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale_out,
    int64_t size)
{
    // First pass: find max absolute value (reduction)
    __shared__ float shared_max[256];
    
    float thread_max = 0.0f;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < size; 
         i += blockDim.x * gridDim.x) {
        thread_max = fmaxf(thread_max, fabsf(input[i]));
    }
    
    // Block reduction
    shared_max[threadIdx.x] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = fmaxf(
                shared_max[threadIdx.x], 
                shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    // First thread writes scale
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *scale_out = shared_max[0] / 127.0f;
    }
    
    __syncthreads();
    float scale = shared_max[0] / 127.0f;
    float inv_scale = 1.0f / scale;
    
    // Second pass: quantize
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < size; 
         i += blockDim.x * gridDim.x) {
        float val = input[i] * inv_scale;
        val = fminf(fmaxf(val, -127.0f), 127.0f);
        output[i] = static_cast<int8_t>(rintf(val));
    }
}

__global__ void dequantize_kernel(
    const int8_t* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int64_t size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<float>(input[idx]) * scale;
    }
}
```

## 5. INT8 GEMM (Quantized Matrix Multiply)

The key operation: `C_f32 = (A_i8 * B_i8) * (scale_a * scale_b)`

### 5.1 Naive Implementation

```cpp
__global__ void gemm_int8_naive(
    const int8_t* __restrict__ A,  // [M, K]
    const int8_t* __restrict__ B,  // [K, N]
    float* __restrict__ C,         // [M, N]
    int M, int K, int N,
    float scale_a, float scale_b)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += static_cast<int32_t>(A[row * K + k]) * 
                   static_cast<int32_t>(B[k * N + col]);
        }
        C[row * N + col] = static_cast<float>(sum) * scale_a * scale_b;
    }
}
```

### 5.2 Optimized INT8 GEMM with Shared Memory

```cpp
#define TILE_SIZE 32

__global__ void gemm_int8_tiled(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N,
    float combined_scale)  // scale_a * scale_b precomputed
{
    __shared__ int8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    int32_t sum = 0;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int k_a = t * TILE_SIZE + threadIdx.x;
        int k_b = t * TILE_SIZE + threadIdx.y;
        
        if (row < M && k_a < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + k_a];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (k_b < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[k_b * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += static_cast<int32_t>(As[threadIdx.y][k]) * 
                   static_cast<int32_t>(Bs[k][threadIdx.x]);
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = static_cast<float>(sum) * combined_scale;
    }
}
```

### 5.3 Using Tensor Cores for INT8 (WMMA)

AMD GPUs with Matrix Core Units (MCUs) support INT8:

```cpp
// Using rocWMMA for INT8 GEMM
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

__global__ void gemm_int8_wmma(
    const int8_t* A, const int8_t* B, float* C,
    int M, int K, int N, float scale)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * WMMA_N;
    
    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;
    
    fill_fragment(c_frag, 0);
    
    for (int k = 0; k < K; k += WMMA_K) {
        load_matrix_sync(a_frag, A + warpM * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warpN, N);
        
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Convert to float and apply scale
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_float;
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_float.x[i] = static_cast<float>(c_frag.x[i]) * scale;
    }
    
    store_matrix_sync(C + warpM * N + warpN, c_float, N, mem_row_major);
}
```

## 6. Dynamic Quantization

Quantize weights ahead of time, quantize activations on-the-fly:

```cpp
// include/vesper/quant/dynamic.h

namespace vesper::quant {

class DynamicQuantizedLinear : public nn::Module {
public:
    DynamicQuantizedLinear(int64_t in_features, int64_t out_features, bool bias = true)
        : in_features_(in_features), out_features_(out_features), has_bias_(bias)
    {
        // Weight is stored quantized
        weight_int8_ = register_buffer("weight_int8", 
            zeros({out_features, in_features}, DType::Int8));
        weight_scale_ = register_buffer("weight_scale", 
            zeros({out_features}, DType::Float32));
        
        if (bias) {
            bias_ = register_parameter("bias", zeros({out_features}));
        }
    }
    
    // Quantize from a regular Linear layer
    static DynamicQuantizedLinear from_float(const nn::Linear& linear) {
        DynamicQuantizedLinear ql(linear.in_features(), linear.out_features(), 
                                   linear.has_bias());
        
        // Quantize weights per-channel (per output feature)
        auto qw = quantize_per_channel(linear.weight(), 0);
        ql.weight_int8_.copy_(qw.data());
        
        // Copy scales
        Tensor scales = tensor(qw.params().scales);
        ql.weight_scale_.copy_(scales);
        
        if (linear.has_bias()) {
            ql.bias_.copy_(linear.bias());
        }
        
        return ql;
    }
    
    Tensor forward(const Tensor& x) override {
        // Dynamically quantize input
        auto qx = quantize(x, QuantScheme::Symmetric);
        
        // INT8 GEMM
        Tensor y = matmul_int8(qx.data(), weight_int8_, 
                               qx.params().scale, weight_scale_);
        
        if (has_bias_) {
            y = y + bias_;
        }
        
        return y;
    }
    
private:
    int64_t in_features_, out_features_;
    bool has_bias_;
    Tensor weight_int8_;
    Tensor weight_scale_;
    Tensor bias_;
};

} // namespace vesper::quant
```

## 7. Static Quantization (Calibration)

For best accuracy, calibrate scale factors using representative data:

```cpp
// include/vesper/quant/calibration.h

namespace vesper::quant {

class CalibrationObserver {
public:
    virtual void observe(const Tensor& x) = 0;
    virtual QuantParams compute_params() const = 0;
    virtual void reset() = 0;
};

// MinMax observer: track min/max values
class MinMaxObserver : public CalibrationObserver {
public:
    void observe(const Tensor& x) override {
        float batch_min = x.min().item<float>();
        float batch_max = x.max().item<float>();
        
        min_val_ = std::min(min_val_, batch_min);
        max_val_ = std::max(max_val_, batch_max);
    }
    
    QuantParams compute_params() const override {
        QuantParams params;
        params.scheme = QuantScheme::Asymmetric;
        params.scale = (max_val_ - min_val_) / 255.0f;
        params.zero_point = static_cast<int32_t>(-min_val_ / params.scale);
        return params;
    }
    
    void reset() override {
        min_val_ = std::numeric_limits<float>::max();
        max_val_ = std::numeric_limits<float>::lowest();
    }
    
private:
    float min_val_ = std::numeric_limits<float>::max();
    float max_val_ = std::numeric_limits<float>::lowest();
};

// Histogram observer for better accuracy (percentile clipping)
class HistogramObserver : public CalibrationObserver {
public:
    HistogramObserver(int n_bins = 2048, float percentile = 99.99f)
        : n_bins_(n_bins), percentile_(percentile), 
          histogram_(n_bins, 0) {}
    
    void observe(const Tensor& x) override {
        // Build histogram of absolute values
        Tensor abs_x = x.abs();
        float batch_max = abs_x.max().item<float>();
        max_observed_ = std::max(max_observed_, batch_max);
        
        // Update histogram
        float bin_width = batch_max / n_bins_;
        Tensor bins = (abs_x / bin_width).floor().clamp(0, n_bins_ - 1).to(DType::Int64);
        
        // Count in each bin
        for (int64_t i = 0; i < bins.numel(); ++i) {
            int bin_idx = bins.data_ptr<int64_t>()[i];
            histogram_[bin_idx]++;
        }
        
        total_count_ += x.numel();
    }
    
    QuantParams compute_params() const override {
        // Find threshold at percentile
        int64_t target_count = static_cast<int64_t>(total_count_ * percentile_ / 100.0f);
        int64_t cumsum = 0;
        float threshold = max_observed_;
        
        for (int i = 0; i < n_bins_; ++i) {
            cumsum += histogram_[i];
            if (cumsum >= target_count) {
                threshold = (i + 1) * (max_observed_ / n_bins_);
                break;
            }
        }
        
        QuantParams params;
        params.scheme = QuantScheme::Symmetric;
        params.scale = threshold / 127.0f;
        params.zero_point = 0;
        return params;
    }
    
    void reset() override {
        std::fill(histogram_.begin(), histogram_.end(), 0);
        max_observed_ = 0;
        total_count_ = 0;
    }
    
private:
    int n_bins_;
    float percentile_;
    std::vector<int64_t> histogram_;
    float max_observed_ = 0;
    int64_t total_count_ = 0;
};

} // namespace vesper::quant
```

### 7.1 Calibration Workflow

```cpp
// Static quantization example
void calibrate_model(nn::Module& model, DataLoader& calib_loader) {
    // Insert observers
    std::unordered_map<std::string, std::unique_ptr<CalibrationObserver>> observers;
    
    // Register hooks to observe activations
    for (auto& [name, module] : model.named_modules()) {
        if (auto* linear = dynamic_cast<nn::Linear*>(module)) {
            observers[name] = std::make_unique<HistogramObserver>();
            
            linear->register_forward_hook([&, obs_name = name](
                const Tensor& input, const Tensor& output) {
                observers[obs_name]->observe(output);
            });
        }
    }
    
    // Run calibration
    model.eval();
    with_no_grad([&]() {
        for (auto& [x, y] : calib_loader) {
            model.forward(x);
        }
    });
    
    // Compute quantization params
    std::unordered_map<std::string, QuantParams> quant_params;
    for (auto& [name, observer] : observers) {
        quant_params[name] = observer->compute_params();
    }
    
    // Apply quantization
    for (auto& [name, module] : model.named_modules()) {
        if (auto* linear = dynamic_cast<nn::Linear*>(module)) {
            // Replace with quantized version using calibrated scales
            // ... implementation details ...
        }
    }
}
```

## 8. Complete INT8 Inference Engine

```cpp
// include/vesper/quant/int8_model.h

namespace vesper::quant {

class QuantizedModel {
public:
    // Quantize a model
    static QuantizedModel from_float(
        const models::Transformer& model,
        const std::optional<DataLoader>& calibration_data = std::nullopt);
    
    // Forward pass with INT8
    Tensor forward(const Tensor& input);
    
    // Save/load quantized weights
    void save(const std::string& path);
    static QuantizedModel load(const std::string& path);
    
    // Memory comparison
    size_t memory_bytes() const;
    
private:
    // Quantized weight storage
    std::vector<QuantizedTensor> weights_;
    
    // Activation scales (from calibration)
    std::unordered_map<std::string, QuantParams> activation_params_;
    
    // Model config
    models::TransformerConfig config_;
};

} // namespace vesper::quant
```

## 9. Testing Strategy

### 9.1 Unit Tests

```cpp
// tests/quant/test_int8.cpp

TEST(Quantize, SymmetricBasic) {
    Tensor x = tensor({-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
    
    auto qx = quantize(x, QuantScheme::Symmetric);
    
    EXPECT_NEAR(qx.params().scale, 1.0f / 127.0f, 1e-6);
    EXPECT_EQ(qx.params().zero_point, 0);
    
    // Check quantized values
    auto data = qx.data().to(DType::Int32);
    EXPECT_EQ(data[0].item<int32_t>(), -127);
    EXPECT_EQ(data[2].item<int32_t>(), 0);
    EXPECT_EQ(data[4].item<int32_t>(), 127);
}

TEST(Quantize, AsymmetricBasic) {
    Tensor x = tensor({0.0f, 0.5f, 1.0f, 1.5f, 2.0f});
    
    auto qx = quantize(x, QuantScheme::Asymmetric);
    
    // Range [0, 2], scale = 2/255
    EXPECT_NEAR(qx.params().scale, 2.0f / 255.0f, 1e-5);
    
    // zero_point maps 0 -> 0
    EXPECT_EQ(qx.params().zero_point, 0);
}

TEST(Quantize, RoundTrip) {
    Tensor x = randn({100, 100});
    
    auto qx = quantize(x, QuantScheme::Symmetric);
    Tensor x_rec = dequantize(qx);
    
    // Check reconstruction error
    float mse = ((x - x_rec).pow(2)).mean().item<float>();
    float rmse = std::sqrt(mse);
    
    // RMSE should be small (within quantization noise)
    EXPECT_LT(rmse, 0.02f);
}

TEST(Quantize, PerChannel) {
    Tensor weights = randn({64, 128});
    
    // Scale different rows differently
    weights[0] = weights[0] * 10;  // Row 0 has 10x larger values
    
    auto qw = quantize_per_channel(weights, 0);
    
    // Check per-channel scales
    EXPECT_GT(qw.params().scales[0], qw.params().scales[32]);
}

TEST(GemmInt8, Correctness) {
    int M = 64, K = 128, N = 32;
    
    Tensor A = randn({M, K});
    Tensor B = randn({K, N});
    
    // Reference: float matmul
    Tensor C_ref = matmul(A, B);
    
    // Quantized matmul
    auto qA = quantize(A, QuantScheme::Symmetric);
    auto qB = quantize(B, QuantScheme::Symmetric);
    
    Tensor C_q = matmul_int8(qA.data(), qB.data(), 
                             qA.params().scale, qB.params().scale);
    
    // Check relative error
    float rel_error = (C_ref - C_q).abs().mean().item<float>() / 
                      C_ref.abs().mean().item<float>();
    
    EXPECT_LT(rel_error, 0.05f);  // Less than 5% relative error
}
```

### 9.2 Stress Tests

```cpp
TEST(QuantInt8, StressTest_LargeGEMM) {
    int M = 4096, K = 4096, N = 4096;
    
    Tensor A = randn({M, K}, DType::Float32, Device::HIP);
    Tensor B = randn({K, N}, DType::Float32, Device::HIP);
    
    // Warmup
    auto qA = quantize(A, QuantScheme::Symmetric);
    auto qB = quantize(B, QuantScheme::Symmetric);
    Tensor C = matmul_int8(qA.data(), qB.data(), 
                           qA.params().scale, qB.params().scale);
    hip_sync();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        C = matmul_int8(qA.data(), qB.data(),
                        qA.params().scale, qB.params().scale);
    }
    hip_sync();
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // TFLOPS calculation
    double flops = 2.0 * M * K * N * 100;
    double tflops = flops / (ms * 1e9);
    
    std::cout << "INT8 GEMM: " << tflops << " TFLOPS" << std::endl;
    
    // Compare to float GEMM
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        Tensor C_f = matmul(A, B);
    }
    hip_sync();
    end = std::chrono::high_resolution_clock::now();
    double ms_float = std::chrono::duration<double, std::milli>(end - start).count();
    double tflops_float = flops / (ms_float * 1e9);
    
    std::cout << "FP32 GEMM: " << tflops_float << " TFLOPS" << std::endl;
    std::cout << "Speedup: " << (ms_float / ms) << "x" << std::endl;
    
    // INT8 should be at least 1.5x faster
    EXPECT_GT(ms_float / ms, 1.5);
}

TEST(QuantInt8, StressTest_ModelAccuracy) {
    // Create small test model
    auto config = models::TransformerConfig::gpt2_small();
    config.n_layers = 2;  // Reduce for testing
    
    auto float_model = std::make_unique<models::Transformer>(config);
    float_model->to(Device::HIP);
    
    // Generate test inputs
    std::vector<Tensor> test_inputs;
    for (int i = 0; i < 100; ++i) {
        test_inputs.push_back(randint(0, config.vocab_size, {1, 32}, DType::Int64));
    }
    
    // Get float outputs
    std::vector<Tensor> float_outputs;
    for (const auto& x : test_inputs) {
        float_outputs.push_back(float_model->forward(x.to(Device::HIP)));
    }
    
    // Quantize model (dynamic quantization)
    auto quant_model = QuantizedModel::from_float(*float_model);
    
    // Compare outputs
    float total_mse = 0;
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        Tensor q_out = quant_model.forward(test_inputs[i].to(Device::HIP));
        float mse = ((float_outputs[i] - q_out).pow(2)).mean().item<float>();
        total_mse += mse;
    }
    
    float avg_mse = total_mse / test_inputs.size();
    std::cout << "Average MSE: " << avg_mse << std::endl;
    
    // MSE should be small
    EXPECT_LT(avg_mse, 0.1f);
}

TEST(QuantInt8, StressTest_MemoryReduction) {
    auto config = models::TransformerConfig::llama2_7b();
    
    // Calculate float model size
    size_t float_bytes = 0;
    float_bytes += config.vocab_size * config.dim * 4;  // embeddings
    float_bytes += config.n_layers * (
        4 * config.dim * config.dim * 4 +  // Q, K, V, O projections
        3 * config.dim * config.ffn_hidden_dim * 4 +  // FFN
        2 * config.dim * 4  // norms
    );
    float_bytes += config.dim * 4 + config.vocab_size * config.dim * 4;  // final norm + lm_head
    
    // INT8 model size
    size_t int8_bytes = float_bytes / 4;  // 4x reduction
    int8_bytes += config.n_layers * 7 * 4;  // Scales (one per linear layer per layer)
    
    float reduction = static_cast<float>(float_bytes) / int8_bytes;
    std::cout << "Float model: " << (float_bytes / 1e9) << " GB" << std::endl;
    std::cout << "INT8 model: " << (int8_bytes / 1e9) << " GB" << std::endl;
    std::cout << "Reduction: " << reduction << "x" << std::endl;
    
    EXPECT_GT(reduction, 3.5f);  // At least 3.5x reduction
}
```

### 9.3 Calibration Tests

```cpp
TEST(Calibration, MinMaxObserver) {
    MinMaxObserver observer;
    
    observer.observe(tensor({1.0f, 2.0f, 3.0f}));
    observer.observe(tensor({-1.0f, 0.0f, 5.0f}));
    
    auto params = observer.compute_params();
    
    // Range [-1, 5]
    EXPECT_NEAR(params.scale, 6.0f / 255.0f, 1e-6);
}

TEST(Calibration, HistogramObserver) {
    HistogramObserver observer(1024, 99.9f);
    
    // Most values in [-1, 1], but some outliers
    Tensor normal = randn({10000});
    observer.observe(normal);
    observer.observe(tensor({100.0f}));  // Outlier
    
    auto params = observer.compute_params();
    
    // Should clip outlier due to percentile
    EXPECT_LT(params.scale, 100.0f / 127.0f);
}
```

## 10. Summary

This chapter covered:

1. **Quantization fundamentals**: Scale, zero-point, symmetric vs asymmetric
2. **Per-channel quantization**: Better accuracy for weights
3. **INT8 GEMM kernels**: Tiled implementation with accumulator
4. **Dynamic quantization**: Quantize activations on-the-fly
5. **Static quantization**: Calibrate with representative data
6. **Histogram observers**: Percentile-based clipping for robustness

Key insights:
- **4x memory reduction** with INT8
- **2-4x speedup** on hardware with INT8 support
- **Per-channel quantization** is crucial for weight accuracy
- **Calibration** significantly improves static quantization quality

Next chapter: INT4 and GPTQ for even more aggressive quantization.
```
