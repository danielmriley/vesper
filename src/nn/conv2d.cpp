#include <vesper/nn/conv2d.h>
#include <vesper/core/factories.h>
#include <vesper/ops/im2col.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h>
#include <cmath>
#include <random>

namespace vesper::nn {

// Helper for Kaiming init (duplicated from linear.cpp, should probably move to utils/init)
void kaiming_uniform_init_conv(Tensor& t, int64_t fan_in) {
    const float bound = std::sqrt(6.0f / fan_in);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-bound, bound);
    std::vector<float> data(t.numel());
    for (float& val : data) val = dist(rng);
    t.copy_from_host(data.data());
}

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, bool use_bias, Device device)
    : Conv2d(in_channels, out_channels, {kernel_size, kernel_size}, {stride, stride}, {padding, padding}, use_bias, device) {}

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, std::pair<int64_t, int64_t> kernel_size, std::pair<int64_t, int64_t> stride, std::pair<int64_t, int64_t> padding, bool use_bias, Device device)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), use_bias_(use_bias)
{
    // Weight shape: [OutCh, InCh, KH, KW] -> Flattened to [OutCh, InCh*KH*KW] for 2D GEMM
    // Standard PyTorch stores as 4D. Vesper 2D GEMM needs 2D.
    // We can store as 4D and view as 2D for matmul.
    
    weight = empty({out_channels, in_channels, kernel_size.first, kernel_size.second}, DType::Float32, device);
    
    // Kaiming init
    int64_t fan_in = in_channels * kernel_size.first * kernel_size.second;
    kaiming_uniform_init_conv(weight, fan_in);
    register_parameter("weight", weight);

    if (use_bias) {
        bias = zeros({out_channels}, DType::Float32, device);
        register_parameter("bias", bias);
    }
}

Tensor Conv2d::forward(const Tensor& input) {
    // Input: [B, C, H, W]
    int64_t B = input.shape()[0];
    int64_t H = input.shape()[2];
    int64_t W = input.shape()[3];
    
    int64_t kh = kernel_size_.first;
    int64_t kw = kernel_size_.second;
    int64_t sh = stride_.first;
    int64_t sw = stride_.second;
    int64_t ph = padding_.first;
    int64_t pw = padding_.second;
    
    int64_t out_h = (H + 2 * ph - kh) / sh + 1;
    int64_t out_w = (W + 2 * pw - kw) / sw + 1;

    // 1. Im2Col
    // Output: [InCh * KH * KW, B * OutH * OutW]
    Tensor col = ops::im2col(input, kh, kw, sh, sw, ph, pw);
    
    // 2. GEMM
    // Weight: [OutCh, InCh, KH, KW] -> View as [OutCh, InCh*KH*KW]
    // Col:    [InCh*KH*KW, B*OutH*OutW]
    // Result: [OutCh, B*OutH*OutW]
    Tensor weight_flat = weight.view({out_channels_, in_channels_ * kh * kw});
    Tensor out_flat = ops::matmul(weight_flat, col);
    
    // 3. Reshape to [OutCh, B, OutH, OutW] ? No, standard is [B, OutCh, OutH, OutW]
    // out_flat is [OutCh, B*OutH*OutW].
    // Reshape to [OutCh, B, OutH*OutW]
    // Transpose to [B, OutCh, OutH*OutW]
    // Reshape to [B, OutCh, OutH, OutW]
    
    Tensor out_reshaped = out_flat.view({out_channels_, B, out_h * out_w});
    Tensor out_transposed = out_reshaped.permute({1, 0, 2}); // [B, OutCh, H*W]
    Tensor output = out_transposed.reshape({B, out_channels_, out_h, out_w});
    
    // 4. Add Bias
    if (use_bias_) {
        // Bias is [OutCh]. Need to broadcast to [B, OutCh, OutH, OutW].
        // Vesper broadcasting handles [B, C, H, W] + [C] ? No.
        // [B, C, H, W] + [C, 1, 1] works.
        Tensor bias_view = bias.view({1, out_channels_, 1, 1});
        output = ops::add(output, bias_view);
    }
    
    // Autograd hook for input (col2im) is implicit via im2col?
    // im2col returns a tensor. If input required grad, im2col output tracks it?
    // Current im2col implementation does NOT attach a grad_node! 
    // I need to fix im2col to support autograd.
    // See ops/im2col.cpp: `Tensor output = empty(..., input.requires_grad())`
    // But no `node->backward_fn` is set!
    
    // I must update im2col to support backward (col2im).
    
    return output;
}

} // namespace vesper::nn
