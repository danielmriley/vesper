#include <vesper/nn/conv2d.h>
#include <vesper/core/factories.h>
#include <vesper/ops/im2col.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h>
#include <vesper/nn/init.h> // New include
#include <cmath>

namespace vesper::nn {

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, bool use_bias, Device device)
    : Conv2d(in_channels, out_channels, {kernel_size, kernel_size}, {stride, stride}, {padding, padding}, use_bias, device) {}

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, std::pair<int64_t, int64_t> kernel_size, std::pair<int64_t, int64_t> stride, std::pair<int64_t, int64_t> padding, bool use_bias, Device device)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), use_bias_(use_bias)
{
    weight = empty({out_channels, in_channels, kernel_size.first, kernel_size.second}, DType::Float32, device);
    
    init::kaiming_uniform_(weight, std::sqrt(5.0f));
    register_parameter("weight", &weight);  // Pass pointer to member variable!

    if (use_bias) {
        bias = empty({out_channels}, DType::Float32, device);
        
        int64_t fan_in = in_channels * kernel_size.first * kernel_size.second;
        float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        init::uniform_(bias, -bound, bound);
        
        register_parameter("bias", &bias);  // Pass pointer to member variable!
    }
}

Tensor Conv2d::forward(const Tensor& input) {
    // ... (rest unchanged) ...
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
    // Transpose to [B, OutCh, H*W]
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
    
    return output;
}

} // namespace vesper::nn
