#include <vesper/ops/pooling.h>
#include <vesper/core/tensor.h>
#include <limits>
#include <cmath>

namespace vesper::ops {

void max_pool2d_cpu_dispatch(const Tensor& input, Tensor& output, Tensor& indices, int kh, int kw, int sh, int sw, int ph, int pw) {
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    int64_t* idx_ptr = indices.data_ptr<int64_t>();
    
    int64_t B = input.shape()[0];
    int64_t C = input.shape()[1];
    int64_t H = input.shape()[2];
    int64_t W = input.shape()[3];
    
    int64_t out_h = output.shape()[2];
    int64_t out_w = output.shape()[3];
    
    // Parallelize over B, C? Or just simple loops.
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    
                    int h_start = oh * sh - ph;
                    int w_start = ow * sw - pw;
                    int h_end = std::min(h_start + kh, (int)H);
                    int w_end = std::min(w_start + kw, (int)W);
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    
                    float max_val = -std::numeric_limits<float>::infinity();
                    int64_t max_idx = -1;
                    
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            int64_t flat_idx = ((b * C + c) * H + h) * W + w;
                            float val = in_ptr[flat_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = flat_idx;
                            }
                        }
                    }
                    
                    int64_t out_idx = ((b * C + c) * out_h + oh) * out_w + ow;
                    out_ptr[out_idx] = max_val;
                    idx_ptr[out_idx] = max_idx;
                }
            }
        }
    }
}

void max_pool2d_backward_cpu_dispatch(const Tensor& grad_output, const Tensor& indices, Tensor& grad_input) {
    const float* go_ptr = grad_output.data_ptr<float>();
    const int64_t* idx_ptr = indices.data_ptr<int64_t>();
    float* gi_ptr = grad_input.data_ptr<float>(); // Already zeroed
    
    size_t num_outputs = grad_output.numel();
    
    // Scatter gradients
    // Note: This naive loop assumes indices are unique-ish per output, but max pooling means one input maps to one output (mostly).
    // Actually, if stride < kernel_size, one input pixel could be max in multiple windows?
    // No, indices store the "winner" for each output pixel.
    // If an input pixel wins in multiple windows, its gradient accumulates.
    // This loop handles accumulation correctly because we +=.
    
    for (size_t i = 0; i < num_outputs; ++i) {
        int64_t input_idx = idx_ptr[i];
        if (input_idx != -1) { // -1 shouldn't happen if implemented correctly
            gi_ptr[input_idx] += go_ptr[i];
        }
    }
}

} // namespace vesper::ops
