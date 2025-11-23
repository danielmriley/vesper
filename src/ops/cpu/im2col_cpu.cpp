#include <vesper/ops/im2col.h>
#include <vesper/core/tensor.h>
#include <cstring>

namespace vesper::ops {

void im2col_cpu_dispatch(const Tensor& input, Tensor& output, int kh, int kw, int sh, int sw, int ph, int pw) {
    // Input: [B, C, H, W]
    // Output: [C*KH*KW, B*OH*OW]
    
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    int64_t B = input.shape()[0];
    int64_t C = input.shape()[1];
    int64_t H = input.shape()[2];
    int64_t W = input.shape()[3];
    
    int64_t out_h = (H + 2 * ph - kh) / sh + 1;
    int64_t out_w = (W + 2 * pw - kw) / sw + 1;
    
    int64_t channel_size = H * W;
    
    // We iterate over the Output columns (each corresponds to one sliding window position)
    // Output column index: b * (out_h * out_w) + oh * out_w + ow
    
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    
                    int64_t output_row = c * kh * kw + ky * kw + kx;
                    
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            
                            int64_t output_col = b * out_h * out_w + oh * out_w + ow;
                            
                            int in_y = oh * sh - ph + ky;
                            int in_x = ow * sw - pw + kx;
                            
                            float val = 0.0f;
                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                val = in_ptr[b * C * channel_size + c * channel_size + in_y * W + in_x];
                            }
                            
                            // output is column-major relative to [C*KH*KW, B*OH*OW]? 
                            // No, Tensor is row-major.
                            // output[output_row, output_col]
                            out_ptr[output_row * output.shape()[1] + output_col] = val;
                        }
                    }
                }
            }
        }
    }
}

void col2im_cpu_dispatch(const Tensor& grad_col, Tensor& grad_img, int kh, int kw, int sh, int sw, int ph, int pw) {
    // grad_col: [C*KH*KW, B*OH*OW]
    // grad_img: [B, C, H, W] (accumulate here)
    
    const float* col_ptr = grad_col.data_ptr<float>();
    float* img_ptr = grad_img.data_ptr<float>();
    
    int64_t B = grad_img.shape()[0];
    int64_t C = grad_img.shape()[1];
    int64_t H = grad_img.shape()[2];
    int64_t W = grad_img.shape()[3];
    
    int64_t out_h = (H + 2 * ph - kh) / sh + 1;
    int64_t out_w = (W + 2 * pw - kw) / sw + 1;
    int64_t channel_size = H * W;

    // Initialize grad_img to 0 (done by factory, but good to ensure if we didn't use zeros())
    // We assume grad_img is zeros.
    
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    
                    int64_t input_row = c * kh * kw + ky * kw + kx;
                    
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            
                            int64_t input_col = b * out_h * out_w + oh * out_w + ow;
                            float val = col_ptr[input_row * grad_col.shape()[1] + input_col];
                            
                            int in_y = oh * sh - ph + ky;
                            int in_x = ow * sw - pw + kx;
                            
                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                img_ptr[b * C * channel_size + c * channel_size + in_y * W + in_x] += val;
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace vesper::ops
