#include <vesper/nn/rope.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace vesper::nn {

// Simple check macro
#define ROPE_CHECK(condition, message) \
    if (!(condition)) { \
        throw std::runtime_error(std::string("RoPE error: ") + (message)); \
    }

// Forward declarations for GPU dispatch (defined in rope.hip / rope.cu)
#if defined(USE_HIP_BACKEND)
void apply_rope_hip(float* x, const float* freqs_cos, const float* freqs_sin,
                    int batch, int heads, int seq_len, int head_dim);
void apply_rope_inverse_hip(float* x, const float* freqs_cos, const float* freqs_sin,
                            int batch, int heads, int seq_len, int head_dim);
#endif

#if defined(USE_CUDA_BACKEND)
void apply_rope_cuda(float* x, const float* freqs_cos, const float* freqs_sin,
                     int batch, int heads, int seq_len, int head_dim);
void apply_rope_inverse_cuda(float* x, const float* freqs_cos, const float* freqs_sin,
                             int batch, int heads, int seq_len, int head_dim);
#endif

// CPU implementations
namespace {

void apply_rope_cpu(float* x, const float* freqs_cos, const float* freqs_sin,
                    int batch, int heads, int seq_len, int head_dim) {
    int half_dim = head_dim / 2;
    
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < half_dim; ++d) {
                    // Index into x: [b, h, s, d*2] and [b, h, s, d*2+1]
                    int base_idx = ((b * heads + h) * seq_len + s) * head_dim;
                    int idx0 = base_idx + d * 2;
                    int idx1 = base_idx + d * 2 + 1;
                    
                    // Index into freqs: [s, d]
                    int freq_idx = s * half_dim + d;
                    
                    float x0 = x[idx0];
                    float x1 = x[idx1];
                    float cos_val = freqs_cos[freq_idx];
                    float sin_val = freqs_sin[freq_idx];
                    
                    // Rotation: [cos -sin; sin cos] * [x0; x1]
                    x[idx0] = x0 * cos_val - x1 * sin_val;
                    x[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

void apply_rope_inverse_cpu(float* x, const float* freqs_cos, const float* freqs_sin,
                            int batch, int heads, int seq_len, int head_dim) {
    int half_dim = head_dim / 2;
    
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < half_dim; ++d) {
                    int base_idx = ((b * heads + h) * seq_len + s) * head_dim;
                    int idx0 = base_idx + d * 2;
                    int idx1 = base_idx + d * 2 + 1;
                    
                    int freq_idx = s * half_dim + d;
                    
                    float x0 = x[idx0];
                    float x1 = x[idx1];
                    float cos_val = freqs_cos[freq_idx];
                    float sin_val = freqs_sin[freq_idx];  // Will negate below
                    
                    // Inverse rotation: [cos sin; -sin cos] * [x0; x1]
                    x[idx0] = x0 * cos_val + x1 * sin_val;
                    x[idx1] = -x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

} // anonymous namespace

RoPEFrequencies::RoPEFrequencies(int64_t max_seq_len, int64_t head_dim, 
                                  float base, Device device)
    : max_seq_len_(max_seq_len), head_dim_(head_dim), base_(base), device_(device) 
{
    ROPE_CHECK(head_dim % 2 == 0, "head_dim must be even for RoPE");
    ROPE_CHECK(max_seq_len > 0, "max_seq_len must be positive");
    ROPE_CHECK(base > 0, "base must be positive");
    
    compute_frequencies();
}

void RoPEFrequencies::compute_frequencies() {
    int64_t half_dim = head_dim_ / 2;
    
    // Compute inverse frequencies: theta_i = base^(-2i/d)
    std::vector<float> inv_freq(half_dim);
    for (int64_t i = 0; i < half_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(base_, static_cast<float>(2 * i) / head_dim_);
    }
    
    // Compute angles for all positions: [max_seq_len, half_dim]
    std::vector<float> angles_data(max_seq_len_ * half_dim);
    for (int64_t pos = 0; pos < max_seq_len_; ++pos) {
        for (int64_t i = 0; i < half_dim; ++i) {
            angles_data[pos * half_dim + i] = pos * inv_freq[i];
        }
    }
    
    // Store angles
    angles_ = vesper::empty({max_seq_len_, half_dim}, DType::Float32, Device::CPU);
    angles_.copy_from_host(angles_data.data());
    
    // Compute cos/sin values: [max_seq_len, half_dim, 2]
    std::vector<float> freqs_data(max_seq_len_ * half_dim * 2);
    for (int64_t pos = 0; pos < max_seq_len_; ++pos) {
        for (int64_t i = 0; i < half_dim; ++i) {
            float angle = angles_data[pos * half_dim + i];
            int64_t idx = (pos * half_dim + i) * 2;
            freqs_data[idx]     = std::cos(angle);  // cos
            freqs_data[idx + 1] = std::sin(angle);  // sin
        }
    }
    
    freqs_ = vesper::empty({max_seq_len_, half_dim, 2}, DType::Float32, Device::CPU);
    freqs_.copy_from_host(freqs_data.data());
    
    // Move to target device if needed
    if (device_ != Device::CPU) {
        freqs_ = freqs_.to(device_);
        angles_ = angles_.to(device_);
    }
}

Tensor RoPEFrequencies::get(int64_t start_pos, int64_t seq_len) const {
    ROPE_CHECK(start_pos >= 0, "start_pos must be non-negative");
    ROPE_CHECK(seq_len > 0, "seq_len must be positive");
    ROPE_CHECK(start_pos + seq_len <= max_seq_len_, "Position range exceeds max_seq_len");
    
    // Return a slice: [start_pos : start_pos + seq_len, :, :]
    // slice(start, stop, step) slices on dim 0
    return freqs_.slice(start_pos, start_pos + seq_len, 1);
}

Tensor RoPEFrequencies::get_angles(int64_t start_pos, int64_t seq_len) const {
    ROPE_CHECK(start_pos >= 0, "start_pos must be non-negative");
    ROPE_CHECK(seq_len > 0, "seq_len must be positive");
    ROPE_CHECK(start_pos + seq_len <= max_seq_len_, "Position range exceeds max_seq_len");
    
    return angles_.slice(start_pos, start_pos + seq_len, 1);
}

void RoPEFrequencies::update_base(float new_base) {
    ROPE_CHECK(new_base > 0, "new_base must be positive");
    base_ = new_base;
    compute_frequencies();
}

void RoPEFrequencies::to(Device device) {
    if (device_ != device) {
        freqs_ = freqs_.to(device);
        angles_ = angles_.to(device);
        device_ = device;
    }
}

namespace functional {

void apply_rope_inplace(Tensor& x, const Tensor& freqs) {
    auto shape = x.shape();
    ROPE_CHECK(shape.size() == 4, "Input must be 4D [B, H, S, D]");
    
    int batch = static_cast<int>(shape[0]);
    int heads = static_cast<int>(shape[1]);
    int seq_len = static_cast<int>(shape[2]);
    int head_dim = static_cast<int>(shape[3]);
    
    ROPE_CHECK(head_dim % 2 == 0, "head_dim must be even");
    
    auto freqs_shape = freqs.shape();
    ROPE_CHECK(freqs_shape.size() == 3, "freqs must be 3D [S, D/2, 2]");
    ROPE_CHECK(freqs_shape[0] == seq_len, "freqs seq_len mismatch");
    ROPE_CHECK(freqs_shape[1] == head_dim / 2, "freqs head_dim/2 mismatch");
    ROPE_CHECK(freqs_shape[2] == 2, "freqs last dim must be 2 for [cos, sin]");
    ROPE_CHECK(x.device() == freqs.device(), "x and freqs must be on same device");
    
    // Make x contiguous if needed
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    // Extract cos and sin as separate views: [S, D/2]
    int half_dim = head_dim / 2;
    
    // Slice freqs to get cos (index 0) and sin (index 1) along last dimension
    Tensor freqs_cos = freqs.index({Slice(), Slice(), Slice(0, 1)}).view({seq_len, half_dim});
    Tensor freqs_sin = freqs.index({Slice(), Slice(), Slice(1, 2)}).view({seq_len, half_dim});
    
    // Need contiguous data for kernels
    Tensor freqs_cos_cont = freqs_cos.contiguous();
    Tensor freqs_sin_cont = freqs_sin.contiguous();
    
    if (x.device() == Device::CPU) {
        apply_rope_cpu(x.data_ptr<float>(), 
                       freqs_cos_cont.data_ptr<float>(),
                       freqs_sin_cont.data_ptr<float>(),
                       batch, heads, seq_len, head_dim);
    } 
#if defined(USE_HIP_BACKEND)
    else if (x.device() == Device::HIP) {
        apply_rope_hip(x.data_ptr<float>(),
                       freqs_cos_cont.data_ptr<float>(),
                       freqs_sin_cont.data_ptr<float>(),
                       batch, heads, seq_len, head_dim);
    }
#endif
#if defined(USE_CUDA_BACKEND)
    else if (x.device() == Device::CUDA) {
        apply_rope_cuda(x.data_ptr<float>(),
                        freqs_cos_cont.data_ptr<float>(),
                        freqs_sin_cont.data_ptr<float>(),
                        batch, heads, seq_len, head_dim);
    }
#endif
    else {
        ROPE_CHECK(false, "Unsupported device for RoPE");
    }
}

void apply_rope(Tensor& q, Tensor& k, const Tensor& freqs) {
    apply_rope_inplace(q, freqs);
    apply_rope_inplace(k, freqs);
}

void apply_rope_inverse(Tensor& x, const Tensor& freqs) {
    auto shape = x.shape();
    ROPE_CHECK(shape.size() == 4, "Input must be 4D [B, H, S, D]");
    
    int batch = static_cast<int>(shape[0]);
    int heads = static_cast<int>(shape[1]);
    int seq_len = static_cast<int>(shape[2]);
    int head_dim = static_cast<int>(shape[3]);
    
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    int half_dim = head_dim / 2;
    Tensor freqs_cos = freqs.index({Slice(), Slice(), Slice(0, 1)}).view({seq_len, half_dim});
    Tensor freqs_sin = freqs.index({Slice(), Slice(), Slice(1, 2)}).view({seq_len, half_dim});
    
    Tensor freqs_cos_cont = freqs_cos.contiguous();
    Tensor freqs_sin_cont = freqs_sin.contiguous();
    
    if (x.device() == Device::CPU) {
        apply_rope_inverse_cpu(x.data_ptr<float>(),
                               freqs_cos_cont.data_ptr<float>(),
                               freqs_sin_cont.data_ptr<float>(),
                               batch, heads, seq_len, head_dim);
    }
#if defined(USE_HIP_BACKEND)
    else if (x.device() == Device::HIP) {
        apply_rope_inverse_hip(x.data_ptr<float>(),
                               freqs_cos_cont.data_ptr<float>(),
                               freqs_sin_cont.data_ptr<float>(),
                               batch, heads, seq_len, head_dim);
    }
#endif
#if defined(USE_CUDA_BACKEND)
    else if (x.device() == Device::CUDA) {
        apply_rope_inverse_cuda(x.data_ptr<float>(),
                                freqs_cos_cont.data_ptr<float>(),
                                freqs_sin_cont.data_ptr<float>(),
                                batch, heads, seq_len, head_dim);
    }
#endif
    else {
        ROPE_CHECK(false, "Unsupported device for RoPE inverse");
    }
}

} // namespace functional

namespace rope_utils {

float compute_ntk_base(int64_t original_max_len, int64_t current_len,
                       float original_base, float alpha) {
    if (current_len <= original_max_len) {
        return original_base;
    }
    
    // Scale factor based on how much we're extrapolating
    float scale = static_cast<float>(current_len) / static_cast<float>(original_max_len);
    
    // NTK-aware base adjustment
    // This spreads the rotations more evenly across the extended context
    float new_base = original_base * std::pow(scale, alpha);
    return new_base;
}

} // namespace rope_utils

} // namespace vesper::nn
