#include "vesper/kernels.h"

#include "vesper/q4k.h"
#include "vesper/q5k.h"
#include "vesper/q6k.h"
#include "vesper/q8.h"
#include "vesper/rdna4.h"
#include "vesper/types.h"
#include "vesper/weight.h"

#include <cstring>

namespace vesper {
namespace {

void add_cpu(float* dst, const float* src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

}  // namespace

void rmsnorm(Device device, float* out, const float* x, const float* weight, int n,
             float eps) {
    switch (device) {
        case Device::CPU:
            rmsnorm(out, x, weight, n, eps);
            return;
        case Device::HIP:
            rdna4::rmsnorm(out, x, weight, n, eps);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void rope_neox(Device device, float* q, float* k, int n_q_heads, int n_kv_heads,
               int head_dim, int rotary_dim, int pos, float theta) {
    switch (device) {
        case Device::CPU:
            rope_neox(q, k, n_q_heads, n_kv_heads, head_dim, rotary_dim, pos, theta);
            return;
        case Device::HIP:
            rdna4::rope_neox(q, k, n_q_heads, n_kv_heads, head_dim, rotary_dim, pos, theta);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void rope_neox(Device device, float* q, float* k, int n_q_heads, int n_kv_heads,
               int head_dim, int pos, float theta) {
    rope_neox(device, q, k, n_q_heads, n_kv_heads, head_dim, head_dim, pos, theta);
}

void gemv(Device device, float* y, const float* weight, const float* x, int out_features,
          int in_features) {
    switch (device) {
        case Device::CPU:
            gemv(y, weight, x, out_features, in_features);
            return;
        case Device::HIP:
            rdna4::gemv(y, weight, x, out_features, in_features);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void gemv(float* y, const WeightMatrix& weight, const float* x) {
    check(weight.device() == Device::CPU, "CPU gemv on non-CPU weight");
    switch (weight.kind()) {
        case WeightKind::F32:
            gemv(y, weight.f32_data(), x, weight.rows(), weight.cols());
            return;
        case WeightKind::Q8_0:
            gemv_q8(y, weight.packed(), x, weight.rows(), weight.cols());
            return;
        case WeightKind::Q4_K:
            gemv_q4k(y, weight.packed(), x, weight.rows(), weight.cols());
            return;
        case WeightKind::Q5_K:
            gemv_q5k(y, weight.packed(), x, weight.rows(), weight.cols());
            return;
        case WeightKind::Q6_K:
            gemv_q6k(y, weight.packed(), x, weight.rows(), weight.cols());
            return;
    }
    throw std::logic_error("unhandled WeightKind");
}

void gemv(Device device, float* y, const WeightMatrix& weight, const float* x) {
    check(weight.device() == device, "gemv weight device mismatch");
    switch (weight.kind()) {
        case WeightKind::F32:
            gemv(device, y, weight.f32_data(), x, weight.rows(), weight.cols());
            return;
        case WeightKind::Q8_0:
            switch (device) {
                case Device::CPU:
                    gemv_q8(y, weight.packed(), x, weight.rows(), weight.cols());
                    return;
                case Device::HIP:
                    rdna4::gemv_q8(y, weight.packed(), x, weight.rows(), weight.cols());
                    return;
            }
            throw std::logic_error("unhandled Device");
        case WeightKind::Q4_K:
            switch (device) {
                case Device::CPU:
                    gemv_q4k(y, weight.packed(), x, weight.rows(), weight.cols());
                    return;
                case Device::HIP:
                    rdna4::gemv_q4k(y, weight.packed(), x, weight.rows(), weight.cols());
                    return;
            }
            throw std::logic_error("unhandled Device");
        case WeightKind::Q5_K:
            switch (device) {
                case Device::CPU:
                    gemv_q5k(y, weight.packed(), x, weight.rows(), weight.cols());
                    return;
                case Device::HIP:
                    rdna4::gemv_q5k(y, weight.packed(), x, weight.rows(), weight.cols());
                    return;
            }
            throw std::logic_error("unhandled Device");
        case WeightKind::Q6_K:
            switch (device) {
                case Device::CPU:
                    gemv_q6k(y, weight.packed(), x, weight.rows(), weight.cols());
                    return;
                case Device::HIP:
                    rdna4::gemv_q6k(y, weight.packed(), x, weight.rows(), weight.cols());
                    return;
            }
            throw std::logic_error("unhandled Device");
    }
    throw std::logic_error("unhandled WeightKind");
}

void swiglu(Device device, float* out, const float* gate, const float* up, int n) {
    switch (device) {
        case Device::CPU:
            swiglu(out, gate, up, n);
            return;
        case Device::HIP:
            rdna4::swiglu(out, gate, up, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void softmax_inplace(Device device, float* x, int n) {
    switch (device) {
        case Device::CPU:
            softmax_inplace(x, n);
            return;
        case Device::HIP:
            rdna4::softmax_inplace(x, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void embed_row(Device device, float* out, const float* table, int token, int hidden) {
    switch (device) {
        case Device::CPU:
            embed_row(out, table, token, hidden);
            return;
        case Device::HIP:
            rdna4::embed_row(out, table, token, hidden);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void split_gated_q(Device device, float* q, float* gate, const float* q_full, int n_heads,
                   int head_dim) {
    switch (device) {
        case Device::CPU:
            split_gated_q(q, gate, q_full, n_heads, head_dim);
            return;
        case Device::HIP:
            for (int h = 0; h < n_heads; ++h) {
                const float* src = q_full + h * 2 * head_dim;
                copy_vec(device, q + h * head_dim, src, head_dim);
                copy_vec(device, gate + h * head_dim, src + head_dim, head_dim);
            }
            return;
    }
    throw std::logic_error("unhandled Device");
}

void embed_row(Device device, float* out, const WeightMatrix& table, int token) {
    if (table.device() == Device::CPU) {
        if (device == Device::CPU) {
            embed_row(out, table, token);
            return;
        }
        fail("packed embed on HIP must copy from a CPU row");
    }
    check(table.device() == device, "embed table device mismatch");
    check(table.kind() == WeightKind::F32, "device embed requires F32 table");
    embed_row(device, out, table.f32_data(), token, table.cols());
}

void sigmoid_inplace(Device device, float* x, int n) {
    switch (device) {
        case Device::CPU:
            sigmoid_inplace(x, n);
            return;
        case Device::HIP:
            rdna4::sigmoid_inplace(x, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void silu_inplace(Device device, float* x, int n) {
    switch (device) {
        case Device::CPU:
            silu_inplace(x, n);
            return;
        case Device::HIP:
            rdna4::silu_inplace(x, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void softplus_inplace(Device device, float* x, int n) {
    switch (device) {
        case Device::CPU:
            softplus_inplace(x, n);
            return;
        case Device::HIP:
            rdna4::softplus_inplace(x, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void exp_inplace(Device device, float* x, int n) {
    switch (device) {
        case Device::CPU:
            exp_inplace(x, n);
            return;
        case Device::HIP:
            rdna4::exp_inplace(x, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void mul_inplace(Device device, float* dst, const float* src, int n) {
    switch (device) {
        case Device::CPU:
            mul_inplace(dst, src, n);
            return;
        case Device::HIP:
            rdna4::mul_inplace(dst, src, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void scale_inplace(Device device, float* x, float scale, int n) {
    switch (device) {
        case Device::CPU:
            scale_inplace(x, scale, n);
            return;
        case Device::HIP:
            rdna4::scale_inplace(x, scale, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void l2_normalize_rows(Device device, float* x, int rows, int dim, float eps) {
    switch (device) {
        case Device::CPU:
            l2_normalize_rows(x, rows, dim, eps);
            return;
        case Device::HIP:
            rdna4::l2_normalize_rows(x, rows, dim, eps);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void attn_scores(Device device, float* scores, const float* q, const float* k, int seq,
                 int n_kv_heads, int kv_head, int head_dim) {
    switch (device) {
        case Device::CPU:
            attn_scores(scores, q, k, seq, n_kv_heads, kv_head, head_dim);
            return;
        case Device::HIP:
            rdna4::attn_scores(scores, q, k, seq, n_kv_heads, kv_head, head_dim);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void attn_mix(Device device, float* out, const float* scores, const float* v, int seq,
              int n_kv_heads, int kv_head, int head_dim) {
    switch (device) {
        case Device::CPU:
            attn_mix(out, scores, v, seq, n_kv_heads, kv_head, head_dim);
            return;
        case Device::HIP:
            rdna4::attn_mix(out, scores, v, seq, n_kv_heads, kv_head, head_dim);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void add_inplace(Device device, float* dst, const float* src, int n) {
    switch (device) {
        case Device::CPU:
            add_cpu(dst, src, n);
            return;
        case Device::HIP:
            rdna4::add_inplace(dst, src, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void copy_vec(Device device, float* dst, const float* src, int n) {
    switch (device) {
        case Device::CPU:
            std::memcpy(dst, src, static_cast<std::size_t>(n) * sizeof(float));
            return;
        case Device::HIP:
            rdna4::copy(dst, src, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

}  // namespace vesper
