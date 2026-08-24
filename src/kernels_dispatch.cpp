#include "vesper/kernels.h"

#include "vesper/rdna4.h"
#include "vesper/types.h"

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
               int head_dim, int pos, float theta) {
    switch (device) {
        case Device::CPU:
            rope_neox(q, k, n_q_heads, n_kv_heads, head_dim, pos, theta);
            return;
        case Device::HIP:
            rdna4::rope_neox(q, k, n_q_heads, n_kv_heads, head_dim, pos, theta);
            return;
    }
    throw std::logic_error("unhandled Device");
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
