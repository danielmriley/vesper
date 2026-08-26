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

void rope_neox_k_norm(Device device, float* q, float* k, const float* k_weight, int n_q_heads,
                      int n_kv_heads, int head_dim, int rotary_dim, int pos, float theta,
                      float eps) {
    switch (device) {
        case Device::CPU:
            rope_neox_k_norm(q, k, k_weight, n_q_heads, n_kv_heads, head_dim, rotary_dim, pos,
                             theta, eps);
            return;
        case Device::HIP:
            rdna4::rope_neox_k_norm(q, k, k_weight, n_q_heads, n_kv_heads, head_dim, rotary_dim,
                                    pos, theta, eps);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void rope_neox(Device device, float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim,
               int rotary_dim, const int* pos, float theta) {
    switch (device) {
        case Device::CPU:
            check(pos != nullptr, "rope pos");
            rope_neox(q, k, n_q_heads, n_kv_heads, head_dim, rotary_dim, *pos, theta);
            return;
        case Device::HIP:
            rdna4::rope_neox(q, k, n_q_heads, n_kv_heads, head_dim, rotary_dim, pos, theta);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void rope_neox_k_norm(Device device, float* q, float* k, const float* k_weight, int n_q_heads,
                      int n_kv_heads, int head_dim, int rotary_dim, const int* pos, float theta,
                      float eps) {
    switch (device) {
        case Device::CPU:
            check(pos != nullptr, "rope pos");
            rope_neox_k_norm(q, k, k_weight, n_q_heads, n_kv_heads, head_dim, rotary_dim, *pos,
                             theta, eps);
            return;
        case Device::HIP:
            rdna4::rope_neox_k_norm(q, k, k_weight, n_q_heads, n_kv_heads, head_dim, rotary_dim,
                                    pos, theta, eps);
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
            rdna4::split_gated_q(q, gate, q_full, n_heads, head_dim);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void split_gated_q_norm(Device device, float* q, float* gate, const float* q_full, const float* weight,
                        int n_heads, int head_dim, float eps) {
    switch (device) {
        case Device::CPU:
            split_gated_q_norm(q, gate, q_full, weight, n_heads, head_dim, eps);
            return;
        case Device::HIP:
            rdna4::split_gated_q_norm(q, gate, q_full, weight, n_heads, head_dim, eps);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void rmsnorm_silu_mul(Device device, float* y, const float* z, const float* weight, int rows, int dim,
                      float eps) {
    switch (device) {
        case Device::CPU:
            rmsnorm_silu_mul(y, z, weight, rows, dim, eps);
            return;
        case Device::HIP:
            rdna4::rmsnorm_silu_mul(y, z, weight, rows, dim, eps);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void embed_row(Device device, float* out, const WeightMatrix& table, int token) {
    check(table.device() == device, "embed table device mismatch");
    switch (device) {
        case Device::CPU:
            embed_row(out, table, token);
            return;
        case Device::HIP:
            rdna4::embed_row(out, table, token);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void embed_row(Device device, float* out, const float* table, const int* token, int hidden) {
    switch (device) {
        case Device::CPU:
            check(token != nullptr, "embed token");
            embed_row(out, table, *token, hidden);
            return;
        case Device::HIP:
            rdna4::embed_row(out, table, token, hidden);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void embed_row(Device device, float* out, const WeightMatrix& table, const int* token) {
    check(table.device() == device, "embed table device mismatch");
    switch (device) {
        case Device::CPU:
            check(token != nullptr, "embed token");
            embed_row(out, table, *token);
            return;
        case Device::HIP:
            rdna4::embed_row(out, table, token);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void scatter_row(Device device, float* base, const float* row, const int* pos, int n) {
    switch (device) {
        case Device::CPU:
            scatter_row(base, row, pos, n);
            return;
        case Device::HIP:
            rdna4::scatter_row(base, row, pos, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void scatter_kv(Device device, float* k_base, float* v_base, const float* k, const float* v,
                const int* pos, int n) {
    switch (device) {
        case Device::CPU:
            scatter_kv(k_base, v_base, k, v, pos, n);
            return;
        case Device::HIP:
            rdna4::scatter_kv(k_base, v_base, k, v, pos, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void attn_prepare(Device device, float* q, float* gate, float* k, float* v, const float* q_full,
                  const float* q_weight, const float* k_weight, float* k_base, float* v_base,
                  const int* pos, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
                  float theta, float eps) {
    switch (device) {
        case Device::CPU:
            attn_prepare(q, gate, k, v, q_full, q_weight, k_weight, k_base, v_base, pos, n_q_heads,
                         n_kv_heads, head_dim, rotary_dim, theta, eps);
            return;
        case Device::HIP:
            rdna4::attn_prepare(q, gate, k, v, q_full, q_weight, k_weight, k_base, v_base, pos,
                                n_q_heads, n_kv_heads, head_dim, rotary_dim, theta, eps);
            return;
    }
    throw std::logic_error("unhandled Device");
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

void rmsnorm_rows(Device device, float* x, const float* weight, int rows, int dim, float eps) {
    switch (device) {
        case Device::CPU:
            rmsnorm_rows(x, weight, rows, dim, eps);
            return;
        case Device::HIP:
            rdna4::rmsnorm_rows(x, weight, rows, dim, eps);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void tile_heads(Device device, float* dst, const float* src, int n_dst, int n_src, int dim) {
    switch (device) {
        case Device::CPU:
            tile_heads(dst, src, n_dst, n_src, dim);
            return;
        case Device::HIP:
            rdna4::tile_heads(dst, src, n_dst, n_src, dim);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void attn_decode(Device device, float* out, float* scores, const float* q, const float* k,
                 const float* v, int seq, int n_q_heads, int n_kv_heads, int head_dim) {
    attn_decode(device, out, scores, q, k, v, nullptr, seq, n_q_heads, n_kv_heads, head_dim);
}

void attn_decode(Device device, float* out, float* scores, const float* q, const float* k,
                 const float* v, const float* gate, int seq, int n_q_heads, int n_kv_heads,
                 int head_dim) {
    switch (device) {
        case Device::CPU:
            attn_decode(out, scores, q, k, v, gate, seq, n_q_heads, n_kv_heads, head_dim);
            return;
        case Device::HIP:
            rdna4::attn_decode(out, q, k, v, gate, seq, n_q_heads, n_kv_heads, head_dim);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void attn_decode(Device device, float* out, float* scores, const float* q, const float* k,
                 const float* v, const float* gate, const int* pos, int n_q_heads, int n_kv_heads,
                 int head_dim) {
    switch (device) {
        case Device::CPU:
            check(pos != nullptr, "attn pos");
            attn_decode(out, scores, q, k, v, gate, *pos + 1, n_q_heads, n_kv_heads, head_dim);
            return;
        case Device::HIP:
            rdna4::attn_decode(out, q, k, v, gate, pos, n_q_heads, n_kv_heads, head_dim);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void gemv_swiglu(Device device, float* hidden, float* gate_tmp, float* up_tmp,
                 const WeightMatrix& gate, const WeightMatrix& up, const float* x) {
    check(gate.device() == device && up.device() == device, "gemv_swiglu device mismatch");
    check(gate.rows() == up.rows() && gate.cols() == up.cols(), "gemv_swiglu shape mismatch");
    switch (device) {
        case Device::CPU:
            gemv_swiglu(hidden, gate_tmp, up_tmp, gate, up, x);
            return;
        case Device::HIP:
            if (gate.kind() == up.kind()) {
                rdna4::gemv_swiglu(hidden, gate, up, x);
                return;
            }
            gemv(device, gate_tmp, gate, x);
            gemv(device, up_tmp, up, x);
            swiglu(device, hidden, gate_tmp, up_tmp, gate.rows());
            return;
    }
    throw std::logic_error("unhandled Device");
}

void gemv_add(Device device, float* y, const WeightMatrix& weight, const float* x,
              const float* addend) {
    check(weight.device() == device, "gemv_add weight device mismatch");
    switch (device) {
        case Device::CPU:
            gemv_add(y, weight, x, addend);
            return;
        case Device::HIP:
            switch (weight.kind()) {
                case WeightKind::F32:
                    rdna4::gemv(y, weight.f32_data(), x, weight.rows(), weight.cols(), addend);
                    return;
                case WeightKind::Q8_0:
                    rdna4::gemv_q8(y, weight.packed(), x, weight.rows(), weight.cols(), addend);
                    return;
                case WeightKind::Q4_K:
                    rdna4::gemv_q4k(y, weight.packed(), x, weight.rows(), weight.cols(), addend);
                    return;
                case WeightKind::Q5_K:
                    rdna4::gemv_q5k(y, weight.packed(), x, weight.rows(), weight.cols(), addend);
                    return;
                case WeightKind::Q6_K:
                    rdna4::gemv_q6k(y, weight.packed(), x, weight.rows(), weight.cols(), addend);
                    return;
            }
            throw std::logic_error("unhandled WeightKind");
    }
    throw std::logic_error("unhandled Device");
}

void gemv3(Device device, float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1,
           float* y2, const WeightMatrix& w2, const float* x) {
    check(w0.device() == device && w1.device() == device && w2.device() == device,
          "gemv3 device mismatch");
    check(w0.cols() == w1.cols() && w1.cols() == w2.cols(), "gemv3 cols mismatch");
    switch (device) {
        case Device::CPU:
            gemv3(y0, w0, y1, w1, y2, w2, x);
            return;
        case Device::HIP:
            if (w0.kind() == w1.kind() && w1.kind() == w2.kind()) {
                rdna4::gemv3(y0, w0, y1, w1, y2, w2, x);
                return;
            }
            gemv(device, y0, w0, x);
            gemv(device, y1, w1, x);
            gemv(device, y2, w2, x);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void gemv4(Device device, float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1,
           float* y2, const WeightMatrix& w2, float* y3, const WeightMatrix& w3, const float* x) {
    check(w0.device() == device && w1.device() == device && w2.device() == device &&
              w3.device() == device,
          "gemv4 device mismatch");
    check(w0.cols() == w1.cols() && w1.cols() == w2.cols() && w2.cols() == w3.cols(),
          "gemv4 cols mismatch");
    switch (device) {
        case Device::CPU:
            gemv4(y0, w0, y1, w1, y2, w2, y3, w3, x);
            return;
        case Device::HIP:
            if (w0.kind() == w1.kind() && w1.kind() == w2.kind() && w2.kind() == w3.kind()) {
                rdna4::gemv4(y0, w0, y1, w1, y2, w2, y3, w3, x);
                return;
            }
            gemv(device, y0, w0, x);
            gemv(device, y1, w1, x);
            gemv(device, y2, w2, x);
            gemv(device, y3, w3, x);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void tile_l2_scale(Device device, float* dst, const float* src, int n_dst, int n_src, int dim,
                   float eps, float scale) {
    switch (device) {
        case Device::CPU:
            tile_l2_scale(dst, src, n_dst, n_src, dim, eps, scale);
            return;
        case Device::HIP:
            rdna4::tile_l2_scale(dst, src, n_dst, n_src, dim, eps, scale);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void tile_l2_pair(Device device, float* q_dst, const float* q_src, float* k_dst, const float* k_src,
                  int n_dst, int n_src, int dim, float eps, float q_scale, float k_scale) {
    switch (device) {
        case Device::CPU:
            tile_l2_pair(q_dst, q_src, k_dst, k_src, n_dst, n_src, dim, eps, q_scale, k_scale);
            return;
        case Device::HIP:
            rdna4::tile_l2_pair(q_dst, q_src, k_dst, k_src, n_dst, n_src, dim, eps, q_scale,
                                k_scale);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void add_rmsnorm(Device device, float* x, float* residual, const float* weight, int n, float eps) {
    switch (device) {
        case Device::CPU:
            add_rmsnorm(x, residual, weight, n, eps);
            return;
        case Device::HIP:
            rdna4::add_rmsnorm(x, residual, weight, n, eps);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void copy_rmsnorm(Device device, float* x, float* residual, const float* weight, int n, float eps) {
    switch (device) {
        case Device::CPU:
            copy_rmsnorm(x, residual, weight, n, eps);
            return;
        case Device::HIP:
            rdna4::copy_rmsnorm(x, residual, weight, n, eps);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void silu_mul(Device device, float* y, const float* z, int n) {
    switch (device) {
        case Device::CPU:
            silu_mul(y, z, n);
            return;
        case Device::HIP:
            rdna4::silu_mul(y, z, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void gdn_gates(Device device, float* decay, float* beta, const float* alpha, const float* dt,
               const float* a, int n) {
    switch (device) {
        case Device::CPU:
            gdn_gates(decay, beta, alpha, dt, a, n);
            return;
        case Device::HIP:
            rdna4::gdn_gates(decay, beta, alpha, dt, a, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void gdn_tile_gates(Device device, float* q_dst, const float* q_src, float* k_dst, const float* k_src,
                    float* decay, float* beta, const float* alpha, const float* dt, const float* a,
                    int n_dst, int n_src, int dim, float eps, float q_scale, float k_scale) {
    switch (device) {
        case Device::CPU:
            gdn_tile_gates(q_dst, q_src, k_dst, k_src, decay, beta, alpha, dt, a, n_dst, n_src, dim,
                           eps, q_scale, k_scale);
            return;
        case Device::HIP:
            rdna4::gdn_tile_gates(q_dst, q_src, k_dst, k_src, decay, beta, alpha, dt, a, n_dst, n_src,
                                  dim, eps, q_scale, k_scale);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void split_qkv(Device device, float* q, float* k, float* v, const float* qkv, int key_dim,
               int value_dim) {
    switch (device) {
        case Device::CPU:
            split_qkv(q, k, v, qkv, key_dim, value_dim);
            return;
        case Device::HIP:
            rdna4::split_qkv(q, k, v, qkv, key_dim, value_dim);
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

int argmax(Device device, const float* x, int n) {
    switch (device) {
        case Device::CPU:
            return argmax(x, n);
        case Device::HIP:
            return rdna4::argmax(x, n);
    }
    throw std::logic_error("unhandled Device");
}

void argmax_write(Device device, int* dst, const float* x, int n) {
    switch (device) {
        case Device::CPU:
            check(dst != nullptr, "argmax_write null");
            *dst = argmax(x, n);
            return;
        case Device::HIP:
            rdna4::argmax_write(dst, x, n);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void seed_generated(Device device, int* ids, int* index, const int* token) {
    switch (device) {
        case Device::CPU:
            seed_generated(ids, index, token);
            return;
        case Device::HIP:
            rdna4::seed_generated(ids, index, token);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void commit_generated(Device device, int* ids, int* index, const int* token, int* pos) {
    switch (device) {
        case Device::CPU:
            commit_generated(ids, index, token, pos);
            return;
        case Device::HIP:
            rdna4::commit_generated(ids, index, token, pos);
            return;
    }
    throw std::logic_error("unhandled Device");
}

}  // namespace vesper
