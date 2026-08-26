#include "vesper/kernels.h"

#include "vesper/q4k.h"
#include "vesper/q5k.h"
#include "vesper/q6k.h"
#include "vesper/q8.h"
#include "vesper/types.h"

#include <cmath>
#include <cstring>

namespace vesper {

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) {
        ss += x[i] * x[i];
    }
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] * inv * weight[i];
    }
}

void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim,
               int rotary_dim, int pos, float theta) {
    check(rotary_dim > 0 && (rotary_dim % 2) == 0 && rotary_dim <= head_dim,
          "rope rotary_dim must be even and <= head_dim");
    const int half = rotary_dim / 2;
    auto rotate = [&](float* vec, int n_heads) {
        for (int h = 0; h < n_heads; ++h) {
            float* head = vec + h * head_dim;
            for (int i = 0; i < half; ++i) {
                const float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) /
                                                              static_cast<float>(rotary_dim));
                const float angle = static_cast<float>(pos) * freq;
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const float x0 = head[i];
                const float x1 = head[i + half];
                head[i] = x0 * c - x1 * s;
                head[i + half] = x0 * s + x1 * c;
            }
        }
    };
    rotate(q, n_q_heads);
    rotate(k, n_kv_heads);
}

void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim,
               int pos, float theta) {
    rope_neox(q, k, n_q_heads, n_kv_heads, head_dim, head_dim, pos, theta);
}

void rope_neox_k_norm(float* q, float* k, const float* k_weight, int n_q_heads, int n_kv_heads,
                      int head_dim, int rotary_dim, int pos, float theta, float eps) {
    rmsnorm_rows(k, k_weight, n_kv_heads, head_dim, eps);
    rope_neox(q, k, n_q_heads, n_kv_heads, head_dim, rotary_dim, pos, theta);
}

void gemv(float* y, const float* weight, const float* x, int out_features,
          int in_features) {
    for (int i = 0; i < out_features; ++i) {
        const float* row = weight + static_cast<std::size_t>(i) * in_features;
        float acc = 0.0f;
        for (int j = 0; j < in_features; ++j) {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
}

void swiglu(float* out, const float* gate, const float* up, int n) {
    for (int i = 0; i < n; ++i) {
        const float g = gate[i];
        const float silu = g / (1.0f + std::exp(-g));
        out[i] = silu * up[i];
    }
}

void softmax_inplace(float* x, int n) {
    float max_v = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_v) {
            max_v = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_v);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        x[i] *= inv;
    }
}

int argmax(const float* x, int n) {
    int best = 0;
    float best_v = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > best_v) {
            best_v = x[i];
            best = i;
        }
    }
    return best;
}

void sigmoid_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        if (x[i] >= 0.0f) {
            const float z = std::exp(-x[i]);
            x[i] = 1.0f / (1.0f + z);
        } else {
            const float z = std::exp(x[i]);
            x[i] = z / (1.0f + z);
        }
    }
}

void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void softplus_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        if (x[i] > 20.0f) {
            continue;
        }
        if (x[i] < -20.0f) {
            x[i] = std::exp(x[i]);
            continue;
        }
        x[i] = std::log1p(std::exp(x[i]));
    }
}

void exp_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i]);
    }
}

void mul_inplace(float* dst, const float* src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] *= src[i];
    }
}

void scale_inplace(float* x, float scale, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] *= scale;
    }
}

void l2_normalize_rows(float* x, int rows, int dim, float eps) {
    for (int r = 0; r < rows; ++r) {
        float* row = x + r * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; ++i) {
            ss += row[i] * row[i];
        }
        const float inv = 1.0f / std::sqrt(ss + eps);
        for (int i = 0; i < dim; ++i) {
            row[i] *= inv;
        }
    }
}

void rmsnorm_rows(float* x, const float* weight, int rows, int dim, float eps) {
    check(rows >= 0 && dim > 0, "rmsnorm_rows empty shape");
    for (int r = 0; r < rows; ++r) {
        float* row = x + r * dim;
        rmsnorm(row, row, weight, dim, eps);
    }
}

void tile_heads(float* dst, const float* src, int n_dst, int n_src, int dim) {
    check(n_dst > 0 && n_src > 0 && dim > 0, "tile_heads empty shape");
    for (int d = 0; d < n_dst; ++d) {
        const int s = d % n_src;
        std::memcpy(dst + d * dim, src + s * dim, static_cast<std::size_t>(dim) * sizeof(float));
    }
}

void attn_decode(float* out, float* scores, const float* q, const float* k, const float* v,
                 int seq, int n_q_heads, int n_kv_heads, int head_dim) {
    attn_decode(out, scores, q, k, v, nullptr, seq, n_q_heads, n_kv_heads, head_dim);
}

void attn_decode(float* out, float* scores, const float* q, const float* k, const float* v,
                 const float* gate, int seq, int n_q_heads, int n_kv_heads, int head_dim) {
    check(seq > 0 && n_q_heads > 0 && n_kv_heads > 0 && head_dim > 0, "attn_decode empty shape");
    check(n_q_heads % n_kv_heads == 0, "attn_decode GQA");
    const int group = n_q_heads / n_kv_heads;
    for (int qh = 0; qh < n_q_heads; ++qh) {
        const int kvh = qh / group;
        attn_scores(scores, q + qh * head_dim, k, seq, n_kv_heads, kvh, head_dim);
        softmax_inplace(scores, seq);
        attn_mix(out + qh * head_dim, scores, v, seq, n_kv_heads, kvh, head_dim);
    }
    if (gate != nullptr) {
        const int n = n_q_heads * head_dim;
        for (int i = 0; i < n; ++i) {
            const float g = gate[i];
            const float s = (g >= 0.0f) ? (1.0f / (1.0f + std::exp(-g)))
                                        : (std::exp(g) / (1.0f + std::exp(g)));
            out[i] *= s;
        }
    }
}

void gemv_swiglu(float* hidden, float* gate_tmp, float* up_tmp, const WeightMatrix& gate,
                 const WeightMatrix& up, const float* x) {
    check(gate.rows() == up.rows() && gate.cols() == up.cols(), "gemv_swiglu shape mismatch");
    gemv(gate_tmp, gate, x);
    gemv(up_tmp, up, x);
    swiglu(hidden, gate_tmp, up_tmp, gate.rows());
}

void gemv_add(float* y, const WeightMatrix& weight, const float* x, const float* addend) {
    gemv(y, weight, x);
    for (int i = 0; i < weight.rows(); ++i) {
        y[i] += addend[i];
    }
}

void gemv3(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1, float* y2,
           const WeightMatrix& w2, const float* x) {
    gemv(y0, w0, x);
    gemv(y1, w1, x);
    gemv(y2, w2, x);
}

void gemv4(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1, float* y2,
           const WeightMatrix& w2, float* y3, const WeightMatrix& w3, const float* x) {
    gemv(y0, w0, x);
    gemv(y1, w1, x);
    gemv(y2, w2, x);
    gemv(y3, w3, x);
}

void tile_l2_scale(float* dst, const float* src, int n_dst, int n_src, int dim, float eps,
                   float scale) {
    tile_heads(dst, src, n_dst, n_src, dim);
    l2_normalize_rows(dst, n_dst, dim, eps);
    if (scale != 1.0f) {
        scale_inplace(dst, scale, n_dst * dim);
    }
}

void tile_l2_pair(float* q_dst, const float* q_src, float* k_dst, const float* k_src, int n_dst,
                  int n_src, int dim, float eps, float q_scale, float k_scale) {
    tile_l2_scale(q_dst, q_src, n_dst, n_src, dim, eps, q_scale);
    tile_l2_scale(k_dst, k_src, n_dst, n_src, dim, eps, k_scale);
}

void add_rmsnorm(float* x, float* residual, const float* weight, int n, float eps) {
    for (int i = 0; i < n; ++i) {
        residual[i] += x[i];
    }
    rmsnorm(x, residual, weight, n, eps);
}

void copy_rmsnorm(float* x, float* residual, const float* weight, int n, float eps) {
    for (int i = 0; i < n; ++i) {
        residual[i] = x[i];
    }
    rmsnorm(x, residual, weight, n, eps);
}

void silu_mul(float* y, const float* z, int n) {
    for (int i = 0; i < n; ++i) {
        const float g = z[i];
        y[i] *= g / (1.0f + std::exp(-g));
    }
}

void gdn_gates(float* decay, float* beta, const float* alpha, const float* dt, const float* a,
               int n) {
    for (int i = 0; i < n; ++i) {
        float t = alpha[i] + dt[i];
        if (t > 20.0f) {
            // softplus(t) ~= t
        } else if (t < -20.0f) {
            t = std::exp(t);
        } else {
            t = std::log1p(std::exp(t));
        }
        decay[i] = std::exp(a[i] * t);
        const float b = beta[i];
        if (b >= 0.0f) {
            beta[i] = 1.0f / (1.0f + std::exp(-b));
        } else {
            const float z = std::exp(b);
            beta[i] = z / (1.0f + z);
        }
    }
}

void split_qkv(float* q, float* k, float* v, const float* qkv, int key_dim, int value_dim) {
    check(key_dim > 0 && value_dim > 0, "split_qkv empty shape");
    std::memcpy(q, qkv, static_cast<std::size_t>(key_dim) * sizeof(float));
    std::memcpy(k, qkv + key_dim, static_cast<std::size_t>(key_dim) * sizeof(float));
    std::memcpy(v, qkv + 2 * key_dim, static_cast<std::size_t>(value_dim) * sizeof(float));
}

void split_gated_q(float* q, float* gate, const float* q_full, int n_heads, int head_dim) {
    check(n_heads > 0 && head_dim > 0, "split_gated_q empty shape");
    for (int h = 0; h < n_heads; ++h) {
        const float* src = q_full + h * 2 * head_dim;
        for (int i = 0; i < head_dim; ++i) {
            q[h * head_dim + i] = src[i];
            gate[h * head_dim + i] = src[head_dim + i];
        }
    }
}

void split_gated_q_norm(float* q, float* gate, const float* q_full, const float* weight, int n_heads,
                        int head_dim, float eps) {
    split_gated_q(q, gate, q_full, n_heads, head_dim);
    rmsnorm_rows(q, weight, n_heads, head_dim, eps);
}

void rmsnorm_silu_mul(float* y, const float* z, const float* weight, int rows, int dim, float eps) {
    rmsnorm_rows(y, weight, rows, dim, eps);
    silu_mul(y, z, rows * dim);
}

void embed_row(float* out, const float* table, int token, int hidden) {
    const float* row = table + static_cast<std::size_t>(token) * hidden;
    for (int i = 0; i < hidden; ++i) {
        out[i] = row[i];
    }
}

void embed_row(float* out, const WeightMatrix& table, int token) {
    check(token >= 0 && token < table.rows(), "embed token out of range");
    check(table.device() == Device::CPU, "CPU embed needs a CPU table");
    const int cols = table.cols();
    switch (table.kind()) {
        case WeightKind::F32:
            embed_row(out, table.f32_data(), token, cols);
            return;
        case WeightKind::Q8_0:
            dequant_q8_row(out, table.packed(), token, cols);
            return;
        case WeightKind::Q4_K:
            dequant_q4k_row(out, table.packed(), token, cols);
            return;
        case WeightKind::Q5_K:
            dequant_q5k_row(out, table.packed(), token, cols);
            return;
        case WeightKind::Q6_K:
            dequant_q6k_row(out, table.packed(), token, cols);
            return;
    }
    throw std::logic_error("unhandled WeightKind");
}

void attn_scores(float* scores, const float* q, const float* k, int seq,
                 int n_kv_heads, int kv_head, int head_dim) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int t = 0; t < seq; ++t) {
        const float* key = k + (static_cast<std::size_t>(t) * n_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            dot += q[i] * key[i];
        }
        scores[t] = dot * scale;
    }
}

void attn_mix(float* out, const float* scores, const float* v, int seq,
              int n_kv_heads, int kv_head, int head_dim) {
    for (int i = 0; i < head_dim; ++i) {
        out[i] = 0.0f;
    }
    for (int t = 0; t < seq; ++t) {
        const float* val = v + (static_cast<std::size_t>(t) * n_kv_heads + kv_head) * head_dim;
        const float w = scores[t];
        for (int i = 0; i < head_dim; ++i) {
            out[i] += w * val[i];
        }
    }
}

}  // namespace vesper
