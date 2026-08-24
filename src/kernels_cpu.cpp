#include "vesper/kernels.h"

#include "vesper/types.h"

#include <cmath>

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
               int pos, float theta) {
    const int half = head_dim / 2;
    auto rotate = [&](float* vec, int n_heads) {
        for (int h = 0; h < n_heads; ++h) {
            float* head = vec + h * head_dim;
            for (int i = 0; i < half; ++i) {
                const float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) /
                                                              static_cast<float>(head_dim));
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

void embed_row(float* out, const float* table, int token, int hidden) {
    const float* row = table + static_cast<std::size_t>(token) * hidden;
    for (int i = 0; i < hidden; ++i) {
        out[i] = row[i];
    }
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
