#pragma once

namespace vesper {

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps);
void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim,
               int pos, float theta);
void gemv(float* y, const float* weight, const float* x, int out_features,
          int in_features);
void swiglu(float* out, const float* gate, const float* up, int n);
void softmax_inplace(float* x, int n);
int argmax(const float* x, int n);
void embed_row(float* out, const float* table, int token, int hidden);

// scores[t] = dot(q, k[t]) / sqrt(head_dim)
void attn_scores(float* scores, const float* q, const float* k, int seq,
                 int n_kv_heads, int kv_head, int head_dim);
void attn_mix(float* out, const float* scores, const float* v, int seq,
              int n_kv_heads, int kv_head, int head_dim);

}  // namespace vesper
