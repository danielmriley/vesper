#pragma once

#include "vesper/types.h"
#include "vesper/weight.h"

namespace vesper {

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps);
void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim,
               int pos, float theta);
void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim,
               int rotary_dim, int pos, float theta);
void gemv(float* y, const float* weight, const float* x, int out_features,
          int in_features);
void gemv(float* y, const WeightMatrix& weight, const float* x);
void swiglu(float* out, const float* gate, const float* up, int n);
void softmax_inplace(float* x, int n);
void sigmoid_inplace(float* x, int n);
void silu_inplace(float* x, int n);
void softplus_inplace(float* x, int n);
void exp_inplace(float* x, int n);
void mul_inplace(float* dst, const float* src, int n);
void scale_inplace(float* x, float scale, int n);
void l2_normalize_rows(float* x, int rows, int dim, float eps);
int argmax(const float* x, int n);
void embed_row(float* out, const float* table, int token, int hidden);
void embed_row(float* out, const WeightMatrix& table, int token);
void split_gated_q(float* q, float* gate, const float* q_full, int n_heads, int head_dim);

// scores[t] = dot(q, k[t]) / sqrt(head_dim)
void attn_scores(float* scores, const float* q, const float* k, int seq,
                 int n_kv_heads, int kv_head, int head_dim);
void attn_mix(float* out, const float* scores, const float* v, int seq,
              int n_kv_heads, int kv_head, int head_dim);

void rmsnorm(Device device, float* out, const float* x, const float* weight, int n,
             float eps);
void rope_neox(Device device, float* q, float* k, int n_q_heads, int n_kv_heads,
               int head_dim, int pos, float theta);
void rope_neox(Device device, float* q, float* k, int n_q_heads, int n_kv_heads,
               int head_dim, int rotary_dim, int pos, float theta);
void gemv(Device device, float* y, const float* weight, const float* x,
          int out_features, int in_features);
void gemv(Device device, float* y, const WeightMatrix& weight, const float* x);
void swiglu(Device device, float* out, const float* gate, const float* up, int n);
void softmax_inplace(Device device, float* x, int n);
void sigmoid_inplace(Device device, float* x, int n);
void silu_inplace(Device device, float* x, int n);
void softplus_inplace(Device device, float* x, int n);
void exp_inplace(Device device, float* x, int n);
void mul_inplace(Device device, float* dst, const float* src, int n);
void scale_inplace(Device device, float* x, float scale, int n);
void l2_normalize_rows(Device device, float* x, int rows, int dim, float eps);
void embed_row(Device device, float* out, const float* table, int token, int hidden);
void embed_row(Device device, float* out, const WeightMatrix& table, int token);
void split_gated_q(Device device, float* q, float* gate, const float* q_full, int n_heads,
                   int head_dim);
void attn_scores(Device device, float* scores, const float* q, const float* k, int seq,
                 int n_kv_heads, int kv_head, int head_dim);
void attn_mix(Device device, float* out, const float* scores, const float* v, int seq,
              int n_kv_heads, int kv_head, int head_dim);
void add_inplace(Device device, float* dst, const float* src, int n);
void copy_vec(Device device, float* dst, const float* src, int n);

}  // namespace vesper
