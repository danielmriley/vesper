#pragma once

#include "vesper/types.h"
#include "vesper/weight.h"

namespace vesper {

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps);
void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim,
               int pos, float theta);
void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim,
               int rotary_dim, int pos, float theta);
void rope_neox_k_norm(float* q, float* k, const float* k_weight, int n_q_heads, int n_kv_heads,
                      int head_dim, int rotary_dim, int pos, float theta, float eps);
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
void split_gated_q_norm(float* q, float* gate, const float* q_full, const float* weight, int n_heads,
                        int head_dim, float eps);
void rmsnorm_rows(float* x, const float* weight, int rows, int dim, float eps);
void rmsnorm_silu_mul(float* y, const float* z, const float* weight, int rows, int dim, float eps);
void tile_heads(float* dst, const float* src, int n_dst, int n_src, int dim);
void attn_decode(float* out, float* scores, const float* q, const float* k, const float* v,
                 int seq, int n_q_heads, int n_kv_heads, int head_dim);
void attn_decode(float* out, float* scores, const float* q, const float* k, const float* v,
                 const float* gate, int seq, int n_q_heads, int n_kv_heads, int head_dim);
void gemv_swiglu(float* hidden, float* gate_tmp, float* up_tmp, const WeightMatrix& gate,
                 const WeightMatrix& up, const float* x);
void gemv_add(float* y, const WeightMatrix& weight, const float* x, const float* addend);
void gemv3(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1, float* y2,
           const WeightMatrix& w2, const float* x);
void gemv4(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1, float* y2,
           const WeightMatrix& w2, float* y3, const WeightMatrix& w3, const float* x);
void tile_l2_scale(float* dst, const float* src, int n_dst, int n_src, int dim, float eps,
                   float scale);
void tile_l2_pair(float* q_dst, const float* q_src, float* k_dst, const float* k_src, int n_dst,
                  int n_src, int dim, float eps, float q_scale, float k_scale);
void add_rmsnorm(float* x, float* residual, const float* weight, int n, float eps);
void copy_rmsnorm(float* x, float* residual, const float* weight, int n, float eps);
void silu_mul(float* y, const float* z, int n);
void gdn_gates(float* decay, float* beta, const float* alpha, const float* dt, const float* a,
               int n);
void split_qkv(float* q, float* k, float* v, const float* qkv, int key_dim, int value_dim);

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
void rope_neox_k_norm(Device device, float* q, float* k, const float* k_weight, int n_q_heads,
                      int n_kv_heads, int head_dim, int rotary_dim, int pos, float theta,
                      float eps);
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
void split_gated_q_norm(Device device, float* q, float* gate, const float* q_full, const float* weight,
                        int n_heads, int head_dim, float eps);
void rmsnorm_rows(Device device, float* x, const float* weight, int rows, int dim, float eps);
void rmsnorm_silu_mul(Device device, float* y, const float* z, const float* weight, int rows, int dim,
                      float eps);
void tile_heads(Device device, float* dst, const float* src, int n_dst, int n_src, int dim);
void attn_decode(Device device, float* out, float* scores, const float* q, const float* k,
                 const float* v, int seq, int n_q_heads, int n_kv_heads, int head_dim);
void attn_decode(Device device, float* out, float* scores, const float* q, const float* k,
                 const float* v, const float* gate, int seq, int n_q_heads, int n_kv_heads,
                 int head_dim);
void gemv_swiglu(Device device, float* hidden, float* gate_tmp, float* up_tmp,
                 const WeightMatrix& gate, const WeightMatrix& up, const float* x);
void gemv_add(Device device, float* y, const WeightMatrix& weight, const float* x,
              const float* addend);
void gemv3(Device device, float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1,
           float* y2, const WeightMatrix& w2, const float* x);
void gemv4(Device device, float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1,
           float* y2, const WeightMatrix& w2, float* y3, const WeightMatrix& w3, const float* x);
void tile_l2_scale(Device device, float* dst, const float* src, int n_dst, int n_src, int dim,
                   float eps, float scale);
void tile_l2_pair(Device device, float* q_dst, const float* q_src, float* k_dst, const float* k_src,
                  int n_dst, int n_src, int dim, float eps, float q_scale, float k_scale);
void add_rmsnorm(Device device, float* x, float* residual, const float* weight, int n, float eps);
void copy_rmsnorm(Device device, float* x, float* residual, const float* weight, int n, float eps);
void silu_mul(Device device, float* y, const float* z, int n);
void gdn_gates(Device device, float* decay, float* beta, const float* alpha, const float* dt,
               const float* a, int n);
void split_qkv(Device device, float* q, float* k, float* v, const float* qkv, int key_dim,
               int value_dim);
void attn_scores(Device device, float* scores, const float* q, const float* k, int seq,
                 int n_kv_heads, int kv_head, int head_dim);
void attn_mix(Device device, float* out, const float* scores, const float* v, int seq,
              int n_kv_heads, int kv_head, int head_dim);
void add_inplace(Device device, float* dst, const float* src, int n);
void copy_vec(Device device, float* dst, const float* src, int n);

}  // namespace vesper
