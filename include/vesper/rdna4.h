#pragma once

#include "vesper/weight.h"

#include <cstddef>

// gfx1201 (RDNA 4 / R9700) kernels. Pointers are device pointers.
// The CPU twins in kernels.h are the numeric gate.

namespace vesper {
namespace rdna4 {

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps);
void rmsnorm_rows(float* x, const float* weight, int rows, int dim, float eps);
void split_gated_q(float* q, float* gate, const float* q_full, int n_heads, int head_dim);
void split_gated_q_norm(float* q, float* gate, const float* q_full, const float* weight, int n_heads,
                        int head_dim, float eps);
void rmsnorm_silu_mul(float* y, const float* z, const float* weight, int rows, int dim, float eps);
void gdn_conv_split(float* q, float* k, float* v, float* state, const float* x, const float* weight,
                    int key_dim, int value_dim, int kernel);
void tile_heads(float* dst, const float* src, int n_dst, int n_src, int dim);
void attn_decode(float* out, const float* q, const float* k, const float* v, const float* gate,
                 int seq, int n_q_heads, int n_kv_heads, int head_dim);
void attn_decode(float* out, const float* q, const float* k, const float* v, const float* gate,
                 const int* pos, int n_q_heads, int n_kv_heads, int head_dim);
void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim,
               int rotary_dim, int pos, float theta);
void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
               const int* pos, float theta);
void rope_neox_k_norm(float* q, float* k, const float* k_weight, int n_q_heads, int n_kv_heads,
                      int head_dim, int rotary_dim, int pos, float theta, float eps);
void rope_neox_k_norm(float* q, float* k, const float* k_weight, int n_q_heads, int n_kv_heads,
                      int head_dim, int rotary_dim, const int* pos, float theta, float eps);
void gemv(float* y, const float* weight, const float* x, int out_features,
          int in_features, const float* addend = nullptr);
void gemv_q8(float* y, const std::byte* packed, const float* x, int rows, int cols,
             const float* addend = nullptr);
void gemv_q4k(float* y, const std::byte* packed, const float* x, int rows, int cols,
              const float* addend = nullptr);
void gemv_q5k(float* y, const std::byte* packed, const float* x, int rows, int cols,
              const float* addend = nullptr);
void gemv_q6k(float* y, const std::byte* packed, const float* x, int rows, int cols,
              const float* addend = nullptr);
// Packed kinds launch one MMVQ grid after a single Q8_1 quantize of x.
void gemv3(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1, float* y2,
           const WeightMatrix& w2, const float* x);
void gemv4(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1, float* y2,
           const WeightMatrix& w2, float* y3, const WeightMatrix& w3, const float* x);
void tile_l2_scale(float* dst, const float* src, int n_dst, int n_src, int dim, float eps,
                   float scale);
void tile_l2_pair(float* q_dst, const float* q_src, float* k_dst, const float* k_src, int n_dst,
                  int n_src, int dim, float eps, float q_scale, float k_scale);
void gemv_swiglu(float* hidden, const WeightMatrix& gate, const WeightMatrix& up, const float* x);
void add_rmsnorm(float* x, float* residual, const float* weight, int n, float eps);
void copy_rmsnorm(float* x, float* residual, const float* weight, int n, float eps);
void silu_mul(float* y, const float* z, int n);
void gdn_gates(float* decay, float* beta, const float* alpha, const float* dt, const float* a,
               int n);
void split_qkv(float* q, float* k, float* v, const float* qkv, int key_dim, int value_dim);
void swiglu(float* out, const float* gate, const float* up, int n);
void softmax_inplace(float* x, int n);
void sigmoid_inplace(float* x, int n);
void silu_inplace(float* x, int n);
void softplus_inplace(float* x, int n);
void exp_inplace(float* x, int n);
void mul_inplace(float* dst, const float* src, int n);
void scale_inplace(float* x, float scale, int n);
void l2_normalize_rows(float* x, int rows, int dim, float eps);
void embed_row(float* out, const float* table, int token, int hidden);
void embed_row(float* out, const float* table, const int* token, int hidden);
void embed_row(float* out, const WeightMatrix& table, int token);
void embed_row(float* out, const WeightMatrix& table, const int* token);
void scatter_row(float* base, const float* row, const int* pos, int n);
void scatter_kv(float* k_base, float* v_base, const float* k, const float* v, const int* pos, int n);
void attn_prepare(float* q, float* gate, float* k, float* v, const float* q_full,
                  const float* q_weight, const float* k_weight, float* k_base, float* v_base,
                  const int* pos, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
                  float theta, float eps);
void gdn_tile_gates(float* q_dst, const float* q_src, float* k_dst, const float* k_src,
                    float* decay, float* beta, const float* alpha, const float* dt, const float* a,
                    int n_dst, int n_src, int dim, float eps, float q_scale, float k_scale);
void attn_scores(float* scores, const float* q, const float* k, int seq,
                 int n_kv_heads, int kv_head, int head_dim);
void attn_mix(float* out, const float* scores, const float* v, int seq,
              int n_kv_heads, int kv_head, int head_dim);
void add_inplace(float* dst, const float* src, int n);
void copy(float* dst, const float* src, int n);
void gdn_conv_update(float* y, float* state, const float* x, const float* weight,
                     int conv_dim, int kernel);
void gdn_delta_rule(float* y, float* rec, const float* q, const float* k, const float* v,
                    const float* decay, const float* beta, int n_heads, int dim);
int argmax(const float* x, int n);
void argmax_write(int* dst, const float* x, int n);
void seed_generated(int* ids, int* index, const int* token);
void commit_generated(int* ids, int* index, const int* token, int* pos);
void argmax_write_commit(int* ids, int* index, int* token, int* pos, const float* x, int n);
// Size Q8_1 x and argmax scratch. Graph capture cannot malloc or synchronize.
void warmup_decode(int max_cols, int vocab);

}  // namespace rdna4
}  // namespace vesper
