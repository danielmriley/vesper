#pragma once

#include "vesper/buffer.h"
#include "vesper/config.h"
#include "vesper/weights.h"

namespace vesper {

struct GdnScratch {
    Buffer qkv;
    Buffer z;
    Buffer beta;
    Buffer alpha;
    Buffer conv_y;
    Buffer q;
    Buffer k;
    Buffer v;
    Buffer q_rep;
    Buffer k_rep;
    Buffer decay;
    Buffer y;
};

void gdn_scratch_init(GdnScratch* scratch, const ModelConfig& cfg, Device device);

void gdn_conv_update(Device device, float* y, float* state, const float* x, const float* weight,
                     int conv_dim, int kernel);
void gdn_conv_split(Device device, float* q, float* k, float* v, float* conv_y, float* state,
                    const float* x, const float* weight, int key_dim, int value_dim, int kernel);

// rec is [n_v][dv][dk], column-contiguous. HIP keeps each column in registers.
void gdn_delta_rule(Device device, float* y, float* rec, const float* q, const float* k,
                    const float* v, const float* decay, const float* beta, int n_heads, int dim);
void gdn_delta_rmsnorm_silu(Device device, float* y, float* rec, const float* q, const float* k,
                            const float* v, const float* decay, const float* beta, const float* z,
                            const float* weight, int n_heads, int dim, float eps);

void gdn_layer(Device device, float* y, const float* x, const LayerWeights& layer,
               const ModelConfig& cfg, float* rec, float* conv, GdnScratch* scratch);

}  // namespace vesper
