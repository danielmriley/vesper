#pragma once

#include "vesper/buffer.h"
#include "vesper/config.h"
#include "vesper/types.h"
#include "vesper/weight.h"

#include <cstdint>
#include <vector>

namespace vesper {

struct LayerWeights {
    Buffer rms_attn;
    WeightMatrix q_proj;
    WeightMatrix k_proj;
    WeightMatrix v_proj;
    WeightMatrix o_proj;
    Buffer q_norm;
    Buffer k_norm;
    Buffer rms_mlp;
    WeightMatrix gate_proj;
    WeightMatrix up_proj;
    WeightMatrix down_proj;
    WeightMatrix qkv_proj;
    WeightMatrix z_proj;
    WeightMatrix beta_proj;
    WeightMatrix alpha_proj;
    WeightMatrix ssm_out;
    Buffer conv1d;
    Buffer ssm_dt;
    Buffer ssm_a;
    Buffer ssm_norm;
};

struct ModelWeights {
    ModelConfig config;
    WeightMatrix tok_emb;
    std::vector<LayerWeights> layers;
    Buffer final_norm;
    WeightMatrix lm_head;

    static ModelWeights random(const ModelConfig& config, std::uint32_t seed);
    ModelWeights to(Device device) const;
    ModelWeights to_q8() const;
    ModelWeights to_q4() const;
    ModelWeights dequant() const;
    Device device() const { return lm_head.device(); }
    std::size_t linear_bytes() const;
    const char* quant_name() const;
};

const char* weight_kind_name(WeightKind kind);

// Packed linear bytes for official Qwen3.8-27B Q4_K_M (no embeddings).
// Same number the compare table uses for llama.cpp and Vesper GB/s.
std::size_t qwen38_27b_q4km_linear_bytes();

}  // namespace vesper
