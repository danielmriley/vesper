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
};

struct ModelWeights {
    ModelConfig config;
    Buffer tok_emb;
    std::vector<LayerWeights> layers;
    Buffer final_norm;
    WeightMatrix lm_head;

    static ModelWeights random(const ModelConfig& config, std::uint32_t seed);
    ModelWeights to(Device device) const;
    ModelWeights to_q8() const;
    ModelWeights dequant() const;
    Device device() const { return tok_emb.device(); }
    std::size_t linear_bytes() const;
};

}  // namespace vesper
