#pragma once

#include "vesper/buffer.h"
#include "vesper/config.h"
#include "vesper/types.h"

#include <cstdint>
#include <vector>

namespace vesper {

struct LayerWeights {
    Buffer rms_attn;
    Buffer q_proj;
    Buffer k_proj;
    Buffer v_proj;
    Buffer o_proj;
    Buffer q_norm;
    Buffer k_norm;
    Buffer rms_mlp;
    Buffer gate_proj;
    Buffer up_proj;
    Buffer down_proj;
};

struct ModelWeights {
    ModelConfig config;
    Buffer tok_emb;
    std::vector<LayerWeights> layers;
    Buffer final_norm;
    Buffer lm_head;

    static ModelWeights random(const ModelConfig& config, std::uint32_t seed);
    ModelWeights to(Device device) const;
    Device device() const { return tok_emb.device(); }
};

}  // namespace vesper
