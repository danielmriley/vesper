#pragma once

#include "vesper/buffer.h"
#include "vesper/config.h"
#include "vesper/types.h"

#include <vector>

namespace vesper {

struct KVCache {
    ModelConfig config;
    std::vector<Buffer> k;
    std::vector<Buffer> v;
    int pos = 0;

    static KVCache create(const ModelConfig& config, Device device = Device::CPU);
    void reset();

    float* k_at(int layer, int position);
    float* v_at(int layer, int position);
    const float* k_at(int layer, int position) const;
    const float* v_at(int layer, int position) const;
};

}  // namespace vesper
