#include "vesper/kv_cache.h"

#include "vesper/types.h"

namespace vesper {

KVCache KVCache::create(const ModelConfig& config) {
    config.validate();
    KVCache cache;
    cache.config = config;
    const std::size_t elems =
        static_cast<std::size_t>(config.max_seq_len) * config.kv_dim();
    cache.k.reserve(static_cast<std::size_t>(config.n_layers));
    cache.v.reserve(static_cast<std::size_t>(config.n_layers));
    for (int i = 0; i < config.n_layers; ++i) {
        cache.k.emplace_back(elems, Device::CPU);
        cache.v.emplace_back(elems, Device::CPU);
    }
    cache.pos = 0;
    return cache;
}

void KVCache::reset() {
    pos = 0;
}

float* KVCache::k_at(int layer, int position) {
    check(layer >= 0 && layer < static_cast<int>(k.size()), "K layer out of range");
    check(position >= 0 && position < config.max_seq_len, "K position out of range");
    return k[static_cast<std::size_t>(layer)].data() +
           static_cast<std::size_t>(position) * config.kv_dim();
}

float* KVCache::v_at(int layer, int position) {
    check(layer >= 0 && layer < static_cast<int>(v.size()), "V layer out of range");
    check(position >= 0 && position < config.max_seq_len, "V position out of range");
    return v[static_cast<std::size_t>(layer)].data() +
           static_cast<std::size_t>(position) * config.kv_dim();
}

const float* KVCache::k_at(int layer, int position) const {
    check(layer >= 0 && layer < static_cast<int>(k.size()), "K layer out of range");
    check(position >= 0 && position < config.max_seq_len, "K position out of range");
    return k[static_cast<std::size_t>(layer)].data() +
           static_cast<std::size_t>(position) * config.kv_dim();
}

const float* KVCache::v_at(int layer, int position) const {
    check(layer >= 0 && layer < static_cast<int>(v.size()), "V layer out of range");
    check(position >= 0 && position < config.max_seq_len, "V position out of range");
    return v[static_cast<std::size_t>(layer)].data() +
           static_cast<std::size_t>(position) * config.kv_dim();
}

}  // namespace vesper
