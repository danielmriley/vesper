#include "vesper/kv_cache.h"

#include "vesper/types.h"

namespace vesper {

KVCache KVCache::create(const ModelConfig& config, Device device) {
    config.validate();
    KVCache cache;
    cache.config = config;
    cache.k.reserve(static_cast<std::size_t>(config.n_layers));
    cache.v.reserve(static_cast<std::size_t>(config.n_layers));
    cache.rec.reserve(static_cast<std::size_t>(config.n_layers));
    cache.conv.reserve(static_cast<std::size_t>(config.n_layers));
    const std::size_t kv_elems =
        static_cast<std::size_t>(config.max_seq_len) * static_cast<std::size_t>(config.kv_dim());
    for (int i = 0; i < config.n_layers; ++i) {
        switch (config.layer_kind(i)) {
            case LayerKind::Attention:
                cache.k.emplace_back(kv_elems, device);
                cache.v.emplace_back(kv_elems, device);
                cache.rec.emplace_back(0, device);
                cache.conv.emplace_back(0, device);
                break;
            case LayerKind::DeltaNet:
                cache.k.emplace_back(0, device);
                cache.v.emplace_back(0, device);
                cache.rec.emplace_back(static_cast<std::size_t>(config.gdn_rec_elems()), device);
                cache.conv.emplace_back(static_cast<std::size_t>(config.gdn_conv_state_elems()),
                                        device);
                break;
        }
    }
    cache.pos = 0;
    return cache;
}

void KVCache::reset() {
    pos = 0;
    for (int i = 0; i < config.n_layers; ++i) {
        if (config.layer_kind(i) != LayerKind::DeltaNet) {
            continue;
        }
        rec[static_cast<std::size_t>(i)].fill(0.0f);
        conv[static_cast<std::size_t>(i)].fill(0.0f);
    }
}

float* KVCache::k_at(int layer, int position) {
    check(layer >= 0 && layer < static_cast<int>(k.size()), "K layer out of range");
    check(k[static_cast<std::size_t>(layer)].size() > 0, "K lookup on a DeltaNet layer");
    check(position >= 0 && position < config.max_seq_len, "K position out of range");
    return k[static_cast<std::size_t>(layer)].data() +
           static_cast<std::size_t>(position) * config.kv_dim();
}

float* KVCache::v_at(int layer, int position) {
    check(layer >= 0 && layer < static_cast<int>(v.size()), "V layer out of range");
    check(v[static_cast<std::size_t>(layer)].size() > 0, "V lookup on a DeltaNet layer");
    check(position >= 0 && position < config.max_seq_len, "V position out of range");
    return v[static_cast<std::size_t>(layer)].data() +
           static_cast<std::size_t>(position) * config.kv_dim();
}

const float* KVCache::k_at(int layer, int position) const {
    check(layer >= 0 && layer < static_cast<int>(k.size()), "K layer out of range");
    check(k[static_cast<std::size_t>(layer)].size() > 0, "K lookup on a DeltaNet layer");
    check(position >= 0 && position < config.max_seq_len, "K position out of range");
    return k[static_cast<std::size_t>(layer)].data() +
           static_cast<std::size_t>(position) * config.kv_dim();
}

const float* KVCache::v_at(int layer, int position) const {
    check(layer >= 0 && layer < static_cast<int>(v.size()), "V layer out of range");
    check(v[static_cast<std::size_t>(layer)].size() > 0, "V lookup on a DeltaNet layer");
    check(position >= 0 && position < config.max_seq_len, "V position out of range");
    return v[static_cast<std::size_t>(layer)].data() +
           static_cast<std::size_t>(position) * config.kv_dim();
}

float* KVCache::rec_at(int layer) {
    check(layer >= 0 && layer < static_cast<int>(rec.size()), "rec layer out of range");
    check(rec[static_cast<std::size_t>(layer)].size() > 0, "rec lookup on an Attention layer");
    return rec[static_cast<std::size_t>(layer)].data();
}

float* KVCache::conv_at(int layer) {
    check(layer >= 0 && layer < static_cast<int>(conv.size()), "conv layer out of range");
    check(conv[static_cast<std::size_t>(layer)].size() > 0, "conv lookup on an Attention layer");
    return conv[static_cast<std::size_t>(layer)].data();
}

}  // namespace vesper
