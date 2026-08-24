#include "vesper/weights.h"

#include "vesper/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vesper {
namespace {

class Lcg {
public:
    explicit Lcg(std::uint32_t seed) : state_(seed ? seed : 1u) {}

    float uniform() {
        state_ = state_ * 1664525u + 1013904223u;
        return static_cast<float>(state_ >> 8) / static_cast<float>(1u << 24);
    }

    float normal() {
        const float u1 = std::max(uniform(), 1e-7f);
        const float u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
    }

private:
    std::uint32_t state_;
};

void fill_normal(Buffer* buf, Lcg* rng, float scale) {
    float* p = buf->data();
    for (std::size_t i = 0; i < buf->size(); ++i) {
        p[i] = rng->normal() * scale;
    }
}

void fill_ones(Buffer* buf) {
    buf->fill(1.0f);
}

}  // namespace

ModelWeights ModelWeights::random(const ModelConfig& config, std::uint32_t seed) {
    config.validate();
    ModelWeights w;
    w.config = config;

    const Device cpu = Device::CPU;
    const int h = config.hidden_size;
    const int q = config.q_dim();
    const int kv = config.kv_dim();
    const int inter = config.intermediate_size;
    const int v = config.vocab_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(h));
    Lcg rng(seed);

    w.tok_emb = Buffer(static_cast<std::size_t>(v) * h, cpu);
    fill_normal(&w.tok_emb, &rng, scale);

    w.layers.reserve(static_cast<std::size_t>(config.n_layers));
    for (int i = 0; i < config.n_layers; ++i) {
        LayerWeights layer;
        layer.rms_attn = Buffer(static_cast<std::size_t>(h), cpu);
        layer.q_proj = Buffer(static_cast<std::size_t>(q) * h, cpu);
        layer.k_proj = Buffer(static_cast<std::size_t>(kv) * h, cpu);
        layer.v_proj = Buffer(static_cast<std::size_t>(kv) * h, cpu);
        layer.o_proj = Buffer(static_cast<std::size_t>(h) * q, cpu);
        layer.q_norm = Buffer(static_cast<std::size_t>(config.head_dim), cpu);
        layer.k_norm = Buffer(static_cast<std::size_t>(config.head_dim), cpu);
        layer.rms_mlp = Buffer(static_cast<std::size_t>(h), cpu);
        layer.gate_proj = Buffer(static_cast<std::size_t>(inter) * h, cpu);
        layer.up_proj = Buffer(static_cast<std::size_t>(inter) * h, cpu);
        layer.down_proj = Buffer(static_cast<std::size_t>(h) * inter, cpu);
        fill_ones(&layer.rms_attn);
        fill_ones(&layer.rms_mlp);
        fill_ones(&layer.q_norm);
        fill_ones(&layer.k_norm);
        fill_normal(&layer.q_proj, &rng, scale);
        fill_normal(&layer.k_proj, &rng, scale);
        fill_normal(&layer.v_proj, &rng, scale);
        fill_normal(&layer.o_proj, &rng, scale);
        fill_normal(&layer.gate_proj, &rng, scale);
        fill_normal(&layer.up_proj, &rng, scale);
        fill_normal(&layer.down_proj, &rng, scale);
        w.layers.push_back(std::move(layer));
    }

    w.final_norm = Buffer(static_cast<std::size_t>(h), cpu);
    fill_ones(&w.final_norm);
    w.lm_head = Buffer(static_cast<std::size_t>(v) * h, cpu);
    if (config.tie_word_embeddings) {
        w.lm_head.copy_from(w.tok_emb.data(), w.tok_emb.size());
    } else {
        fill_normal(&w.lm_head, &rng, scale);
    }
    return w;
}

ModelWeights ModelWeights::to(Device device) const {
    if (this->device() == device) {
        return *this;
    }
    ModelWeights w;
    w.config = config;
    w.tok_emb = tok_emb.to(device);
    w.final_norm = final_norm.to(device);
    w.lm_head = lm_head.to(device);
    w.layers.reserve(layers.size());
    for (const LayerWeights& layer : layers) {
        LayerWeights out;
        out.rms_attn = layer.rms_attn.to(device);
        out.q_proj = layer.q_proj.to(device);
        out.k_proj = layer.k_proj.to(device);
        out.v_proj = layer.v_proj.to(device);
        out.o_proj = layer.o_proj.to(device);
        out.q_norm = layer.q_norm.to(device);
        out.k_norm = layer.k_norm.to(device);
        out.rms_mlp = layer.rms_mlp.to(device);
        out.gate_proj = layer.gate_proj.to(device);
        out.up_proj = layer.up_proj.to(device);
        out.down_proj = layer.down_proj.to(device);
        w.layers.push_back(std::move(out));
    }
    return w;
}

}  // namespace vesper
