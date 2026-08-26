#include "vesper/weights.h"

#include "vesper/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

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

WeightMatrix take_f32(Buffer&& buf, int rows, int cols) {
    return WeightMatrix::from_f32(std::move(buf), rows, cols);
}

WeightMatrix as_q8(const WeightMatrix& w) {
    switch (w.kind()) {
        case WeightKind::F32:
            check(w.device() == Device::CPU, "to_q8 needs CPU F32 weights");
            return WeightMatrix::q8_from_f32(w.f32_data(), w.rows(), w.cols());
        case WeightKind::Q8_0:
            check(w.device() == Device::CPU, "to_q8 needs CPU Q8 weights");
            return w;
    }
    throw std::logic_error("unhandled WeightKind");
}

WeightMatrix as_f32(const WeightMatrix& w) {
    switch (w.kind()) {
        case WeightKind::F32:
            return w;
        case WeightKind::Q8_0:
            return w.dequant_f32();
    }
    throw std::logic_error("unhandled WeightKind");
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
        Buffer q_proj(static_cast<std::size_t>(q) * h, cpu);
        Buffer k_proj(static_cast<std::size_t>(kv) * h, cpu);
        Buffer v_proj(static_cast<std::size_t>(kv) * h, cpu);
        Buffer o_proj(static_cast<std::size_t>(h) * q, cpu);
        layer.q_norm = Buffer(static_cast<std::size_t>(config.head_dim), cpu);
        layer.k_norm = Buffer(static_cast<std::size_t>(config.head_dim), cpu);
        layer.rms_mlp = Buffer(static_cast<std::size_t>(h), cpu);
        Buffer gate_proj(static_cast<std::size_t>(inter) * h, cpu);
        Buffer up_proj(static_cast<std::size_t>(inter) * h, cpu);
        Buffer down_proj(static_cast<std::size_t>(h) * inter, cpu);
        fill_ones(&layer.rms_attn);
        fill_ones(&layer.rms_mlp);
        fill_ones(&layer.q_norm);
        fill_ones(&layer.k_norm);
        fill_normal(&q_proj, &rng, scale);
        fill_normal(&k_proj, &rng, scale);
        fill_normal(&v_proj, &rng, scale);
        fill_normal(&o_proj, &rng, scale);
        fill_normal(&gate_proj, &rng, scale);
        fill_normal(&up_proj, &rng, scale);
        fill_normal(&down_proj, &rng, scale);
        layer.q_proj = take_f32(std::move(q_proj), q, h);
        layer.k_proj = take_f32(std::move(k_proj), kv, h);
        layer.v_proj = take_f32(std::move(v_proj), kv, h);
        layer.o_proj = take_f32(std::move(o_proj), h, q);
        layer.gate_proj = take_f32(std::move(gate_proj), inter, h);
        layer.up_proj = take_f32(std::move(up_proj), inter, h);
        layer.down_proj = take_f32(std::move(down_proj), h, inter);
        w.layers.push_back(std::move(layer));
    }

    w.final_norm = Buffer(static_cast<std::size_t>(h), cpu);
    fill_ones(&w.final_norm);
    Buffer lm_head(static_cast<std::size_t>(v) * h, cpu);
    if (config.tie_word_embeddings) {
        lm_head.copy_from(w.tok_emb.data(), w.tok_emb.size());
    } else {
        fill_normal(&lm_head, &rng, scale);
    }
    w.lm_head = take_f32(std::move(lm_head), v, h);
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

ModelWeights ModelWeights::to_q8() const {
    check(device() == Device::CPU, "to_q8 is CPU-only");
    ModelWeights w;
    w.config = config;
    w.tok_emb = tok_emb;
    w.final_norm = final_norm;
    w.lm_head = as_q8(lm_head);
    w.layers.reserve(layers.size());
    for (const LayerWeights& layer : layers) {
        LayerWeights out;
        out.rms_attn = layer.rms_attn;
        out.q_proj = as_q8(layer.q_proj);
        out.k_proj = as_q8(layer.k_proj);
        out.v_proj = as_q8(layer.v_proj);
        out.o_proj = as_q8(layer.o_proj);
        out.q_norm = layer.q_norm;
        out.k_norm = layer.k_norm;
        out.rms_mlp = layer.rms_mlp;
        out.gate_proj = as_q8(layer.gate_proj);
        out.up_proj = as_q8(layer.up_proj);
        out.down_proj = as_q8(layer.down_proj);
        w.layers.push_back(std::move(out));
    }
    return w;
}

ModelWeights ModelWeights::dequant() const {
    check(device() == Device::CPU, "dequant is CPU-only");
    ModelWeights w;
    w.config = config;
    w.tok_emb = tok_emb;
    w.final_norm = final_norm;
    w.lm_head = as_f32(lm_head);
    w.layers.reserve(layers.size());
    for (const LayerWeights& layer : layers) {
        LayerWeights out;
        out.rms_attn = layer.rms_attn;
        out.q_proj = as_f32(layer.q_proj);
        out.k_proj = as_f32(layer.k_proj);
        out.v_proj = as_f32(layer.v_proj);
        out.o_proj = as_f32(layer.o_proj);
        out.q_norm = layer.q_norm;
        out.k_norm = layer.k_norm;
        out.rms_mlp = layer.rms_mlp;
        out.gate_proj = as_f32(layer.gate_proj);
        out.up_proj = as_f32(layer.up_proj);
        out.down_proj = as_f32(layer.down_proj);
        w.layers.push_back(std::move(out));
    }
    return w;
}

std::size_t ModelWeights::linear_bytes() const {
    std::size_t n = lm_head.bytes();
    for (const LayerWeights& layer : layers) {
        n += layer.q_proj.bytes();
        n += layer.k_proj.bytes();
        n += layer.v_proj.bytes();
        n += layer.o_proj.bytes();
        n += layer.gate_proj.bytes();
        n += layer.up_proj.bytes();
        n += layer.down_proj.bytes();
    }
    return n;
}

}  // namespace vesper
