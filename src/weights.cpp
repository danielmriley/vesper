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

WeightMatrix maybe_q8(const WeightMatrix& w) {
    if (w.rows() == 0) {
        return w;
    }
    switch (w.kind()) {
        case WeightKind::F32:
            check(w.device() == Device::CPU, "to_q8 needs CPU F32 weights");
            return WeightMatrix::q8_from_f32(w.f32_data(), w.rows(), w.cols());
        case WeightKind::Q8_0:
            check(w.device() == Device::CPU, "to_q8 needs CPU Q8 weights");
            return w;
        case WeightKind::Q4_K:
            fail("to_q8 cannot convert Q4_K");
        case WeightKind::Q5_K:
            fail("to_q8 cannot convert Q5_K");
        case WeightKind::Q6_K:
            fail("to_q8 cannot convert Q6_K");
    }
    throw std::logic_error("unhandled WeightKind");
}

WeightMatrix maybe_q4(const WeightMatrix& w) {
    if (w.rows() == 0) {
        return w;
    }
    switch (w.kind()) {
        case WeightKind::F32:
            check(w.device() == Device::CPU, "to_q4 needs CPU F32 weights");
            return WeightMatrix::q4_from_f32(w.f32_data(), w.rows(), w.cols());
        case WeightKind::Q4_K:
            check(w.device() == Device::CPU, "to_q4 needs CPU Q4 weights");
            return w;
        case WeightKind::Q8_0:
            fail("to_q4 cannot convert Q8_0");
        case WeightKind::Q5_K:
            fail("to_q4 cannot convert Q5_K");
        case WeightKind::Q6_K:
            fail("to_q4 cannot convert Q6_K");
    }
    throw std::logic_error("unhandled WeightKind");
}

WeightMatrix as_f32(const WeightMatrix& w) {
    if (w.rows() == 0) {
        return w;
    }
    switch (w.kind()) {
        case WeightKind::F32:
            return w;
        case WeightKind::Q8_0:
        case WeightKind::Q4_K:
        case WeightKind::Q5_K:
        case WeightKind::Q6_K:
            return w.dequant_f32();
    }
    throw std::logic_error("unhandled WeightKind");
}

void copy_layer_device(LayerWeights* out, const LayerWeights& layer, Device device) {
    out->rms_attn = layer.rms_attn.to(device);
    out->q_proj = layer.q_proj.to(device);
    out->k_proj = layer.k_proj.to(device);
    out->v_proj = layer.v_proj.to(device);
    out->o_proj = layer.o_proj.to(device);
    out->q_norm = layer.q_norm.to(device);
    out->k_norm = layer.k_norm.to(device);
    out->rms_mlp = layer.rms_mlp.to(device);
    out->gate_proj = layer.gate_proj.to(device);
    out->up_proj = layer.up_proj.to(device);
    out->down_proj = layer.down_proj.to(device);
    out->qkv_proj = layer.qkv_proj.to(device);
    out->z_proj = layer.z_proj.to(device);
    out->beta_proj = layer.beta_proj.to(device);
    out->alpha_proj = layer.alpha_proj.to(device);
    out->ssm_out = layer.ssm_out.to(device);
    out->conv1d = layer.conv1d.to(device);
    out->ssm_dt = layer.ssm_dt.to(device);
    out->ssm_a = layer.ssm_a.to(device);
    out->ssm_norm = layer.ssm_norm.to(device);
}

void fill_ffn(LayerWeights* layer, Lcg* rng, int hidden, int inter, float scale) {
    const Device cpu = Device::CPU;
    layer->rms_mlp = Buffer(static_cast<std::size_t>(hidden), cpu);
    Buffer gate_proj(static_cast<std::size_t>(inter) * hidden, cpu);
    Buffer up_proj(static_cast<std::size_t>(inter) * hidden, cpu);
    Buffer down_proj(static_cast<std::size_t>(hidden) * inter, cpu);
    fill_ones(&layer->rms_mlp);
    fill_normal(&gate_proj, rng, scale);
    fill_normal(&up_proj, rng, scale);
    fill_normal(&down_proj, rng, scale);
    layer->gate_proj = take_f32(std::move(gate_proj), inter, hidden);
    layer->up_proj = take_f32(std::move(up_proj), inter, hidden);
    layer->down_proj = take_f32(std::move(down_proj), hidden, inter);
}

}  // namespace

const char* weight_kind_name(WeightKind kind) {
    switch (kind) {
        case WeightKind::F32:
            return "f32";
        case WeightKind::Q8_0:
            return "Q8_0";
        case WeightKind::Q4_K:
            return "Q4_K";
        case WeightKind::Q5_K:
            return "Q5_K";
        case WeightKind::Q6_K:
            return "Q6_K";
    }
    throw std::logic_error("unhandled WeightKind");
}

ModelWeights ModelWeights::random(const ModelConfig& config, std::uint32_t seed) {
    config.validate();
    ModelWeights w;
    w.config = config;

    const Device cpu = Device::CPU;
    const int h = config.hidden_size;
    const int q = config.q_dim();
    const int q_rows = config.q_proj_rows();
    const int kv = config.kv_dim();
    const int inter = config.intermediate_size;
    const int v = config.vocab_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(h));
    Lcg rng(seed);

    Buffer tok(static_cast<std::size_t>(v) * h, cpu);
    fill_normal(&tok, &rng, scale);
    w.tok_emb = take_f32(std::move(tok), v, h);

    w.layers.reserve(static_cast<std::size_t>(config.n_layers));
    for (int i = 0; i < config.n_layers; ++i) {
        LayerWeights layer;
        layer.rms_attn = Buffer(static_cast<std::size_t>(h), cpu);
        fill_ones(&layer.rms_attn);
        const LayerKind kind = config.layer_kind(i);
        switch (kind) {
            case LayerKind::Attention: {
                Buffer q_proj(static_cast<std::size_t>(q_rows) * h, cpu);
                Buffer k_proj(static_cast<std::size_t>(kv) * h, cpu);
                Buffer v_proj(static_cast<std::size_t>(kv) * h, cpu);
                Buffer o_proj(static_cast<std::size_t>(h) * q, cpu);
                layer.q_norm = Buffer(static_cast<std::size_t>(config.head_dim), cpu);
                layer.k_norm = Buffer(static_cast<std::size_t>(config.head_dim), cpu);
                fill_ones(&layer.q_norm);
                fill_ones(&layer.k_norm);
                fill_normal(&q_proj, &rng, scale);
                fill_normal(&k_proj, &rng, scale);
                fill_normal(&v_proj, &rng, scale);
                fill_normal(&o_proj, &rng, scale);
                layer.q_proj = take_f32(std::move(q_proj), q_rows, h);
                layer.k_proj = take_f32(std::move(k_proj), kv, h);
                layer.v_proj = take_f32(std::move(v_proj), kv, h);
                layer.o_proj = take_f32(std::move(o_proj), h, q);
                break;
            }
            case LayerKind::DeltaNet: {
                const int qkv = config.gdn_qkv_dim();
                const int vd = config.gdn_value_dim();
                Buffer qkv_proj(static_cast<std::size_t>(qkv) * h, cpu);
                Buffer z_proj(static_cast<std::size_t>(vd) * h, cpu);
                Buffer beta_proj(static_cast<std::size_t>(config.gdn_v_heads) * h, cpu);
                Buffer alpha_proj(static_cast<std::size_t>(config.gdn_v_heads) * h, cpu);
                Buffer ssm_out(static_cast<std::size_t>(h) * vd, cpu);
                layer.conv1d = Buffer(static_cast<std::size_t>(qkv) * config.gdn_conv_kernel, cpu);
                layer.ssm_dt = Buffer(static_cast<std::size_t>(config.gdn_v_heads), cpu);
                layer.ssm_a = Buffer(static_cast<std::size_t>(config.gdn_v_heads), cpu);
                layer.ssm_norm = Buffer(static_cast<std::size_t>(config.gdn_head_dim), cpu);
                fill_normal(&qkv_proj, &rng, scale);
                fill_normal(&z_proj, &rng, scale);
                fill_normal(&beta_proj, &rng, scale);
                fill_normal(&alpha_proj, &rng, scale);
                fill_normal(&ssm_out, &rng, scale);
                fill_normal(&layer.conv1d, &rng, scale);
                fill_normal(&layer.ssm_dt, &rng, 0.1f);
                for (int j = 0; j < config.gdn_v_heads; ++j) {
                    layer.ssm_a.data()[j] = -0.5f;
                }
                fill_ones(&layer.ssm_norm);
                layer.qkv_proj = take_f32(std::move(qkv_proj), qkv, h);
                layer.z_proj = take_f32(std::move(z_proj), vd, h);
                layer.beta_proj = take_f32(std::move(beta_proj), config.gdn_v_heads, h);
                layer.alpha_proj = take_f32(std::move(alpha_proj), config.gdn_v_heads, h);
                layer.ssm_out = take_f32(std::move(ssm_out), h, vd);
                break;
            }
        }
        fill_ffn(&layer, &rng, h, inter, scale);
        w.layers.push_back(std::move(layer));
    }

    w.final_norm = Buffer(static_cast<std::size_t>(h), cpu);
    fill_ones(&w.final_norm);
    Buffer lm_head(static_cast<std::size_t>(v) * h, cpu);
    if (config.tie_word_embeddings) {
        lm_head.copy_from(w.tok_emb.f32_data(), static_cast<std::size_t>(v) * h);
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
    if (tok_emb.kind() == WeightKind::F32) {
        w.tok_emb = tok_emb.to(device);
    } else {
        w.tok_emb = tok_emb;
    }
    w.final_norm = final_norm.to(device);
    w.lm_head = lm_head.to(device);
    w.layers.reserve(layers.size());
    for (const LayerWeights& layer : layers) {
        LayerWeights out;
        copy_layer_device(&out, layer, device);
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
    w.lm_head = maybe_q8(lm_head);
    w.layers.reserve(layers.size());
    for (const LayerWeights& layer : layers) {
        LayerWeights out;
        out.rms_attn = layer.rms_attn;
        out.q_proj = maybe_q8(layer.q_proj);
        out.k_proj = maybe_q8(layer.k_proj);
        out.v_proj = maybe_q8(layer.v_proj);
        out.o_proj = maybe_q8(layer.o_proj);
        out.q_norm = layer.q_norm;
        out.k_norm = layer.k_norm;
        out.rms_mlp = layer.rms_mlp;
        out.gate_proj = maybe_q8(layer.gate_proj);
        out.up_proj = maybe_q8(layer.up_proj);
        out.down_proj = maybe_q8(layer.down_proj);
        out.qkv_proj = maybe_q8(layer.qkv_proj);
        out.z_proj = maybe_q8(layer.z_proj);
        out.beta_proj = maybe_q8(layer.beta_proj);
        out.alpha_proj = maybe_q8(layer.alpha_proj);
        out.ssm_out = maybe_q8(layer.ssm_out);
        out.conv1d = layer.conv1d;
        out.ssm_dt = layer.ssm_dt;
        out.ssm_a = layer.ssm_a;
        out.ssm_norm = layer.ssm_norm;
        w.layers.push_back(std::move(out));
    }
    return w;
}

ModelWeights ModelWeights::to_q4() const {
    check(device() == Device::CPU, "to_q4 is CPU-only");
    ModelWeights w;
    w.config = config;
    w.tok_emb = tok_emb;
    w.final_norm = final_norm;
    w.lm_head = maybe_q4(lm_head);
    w.layers.reserve(layers.size());
    for (const LayerWeights& layer : layers) {
        LayerWeights out;
        out.rms_attn = layer.rms_attn;
        out.q_proj = maybe_q4(layer.q_proj);
        out.k_proj = maybe_q4(layer.k_proj);
        out.v_proj = maybe_q4(layer.v_proj);
        out.o_proj = maybe_q4(layer.o_proj);
        out.q_norm = layer.q_norm;
        out.k_norm = layer.k_norm;
        out.rms_mlp = layer.rms_mlp;
        out.gate_proj = maybe_q4(layer.gate_proj);
        out.up_proj = maybe_q4(layer.up_proj);
        out.down_proj = maybe_q4(layer.down_proj);
        out.qkv_proj = maybe_q4(layer.qkv_proj);
        out.z_proj = maybe_q4(layer.z_proj);
        out.beta_proj = maybe_q4(layer.beta_proj);
        out.alpha_proj = maybe_q4(layer.alpha_proj);
        out.ssm_out = maybe_q4(layer.ssm_out);
        out.conv1d = layer.conv1d;
        out.ssm_dt = layer.ssm_dt;
        out.ssm_a = layer.ssm_a;
        out.ssm_norm = layer.ssm_norm;
        w.layers.push_back(std::move(out));
    }
    return w;
}

ModelWeights ModelWeights::dequant() const {
    check(device() == Device::CPU, "dequant is CPU-only");
    ModelWeights w;
    w.config = config;
    w.tok_emb = as_f32(tok_emb);
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
        out.qkv_proj = as_f32(layer.qkv_proj);
        out.z_proj = as_f32(layer.z_proj);
        out.beta_proj = as_f32(layer.beta_proj);
        out.alpha_proj = as_f32(layer.alpha_proj);
        out.ssm_out = as_f32(layer.ssm_out);
        out.conv1d = layer.conv1d;
        out.ssm_dt = layer.ssm_dt;
        out.ssm_a = layer.ssm_a;
        out.ssm_norm = layer.ssm_norm;
        w.layers.push_back(std::move(out));
    }
    return w;
}

std::size_t qwen38_27b_q4km_linear_bytes() {
    const ModelConfig cfg = ModelConfig::qwen38_27b();
    const int h = cfg.hidden_size;
    const int inter = cfg.intermediate_size;
    std::size_t n = packed_bytes(WeightKind::Q6_K, cfg.vocab_size, h);
    n += static_cast<std::size_t>(cfg.n_layers) *
         (packed_bytes(WeightKind::Q4_K, inter, h) + packed_bytes(WeightKind::Q4_K, inter, h) +
          packed_bytes(WeightKind::Q4_K, h, inter));
    for (int i = 0; i < cfg.n_layers; ++i) {
        switch (cfg.layer_kind(i)) {
            case LayerKind::Attention:
                n += packed_bytes(WeightKind::Q8_0, cfg.q_proj_rows(), h);
                n += packed_bytes(WeightKind::Q8_0, cfg.kv_dim(), h);
                n += packed_bytes(WeightKind::Q8_0, cfg.kv_dim(), h);
                n += packed_bytes(WeightKind::Q6_K, h, cfg.q_dim());
                continue;
            case LayerKind::DeltaNet:
                n += packed_bytes(WeightKind::Q8_0, cfg.gdn_qkv_dim(), h);
                n += packed_bytes(WeightKind::Q8_0, cfg.gdn_value_dim(), h);
                n += packed_bytes(WeightKind::Q8_0, cfg.gdn_v_heads, h);
                n += packed_bytes(WeightKind::Q8_0, cfg.gdn_v_heads, h);
                n += packed_bytes(WeightKind::Q8_0, h, cfg.gdn_value_dim());
                continue;
        }
        throw std::logic_error("unhandled LayerKind");
    }
    return n;
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
        n += layer.qkv_proj.bytes();
        n += layer.z_proj.bytes();
        n += layer.beta_proj.bytes();
        n += layer.alpha_proj.bytes();
        n += layer.ssm_out.bytes();
    }
    return n;
}

const char* ModelWeights::quant_name() const {
    bool f32 = false;
    bool q8 = false;
    bool q4 = false;
    bool q5 = false;
    bool q6 = false;
    auto mark = [&](const WeightMatrix& w) {
        if (w.rows() <= 0) {
            return;
        }
        switch (w.kind()) {
            case WeightKind::F32:
                f32 = true;
                return;
            case WeightKind::Q8_0:
                q8 = true;
                return;
            case WeightKind::Q4_K:
                q4 = true;
                return;
            case WeightKind::Q5_K:
                q5 = true;
                return;
            case WeightKind::Q6_K:
                q6 = true;
                return;
        }
        throw std::logic_error("unhandled WeightKind");
    };
    mark(tok_emb);
    mark(lm_head);
    for (const LayerWeights& layer : layers) {
        mark(layer.q_proj);
        mark(layer.k_proj);
        mark(layer.v_proj);
        mark(layer.o_proj);
        mark(layer.gate_proj);
        mark(layer.up_proj);
        mark(layer.down_proj);
        mark(layer.qkv_proj);
        mark(layer.z_proj);
        mark(layer.beta_proj);
        mark(layer.alpha_proj);
        mark(layer.ssm_out);
    }
    if (q4 && (q5 || q6 || q8)) {
        return "Q4_K_M";
    }
    if (q4) {
        return "Q4_K";
    }
    if (q5) {
        return "Q5_K";
    }
    if (q6) {
        return "Q6_K";
    }
    if (q8) {
        return "Q8_0";
    }
    if (f32) {
        return "F32";
    }
    return weight_kind_name(lm_head.kind());
}

}  // namespace vesper
