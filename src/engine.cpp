#include "vesper/engine.h"

#include "vesper/kernels.h"
#include "vesper/types.h"

#include <chrono>
#include <cstring>

namespace vesper {
namespace {

void add_inplace(float* dst, const float* src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

void copy_vec(float* dst, const float* src, int n) {
    std::memcpy(dst, src, static_cast<std::size_t>(n) * sizeof(float));
}

}  // namespace

double GenerateStats::decode_tps() const {
    if (decode_ms <= 0.0 || generated_tokens <= 0) {
        return 0.0;
    }
    return static_cast<double>(generated_tokens) / (decode_ms / 1000.0);
}

double GenerateStats::prefill_tps() const {
    if (prefill_ms <= 0.0 || prompt_tokens <= 0) {
        return 0.0;
    }
    return static_cast<double>(prompt_tokens) / (prefill_ms / 1000.0);
}

Engine::Engine(ModelWeights weights)
    : weights_(std::move(weights)), cache_(KVCache::create(weights_.config)) {
    const ModelConfig& cfg = weights_.config;
    const Device cpu = Device::CPU;
    scratch_.x = Buffer(static_cast<std::size_t>(cfg.hidden_size), cpu);
    scratch_.residual = Buffer(static_cast<std::size_t>(cfg.hidden_size), cpu);
    scratch_.q = Buffer(static_cast<std::size_t>(cfg.q_dim()), cpu);
    scratch_.k = Buffer(static_cast<std::size_t>(cfg.kv_dim()), cpu);
    scratch_.v = Buffer(static_cast<std::size_t>(cfg.kv_dim()), cpu);
    scratch_.attn = Buffer(static_cast<std::size_t>(cfg.q_dim()), cpu);
    scratch_.gate = Buffer(static_cast<std::size_t>(cfg.intermediate_size), cpu);
    scratch_.up = Buffer(static_cast<std::size_t>(cfg.intermediate_size), cpu);
    scratch_.hidden = Buffer(static_cast<std::size_t>(cfg.intermediate_size), cpu);
    scratch_.logits = Buffer(static_cast<std::size_t>(cfg.vocab_size), cpu);
    scratch_.scores = Buffer(static_cast<std::size_t>(cfg.max_seq_len), cpu);
}

void Engine::reset() {
    cache_.reset();
}

void Engine::ensure_room() const {
    check(cache_.pos < weights_.config.max_seq_len, "sequence exceeds max_seq_len");
}

void Engine::forward_token(int token) {
    const ModelConfig& cfg = weights_.config;
    check(token >= 0 && token < cfg.vocab_size, "token id out of range");
    ensure_room();

    const int h = cfg.hidden_size;
    const int pos = cache_.pos;
    float* x = scratch_.x.data();
    float* residual = scratch_.residual.data();

    embed_row(x, weights_.tok_emb.data(), token, h);

    for (int layer_i = 0; layer_i < cfg.n_layers; ++layer_i) {
        const LayerWeights& layer = weights_.layers[static_cast<std::size_t>(layer_i)];
        copy_vec(residual, x, h);
        rmsnorm(x, residual, layer.rms_attn.data(), h, cfg.rms_eps);

        gemv(scratch_.q.data(), layer.q_proj.data(), x, cfg.q_dim(), h);
        gemv(scratch_.k.data(), layer.k_proj.data(), x, cfg.kv_dim(), h);
        gemv(scratch_.v.data(), layer.v_proj.data(), x, cfg.kv_dim(), h);

        if (cfg.qk_norm) {
            for (int head = 0; head < cfg.n_heads; ++head) {
                float* qh = scratch_.q.data() + head * cfg.head_dim;
                rmsnorm(qh, qh, layer.q_norm.data(), cfg.head_dim, cfg.rms_eps);
            }
            for (int head = 0; head < cfg.n_kv_heads; ++head) {
                float* kh = scratch_.k.data() + head * cfg.head_dim;
                rmsnorm(kh, kh, layer.k_norm.data(), cfg.head_dim, cfg.rms_eps);
            }
        }

        rope_neox(scratch_.q.data(), scratch_.k.data(), cfg.n_heads, cfg.n_kv_heads,
                  cfg.head_dim, pos, cfg.rope_theta);

        copy_vec(cache_.k_at(layer_i, pos), scratch_.k.data(), cfg.kv_dim());
        copy_vec(cache_.v_at(layer_i, pos), scratch_.v.data(), cfg.kv_dim());

        const int seq = pos + 1;
        const int group = cfg.gqa_group();
        for (int qh = 0; qh < cfg.n_heads; ++qh) {
            const int kvh = qh / group;
            float* q_head = scratch_.q.data() + qh * cfg.head_dim;
            attn_scores(scratch_.scores.data(), q_head, cache_.k[static_cast<std::size_t>(layer_i)].data(),
                        seq, cfg.n_kv_heads, kvh, cfg.head_dim);
            softmax_inplace(scratch_.scores.data(), seq);
            attn_mix(scratch_.attn.data() + qh * cfg.head_dim, scratch_.scores.data(),
                     cache_.v[static_cast<std::size_t>(layer_i)].data(), seq, cfg.n_kv_heads, kvh,
                     cfg.head_dim);
        }

        gemv(x, layer.o_proj.data(), scratch_.attn.data(), h, cfg.q_dim());
        add_inplace(x, residual, h);

        copy_vec(residual, x, h);
        rmsnorm(x, residual, layer.rms_mlp.data(), h, cfg.rms_eps);
        gemv(scratch_.gate.data(), layer.gate_proj.data(), x, cfg.intermediate_size, h);
        gemv(scratch_.up.data(), layer.up_proj.data(), x, cfg.intermediate_size, h);
        swiglu(scratch_.hidden.data(), scratch_.gate.data(), scratch_.up.data(),
               cfg.intermediate_size);
        gemv(x, layer.down_proj.data(), scratch_.hidden.data(), h, cfg.intermediate_size);
        add_inplace(x, residual, h);
    }

    copy_vec(residual, x, h);
    rmsnorm(x, residual, weights_.final_norm.data(), h, cfg.rms_eps);
    gemv(scratch_.logits.data(), weights_.lm_head.data(), x, cfg.vocab_size, h);
    cache_.pos = pos + 1;
}

void Engine::step(int token) {
    forward_token(token);
}

std::vector<int> Engine::generate(const std::vector<int>& prompt, int max_new_tokens) {
    check(!prompt.empty(), "prompt must not be empty");
    check(max_new_tokens >= 0, "max_new_tokens must be non-negative");
    reset();

    GenerateStats stats;
    stats.prompt_tokens = static_cast<int>(prompt.size());

    const auto t0 = std::chrono::steady_clock::now();
    for (int token : prompt) {
        forward_token(token);
    }
    const auto t1 = std::chrono::steady_clock::now();
    stats.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<int> out = prompt;
    const auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new_tokens; ++i) {
        if (cache_.pos >= weights_.config.max_seq_len) {
            break;
        }
        const int next = argmax(scratch_.logits.data(), weights_.config.vocab_size);
        out.push_back(next);
        forward_token(next);
        ++stats.generated_tokens;
    }
    const auto t3 = std::chrono::steady_clock::now();
    stats.decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    stats_ = stats;
    return out;
}

std::vector<int> encode_bytes(const std::string& text) {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (unsigned char ch : text) {
        ids.push_back(static_cast<int>(ch));
    }
    if (ids.empty()) {
        ids.push_back(0);
    }
    return ids;
}

std::string decode_bytes(const std::vector<int>& tokens) {
    std::string text;
    text.reserve(tokens.size());
    for (int id : tokens) {
        check(id >= 0 && id < 256, "byte tokenizer only accepts ids in [0, 255]");
        text.push_back(static_cast<char>(id));
    }
    return text;
}

}  // namespace vesper
