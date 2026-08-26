#include "vesper/engine.h"

#include "vesper/hip.h"
#include "vesper/kernels.h"
#include "vesper/types.h"

#include <chrono>
#include <cstring>

namespace vesper {

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

const float* Engine::logits() const {
    switch (device_) {
        case Device::CPU:
            return scratch_.logits.data();
        case Device::HIP:
            return host_logits_.data();
    }
    throw std::logic_error("unhandled Device");
}

DecodeReport Engine::last_report() const {
    DecodeReport report;
    report.engine = ReportEngine::Vesper;
    report.backend = report_backend(device_);
    if (weights_.config.arch == "vesper_tiny") {
        report.model = "tiny_demo";
    } else if (weights_.config.arch == "vesper_hybrid") {
        report.model = "tiny_hybrid";
    } else {
        report.model = weights_.config.arch;
    }
    report.quant = weights_.quant_name();
    report.arch = weights_.config.arch;
    report.prompt_tokens = stats_.prompt_tokens;
    report.new_tokens = stats_.generated_tokens;
    report.prefill_tps = stats_.prefill_tps();
    report.decode_tps = stats_.decode_tps();
    report.bytes_per_token = weights_.linear_bytes();
    report.context = cache_.pos;
    report.status = ReportStatus::Ok;
    report.fill_roofline(stats_.decode_ms);
    return report;
}

Engine::Engine(ModelWeights weights, Device device)
    : device_(device),
      host_logits_(static_cast<std::size_t>(weights.config.vocab_size), 0.0f),
      host_embed_(static_cast<std::size_t>(weights.config.hidden_size), 0.0f) {
    if (device == Device::HIP) {
        hip_init();
    }
    if (weights.device() != device) {
        weights_ = weights.to(device);
    } else {
        weights_ = std::move(weights);
    }
    cache_ = KVCache::create(weights_.config, device);

    const ModelConfig& cfg = weights_.config;
    scratch_.x = Buffer(static_cast<std::size_t>(cfg.hidden_size), device);
    scratch_.residual = Buffer(static_cast<std::size_t>(cfg.hidden_size), device);
    scratch_.q = Buffer(static_cast<std::size_t>(cfg.q_dim()), device);
    scratch_.k = Buffer(static_cast<std::size_t>(cfg.kv_dim()), device);
    scratch_.v = Buffer(static_cast<std::size_t>(cfg.kv_dim()), device);
    scratch_.attn = Buffer(static_cast<std::size_t>(cfg.q_dim()), device);
    scratch_.q_full = Buffer(static_cast<std::size_t>(cfg.q_proj_rows()), device);
    scratch_.attn_gate = Buffer(static_cast<std::size_t>(cfg.q_dim()), device);
    scratch_.gate = Buffer(static_cast<std::size_t>(cfg.intermediate_size), device);
    scratch_.up = Buffer(static_cast<std::size_t>(cfg.intermediate_size), device);
    scratch_.hidden = Buffer(static_cast<std::size_t>(cfg.intermediate_size), device);
    scratch_.logits = Buffer(static_cast<std::size_t>(cfg.vocab_size), device);
    scratch_.scores = Buffer(static_cast<std::size_t>(cfg.max_seq_len), device);
    if (cfg.is_hybrid()) {
        gdn_scratch_init(&scratch_.gdn, cfg, device);
    }
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

    if (weights_.tok_emb.kind() == WeightKind::F32 && weights_.tok_emb.device() == device_) {
        embed_row(device_, x, weights_.tok_emb.f32_data(), token, h);
    } else {
        check(weights_.tok_emb.device() == Device::CPU, "packed embed stays on CPU");
        embed_row(host_embed_.data(), weights_.tok_emb, token);
        scratch_.x.copy_from(host_embed_.data(), static_cast<std::size_t>(h));
    }

    for (int layer_i = 0; layer_i < cfg.n_layers; ++layer_i) {
        const LayerWeights& layer = weights_.layers[static_cast<std::size_t>(layer_i)];
        copy_vec(device_, residual, x, h);
        rmsnorm(device_, x, residual, layer.rms_attn.data(), h, cfg.rms_eps);

        switch (cfg.layer_kind(layer_i)) {
            case LayerKind::Attention: {
                gemv(device_, scratch_.q_full.data(), layer.q_proj, x);
                if (cfg.attn_gate) {
                    copy_vec(device_, scratch_.q.data(), scratch_.q_full.data(), cfg.q_dim());
                    copy_vec(device_, scratch_.attn_gate.data(),
                             scratch_.q_full.data() + cfg.q_dim(), cfg.q_dim());
                } else {
                    copy_vec(device_, scratch_.q.data(), scratch_.q_full.data(), cfg.q_dim());
                }
                gemv(device_, scratch_.k.data(), layer.k_proj, x);
                gemv(device_, scratch_.v.data(), layer.v_proj, x);

                if (cfg.qk_norm) {
                    for (int head = 0; head < cfg.n_heads; ++head) {
                        float* qh = scratch_.q.data() + head * cfg.head_dim;
                        rmsnorm(device_, qh, qh, layer.q_norm.data(), cfg.head_dim, cfg.rms_eps);
                    }
                    for (int head = 0; head < cfg.n_kv_heads; ++head) {
                        float* kh = scratch_.k.data() + head * cfg.head_dim;
                        rmsnorm(device_, kh, kh, layer.k_norm.data(), cfg.head_dim, cfg.rms_eps);
                    }
                }

                rope_neox(device_, scratch_.q.data(), scratch_.k.data(), cfg.n_heads,
                          cfg.n_kv_heads, cfg.head_dim, cfg.rotary_dim(), pos, cfg.rope_theta);

                copy_vec(device_, cache_.k_at(layer_i, pos), scratch_.k.data(), cfg.kv_dim());
                copy_vec(device_, cache_.v_at(layer_i, pos), scratch_.v.data(), cfg.kv_dim());

                const int seq = pos + 1;
                const int group = cfg.gqa_group();
                for (int qh = 0; qh < cfg.n_heads; ++qh) {
                    const int kvh = qh / group;
                    float* q_head = scratch_.q.data() + qh * cfg.head_dim;
                    attn_scores(device_, scratch_.scores.data(), q_head,
                                cache_.k[static_cast<std::size_t>(layer_i)].data(), seq,
                                cfg.n_kv_heads, kvh, cfg.head_dim);
                    softmax_inplace(device_, scratch_.scores.data(), seq);
                    attn_mix(device_, scratch_.attn.data() + qh * cfg.head_dim,
                             scratch_.scores.data(),
                             cache_.v[static_cast<std::size_t>(layer_i)].data(), seq,
                             cfg.n_kv_heads, kvh, cfg.head_dim);
                }

                if (cfg.attn_gate) {
                    sigmoid_inplace(device_, scratch_.attn_gate.data(), cfg.q_dim());
                    mul_inplace(device_, scratch_.attn.data(), scratch_.attn_gate.data(),
                                cfg.q_dim());
                }

                gemv(device_, x, layer.o_proj, scratch_.attn.data());
                break;
            }
            case LayerKind::DeltaNet:
                gdn_layer(device_, x, x, layer, cfg, cache_.rec_at(layer_i),
                          cache_.conv_at(layer_i), &scratch_.gdn);
                break;
        }
        add_inplace(device_, x, residual, h);

        copy_vec(device_, residual, x, h);
        rmsnorm(device_, x, residual, layer.rms_mlp.data(), h, cfg.rms_eps);
        gemv(device_, scratch_.gate.data(), layer.gate_proj, x);
        gemv(device_, scratch_.up.data(), layer.up_proj, x);
        swiglu(device_, scratch_.hidden.data(), scratch_.gate.data(), scratch_.up.data(),
               cfg.intermediate_size);
        gemv(device_, x, layer.down_proj, scratch_.hidden.data());
        add_inplace(device_, x, residual, h);
    }

    copy_vec(device_, residual, x, h);
    rmsnorm(device_, x, residual, weights_.final_norm.data(), h, cfg.rms_eps);
    gemv(device_, scratch_.logits.data(), weights_.lm_head, x);

    if (device_ == Device::HIP) {
        scratch_.logits.copy_to(host_logits_.data(), host_logits_.size());
    }
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
        const int next = argmax(logits(), weights_.config.vocab_size);
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
