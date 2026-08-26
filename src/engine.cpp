#include "vesper/engine.h"

#include "vesper/hip.h"
#include "vesper/kernels.h"
#include "vesper/types.h"

#include <chrono>
#include <cstring>
#include <string>

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
            scratch_.logits.copy_to(host_logits_.data(), host_logits_.size());
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
    if (!last_new_ids_.empty()) {
        report.ids.reserve(static_cast<std::size_t>(last_new_ids_.size()) * 6u);
        for (std::size_t i = 0; i < last_new_ids_.size(); ++i) {
            if (i > 0) {
                report.ids += ',';
            }
            report.ids += std::to_string(last_new_ids_[i]);
        }
    }
    return report;
}

Engine::Engine(ModelWeights weights, Device device, int context)
    : device_(device),
      host_logits_(static_cast<std::size_t>(weights.config.vocab_size), 0.0f) {
    if (context > 0) {
        weights.config.cap_seq_len(context);
    }
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
    if (device == Device::HIP) {
        d_token_ = static_cast<int*>(hip_alloc(sizeof(int)));
        d_pos_ = static_cast<int*>(hip_alloc(sizeof(int)));
        d_gen_i_ = static_cast<int*>(hip_alloc(sizeof(int)));
        d_ids_ = static_cast<int*>(
            hip_alloc(sizeof(int) * (static_cast<std::size_t>(weights_.config.max_seq_len) + 1u)));
    }
}

void Engine::reset() {
    cache_.reset();
}

Engine::~Engine() {
    if (device_ == Device::HIP) {
        hip_graph_destroy_all();
        hip_free(d_token_);
        hip_free(d_pos_);
        hip_free(d_ids_);
        hip_free(d_gen_i_);
        d_token_ = nullptr;
        d_pos_ = nullptr;
        d_ids_ = nullptr;
        d_gen_i_ = nullptr;
    }
}

void Engine::ensure_room() const {
    check(cache_.pos < weights_.config.max_seq_len, "sequence exceeds max_seq_len");
}

void Engine::apply_layer(int layer_i) {
    const ModelConfig& cfg = weights_.config;
    const int h = cfg.hidden_size;
    const int kv = cfg.kv_dim();
    const int pos = cache_.pos;
    const int* pos_ptr = (device_ == Device::HIP) ? d_pos_ : &pos;
    float* x = scratch_.x.data();
    float* residual = scratch_.residual.data();
    const LayerWeights& layer = weights_.layers[static_cast<std::size_t>(layer_i)];
    copy_rmsnorm(device_, x, residual, layer.rms_attn.data(), h, cfg.rms_eps);

    switch (cfg.layer_kind(layer_i)) {
        case LayerKind::Attention: {
            float* k_row = scratch_.k.data();
            float* v_row = scratch_.v.data();
            gemv3(device_, scratch_.q_full.data(), layer.q_proj, k_row, layer.k_proj, v_row,
                  layer.v_proj, x);
            if (cfg.attn_gate) {
                if (cfg.qk_norm) {
                    split_gated_q_norm(device_, scratch_.q.data(), scratch_.attn_gate.data(),
                                       scratch_.q_full.data(), layer.q_norm.data(), cfg.n_heads,
                                       cfg.head_dim, cfg.rms_eps);
                } else {
                    split_gated_q(device_, scratch_.q.data(), scratch_.attn_gate.data(),
                                  scratch_.q_full.data(), cfg.n_heads, cfg.head_dim);
                }
            } else {
                copy_vec(device_, scratch_.q.data(), scratch_.q_full.data(), cfg.q_dim());
                if (cfg.qk_norm) {
                    rmsnorm_rows(device_, scratch_.q.data(), layer.q_norm.data(), cfg.n_heads,
                                 cfg.head_dim, cfg.rms_eps);
                }
            }
            if (cfg.qk_norm) {
                rope_neox_k_norm(device_, scratch_.q.data(), k_row, layer.k_norm.data(),
                                 cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rotary_dim(),
                                 pos_ptr, cfg.rope_theta, cfg.rms_eps);
            } else {
                rope_neox(device_, scratch_.q.data(), k_row, cfg.n_heads, cfg.n_kv_heads,
                          cfg.head_dim, cfg.rotary_dim(), pos_ptr, cfg.rope_theta);
            }
            scatter_row(device_, cache_.k[static_cast<std::size_t>(layer_i)].data(), k_row, pos_ptr,
                        kv);
            scatter_row(device_, cache_.v[static_cast<std::size_t>(layer_i)].data(), v_row, pos_ptr,
                        kv);
            attn_decode(device_, scratch_.attn.data(), scratch_.scores.data(), scratch_.q.data(),
                        cache_.k[static_cast<std::size_t>(layer_i)].data(),
                        cache_.v[static_cast<std::size_t>(layer_i)].data(),
                        cfg.attn_gate ? scratch_.attn_gate.data() : nullptr, pos_ptr, cfg.n_heads,
                        cfg.n_kv_heads, cfg.head_dim);

            gemv(device_, x, layer.o_proj, scratch_.attn.data());
            break;
        }
        case LayerKind::DeltaNet:
            gdn_layer(device_, x, x, layer, cfg, cache_.rec_at(layer_i), cache_.conv_at(layer_i),
                      &scratch_.gdn);
            break;
    }

    add_rmsnorm(device_, x, residual, layer.rms_mlp.data(), h, cfg.rms_eps);
    gemv_swiglu(device_, scratch_.hidden.data(), scratch_.gate.data(), scratch_.up.data(),
                layer.gate_proj, layer.up_proj, x);
    gemv_add(device_, x, layer.down_proj, scratch_.hidden.data(), residual);
}

void Engine::run_layers_and_head() {
    const ModelConfig& cfg = weights_.config;
    float* x = scratch_.x.data();
    for (int layer_i = 0; layer_i < cfg.n_layers; ++layer_i) {
        apply_layer(layer_i);
    }
    rmsnorm(device_, x, x, weights_.final_norm.data(), cfg.hidden_size, cfg.rms_eps);
    gemv(device_, scratch_.logits.data(), weights_.lm_head, x);
}

void Engine::upload_step_scalars(int token) {
    h_token_ = token;
    h_pos_ = cache_.pos;
    hip_upload_i32(d_token_, &h_token_);
    hip_upload_i32(d_pos_, &h_pos_);
}

void Engine::decode_device_step() {
    embed_row(device_, scratch_.x.data(), weights_.tok_emb, d_token_);
    run_layers_and_head();
    argmax_write(device_, d_token_, scratch_.logits.data(), weights_.config.vocab_size);
    commit_generated(device_, d_ids_, d_gen_i_, d_token_, d_pos_);
}

void Engine::forward_token(int token) {
    const ModelConfig& cfg = weights_.config;
    check(token >= 0 && token < cfg.vocab_size, "token id out of range");
    ensure_room();

    const int pos = cache_.pos;
    float* x = scratch_.x.data();

    if (device_ == Device::HIP) {
        upload_step_scalars(token);
        embed_row(device_, x, weights_.tok_emb, d_token_);
        run_layers_and_head();
        cache_.pos = pos + 1;
        hip_warm_ = true;
        return;
    }

    embed_row(device_, x, weights_.tok_emb, token);
    run_layers_and_head();
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
    last_new_ids_.clear();
    last_new_ids_.reserve(static_cast<std::size_t>(max_new_tokens));
    const auto t2 = std::chrono::steady_clock::now();
    if (device_ == Device::HIP && max_new_tokens > 0) {
        generate_hip_decode(&out, max_new_tokens, &stats);
    } else {
        for (int i = 0; i < max_new_tokens; ++i) {
            if (cache_.pos >= weights_.config.max_seq_len) {
                break;
            }
            const int next = argmax(device_, scratch_.logits.data(), weights_.config.vocab_size);
            out.push_back(next);
            last_new_ids_.push_back(next);
            forward_token(next);
            ++stats.generated_tokens;
        }
    }
    const auto t3 = std::chrono::steady_clock::now();
    stats.decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    stats_ = stats;
    return out;
}

void Engine::generate_hip_decode(std::vector<int>* out, int max_new_tokens, GenerateStats* stats) {
    const int room = weights_.config.max_seq_len - cache_.pos;
    const int n = max_new_tokens < room ? max_new_tokens : room;
    if (n <= 0) {
        return;
    }
    argmax_write(device_, d_token_, scratch_.logits.data(), weights_.config.vocab_size);
    seed_generated(device_, d_ids_, d_gen_i_, d_token_);
    h_pos_ = cache_.pos;
    hip_upload_i32(d_pos_, &h_pos_);

    for (int i = 0; i < n; ++i) {
        if (hip_warm_ && hip_graph_ready(kDecodeGraphSlot)) {
            hip_graph_launch(kDecodeGraphSlot);
            continue;
        }
        bool capturing = false;
        if (hip_warm_) {
            capturing = hip_graph_try_begin(kDecodeGraphSlot);
        }
        try {
            decode_device_step();
        } catch (...) {
            if (capturing) {
                hip_graph_abort();
            }
            throw;
        }
        if (capturing) {
            (void)hip_graph_try_end(kDecodeGraphSlot);
        }
        hip_warm_ = true;
    }

    last_new_ids_.resize(static_cast<std::size_t>(n));
    hip_copy_d2h(last_new_ids_.data(), d_ids_, static_cast<std::size_t>(n) * sizeof(int));
    out->insert(out->end(), last_new_ids_.begin(), last_new_ids_.end());
    cache_.pos += n;
    stats->generated_tokens = n;
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
