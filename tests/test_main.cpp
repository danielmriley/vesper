#include "vesper/buffer.h"
#include "vesper/config.h"
#include "vesper/engine.h"
#include "vesper/kernels.h"
#include "vesper/weights.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failed = 0;
int g_passed = 0;

void expect(bool ok, const std::string& name) {
    if (ok) {
        ++g_passed;
        return;
    }
    ++g_failed;
    std::cerr << "FAIL  " << name << "\n";
}

bool close(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

bool close_vec(const float* a, const float* b, int n, float tol = 1e-5f) {
    for (int i = 0; i < n; ++i) {
        if (!close(a[i], b[i], tol)) {
            return false;
        }
    }
    return true;
}

void test_gemv() {
    const float w[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float x[] = {1.0f, 0.5f, -1.0f};
    float y[2] = {0.0f, 0.0f};
    vesper::gemv(y, w, x, 2, 3);
    expect(close(y[0], 1.0f + 1.0f - 3.0f), "gemv row0");
    expect(close(y[1], 4.0f + 2.5f - 6.0f), "gemv row1");
}

void test_rmsnorm() {
    const float x[] = {3.0f, 4.0f};
    const float w[] = {1.0f, 1.0f};
    float out[2] = {0.0f, 0.0f};
    vesper::rmsnorm(out, x, w, 2, 0.0f);
    const float inv = 1.0f / std::sqrt(12.5f);
    expect(close(out[0], 3.0f * inv), "rmsnorm 0");
    expect(close(out[1], 4.0f * inv), "rmsnorm 1");
}

void test_softmax() {
    float x[] = {1.0f, 2.0f, 3.0f};
    vesper::softmax_inplace(x, 3);
    const float sum = x[0] + x[1] + x[2];
    expect(close(sum, 1.0f, 1e-6f), "softmax sums to 1");
    expect(x[2] > x[1] && x[1] > x[0], "softmax order");
}

void test_rope_norm() {
    float q[] = {0.5f, -0.25f, 1.0f, 0.0f};
    float k[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float q0[4];
    float k0[4];
    std::memcpy(q0, q, sizeof(q0));
    std::memcpy(k0, k, sizeof(k0));
    auto n2 = [](const float* v) {
        return v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    };
    vesper::rope_neox(q, k, 1, 1, 4, 3, 10000.0f);
    expect(close(n2(q), n2(q0), 1e-5f), "rope preserves Q L2");
    expect(close(n2(k), n2(k0), 1e-5f), "rope preserves K L2");
}

void test_byte_roundtrip() {
    const std::string text = "Hi\n";
    const std::vector<int> ids = vesper::encode_bytes(text);
    expect(vesper::decode_bytes(ids) == text, "byte tokenizer roundtrip");
}

void test_hip_buffer_rejected() {
    bool threw = false;
    try {
        vesper::Buffer buf(4, vesper::Device::HIP);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, "HIP buffer rejected until backend exists");
}

void test_qwen_configs() {
    vesper::ModelConfig::tiny_demo().validate();
    vesper::ModelConfig::qwen3_06b().validate();
    vesper::ModelConfig::qwen3_8b().validate();
    const auto q8 = vesper::ModelConfig::qwen3_8b();
    expect(q8.q_dim() == 4096, "qwen3-8b q_dim");
    expect(q8.kv_dim() == 1024, "qwen3-8b kv_dim");
    expect(q8.gqa_group() == 4, "qwen3-8b GQA group");
    ++g_passed;  // validate() did not throw
}

void recompute_layer0_k(const vesper::Engine& engine, int token, int pos, float* out_k) {
    const vesper::ModelConfig& cfg = engine.config();
    const vesper::LayerWeights& layer = engine.weights().layers[0];
    std::vector<float> x(static_cast<std::size_t>(cfg.hidden_size));
    std::vector<float> normed(static_cast<std::size_t>(cfg.hidden_size));
    std::vector<float> q(static_cast<std::size_t>(cfg.q_dim()));
    std::vector<float> k(static_cast<std::size_t>(cfg.kv_dim()));
    vesper::embed_row(x.data(), engine.weights().tok_emb.data(), token, cfg.hidden_size);
    vesper::rmsnorm(normed.data(), x.data(), layer.rms_attn.data(), cfg.hidden_size, cfg.rms_eps);
    vesper::gemv(q.data(), layer.q_proj.data(), normed.data(), cfg.q_dim(), cfg.hidden_size);
    vesper::gemv(k.data(), layer.k_proj.data(), normed.data(), cfg.kv_dim(), cfg.hidden_size);
    if (cfg.qk_norm) {
        for (int head = 0; head < cfg.n_kv_heads; ++head) {
            float* kh = k.data() + head * cfg.head_dim;
            vesper::rmsnorm(kh, kh, layer.k_norm.data(), cfg.head_dim, cfg.rms_eps);
        }
        for (int head = 0; head < cfg.n_heads; ++head) {
            float* qh = q.data() + head * cfg.head_dim;
            vesper::rmsnorm(qh, qh, layer.q_norm.data(), cfg.head_dim, cfg.rms_eps);
        }
    }
    vesper::rope_neox(q.data(), k.data(), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, pos,
                      cfg.rope_theta);
    std::memcpy(out_k, k.data(), static_cast<std::size_t>(cfg.kv_dim()) * sizeof(float));
}

void test_kv_matches_recompute() {
    auto weights = vesper::ModelWeights::random(vesper::ModelConfig::tiny_demo(), 7);
    vesper::Engine engine(std::move(weights));
    engine.step(65);
    engine.step(66);
    std::vector<float> k0(static_cast<std::size_t>(engine.config().kv_dim()));
    std::vector<float> k1(static_cast<std::size_t>(engine.config().kv_dim()));
    recompute_layer0_k(engine, 65, 0, k0.data());
    recompute_layer0_k(engine, 66, 1, k1.data());
    expect(close_vec(engine.cache().k_at(0, 0), k0.data(), engine.config().kv_dim()),
           "cached K pos0 matches recompute");
    expect(close_vec(engine.cache().k_at(0, 1), k1.data(), engine.config().kv_dim()),
           "cached K pos1 matches recompute");
}

void test_greedy_continuation() {
    const auto cfg = vesper::ModelConfig::tiny_demo();
    auto w1 = vesper::ModelWeights::random(cfg, 11);
    auto w2 = vesper::ModelWeights::random(cfg, 11);
    vesper::Engine a(std::move(w1));
    vesper::Engine b(std::move(w2));
    const std::vector<int> prompt = {1, 2, 3};
    const std::vector<int> full = a.generate(prompt, 4);
    expect(full.size() == 7, "generate length");
    std::vector<int> prefix(full.begin(), full.end() - 1);
    const std::vector<int> cont = b.generate(prefix, 1);
    expect(cont.back() == full.back(), "cached continuation matches full greedy");
}

void test_two_engines_match() {
    const auto cfg = vesper::ModelConfig::tiny_demo();
    vesper::Engine a(vesper::ModelWeights::random(cfg, 3));
    vesper::Engine b(vesper::ModelWeights::random(cfg, 3));
    const auto left = a.generate({9, 8, 7}, 8);
    const auto right = b.generate({9, 8, 7}, 8);
    expect(left == right, "two engines same weights same tokens");
}

void test_argmax() {
    const float x[] = {0.1f, 4.0f, 2.0f};
    expect(vesper::argmax(x, 3) == 1, "argmax");
}

}  // namespace

int main() {
    test_gemv();
    test_rmsnorm();
    test_softmax();
    test_rope_norm();
    test_byte_roundtrip();
    test_hip_buffer_rejected();
    test_qwen_configs();
    test_kv_matches_recompute();
    test_greedy_continuation();
    test_two_engines_match();
    test_argmax();

    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
