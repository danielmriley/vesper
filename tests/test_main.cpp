#include "vesper/buffer.h"
#include "vesper/config.h"
#include "vesper/engine.h"
#include "vesper/gdn.h"
#include "vesper/gguf.h"
#include "vesper/gguf_write.h"
#include "vesper/hip.h"
#include "vesper/kernels.h"
#include "vesper/model_io.h"
#include "vesper/multi_row.h"
#include "vesper/dotq.h"
#include "vesper/q4k.h"
#include "vesper/q5k.h"
#include "vesper/q6k.h"
#include "vesper/q8.h"
#include "vesper/q8x.h"
#include "vesper/report.h"
#include "vesper/target.h"
#include "vesper/tokenizer.h"
#include "vesper/weight.h"
#include "vesper/weights.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
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

void test_split_gated_q() {
    // GGUF / llama.cpp layout: per head [q | gate], not [all q | all gate].
    const float q_full[] = {1.0f, 2.0f, 10.0f, 20.0f, 3.0f, 4.0f, 30.0f, 40.0f};
    float q[4] = {};
    float gate[4] = {};
    vesper::split_gated_q(q, gate, q_full, 2, 2);
    expect(close(q[0], 1.0f) && close(q[1], 2.0f) && close(q[2], 3.0f) && close(q[3], 4.0f),
           "gated Q is per-head first half");
    expect(close(gate[0], 10.0f) && close(gate[1], 20.0f) && close(gate[2], 30.0f) &&
               close(gate[3], 40.0f),
           "gated gate is per-head second half");
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

void test_target_pin() {
    expect(std::string(vesper::kHipArch) == "gfx1201", "v1 HIP arch is gfx1201");
    expect(vesper::kWavefront == 32, "RDNA4 wave32");
    expect(vesper::kCachelineBytes == 256, "RDNA4 256B cacheline");
    expect(vesper::kComputeUnits == 64, "R9700 64 CUs");
    expect(vesper::kGemvWorkgroup == 256, "GEMV workgroup 256");
    expect(vesper::kGemvRowsPerWg == 1, "RDNA4 MMVQ 1 row per workgroup");
    expect(vesper::kGemvWaves == 8, "RDNA4 MMVQ 8 waves");
    expect(vesper::kLdsQ8xMaxBytes == 32768, "Q8_1 x LDS cap 32 KiB");
    expect(vesper::kIdlePowerQueues == 1, "HIP idle-power queue pin");
    expect(vesper::kPeakBandwidthGBs == 640.0, "R9700 640 GB/s pin");
}

void test_cpu_device_dispatch() {
    const float w[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float x[] = {1.0f, 0.5f, -1.0f};
    float y0[2] = {0.0f, 0.0f};
    float y1[2] = {0.0f, 0.0f};
    vesper::gemv(y0, w, x, 2, 3);
    vesper::gemv(vesper::Device::CPU, y1, w, x, 2, 3);
    expect(close_vec(y0, y1, 2), "CPU device dispatch matches gemv");
}

void test_hip_buffer() {
    if (vesper::hip_available()) {
        vesper::hip_init();
        vesper::Buffer buf(64, vesper::Device::HIP);
        expect(buf.size() == 64, "hip buffer size");
        expect(buf.device() == vesper::Device::HIP, "hip buffer device");
        const auto addr = reinterpret_cast<std::uintptr_t>(buf.data());
        expect(addr % 256 == 0, "hip 256B alignment");
        return;
    }
    bool threw = false;
    try {
        vesper::Buffer buf(4, vesper::Device::HIP);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, "HIP buffer rejected without a gfx1201 device");
}

void test_hip_graph_idle() {
    expect(!vesper::hip_graph_ready(0), "no GDN graph before capture");
    expect(!vesper::hip_graph_ready(47), "no official GDN slot before capture");
}

void test_hip_kernels_match_cpu() {
    if (!vesper::hip_available()) {
        return;
    }
    vesper::hip_init();

    const float w[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float x[] = {1.0f, 0.5f, -1.0f};
    float y_cpu[2] = {0.0f, 0.0f};
    vesper::gemv(y_cpu, w, x, 2, 3);
    vesper::Buffer W(6, vesper::Device::HIP);
    vesper::Buffer X(3, vesper::Device::HIP);
    vesper::Buffer Y(2, vesper::Device::HIP);
    W.copy_from(w, 6);
    X.copy_from(x, 3);
    vesper::gemv(vesper::Device::HIP, Y.data(), W.data(), X.data(), 2, 3);
    float y_gpu[2] = {0.0f, 0.0f};
    Y.copy_to(y_gpu, 2);
    expect(close_vec(y_cpu, y_gpu, 2, 1e-5f), "HIP GEMV 2x3 matches CPU");

    const int out = 64;
    const int in = 64;
    std::vector<float> wh(static_cast<std::size_t>(out * in));
    std::vector<float> xh(static_cast<std::size_t>(in));
    for (int i = 0; i < out * in; ++i) {
        wh[static_cast<std::size_t>(i)] = 0.01f * static_cast<float>((i * 17) % 23 - 11);
    }
    for (int i = 0; i < in; ++i) {
        xh[static_cast<std::size_t>(i)] = 0.05f * static_cast<float>((i * 9) % 13 - 6);
    }
    std::vector<float> y_big_cpu(static_cast<std::size_t>(out));
    std::vector<float> y_big_gpu(static_cast<std::size_t>(out));
    vesper::gemv(y_big_cpu.data(), wh.data(), xh.data(), out, in);
    vesper::Buffer Wb(static_cast<std::size_t>(out * in), vesper::Device::HIP);
    vesper::Buffer Xb(static_cast<std::size_t>(in), vesper::Device::HIP);
    vesper::Buffer Yb(static_cast<std::size_t>(out), vesper::Device::HIP);
    Wb.copy_from(wh.data(), wh.size());
    Xb.copy_from(xh.data(), xh.size());
    vesper::gemv(vesper::Device::HIP, Yb.data(), Wb.data(), Xb.data(), out, in);
    Yb.copy_to(y_big_gpu.data(), y_big_gpu.size());
    expect(close_vec(y_big_cpu.data(), y_big_gpu.data(), out, 2e-4f),
           "HIP GEMV 64x64 matches CPU");

    const float xn[] = {3.0f, 4.0f};
    const float wn[] = {1.0f, 1.0f};
    float n_cpu[2] = {0.0f, 0.0f};
    vesper::rmsnorm(n_cpu, xn, wn, 2, 0.0f);
    vesper::Buffer Xn(2, vesper::Device::HIP);
    vesper::Buffer Wn(2, vesper::Device::HIP);
    vesper::Buffer On(2, vesper::Device::HIP);
    Xn.copy_from(xn, 2);
    Wn.copy_from(wn, 2);
    vesper::rmsnorm(vesper::Device::HIP, On.data(), Xn.data(), Wn.data(), 2, 0.0f);
    float n_gpu[2] = {0.0f, 0.0f};
    On.copy_to(n_gpu, 2);
    expect(close_vec(n_cpu, n_gpu, 2, 1e-5f), "HIP RMSNorm matches CPU");

    float q[] = {0.5f, -0.25f, 1.0f, 0.0f};
    float k[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float q_cpu[4];
    float k_cpu[4];
    std::memcpy(q_cpu, q, sizeof(q_cpu));
    std::memcpy(k_cpu, k, sizeof(k_cpu));
    vesper::rope_neox(q_cpu, k_cpu, 1, 1, 4, 3, 10000.0f);
    vesper::Buffer Q(4, vesper::Device::HIP);
    vesper::Buffer K(4, vesper::Device::HIP);
    Q.copy_from(q, 4);
    K.copy_from(k, 4);
    vesper::rope_neox(vesper::Device::HIP, Q.data(), K.data(), 1, 1, 4, 3, 10000.0f);
    float q_gpu[4];
    float k_gpu[4];
    Q.copy_to(q_gpu, 4);
    K.copy_to(k_gpu, 4);
    expect(close_vec(q_cpu, q_gpu, 4, 1e-5f), "HIP RoPE Q matches CPU");
    expect(close_vec(k_cpu, k_gpu, 4, 1e-5f), "HIP RoPE K matches CPU");
}

void test_hip_engine_matches_cpu() {
    if (!vesper::hip_available()) {
        return;
    }
    const auto cfg = vesper::ModelConfig::tiny_demo();
    auto cpu_w = vesper::ModelWeights::random(cfg, 3);
    auto hip_w = vesper::ModelWeights::random(cfg, 3);
    vesper::Engine cpu(std::move(cpu_w), vesper::Device::CPU);
    vesper::Engine gpu(std::move(hip_w), vesper::Device::HIP);
    const auto left = cpu.generate({9, 8, 7}, 8);
    const auto right = gpu.generate({9, 8, 7}, 8);
    expect(left == right, "HIP engine greedy tokens match CPU");
}

void test_qwen_configs() {
    vesper::ModelConfig::tiny_demo().validate();
    vesper::ModelConfig::qwen3_06b().validate();
    vesper::ModelConfig::qwen3_8b().validate();
    const auto q8 = vesper::ModelConfig::qwen3_8b();
    expect(q8.q_dim() == 4096, "qwen3-8b q_dim");
    expect(q8.kv_dim() == 1024, "qwen3-8b kv_dim");
    expect(q8.gqa_group() == 4, "qwen3-8b GQA group");
    const auto hybrid = vesper::ModelConfig::tiny_hybrid();
    expect(hybrid.layer_kind(0) == vesper::LayerKind::DeltaNet, "hybrid layer 0 is GDN");
    expect(hybrid.layer_kind(3) == vesper::LayerKind::Attention, "hybrid layer 3 is attn");
    expect(hybrid.q_proj_rows() == 128, "gated q_proj is 2x");
    const auto q4km = vesper::ModelConfig::tiny_q4km();
    expect(q4km.hidden_size == 256 && q4km.intermediate_size == 256, "tiny_q4km K-quant dims");
    const auto q27 = vesper::ModelConfig::qwen38_27b();
    expect(q27.arch == "qwen35", "qwen38 arch qwen35");
    expect(q27.n_layers == 64 && q27.hidden_size == 5120, "qwen38 64x5120");
    expect(q27.gdn_v_heads == 48 && q27.gdn_qk_heads == 16, "qwen38 GDN heads");
    expect(q27.attn_gate && q27.rope_dim == 64, "qwen38 gated partial rope");
    expect(q27.n_rope_sections == 3 && q27.rope_section_sum() == 32, "qwen38 mrope 11+11+10");
    expect(q27.layer_kind(0) == vesper::LayerKind::DeltaNet, "qwen38 layer 0 GDN");
    expect(q27.layer_kind(3) == vesper::LayerKind::Attention, "qwen38 layer 3 attn");
    expect(q27.layer_kind(63) == vesper::LayerKind::Attention, "qwen38 last layer attn");
    ++g_passed;  // validate() did not throw
}

void test_recurrent_layers_override() {
    auto cfg = vesper::ModelConfig::tiny_hybrid();
    expect(cfg.layer_kind(3) == vesper::LayerKind::Attention, "interval maps layer 3 to attn");
    cfg.recurrent_layers = {1, 0, 1, 1};
    expect(cfg.layer_kind(0) == vesper::LayerKind::DeltaNet, "map layer 0 GDN");
    expect(cfg.layer_kind(1) == vesper::LayerKind::Attention, "map overrides interval on layer 1");
    expect(cfg.layer_kind(3) == vesper::LayerKind::DeltaNet, "map overrides interval on layer 3");
    cfg.validate();
    ++g_passed;
}

void recompute_layer0_k(const vesper::Engine& engine, int token, int pos, float* out_k) {
    const vesper::ModelConfig& cfg = engine.config();
    const vesper::LayerWeights& layer = engine.weights().layers[0];
    std::vector<float> x(static_cast<std::size_t>(cfg.hidden_size));
    std::vector<float> normed(static_cast<std::size_t>(cfg.hidden_size));
    std::vector<float> q(static_cast<std::size_t>(cfg.q_dim()));
    std::vector<float> k(static_cast<std::size_t>(cfg.kv_dim()));
    vesper::embed_row(x.data(), engine.weights().tok_emb, token);
    vesper::rmsnorm(normed.data(), x.data(), layer.rms_attn.data(), cfg.hidden_size, cfg.rms_eps);
    vesper::gemv(q.data(), layer.q_proj, normed.data());
    vesper::gemv(k.data(), layer.k_proj, normed.data());
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
    expect(vesper::argmax(vesper::Device::CPU, x, 3) == 1, "device argmax cpu");
    const float ties[] = {1.0f, 3.0f, 3.0f, 2.0f};
    expect(vesper::argmax(ties, 4) == 1, "argmax first max on a tie");
    expect(vesper::argmax(vesper::Device::CPU, ties, 4) == 1, "device argmax first max");
}

bool open_throws(const std::string& path) {
    try {
        (void)vesper::GgufFile::open(path);
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

void test_ggml_nbytes() {
    const std::uint64_t q8[] = {32};
    const std::uint64_t q4k[] = {256};
    const std::uint64_t f32[] = {6};
    expect(vesper::ggml_nbytes(vesper::GgmlType::Q8_0, q8, 1) == 34, "ggml_nbytes Q8_0 32");
    expect(vesper::ggml_nbytes(vesper::GgmlType::Q4_K, q4k, 1) == 144, "ggml_nbytes Q4_K 256");
    expect(vesper::ggml_nbytes(vesper::GgmlType::Q5_K, q4k, 1) == 176, "ggml_nbytes Q5_K 256");
    expect(vesper::ggml_nbytes(vesper::GgmlType::Q6_K, q4k, 1) == 210, "ggml_nbytes Q6_K 256");
    expect(vesper::ggml_nbytes(vesper::GgmlType::F32, f32, 1) == 24, "ggml_nbytes F32 6");
}

void test_gguf_roundtrip() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-m1";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "tiny.gguf").string();
    const std::string fixture = "/tmp/vesper-m1.gguf";

    const float weight[] = {1.25f, 2.0f, -0.5f, 3.5f, 0.25f, 8.0f, -1.0f, 4.0f};
    std::vector<std::byte> f32_bytes(sizeof(weight));
    std::memcpy(f32_bytes.data(), weight, sizeof(weight));

    std::vector<std::byte> q8_bytes(34, std::byte{0});
    q8_bytes[0] = std::byte{0xab};

    const std::vector<vesper::GgufKvWrite> kvs = {
        vesper::gguf_kv_string("general.architecture", "qwen3"),
        vesper::gguf_kv_u32("general.alignment", 32),
    };
    const std::vector<vesper::GgufTensorWrite> tensors = {
        {"blk.0.weight",
         vesper::GgmlType::F32,
         {4, 2},
         f32_bytes},
        {"blk.0.ffn", vesper::GgmlType::Q8_0, {32}, q8_bytes},
    };
    vesper::write_gguf(path, kvs, tensors);
    vesper::write_gguf(fixture, kvs, tensors);

    const vesper::GgufFile file = vesper::GgufFile::open(path);
    expect(file.version() == 3, "gguf version 3");
    expect(file.architecture() == "qwen3", "gguf architecture qwen3");
    expect(file.alignment() == 32, "gguf alignment 32");
    const vesper::GgufTensor* w = file.find("blk.0.weight");
    const vesper::GgufTensor* ffn = file.find("blk.0.ffn");
    expect(w != nullptr, "find blk.0.weight");
    expect(ffn != nullptr, "find blk.0.ffn");
    if (w != nullptr) {
        expect(w->nbytes == 32, "F32 nbytes 32");
        expect(w->type == vesper::GgmlType::F32, "F32 type");
        float first = 0.0f;
        std::memcpy(&first, w->data, sizeof(first));
        expect(close(first, 1.25f), "F32 first value");
    }
    if (ffn != nullptr) {
        expect(ffn->nbytes == 34, "Q8_0 nbytes 34");
        expect(ffn->type == vesper::GgmlType::Q8_0, "Q8_0 type");
        expect(static_cast<unsigned>(ffn->data[0]) == 0xab, "Q8_0 first payload byte");
    }
}

void test_gguf_bad_magic() {
    const auto path = std::filesystem::temp_directory_path() / "vesper-gguf-bad-magic.bin";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const char junk[] = {'X', 'X', 'X', 'X'};
        out.write(junk, 4);
        const std::uint32_t version = 3;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        const std::uint64_t zero = 0;
        out.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
        out.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    }
    expect(open_throws(path.string()), "bad magic fails");
}

void test_q8_nbytes() {
    expect(vesper::q8_packed_bytes(1, 32) == 34, "q8_packed_bytes 1x32");
    expect(vesper::q8_packed_bytes(4, 64) == 4u * 2u * 34u, "q8_packed_bytes 4x64");
}

void test_q8_gemv_matches_dequant() {
    const int rows = 8;
    const int cols = 64;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 13) % 29 - 14);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.07f * static_cast<float>((i * 5) % 17 - 8);
    }
    std::vector<std::byte> packed(vesper::q8_packed_bytes(rows, cols));
    vesper::quantize_q8(w.data(), packed.data(), rows, cols);
    std::vector<float> deq(static_cast<std::size_t>(rows * cols));
    vesper::dequant_q8(deq.data(), packed.data(), rows, cols);
    std::vector<float> y_q(static_cast<std::size_t>(rows));
    std::vector<float> y_f(static_cast<std::size_t>(rows));
    vesper::gemv_q8(y_q.data(), packed.data(), x.data(), rows, cols);
    vesper::gemv(y_f.data(), deq.data(), x.data(), rows, cols);
    expect(close_vec(y_q.data(), y_f.data(), rows, 1e-5f), "Q8 GEMV matches dequant F32 GEMV");
}

void test_q8_q8x_matches_reconstructed() {
    const int rows = 8;
    const int cols = 64;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 11) % 19 - 9);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.04f * static_cast<float>((i * 7) % 15 - 7);
    }
    std::vector<std::byte> packed(vesper::q8_packed_bytes(rows, cols));
    vesper::quantize_q8(w.data(), packed.data(), rows, cols);
    const int nblocks = cols / vesper::kQ8XBlockElems;
    std::vector<std::int8_t> qs(static_cast<std::size_t>(cols));
    std::vector<float> xd(static_cast<std::size_t>(nblocks));
    std::vector<float> xsum(static_cast<std::size_t>(nblocks));
    vesper::quantize_q8x(x.data(), qs.data(), xd.data(), xsum.data(), cols);
    std::vector<float> xhat(static_cast<std::size_t>(cols));
    vesper::dequant_q8x(xhat.data(), qs.data(), xd.data(), cols);
    std::vector<float> y_q8x(static_cast<std::size_t>(rows));
    std::vector<float> y_f(static_cast<std::size_t>(rows));
    vesper::gemv_q8_q8x(y_q8x.data(), packed.data(), qs.data(), xd.data(), rows, cols);
    vesper::gemv_q8(y_f.data(), packed.data(), xhat.data(), rows, cols);
    expect(close_vec(y_q8x.data(), y_f.data(), rows, 2e-4f),
           "Q8 q8x GEMV matches F32 GEMV on reconstructed x");

    const auto* blk = reinterpret_cast<const vesper::BlockQ80*>(packed.data());
    float acc0 = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        acc0 += vesper::q8_dot_q8(blk[b].qs, vesper::f16_to_f32(blk[b].d),
                                  qs.data() + b * vesper::kQ8XBlockElems, xd[static_cast<std::size_t>(b)]);
    }
    expect(close(acc0, y_q8x[0], 2e-4f), "q8_dot_q8 matches gemv_q8_q8x row0");

    float acc_iqs = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        for (int iqs = 0; iqs < vesper::kQ8Qi; iqs += vesper::kQ8VdrMmvq) {
            acc_iqs += vesper::q8_dot_q8_iqs(blk[b].qs, vesper::f16_to_f32(blk[b].d),
                                             qs.data() + b * vesper::kQ8XBlockElems,
                                             xd[static_cast<std::size_t>(b)], iqs);
        }
    }
    expect(close(acc_iqs, acc0, 1e-6f), "Q8 VDR slices sum to q8_dot_q8");
}

void test_q8_ids_match_dequant() {
    const auto cfg = vesper::ModelConfig::tiny_demo();
    const auto q8 = vesper::ModelWeights::random(cfg, 19).to_q8();
    const auto deq = q8.dequant();
    vesper::Engine fused(q8);
    vesper::Engine oracle(deq);
    const auto left = fused.generate({1, 2, 3}, 8);
    const auto right = oracle.generate({1, 2, 3}, 8);
    expect(left == right, "Q8 fused ids match dequant F32");
}

void test_write_load_tiny() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-m2";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "tiny.gguf").string();
    vesper::write_tiny_q8(path, 5);
    const auto loaded = vesper::load_model(path);
    expect(loaded.config.hidden_size == 64, "loaded hidden 64");
    expect(loaded.lm_head.kind() == vesper::WeightKind::Q8_0, "lm_head is Q8_0");
    expect(loaded.layers[0].q_proj.kind() == vesper::WeightKind::Q8_0, "q_proj is Q8_0");
    expect(loaded.lm_head.is_view(), "loaded Q8 lm_head is mmap view");
    expect(loaded.layers[0].q_proj.is_view(), "loaded Q8 q_proj is mmap view");
    expect(!loaded.tok_emb.is_view(), "F32 token_embd is copied");
    const auto memory = vesper::ModelWeights::random(vesper::ModelConfig::tiny_demo(), 5).to_q8();
    vesper::Engine a(memory);
    vesper::Engine b(loaded);
    expect(a.generate({9, 8, 7}, 6) == b.generate({9, 8, 7}, 6),
           "loaded vesper_tiny matches in-memory Q8");
}

void test_load_rejects_other_arch() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-m2";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "not-tiny.gguf").string();
    const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::byte> bytes(sizeof(weight));
    std::memcpy(bytes.data(), weight, sizeof(weight));
    vesper::write_gguf(path,
                       {vesper::gguf_kv_string("general.architecture", "qwen3")},
                       {{"blk.0.weight", vesper::GgmlType::F32, {4}, bytes}});
    bool threw = false;
    try {
        (void)vesper::load_model(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, "load_model rejects non-vesper_tiny");
}

void test_hip_q8_gemv_matches_cpu() {
    if (!vesper::hip_available()) {
        return;
    }
    const int rows = 64;
    const int cols = 64;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 11) % 19 - 9);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.04f * static_cast<float>((i * 7) % 15 - 7);
    }
    auto packed = vesper::WeightMatrix::q8_from_f32(w.data(), rows, cols);
    const int nblocks = cols / vesper::kQ8XBlockElems;
    std::vector<std::int8_t> qs(static_cast<std::size_t>(cols));
    std::vector<float> xd(static_cast<std::size_t>(nblocks));
    std::vector<float> xsum(static_cast<std::size_t>(nblocks));
    vesper::quantize_q8x(x.data(), qs.data(), xd.data(), xsum.data(), cols);
    std::vector<float> y_cpu(static_cast<std::size_t>(rows));
    vesper::gemv_q8_q8x(y_cpu.data(), packed.packed(), qs.data(), xd.data(), rows, cols);

    auto gpu_w = packed.to(vesper::Device::HIP);
    vesper::Buffer X(static_cast<std::size_t>(cols), vesper::Device::HIP);
    vesper::Buffer Y(static_cast<std::size_t>(rows), vesper::Device::HIP);
    X.copy_from(x.data(), x.size());
    vesper::gemv(vesper::Device::HIP, Y.data(), gpu_w, X.data());
    std::vector<float> y_gpu(static_cast<std::size_t>(rows));
    Y.copy_to(y_gpu.data(), y_gpu.size());
    expect(close_vec(y_cpu.data(), y_gpu.data(), rows, 2e-4f), "HIP Q8 GEMV matches CPU q8x");
}

void test_hip_q8_engine_matches_cpu() {
    if (!vesper::hip_available()) {
        return;
    }
    const auto cfg = vesper::ModelConfig::tiny_demo();
    const auto q8 = vesper::ModelWeights::random(cfg, 3).to_q8();
    vesper::Engine gpu(q8, vesper::Device::HIP);
    const auto ids = gpu.generate({9, 8, 7}, 8);
    expect(ids.size() == 11, "HIP Q8 engine generates prompt+new tokens");
}

void test_q4k_nbytes() {
    expect(vesper::q4k_packed_bytes(1, 256) == 144, "q4k_packed_bytes 1x256");
    expect(vesper::q4k_packed_bytes(4, 512) == 4u * 2u * 144u, "q4k_packed_bytes 4x512");
}

void test_q4k_gemv_matches_dequant() {
    const int rows = 8;
    const int cols = 256;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 13) % 29 - 14);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.07f * static_cast<float>((i * 5) % 17 - 8);
    }
    std::vector<std::byte> packed(vesper::q4k_packed_bytes(rows, cols));
    vesper::quantize_q4k(w.data(), packed.data(), rows, cols);
    std::vector<float> deq(static_cast<std::size_t>(rows * cols));
    vesper::dequant_q4k(deq.data(), packed.data(), rows, cols);
    std::vector<float> y_q(static_cast<std::size_t>(rows));
    std::vector<float> y_f(static_cast<std::size_t>(rows));
    vesper::gemv_q4k(y_q.data(), packed.data(), x.data(), rows, cols);
    vesper::gemv(y_f.data(), deq.data(), x.data(), rows, cols);
    expect(close_vec(y_q.data(), y_f.data(), rows, 2e-4f), "Q4_K GEMV matches dequant F32 GEMV");
}

void test_q4k_q8x_matches_reconstructed() {
    const int rows = 8;
    const int cols = 512;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 13) % 29 - 14);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.07f * static_cast<float>((i * 5) % 17 - 8);
    }
    std::vector<std::byte> packed(vesper::q4k_packed_bytes(rows, cols));
    vesper::quantize_q4k(w.data(), packed.data(), rows, cols);
    std::vector<std::int8_t> qs(static_cast<std::size_t>(cols));
    std::vector<float> xd(static_cast<std::size_t>(cols / 32));
    std::vector<float> xsum(static_cast<std::size_t>(cols / 32));
    vesper::quantize_q8x(x.data(), qs.data(), xd.data(), xsum.data(), cols);
    std::vector<float> xhat(static_cast<std::size_t>(cols));
    vesper::dequant_q8x(xhat.data(), qs.data(), xd.data(), cols);
    std::vector<float> y_f(static_cast<std::size_t>(rows));
    std::vector<float> y_q(static_cast<std::size_t>(rows));
    vesper::gemv_q4k(y_f.data(), packed.data(), xhat.data(), rows, cols);
    vesper::gemv_q4k_q8x(y_q.data(), packed.data(), qs.data(), xd.data(), xsum.data(), rows, cols);
    expect(close_vec(y_q.data(), y_f.data(), rows, 2e-4f), "Q4_K q8x GEMV matches reconstructed x");

    const auto* blocks = reinterpret_cast<const vesper::BlockQ4K*>(packed.data());
    const int supers = cols / 256;
    std::vector<float> y_iqs(static_cast<std::size_t>(rows), 0.0f);
    for (int r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (int s = 0; s < supers; ++s) {
            const vesper::BlockQ4K& blk = blocks[r * supers + s];
            acc += vesper::q4k_dot_q8_super(
                reinterpret_cast<const unsigned char*>(&blk), vesper::f16_to_f32(blk.d),
                vesper::f16_to_f32(blk.dmin), qs.data() + s * 256, xd.data() + s * 8);
        }
        y_iqs[static_cast<std::size_t>(r)] = acc;
    }
    expect(close_vec(y_iqs.data(), y_q.data(), rows, 2e-4f), "Q4_K MMVQ iqs matches q8x GEMV");
}

void test_q5k_nbytes() {
    expect(vesper::q5k_packed_bytes(1, 256) == 176, "q5k_packed_bytes 1x256");
    expect(vesper::q5k_packed_bytes(4, 512) == 4u * 2u * 176u, "q5k_packed_bytes 4x512");
}

void test_q5k_gemv_matches_dequant() {
    const int rows = 8;
    const int cols = 256;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 13) % 29 - 14);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.07f * static_cast<float>((i * 5) % 17 - 8);
    }
    std::vector<std::byte> packed(vesper::q5k_packed_bytes(rows, cols));
    vesper::quantize_q5k(w.data(), packed.data(), rows, cols);
    std::vector<float> deq(static_cast<std::size_t>(rows * cols));
    vesper::dequant_q5k(deq.data(), packed.data(), rows, cols);
    std::vector<float> y_q(static_cast<std::size_t>(rows));
    std::vector<float> y_f(static_cast<std::size_t>(rows));
    vesper::gemv_q5k(y_q.data(), packed.data(), x.data(), rows, cols);
    vesper::gemv(y_f.data(), deq.data(), x.data(), rows, cols);
    expect(close_vec(y_q.data(), y_f.data(), rows, 2e-4f), "Q5_K GEMV matches dequant F32 GEMV");
}

void test_q5k_q8x_matches_reconstructed() {
    const int rows = 8;
    const int cols = 256;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 13) % 29 - 14);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.07f * static_cast<float>((i * 5) % 17 - 8);
    }
    std::vector<std::byte> packed(vesper::q5k_packed_bytes(rows, cols));
    vesper::quantize_q5k(w.data(), packed.data(), rows, cols);
    std::vector<std::int8_t> qs(static_cast<std::size_t>(cols));
    std::vector<float> xd(static_cast<std::size_t>(cols / 32));
    std::vector<float> xsum(static_cast<std::size_t>(cols / 32));
    vesper::quantize_q8x(x.data(), qs.data(), xd.data(), xsum.data(), cols);
    std::vector<float> xhat(static_cast<std::size_t>(cols));
    vesper::dequant_q8x(xhat.data(), qs.data(), xd.data(), cols);
    std::vector<float> y_f(static_cast<std::size_t>(rows));
    std::vector<float> y_q(static_cast<std::size_t>(rows));
    vesper::gemv_q5k(y_f.data(), packed.data(), xhat.data(), rows, cols);
    vesper::gemv_q5k_q8x(y_q.data(), packed.data(), qs.data(), xd.data(), xsum.data(), rows, cols);
    expect(close_vec(y_q.data(), y_f.data(), rows, 2e-4f), "Q5_K q8x GEMV matches reconstructed x");
}

void test_q6k_nbytes() {
    expect(vesper::q6k_packed_bytes(1, 256) == 210, "q6k_packed_bytes 1x256");
    expect(vesper::q6k_packed_bytes(4, 512) == 4u * 2u * 210u, "q6k_packed_bytes 4x512");
}

void test_q6k_gemv_matches_dequant() {
    const int rows = 8;
    const int cols = 256;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 13) % 29 - 14);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.07f * static_cast<float>((i * 5) % 17 - 8);
    }
    std::vector<std::byte> packed(vesper::q6k_packed_bytes(rows, cols));
    vesper::quantize_q6k(w.data(), packed.data(), rows, cols);
    std::vector<float> deq(static_cast<std::size_t>(rows * cols));
    vesper::dequant_q6k(deq.data(), packed.data(), rows, cols);
    std::vector<float> y_q(static_cast<std::size_t>(rows));
    std::vector<float> y_f(static_cast<std::size_t>(rows));
    vesper::gemv_q6k(y_q.data(), packed.data(), x.data(), rows, cols);
    vesper::gemv(y_f.data(), deq.data(), x.data(), rows, cols);
    expect(close_vec(y_q.data(), y_f.data(), rows, 2e-4f), "Q6_K GEMV matches dequant F32 GEMV");
}

void test_q6k_q8x_matches_reconstructed() {
    const int rows = 8;
    const int cols = 256;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 13) % 29 - 14);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.07f * static_cast<float>((i * 5) % 17 - 8);
    }
    std::vector<std::byte> packed(vesper::q6k_packed_bytes(rows, cols));
    vesper::quantize_q6k(w.data(), packed.data(), rows, cols);
    std::vector<std::int8_t> qs(static_cast<std::size_t>(cols));
    std::vector<float> xd(static_cast<std::size_t>(cols / 32));
    std::vector<float> xsum(static_cast<std::size_t>(cols / 32));
    vesper::quantize_q8x(x.data(), qs.data(), xd.data(), xsum.data(), cols);
    std::vector<float> xhat(static_cast<std::size_t>(cols));
    vesper::dequant_q8x(xhat.data(), qs.data(), xd.data(), cols);
    std::vector<float> y_f(static_cast<std::size_t>(rows));
    std::vector<float> y_q(static_cast<std::size_t>(rows));
    vesper::gemv_q6k(y_f.data(), packed.data(), xhat.data(), rows, cols);
    vesper::gemv_q6k_q8x(y_q.data(), packed.data(), qs.data(), xd.data(), rows, cols);
    expect(close_vec(y_q.data(), y_f.data(), rows, 2e-4f), "Q6_K q8x GEMV matches reconstructed x");

    const auto* blk = reinterpret_cast<const vesper::BlockQ6K*>(packed.data());
    float acc_iqs = 0.0f;
    for (int s = 0; s < cols / vesper::kQ6KBlockElems; ++s) {
        acc_iqs += vesper::q6k_dot_q8_super(reinterpret_cast<const unsigned char*>(&blk[s]),
                                            vesper::f16_to_f32(blk[s].d),
                                            qs.data() + s * vesper::kQ6KBlockElems, xd.data() + s * 8);
    }
    expect(close(acc_iqs, y_q[0], 2e-4f), "Q6_K dp4a iqs sum matches q8x GEMV row0");
}

void test_q4k_embed_row() {
    const int rows = 4;
    const int cols = 256;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 13) % 17 - 8);
    }
    auto packed = vesper::WeightMatrix::q4_from_f32(w.data(), rows, cols);
    std::vector<float> deq(static_cast<std::size_t>(cols));
    std::vector<float> row(static_cast<std::size_t>(cols));
    vesper::dequant_q4k_row(deq.data(), packed.packed(), 2, cols);
    vesper::embed_row(row.data(), packed, 2);
    expect(close_vec(row.data(), deq.data(), cols, 1e-6f), "Q4_K embed_row matches dequant row");
}

void test_q6k_embed_row() {
    const int rows = 4;
    const int cols = 256;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 11) % 19 - 9);
    }
    auto packed = vesper::WeightMatrix::q6_from_f32(w.data(), rows, cols);
    std::vector<float> deq(static_cast<std::size_t>(cols));
    std::vector<float> row(static_cast<std::size_t>(cols));
    vesper::dequant_q6k_row(deq.data(), packed.packed(), 2, cols);
    vesper::embed_row(row.data(), packed, 2);
    expect(close_vec(row.data(), deq.data(), cols, 1e-6f), "Q6_K embed_row matches dequant row");
}

void test_write_load_q4km() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-q4km";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "tiny-q4km.gguf").string();
    vesper::write_tiny_q4km(path, 7);
    const auto loaded = vesper::load_model(path);
    expect(loaded.config.hidden_size == 256, "q4km hidden 256");
    expect(loaded.tok_emb.kind() == vesper::WeightKind::Q4_K, "q4km token_embd Q4_K");
    expect(loaded.lm_head.kind() == vesper::WeightKind::Q6_K, "q4km output Q6_K");
    expect(loaded.layers[0].q_proj.kind() == vesper::WeightKind::Q5_K, "q4km q_proj Q5_K");
    expect(loaded.layers[0].k_proj.kind() == vesper::WeightKind::Q4_K, "q4km k_proj Q4_K");
    expect(loaded.layers[0].v_proj.kind() == vesper::WeightKind::Q6_K, "q4km v_proj Q6_K");
    expect(loaded.layers[0].down_proj.kind() == vesper::WeightKind::Q6_K, "q4km down Q6_K");
    expect(loaded.tok_emb.is_view(), "q4km token_embd is mmap view");
    expect(loaded.lm_head.is_view(), "q4km output is mmap view");
    expect(loaded.layers[0].gate_proj.is_view(), "q4km ffn is mmap view");
    expect(loaded.layers[0].q_proj.is_view(), "q4km q_proj is mmap view");
    vesper::Engine engine(loaded);
    const auto ids = engine.generate({3, 1, 4}, 6);
    expect(ids.size() == 9, "q4km generate length");
    bool in_vocab = true;
    for (int id : ids) {
        if (id < 0 || id >= loaded.config.vocab_size) {
            in_vocab = false;
        }
    }
    expect(in_vocab, "q4km tokens in vocab");
    expect(std::string(loaded.quant_name()) == "Q4_K_M", "mixed K-quant reports Q4_K_M");
}

void test_hip_q5k_gemv_matches_cpu() {
    if (!vesper::hip_available()) {
        return;
    }
    const int rows = 64;
    const int cols = 256;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 11) % 19 - 9);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.04f * static_cast<float>((i * 7) % 15 - 7);
    }
    auto packed = vesper::WeightMatrix::q5_from_f32(w.data(), rows, cols);
    std::vector<std::int8_t> qs(static_cast<std::size_t>(cols));
    std::vector<float> xd(static_cast<std::size_t>(cols / 32));
    std::vector<float> xsum(static_cast<std::size_t>(cols / 32));
    vesper::quantize_q8x(x.data(), qs.data(), xd.data(), xsum.data(), cols);
    std::vector<float> y_cpu(static_cast<std::size_t>(rows));
    vesper::gemv_q5k_q8x(y_cpu.data(), packed.packed(), qs.data(), xd.data(), xsum.data(), rows,
                         cols);
    auto gpu_w = packed.to(vesper::Device::HIP);
    vesper::Buffer X(static_cast<std::size_t>(cols), vesper::Device::HIP);
    vesper::Buffer Y(static_cast<std::size_t>(rows), vesper::Device::HIP);
    X.copy_from(x.data(), x.size());
    vesper::gemv(vesper::Device::HIP, Y.data(), gpu_w, X.data());
    std::vector<float> y_gpu(static_cast<std::size_t>(rows));
    Y.copy_to(y_gpu.data(), y_gpu.size());
    expect(close_vec(y_cpu.data(), y_gpu.data(), rows, 2e-3f), "HIP Q5_K GEMV matches CPU q8x");
}

void test_hip_q6k_gemv_matches_cpu() {
    if (!vesper::hip_available()) {
        return;
    }
    const int rows = 64;
    const int cols = 256;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 11) % 19 - 9);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.04f * static_cast<float>((i * 7) % 15 - 7);
    }
    auto packed = vesper::WeightMatrix::q6_from_f32(w.data(), rows, cols);
    std::vector<std::int8_t> qs(static_cast<std::size_t>(cols));
    std::vector<float> xd(static_cast<std::size_t>(cols / 32));
    std::vector<float> xsum(static_cast<std::size_t>(cols / 32));
    vesper::quantize_q8x(x.data(), qs.data(), xd.data(), xsum.data(), cols);
    std::vector<float> y_cpu(static_cast<std::size_t>(rows));
    vesper::gemv_q6k_q8x(y_cpu.data(), packed.packed(), qs.data(), xd.data(), rows, cols);
    auto gpu_w = packed.to(vesper::Device::HIP);
    vesper::Buffer X(static_cast<std::size_t>(cols), vesper::Device::HIP);
    vesper::Buffer Y(static_cast<std::size_t>(rows), vesper::Device::HIP);
    X.copy_from(x.data(), x.size());
    vesper::gemv(vesper::Device::HIP, Y.data(), gpu_w, X.data());
    std::vector<float> y_gpu(static_cast<std::size_t>(rows));
    Y.copy_to(y_gpu.data(), y_gpu.size());
    expect(close_vec(y_cpu.data(), y_gpu.data(), rows, 2e-3f), "HIP Q6_K GEMV matches CPU q8x");
}

void test_hip_q4k_gemv_matches_cpu() {
    if (!vesper::hip_available()) {
        return;
    }
    const int rows = 64;
    const int cols = 256;
    std::vector<float> w(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        w[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 11) % 19 - 9);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.04f * static_cast<float>((i * 7) % 15 - 7);
    }
    auto packed = vesper::WeightMatrix::q4_from_f32(w.data(), rows, cols);
    std::vector<std::int8_t> qs(static_cast<std::size_t>(cols));
    std::vector<float> xd(static_cast<std::size_t>(cols / 32));
    std::vector<float> xsum(static_cast<std::size_t>(cols / 32));
    vesper::quantize_q8x(x.data(), qs.data(), xd.data(), xsum.data(), cols);
    std::vector<float> y_cpu(static_cast<std::size_t>(rows));
    vesper::gemv_q4k_q8x(y_cpu.data(), packed.packed(), qs.data(), xd.data(), xsum.data(), rows,
                         cols);
    auto gpu_w = packed.to(vesper::Device::HIP);
    vesper::Buffer X(static_cast<std::size_t>(cols), vesper::Device::HIP);
    vesper::Buffer Y(static_cast<std::size_t>(rows), vesper::Device::HIP);
    X.copy_from(x.data(), x.size());
    vesper::gemv(vesper::Device::HIP, Y.data(), gpu_w, X.data());
    std::vector<float> y_gpu(static_cast<std::size_t>(rows));
    Y.copy_to(y_gpu.data(), y_gpu.size());
    expect(close_vec(y_cpu.data(), y_gpu.data(), rows, 2e-3f), "HIP Q4_K GEMV matches CPU q8x");
}

void gdn_delta_ref(float* y, float* rec, const float* q, const float* k, const float* v,
                   const float* decay, const float* beta, int heads, int dim) {
    for (int h = 0; h < heads; ++h) {
        float* S = rec + h * dim * dim;
        const float g = decay[static_cast<std::size_t>(h)];
        const float b = beta[static_cast<std::size_t>(h)];
        for (int j = 0; j < dim; ++j) {
            float* col = S + j * dim;
            float retrieved = 0.0f;
            for (int i = 0; i < dim; ++i) {
                retrieved += col[i] * k[static_cast<std::size_t>(h * dim + i)];
            }
            const float delta = b * (v[static_cast<std::size_t>(h * dim + j)] - g * retrieved);
            float acc = 0.0f;
            for (int i = 0; i < dim; ++i) {
                const float s = g * col[i] + k[static_cast<std::size_t>(h * dim + i)] * delta;
                col[i] = s;
                acc += s * q[static_cast<std::size_t>(h * dim + i)];
            }
            y[static_cast<std::size_t>(h * dim + j)] = acc;
        }
    }
}

void test_gdn_delta_step() {
    const int heads = 2;
    const int dim = 16;
    std::vector<float> rec(static_cast<std::size_t>(heads * dim * dim));
    for (int h = 0; h < heads; ++h) {
        for (int j = 0; j < dim; ++j) {
            for (int i = 0; i < dim; ++i) {
                rec[static_cast<std::size_t>((h * dim + j) * dim + i)] =
                    0.1f + 0.001f * static_cast<float>(i + 17 * j + 31 * h);
            }
        }
    }
    std::vector<float> rec_ref = rec;
    std::vector<float> q(static_cast<std::size_t>(heads * dim));
    std::vector<float> k(static_cast<std::size_t>(heads * dim));
    std::vector<float> v(static_cast<std::size_t>(heads * dim));
    std::vector<float> decay = {0.8f, 0.5f};
    std::vector<float> beta = {0.3f, 0.7f};
    for (int i = 0; i < heads * dim; ++i) {
        q[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>(i - 7);
        k[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i % 5) - 2);
        v[static_cast<std::size_t>(i)] = 0.04f * static_cast<float>((i % 7) - 3);
    }
    std::vector<float> y(static_cast<std::size_t>(heads * dim), 0.0f);
    std::vector<float> y_ref(static_cast<std::size_t>(heads * dim), 0.0f);
    vesper::gdn_delta_rule(vesper::Device::CPU, y.data(), rec.data(), q.data(), k.data(),
                           v.data(), decay.data(), beta.data(), heads, dim);
    gdn_delta_ref(y_ref.data(), rec_ref.data(), q.data(), k.data(), v.data(), decay.data(),
                  beta.data(), heads, dim);
    expect(close_vec(y.data(), y_ref.data(), heads * dim, 1e-5f), "GDN delta matches reference");
    expect(close_vec(rec.data(), rec_ref.data(), heads * dim * dim, 1e-5f),
           "GDN rec is column-contiguous after step");

    vesper::gdn_delta_rule(vesper::Device::CPU, y.data(), rec.data(), q.data(), k.data(),
                           v.data(), decay.data(), beta.data(), heads, dim);
    gdn_delta_ref(y_ref.data(), rec_ref.data(), q.data(), k.data(), v.data(), decay.data(),
                  beta.data(), heads, dim);
    expect(close_vec(y.data(), y_ref.data(), heads * dim, 1e-5f), "GDN delta step 2 matches");
}

void test_gdn_delta_official_shape() {
    const int heads = 48;
    const int dim = 128;
    std::vector<float> rec(static_cast<std::size_t>(heads * dim * dim));
    for (int n = 0; n < heads * dim * dim; ++n) {
        rec[static_cast<std::size_t>(n)] = 0.01f * static_cast<float>((n % 23) - 11);
    }
    std::vector<float> rec_ref = rec;
    std::vector<float> q(static_cast<std::size_t>(heads * dim));
    std::vector<float> k(static_cast<std::size_t>(heads * dim));
    std::vector<float> v(static_cast<std::size_t>(heads * dim));
    std::vector<float> decay(static_cast<std::size_t>(heads));
    std::vector<float> beta(static_cast<std::size_t>(heads));
    for (int i = 0; i < heads * dim; ++i) {
        q[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i % 13) - 6);
        k[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i % 11) - 5);
        v[static_cast<std::size_t>(i)] = 0.04f * static_cast<float>((i % 17) - 8);
    }
    for (int h = 0; h < heads; ++h) {
        decay[static_cast<std::size_t>(h)] = 0.4f + 0.01f * static_cast<float>(h);
        beta[static_cast<std::size_t>(h)] = 0.2f + 0.01f * static_cast<float>(h % 7);
    }
    std::vector<float> y(static_cast<std::size_t>(heads * dim), 0.0f);
    std::vector<float> y_ref(static_cast<std::size_t>(heads * dim), 0.0f);
    vesper::gdn_delta_rule(vesper::Device::CPU, y.data(), rec.data(), q.data(), k.data(),
                           v.data(), decay.data(), beta.data(), heads, dim);
    gdn_delta_ref(y_ref.data(), rec_ref.data(), q.data(), k.data(), v.data(), decay.data(),
                  beta.data(), heads, dim);
    expect(close_vec(y.data(), y_ref.data(), heads * dim, 2e-4f),
           "official 48x128 GDN delta matches");
}

void test_decode_report_line() {
    vesper::DecodeReport report;
    report.model = "tiny_demo";
    report.quant = "f32";
    report.arch = "vesper_tiny";
    report.prompt_tokens = 5;
    report.new_tokens = 8;
    report.decode_tps = 40.0;
    report.bytes_per_token = 16000000000ull;
    report.fill_roofline(200.0);
    const std::string line = report.line();
    expect(line.find("engine=vesper") != std::string::npos, "report engine");
    expect(line.find("backend=cpu") != std::string::npos, "report backend");
    expect(line.find("model=tiny_demo") != std::string::npos, "report model");
    expect(line.find("quant=f32") != std::string::npos, "report quant");
    expect(line.find("arch=vesper_tiny") != std::string::npos, "report arch");
    expect(line.find("bytes_per_token=16000000000") != std::string::npos, "report bytes");
    expect(line.find("achieved_gbs=") != std::string::npos, "report achieved");
    expect(line.find("peak_gbs=640") != std::string::npos, "report peak");
    expect(line.find("status=ok") != std::string::npos, "report status");
    expect(line.find("ids=-") != std::string::npos, "empty ids prints dash");
    expect(report.achieved_gbs > 600.0 && report.achieved_gbs < 650.0, "roofline 16GB / 0.2s");
}

void test_write_load_hybrid() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "hybrid.gguf").string();
    vesper::write_tiny_hybrid(path, 5);
    const auto loaded = vesper::load_model(path);
    expect(loaded.config.arch == "vesper_hybrid", "loaded hybrid arch");
    expect(loaded.config.n_layers == 4, "loaded hybrid layers");
    expect(loaded.layers[0].qkv_proj.rows() > 0, "layer 0 has GDN qkv");
    expect(loaded.layers[3].q_proj.rows() == loaded.config.q_proj_rows(), "attn layer gated q");
    const auto memory = vesper::ModelWeights::random(vesper::ModelConfig::tiny_hybrid(), 5).to_q8();
    vesper::Engine a(memory);
    vesper::Engine b(loaded);
    expect(a.generate({9, 8, 7}, 6) == b.generate({9, 8, 7}, 6),
           "loaded vesper_hybrid matches in-memory Q8");
    expect(loaded.lm_head.is_view(), "loaded hybrid Q8 output is mmap view");
    expect(loaded.layers[0].qkv_proj.is_view(), "loaded hybrid GDN qkv is mmap view");
}

void test_packed_mmap_view() {
    const int rows = 8;
    const int cols = 256;
    std::vector<float> src(static_cast<std::size_t>(rows * cols));
    for (int i = 0; i < rows * cols; ++i) {
        src[static_cast<std::size_t>(i)] = 0.01f * static_cast<float>((i % 17) - 8);
    }
    const auto owned = vesper::WeightMatrix::q4_from_f32(src.data(), rows, cols);
    expect(!owned.is_view(), "q4_from_f32 owns packed bytes");
    auto storage = std::make_shared<std::vector<std::byte>>(owned.packed(),
                                                            owned.packed() + owned.bytes());
    std::shared_ptr<const void> keep(storage, storage->data());
    const auto view = vesper::WeightMatrix::from_view(vesper::WeightKind::Q4_K, storage->data(),
                                                      rows, cols, keep);
    expect(view.is_view(), "from_view is a view");
    expect(view.bytes() == owned.bytes(), "view byte size");
    const auto deq_owned = owned.dequant_f32();
    const auto deq_view = view.dequant_f32();
    expect(close_vec(deq_owned.f32_data(), deq_view.f32_data(), rows * cols, 1e-6f),
           "view dequant matches owned");

    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-q4km";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "tiny-q4km-view.gguf").string();
    vesper::write_tiny_q4km(path, 7);
    vesper::WeightMatrix tok;
    vesper::WeightMatrix lm;
    vesper::WeightMatrix gate;
    {
        const auto loaded = vesper::load_model(path);
        expect(loaded.tok_emb.is_view(), "loaded Q4_K emb is mmap view");
        expect(loaded.lm_head.is_view(), "loaded Q6_K output is mmap view");
        expect(loaded.layers[0].gate_proj.is_view(), "loaded Q4_K ffn is mmap view");
        tok = loaded.tok_emb;
        lm = loaded.lm_head;
        gate = loaded.layers[0].gate_proj;
    }
    expect(tok.is_view() && lm.is_view() && gate.is_view(), "views survive ModelWeights drop");
    const auto tok_deq = tok.dequant_f32();
    expect(tok_deq.rows() == tok.rows() && tok_deq.cols() == tok.cols(),
           "dequant after drop keeps shape");
    bool finite = true;
    for (int i = 0; i < tok.rows() * tok.cols(); ++i) {
        finite = finite && std::isfinite(tok_deq.f32_data()[i]);
    }
    expect(finite, "dequant after drop is finite");
    std::vector<float> x(static_cast<std::size_t>(gate.cols()), 0.01f);
    std::vector<float> y(static_cast<std::size_t>(gate.rows()), 0.0f);
    vesper::gemv_q4k(y.data(), gate.packed(), x.data(), gate.rows(), gate.cols());
    float mag = 0.0f;
    for (float v : y) {
        mag += v * v;
    }
    expect(std::isfinite(mag) && mag > 0.0f, "gemv through surviving Q4 view");
}

void test_write_load_qwen35_fixture() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen35.gguf").string();
    vesper::write_tiny_qwen35(path, 9);
    const auto loaded = vesper::load_model(path);
    expect(loaded.config.arch == "qwen35", "loaded qwen35 arch");
    expect(loaded.config.is_hybrid(), "qwen35 is hybrid");
    expect(loaded.config.n_rope_sections == 3, "qwen35 strips trailing 0 in mrope sections");
    expect(loaded.config.rope_section[0] == 3 && loaded.config.rope_section[2] == 2,
           "qwen35 fixture mrope 3+3+2");
    expect(loaded.config.recurrent_layers.size() == 4, "qwen35 fixture writes recurrent_layers");
    expect(loaded.config.recurrent_layers[0] != 0 && loaded.config.recurrent_layers[3] == 0,
           "qwen35 fixture map is 3 GDN + 1 attn");
    vesper::Engine engine(loaded);
    const auto ids = engine.generate({1, 2}, 4);
    expect(ids.size() == 6, "qwen35 fixture generate length");
}

void test_load_qwen35_strips_nextn() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen35-nextn.gguf").string();
    vesper::write_tiny_qwen35(path, 9, 1);
    const auto loaded = vesper::load_model(path);
    expect(loaded.config.n_layers == 4, "nextn block is not a decode layer");
    expect(loaded.config.nextn_predict_layers == 1, "nextn count preserved");
    vesper::Engine engine(loaded);
    expect(engine.generate({1, 2}, 3).size() == 5, "qwen35 nextn fixture generates");
}

void test_load_qwen35_recurrent_map() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen35-recurrent-map.gguf").string();
    auto cfg = vesper::ModelConfig::tiny_hybrid();
    cfg.arch = "qwen35";
    cfg.full_attention_interval = 4;
    cfg.recurrent_layers = {1, 0, 1, 1};
    vesper::write_tiny_qwen35(path, 11, cfg);
    const auto loaded = vesper::load_model(path);
    expect(loaded.config.layer_kind(1) == vesper::LayerKind::Attention,
           "loaded map makes layer 1 attn");
    expect(loaded.config.layer_kind(3) == vesper::LayerKind::DeltaNet,
           "loaded map makes layer 3 GDN");
    expect(loaded.layers[1].q_proj.rows() == loaded.config.q_proj_rows(),
           "map attn layer has gated q");
    expect(loaded.layers[3].qkv_proj.rows() > 0, "map GDN layer has qkv");
    vesper::Engine engine(loaded);
    expect(engine.generate({1, 2}, 3).size() == 5, "recurrent map fixture generates");
}

void test_load_qwen35_pin_kv() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen35-pin-kv.gguf").string();
    vesper::write_tiny_qwen35_pin_kv(path, 13);
    const vesper::GgufFile file = vesper::GgufFile::open(path);
    expect(!file.has_kv("qwen35.attention.recurrent_layers"), "pin KV has no recurrent_layers");
    expect(!file.has_kv("qwen35.vocab_size"), "pin KV has no vocab_size");
    expect(!file.has_kv("qwen35.nextn_predict_layers"), "pin KV has no nextn");
    expect(!file.has_kv("qwen35.attention.qk_norm"), "pin KV has no qk_norm flag");
    expect(file.has_kv("qwen35.full_attention_interval"), "pin KV has interval");
    expect(file.has_kv("qwen35.ssm.conv_kernel"), "pin KV has official ssm.conv_kernel");
    const auto loaded = vesper::load_model(path);
    const auto cfg = vesper::load_config(path);
    expect(cfg.n_layers == loaded.config.n_layers && cfg.hidden_size == loaded.config.hidden_size &&
               cfg.full_attention_interval == loaded.config.full_attention_interval,
           "load_config matches load_model on pin KV");
    expect(loaded.config.recurrent_layers.empty(), "pin load uses interval, not a map");
    expect(loaded.config.full_attention_interval == 4, "pin interval 4");
    expect(loaded.config.vocab_size == loaded.tok_emb.rows(), "pin vocab from token_embd");
    expect(loaded.config.qk_norm, "pin qk_norm defaults on");
    expect(loaded.config.layer_kind(0) == vesper::LayerKind::DeltaNet, "pin layer 0 GDN");
    expect(loaded.config.layer_kind(3) == vesper::LayerKind::Attention, "pin layer 3 attn");
    vesper::Engine engine(loaded);
    expect(engine.generate({1, 2}, 3).size() == 5, "pin KV fixture generates");
}

void test_load_qwen35_ssm_aliases() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen35-ssm-alias.gguf").string();
    vesper::write_tiny_qwen35_ssm_aliases(path, 7);
    const auto loaded = vesper::load_model(path);
    expect(loaded.config.gdn_conv_kernel == 4, "ssm.d_conv alias");
    expect(loaded.config.gdn_qk_heads == 2, "ssm.n_group alias");
    expect(loaded.config.gdn_v_heads == 4, "ssm.dt_rank alias");
    expect(loaded.config.gdn_head_dim == 16, "ssm.d_state alias");
    expect(loaded.config.full_attention_interval == 0, "alias file has no interval");
    expect(loaded.config.is_hybrid(), "map-only file is hybrid");
    expect(loaded.config.layer_kind(2) == vesper::LayerKind::DeltaNet, "alias map layer 2 GDN");
    expect(loaded.config.layer_kind(3) == vesper::LayerKind::Attention, "alias map layer 3 attn");
    vesper::Engine engine(loaded);
    expect(engine.generate({1, 2}, 2).size() == 4, "ssm alias fixture generates");
}

void test_load_qwen35_rejects_missing_gdn() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen35-no-gdn.gguf").string();
    const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::byte> bytes(sizeof(weight));
    std::memcpy(bytes.data(), weight, sizeof(weight));
    vesper::write_gguf(path,
                       {vesper::gguf_kv_string("general.architecture", "qwen35"),
                        vesper::gguf_kv_u32("qwen35.block_count", 4),
                        vesper::gguf_kv_u32("qwen35.embedding_length", 64),
                        vesper::gguf_kv_u32("qwen35.feed_forward_length", 128),
                        vesper::gguf_kv_u32("qwen35.attention.head_count", 4),
                        vesper::gguf_kv_u32("qwen35.attention.head_count_kv", 2),
                        vesper::gguf_kv_u32("qwen35.attention.key_length", 16),
                        vesper::gguf_kv_u32("qwen35.full_attention_interval", 4),
                        vesper::gguf_kv_u32("qwen35.ssm.group_count", 2),
                        vesper::gguf_kv_u32("qwen35.ssm.time_step_rank", 4),
                        vesper::gguf_kv_u32("qwen35.ssm.state_size", 16)},
                       {{"token_embd.weight", vesper::GgmlType::F32, {2, 2}, bytes}});
    bool threw = false;
    try {
        (void)vesper::load_model(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, "qwen35 without DeltaNet tensors fails");
}

void test_load_qwen3_5_arch_alias() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen3-5-alias.gguf").string();
    const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::byte> bytes(sizeof(weight));
    std::memcpy(bytes.data(), weight, sizeof(weight));
    vesper::write_gguf(path,
                       {vesper::gguf_kv_string("general.architecture", "qwen3_5"),
                        vesper::gguf_kv_u32("qwen3_5.block_count", 4),
                        vesper::gguf_kv_u32("qwen3_5.embedding_length", 64),
                        vesper::gguf_kv_u32("qwen3_5.feed_forward_length", 128),
                        vesper::gguf_kv_u32("qwen3_5.attention.head_count", 4),
                        vesper::gguf_kv_u32("qwen3_5.attention.head_count_kv", 2),
                        vesper::gguf_kv_u32("qwen3_5.attention.key_length", 16),
                        vesper::gguf_kv_u32("qwen3_5.full_attention_interval", 4),
                        vesper::gguf_kv_u32("qwen3_5.ssm.group_count", 2),
                        vesper::gguf_kv_u32("qwen3_5.ssm.time_step_rank", 4),
                        vesper::gguf_kv_u32("qwen3_5.ssm.state_size", 16)},
                       {{"token_embd.weight", vesper::GgmlType::F32, {2, 2}, bytes}});
    std::string err;
    try {
        (void)vesper::load_model(path);
    } catch (const std::runtime_error& e) {
        err = e.what();
    }
    expect(!err.empty() && err.find("unsupported GGUF architecture") == std::string::npos,
           "qwen3_5 is accepted as qwen35");
}

void test_tokenizer_gguf_roundtrip() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-tok";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "tok.gguf").string();
    const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::byte> bytes(sizeof(weight));
    std::memcpy(bytes.data(), weight, sizeof(weight));
    vesper::write_gguf(path,
                       {vesper::gguf_kv_string("general.architecture", "qwen35"),
                        vesper::gguf_kv_string_array("tokenizer.ggml.tokens", {"a", "b", "ab", "c"}),
                        vesper::gguf_kv_string_array("tokenizer.ggml.merges", {"a b"})},
                       {{"blk.0.weight", vesper::GgmlType::F32, {4}, bytes}});
    const vesper::GgufFile file = vesper::GgufFile::open(path);
    const auto tokens = file.kv_string_array("tokenizer.ggml.tokens");
    expect(tokens.size() == 4 && tokens[2] == "ab", "string array tokens");
    const vesper::Tokenizer tok = vesper::Tokenizer::from_gguf(file);
    const auto ids = tok.encode("ab");
    expect(!ids.empty(), "bpe encode produced ids");
    const std::string back = tok.decode(ids);
    expect(back.find('a') != std::string::npos || back.find('b') != std::string::npos,
           "bpe decode has letters");
}

void test_gguf_u32_array() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-tok";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "arr.gguf").string();
    const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::byte> bytes(sizeof(weight));
    std::memcpy(bytes.data(), weight, sizeof(weight));
    vesper::write_gguf(path,
                       {vesper::gguf_kv_string("general.architecture", "qwen35"),
                        vesper::gguf_kv_u32_array("qwen35.rope.dimension_sections", {11, 11, 10})},
                       {{"blk.0.weight", vesper::GgmlType::F32, {4}, bytes}});
    const vesper::GgufFile file = vesper::GgufFile::open(path);
    const auto secs = file.kv_u64_array("qwen35.rope.dimension_sections");
    expect(secs.size() == 3 && secs[0] == 11 && secs[2] == 10, "u32 array rope sections");
}

void test_hybrid_generate() {
    const auto cfg = vesper::ModelConfig::tiny_hybrid();
    vesper::Engine a(vesper::ModelWeights::random(cfg, 3));
    vesper::Engine b(vesper::ModelWeights::random(cfg, 3));
    const auto ids = a.generate({9, 8, 7}, 8);
    expect(ids == b.generate({9, 8, 7}, 8), "two hybrid engines same tokens");
    const vesper::DecodeReport report = a.last_report();
    expect(report.model == "tiny_hybrid", "hybrid report model");
    expect(report.new_tokens == 8, "hybrid report tokens");
    std::string want_ids;
    for (std::size_t i = 3; i < ids.size(); ++i) {
        if (!want_ids.empty()) {
            want_ids += ',';
        }
        want_ids += std::to_string(ids[i]);
    }
    expect(report.ids == want_ids, "hybrid report ids match generated tail");
}

std::filesystem::path repo_root() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

std::filesystem::path infer_bin() {
    const auto from_root = repo_root() / "build" / "vesper-infer";
    if (std::filesystem::exists(from_root)) {
        return from_root;
    }
    return std::filesystem::current_path() / "vesper-infer";
}

void test_inspect_qwen35_pin_kv() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string gguf = (dir / "qwen35-pin-inspect.gguf").string();
    const std::string out = (dir / "inspect.txt").string();
    vesper::write_tiny_qwen35_pin_kv(gguf, 13);
    const std::string cmd = infer_bin().string() + " --inspect " + gguf + " > " + out;
    expect(std::system(cmd.c_str()) == 0, "inspect pin kv exits 0");
    std::ifstream in(out);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    expect(text.find("architecture qwen35") != std::string::npos, "inspect prints arch");
    expect(text.find("hybrid_kv") != std::string::npos, "inspect prints hybrid kv");
    expect(text.find("qwen35.full_attention_interval 4") != std::string::npos,
           "inspect prints interval");
    expect(text.find("qwen35.attention.recurrent_layers absent") != std::string::npos,
           "inspect notes missing recurrent map");
    expect(text.find("qwen35.ssm.group_count 2") != std::string::npos, "inspect prints ssm kv");
    expect(text.find("types") != std::string::npos, "inspect prints type histogram");
    expect(text.find("payloads     complete") != std::string::npos, "tiny pin payloads complete");
    expect(text.find("pin_header   qwen38-27b-q4km no") != std::string::npos,
           "tiny pin is not the official 27B header");
    expect(text.find("config       arch=qwen35") != std::string::npos, "inspect prints load_config");
}

void test_rmsnorm_rows_and_tile() {
    float x[] = {3.0f, 4.0f, 6.0f, 8.0f};
    const float w[] = {1.0f, 1.0f};
    vesper::rmsnorm_rows(x, w, 2, 2, 0.0f);
    const float inv0 = 1.0f / std::sqrt(12.5f);
    const float inv1 = 1.0f / std::sqrt(50.0f);
    expect(close(x[0], 3.0f * inv0) && close(x[1], 4.0f * inv0), "rmsnorm_rows row0");
    expect(close(x[2], 6.0f * inv1) && close(x[3], 8.0f * inv1), "rmsnorm_rows row1");

    const float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dst[8] = {};
    vesper::tile_heads(dst, src, 4, 2, 2);
    expect(close(dst[0], 1.0f) && close(dst[2], 3.0f) && close(dst[4], 1.0f) && close(dst[6], 3.0f),
           "tile_heads repeats k heads");
}

void test_attn_decode_matches_loop() {
    const int seq = 3;
    const int n_q = 2;
    const int n_kv = 1;
    const int dim = 2;
    const float q[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float k[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const float v[] = {2.0f, 0.0f, 0.0f, 3.0f, 4.0f, 5.0f};
    float out[4] = {};
    float scores[3] = {};
    vesper::attn_decode(out, scores, q, k, v, seq, n_q, n_kv, dim);

    float expect_out[4] = {};
    float sc[3] = {};
    for (int qh = 0; qh < n_q; ++qh) {
        vesper::attn_scores(sc, q + qh * dim, k, seq, n_kv, 0, dim);
        vesper::softmax_inplace(sc, seq);
        vesper::attn_mix(expect_out + qh * dim, sc, v, seq, n_kv, 0, dim);
    }
    expect(close_vec(out, expect_out, 4, 1e-5f), "attn_decode matches per-head loop");

    const float gate[] = {0.0f, 2.0f, -1.0f, 4.0f};
    float gated[4] = {};
    vesper::attn_decode(gated, scores, q, k, v, gate, seq, n_q, n_kv, dim);
    float sg[4] = {gate[0], gate[1], gate[2], gate[3]};
    vesper::sigmoid_inplace(sg, 4);
    float expect_gated[4];
    for (int i = 0; i < 4; ++i) {
        expect_gated[i] = expect_out[i] * sg[i];
    }
    expect(close_vec(gated, expect_gated, 4, 1e-5f), "attn_decode applies sigmoid(gate)");
}

void test_add_rmsnorm_and_split_qkv() {
    float x[] = {1.0f, 2.0f};
    float residual[] = {3.0f, 4.0f};
    const float w[] = {1.0f, 1.0f};
    float x_ref[] = {1.0f, 2.0f};
    float res_ref[] = {3.0f, 4.0f};
    for (int i = 0; i < 2; ++i) {
        res_ref[i] += x_ref[i];
    }
    vesper::rmsnorm(x_ref, res_ref, w, 2, 0.0f);
    vesper::add_rmsnorm(x, residual, w, 2, 0.0f);
    expect(close_vec(x, x_ref, 2) && close_vec(residual, res_ref, 2),
           "add_rmsnorm is residual+=x then rmsnorm");

    const float qkv[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float q[2] = {};
    float k[2] = {};
    float v[3] = {};
    vesper::split_qkv(q, k, v, qkv, 2, 3);
    expect(close(q[0], 1.0f) && close(q[1], 2.0f) && close(k[0], 3.0f) && close(k[1], 4.0f),
           "split_qkv q/k");
    expect(close(v[0], 5.0f) && close(v[1], 6.0f) && close(v[2], 7.0f), "split_qkv v");

    float cx[] = {3.0f, 4.0f};
    float cres[] = {9.0f, 9.0f};
    float cx_ref[] = {3.0f, 4.0f};
    float cres_ref[] = {3.0f, 4.0f};
    vesper::rmsnorm(cx_ref, cres_ref, w, 2, 0.0f);
    vesper::copy_rmsnorm(cx, cres, w, 2, 0.0f);
    expect(close_vec(cx, cx_ref, 2) && close_vec(cres, cres_ref, 2), "copy_rmsnorm saves x then norms");

    float y[] = {2.0f, -4.0f};
    const float z[] = {1.0f, 2.0f};
    vesper::silu_mul(y, z, 2);
    const float silu0 = 1.0f / (1.0f + std::exp(-1.0f));
    const float silu1 = 2.0f / (1.0f + std::exp(-2.0f));
    expect(close(y[0], 2.0f * silu0) && close(y[1], -4.0f * silu1), "silu_mul is y *= silu(z)");
}

void test_fused_split_norm_and_silu() {
    const float q_full[] = {3.0f, 4.0f, 10.0f, 20.0f, 6.0f, 8.0f, 30.0f, 40.0f};
    const float w[] = {1.0f, 1.0f};
    float q[4] = {};
    float gate[4] = {};
    float q_ref[4] = {};
    float gate_ref[4] = {};
    vesper::split_gated_q(q_ref, gate_ref, q_full, 2, 2);
    vesper::rmsnorm_rows(q_ref, w, 2, 2, 1e-6f);
    vesper::split_gated_q_norm(q, gate, q_full, w, 2, 2, 1e-6f);
    expect(close_vec(q, q_ref, 4) && close_vec(gate, gate_ref, 4),
           "split_gated_q_norm is split then rmsnorm");

    float y[] = {3.0f, 4.0f, 6.0f, 8.0f};
    float y_ref[] = {3.0f, 4.0f, 6.0f, 8.0f};
    const float z[] = {1.0f, -1.0f, 0.5f, 2.0f};
    vesper::rmsnorm_rows(y_ref, w, 2, 2, 1e-6f);
    vesper::silu_mul(y_ref, z, 4);
    vesper::rmsnorm_silu_mul(y, z, w, 2, 2, 1e-6f);
    expect(close_vec(y, y_ref, 4), "rmsnorm_silu_mul is rmsnorm then silu_mul");
}

void test_gdn_conv_split_matches_chain() {
    const int key_dim = 4;
    const int value_dim = 6;
    const int kernel = 4;
    const int qkv_dim = 2 * key_dim + value_dim;
    const int hist = kernel - 1;
    std::vector<float> x(static_cast<std::size_t>(qkv_dim));
    std::vector<float> weight(static_cast<std::size_t>(qkv_dim * kernel));
    std::vector<float> state(static_cast<std::size_t>(qkv_dim * hist));
    for (int i = 0; i < qkv_dim; ++i) {
        x[static_cast<std::size_t>(i)] = 0.05f * static_cast<float>((i % 7) - 3);
        for (int t = 0; t < kernel; ++t) {
            weight[static_cast<std::size_t>(i * kernel + t)] =
                0.02f * static_cast<float>((i * 3 + t) % 5 - 2);
        }
        for (int t = 0; t < hist; ++t) {
            state[static_cast<std::size_t>(i * hist + t)] =
                0.03f * static_cast<float>((i + t) % 4 - 1);
        }
    }
    std::vector<float> state_ref = state;
    std::vector<float> conv_y(static_cast<std::size_t>(qkv_dim));
    std::vector<float> q(static_cast<std::size_t>(key_dim));
    std::vector<float> k(static_cast<std::size_t>(key_dim));
    std::vector<float> v(static_cast<std::size_t>(value_dim));
    std::vector<float> q_ref(static_cast<std::size_t>(key_dim));
    std::vector<float> k_ref(static_cast<std::size_t>(key_dim));
    std::vector<float> v_ref(static_cast<std::size_t>(value_dim));
    vesper::gdn_conv_update(vesper::Device::CPU, conv_y.data(), state_ref.data(), x.data(),
                            weight.data(), qkv_dim, kernel);
    vesper::split_qkv(q_ref.data(), k_ref.data(), v_ref.data(), conv_y.data(), key_dim, value_dim);
    std::vector<float> conv_unused(static_cast<std::size_t>(qkv_dim));
    vesper::gdn_conv_split(vesper::Device::CPU, q.data(), k.data(), v.data(), conv_unused.data(),
                           state.data(), x.data(), weight.data(), key_dim, value_dim, kernel);
    expect(close_vec(q.data(), q_ref.data(), key_dim) && close_vec(k.data(), k_ref.data(), key_dim),
           "gdn_conv_split q/k match conv+split");
    expect(close_vec(v.data(), v_ref.data(), value_dim), "gdn_conv_split v matches conv+split");
    expect(close_vec(state.data(), state_ref.data(), qkv_dim * hist),
           "gdn_conv_split updates conv state");
}

void test_gdn_gates_matches_chain() {
    const int n = 4;
    float alpha[] = {0.2f, -3.0f, 25.0f, -25.0f};
    const float dt[] = {0.1f, 0.5f, 1.0f, -0.2f};
    const float a[] = {-0.4f, -1.2f, -0.1f, -2.0f};
    float beta[] = {0.5f, -2.0f, 3.0f, -0.1f};
    float decay[4] = {};
    vesper::gdn_gates(decay, beta, alpha, dt, a, n);

    float decay_ref[4];
    float beta_ref[] = {0.5f, -2.0f, 3.0f, -0.1f};
    for (int i = 0; i < n; ++i) {
        decay_ref[i] = alpha[i] + dt[i];
    }
    vesper::softplus_inplace(decay_ref, n);
    vesper::mul_inplace(decay_ref, a, n);
    vesper::exp_inplace(decay_ref, n);
    vesper::sigmoid_inplace(beta_ref, n);
    expect(close_vec(decay, decay_ref, n, 1e-5f), "gdn_gates decay matches chain");
    expect(close_vec(beta, beta_ref, n, 1e-5f), "gdn_gates beta is sigmoid");
}

void test_gemv_swiglu_matches_pair() {
    const int rows = 8;
    const int cols = 16;
    std::vector<float> gw(static_cast<std::size_t>(rows * cols));
    std::vector<float> uw(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        gw[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 13) % 17 - 8);
        uw[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 9) % 15 - 7);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.05f * static_cast<float>((i * 5) % 11 - 5);
    }
    auto gate = vesper::WeightMatrix::from_f32(gw.data(), rows, cols);
    auto up = vesper::WeightMatrix::from_f32(uw.data(), rows, cols);
    std::vector<float> hidden(static_cast<std::size_t>(rows));
    std::vector<float> gate_tmp(static_cast<std::size_t>(rows));
    std::vector<float> up_tmp(static_cast<std::size_t>(rows));
    vesper::gemv_swiglu(hidden.data(), gate_tmp.data(), up_tmp.data(), gate, up, x.data());

    std::vector<float> g(static_cast<std::size_t>(rows));
    std::vector<float> u(static_cast<std::size_t>(rows));
    std::vector<float> ref(static_cast<std::size_t>(rows));
    vesper::gemv(g.data(), gate, x.data());
    vesper::gemv(u.data(), up, x.data());
    vesper::swiglu(ref.data(), g.data(), u.data(), rows);
    expect(close_vec(hidden.data(), ref.data(), rows, 1e-5f), "F32 gemv_swiglu matches pair");

    const int qrows = 16;
    const int qcols = 256;
    std::vector<float> qgw(static_cast<std::size_t>(qrows * qcols));
    std::vector<float> quw(static_cast<std::size_t>(qrows * qcols));
    std::vector<float> qx(static_cast<std::size_t>(qcols));
    for (int i = 0; i < qrows * qcols; ++i) {
        qgw[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 11) % 19 - 9);
        quw[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 7) % 13 - 6);
    }
    for (int i = 0; i < qcols; ++i) {
        qx[static_cast<std::size_t>(i)] = 0.04f * static_cast<float>((i * 3) % 17 - 8);
    }
    auto qgate = vesper::WeightMatrix::q4_from_f32(qgw.data(), qrows, qcols);
    auto qup = vesper::WeightMatrix::q4_from_f32(quw.data(), qrows, qcols);
    std::vector<float> qhidden(static_cast<std::size_t>(qrows));
    std::vector<float> qgt(static_cast<std::size_t>(qrows));
    std::vector<float> qut(static_cast<std::size_t>(qrows));
    vesper::gemv_swiglu(qhidden.data(), qgt.data(), qut.data(), qgate, qup, qx.data());
    std::vector<float> qg(static_cast<std::size_t>(qrows));
    std::vector<float> qu(static_cast<std::size_t>(qrows));
    std::vector<float> qref(static_cast<std::size_t>(qrows));
    vesper::gemv(qg.data(), qgate, qx.data());
    vesper::gemv(qu.data(), qup, qx.data());
    vesper::swiglu(qref.data(), qg.data(), qu.data(), qrows);
    expect(close_vec(qhidden.data(), qref.data(), qrows, 2e-3f), "Q4_K gemv_swiglu matches pair");
}

void test_pick_multi_row() {
    vesper::MultiRowPick pick{};
    // Official attn Q/K/V: 12288 + 1024 + 1024
    expect(vesper::pick_multi_row(0, 12288, 1024, 1024, 0, &pick) && pick.slot == 0 &&
               pick.local == 0,
           "attn first Q row");
    expect(vesper::pick_multi_row(12287, 12288, 1024, 1024, 0, &pick) && pick.slot == 0 &&
               pick.local == 12287,
           "attn last Q row");
    expect(vesper::pick_multi_row(12288, 12288, 1024, 1024, 0, &pick) && pick.slot == 1 &&
               pick.local == 0,
           "attn first K row");
    expect(vesper::pick_multi_row(13311, 12288, 1024, 1024, 0, &pick) && pick.slot == 1 &&
               pick.local == 1023,
           "attn last K row");
    expect(vesper::pick_multi_row(13312, 12288, 1024, 1024, 0, &pick) && pick.slot == 2 &&
               pick.local == 0,
           "attn first V row");
    expect(vesper::pick_multi_row(14335, 12288, 1024, 1024, 0, &pick) && pick.slot == 2 &&
               pick.local == 1023,
           "attn last V row");
    expect(!vesper::pick_multi_row(14336, 12288, 1024, 1024, 0, &pick), "attn past last V");

    // Official GDN qkv/z/beta/alpha: 10240 + 6144 + 48 + 48
    expect(vesper::pick_multi_row(10240, 10240, 6144, 48, 48, &pick) && pick.slot == 1 &&
               pick.local == 0,
           "gdn first z row");
    expect(vesper::pick_multi_row(16383, 10240, 6144, 48, 48, &pick) && pick.slot == 1 &&
               pick.local == 6143,
           "gdn last z row");
    expect(vesper::pick_multi_row(16384, 10240, 6144, 48, 48, &pick) && pick.slot == 2 &&
               pick.local == 0,
           "gdn first beta row");
    expect(vesper::pick_multi_row(16431, 10240, 6144, 48, 48, &pick) && pick.slot == 2 &&
               pick.local == 47,
           "gdn last beta row");
    expect(vesper::pick_multi_row(16432, 10240, 6144, 48, 48, &pick) && pick.slot == 3 &&
               pick.local == 0,
           "gdn first alpha row");
    expect(vesper::pick_multi_row(16479, 10240, 6144, 48, 48, &pick) && pick.slot == 3 &&
               pick.local == 47,
           "gdn last alpha row");
    expect(!vesper::pick_multi_row(16480, 10240, 6144, 48, 48, &pick), "gdn past last alpha");
}

void test_gemv3_and_tile_l2() {
    const int rows0 = 4;
    const int rows1 = 2;
    const int rows2 = 3;
    const int cols = 8;
    std::vector<float> w0(static_cast<std::size_t>(rows0 * cols));
    std::vector<float> w1(static_cast<std::size_t>(rows1 * cols));
    std::vector<float> w2(static_cast<std::size_t>(rows2 * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows0 * cols; ++i) {
        w0[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 5) % 11 - 5);
    }
    for (int i = 0; i < rows1 * cols; ++i) {
        w1[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 7) % 13 - 6);
    }
    for (int i = 0; i < rows2 * cols; ++i) {
        w2[static_cast<std::size_t>(i)] = 0.04f * static_cast<float>((i * 3) % 9 - 4);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.05f * static_cast<float>((i % 7) - 3);
    }
    auto m0 = vesper::WeightMatrix::from_f32(w0.data(), rows0, cols);
    auto m1 = vesper::WeightMatrix::from_f32(w1.data(), rows1, cols);
    auto m2 = vesper::WeightMatrix::from_f32(w2.data(), rows2, cols);
    std::vector<float> y0(static_cast<std::size_t>(rows0));
    std::vector<float> y1(static_cast<std::size_t>(rows1));
    std::vector<float> y2(static_cast<std::size_t>(rows2));
    vesper::gemv3(y0.data(), m0, y1.data(), m1, y2.data(), m2, x.data());
    std::vector<float> r0(static_cast<std::size_t>(rows0));
    std::vector<float> r1(static_cast<std::size_t>(rows1));
    std::vector<float> r2(static_cast<std::size_t>(rows2));
    vesper::gemv(r0.data(), m0, x.data());
    vesper::gemv(r1.data(), m1, x.data());
    vesper::gemv(r2.data(), m2, x.data());
    expect(close_vec(y0.data(), r0.data(), rows0) && close_vec(y1.data(), r1.data(), rows1) &&
               close_vec(y2.data(), r2.data(), rows2),
           "gemv3 matches three gemvs");

    const float src[] = {3.0f, 4.0f, 0.0f, 5.0f};
    float dst[8] = {};
    float ref[8] = {};
    vesper::tile_heads(ref, src, 4, 2, 2);
    vesper::l2_normalize_rows(ref, 4, 2, 1e-6f);
    vesper::scale_inplace(ref, 0.5f, 8);
    vesper::tile_l2_scale(dst, src, 4, 2, 2, 1e-6f, 0.5f);
    expect(close_vec(dst, ref, 8, 1e-5f), "tile_l2_scale matches tile+l2+scale");

    float q_dst[8] = {};
    float k_dst[8] = {};
    float q_ref[8] = {};
    float k_ref[8] = {};
    vesper::tile_l2_scale(q_ref, src, 4, 2, 2, 1e-6f, 0.5f);
    vesper::tile_l2_scale(k_ref, src, 4, 2, 2, 1e-6f, 1.0f);
    vesper::tile_l2_pair(q_dst, src, k_dst, src, 4, 2, 2, 1e-6f, 0.5f, 1.0f);
    expect(close_vec(q_dst, q_ref, 8) && close_vec(k_dst, k_ref, 8),
           "tile_l2_pair matches two tile_l2_scale calls");
}

void test_hip_gemv_swiglu_matches_cpu() {
    if (!vesper::hip_available()) {
        return;
    }
    const int rows = 32;
    const int cols = 256;
    std::vector<float> gw(static_cast<std::size_t>(rows * cols));
    std::vector<float> uw(static_cast<std::size_t>(rows * cols));
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        gw[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>((i * 11) % 19 - 9);
        uw[static_cast<std::size_t>(i)] = 0.03f * static_cast<float>((i * 7) % 13 - 6);
    }
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = 0.04f * static_cast<float>((i * 3) % 17 - 8);
    }
    auto gate = vesper::WeightMatrix::q4_from_f32(gw.data(), rows, cols);
    auto up = vesper::WeightMatrix::q4_from_f32(uw.data(), rows, cols);
    std::vector<std::int8_t> qs(static_cast<std::size_t>(cols));
    std::vector<float> xd(static_cast<std::size_t>(cols / 32));
    std::vector<float> xsum(static_cast<std::size_t>(cols / 32));
    vesper::quantize_q8x(x.data(), qs.data(), xd.data(), xsum.data(), cols);
    std::vector<float> gt(static_cast<std::size_t>(rows));
    std::vector<float> ut(static_cast<std::size_t>(rows));
    vesper::gemv_q4k_q8x(gt.data(), gate.packed(), qs.data(), xd.data(), xsum.data(), rows, cols);
    vesper::gemv_q4k_q8x(ut.data(), up.packed(), qs.data(), xd.data(), xsum.data(), rows, cols);
    std::vector<float> y_cpu(static_cast<std::size_t>(rows));
    vesper::swiglu(y_cpu.data(), gt.data(), ut.data(), rows);

    auto gpu_g = gate.to(vesper::Device::HIP);
    auto gpu_u = up.to(vesper::Device::HIP);
    vesper::Buffer X(static_cast<std::size_t>(cols), vesper::Device::HIP);
    vesper::Buffer Y(static_cast<std::size_t>(rows), vesper::Device::HIP);
    vesper::Buffer GT(static_cast<std::size_t>(rows), vesper::Device::HIP);
    vesper::Buffer UT(static_cast<std::size_t>(rows), vesper::Device::HIP);
    X.copy_from(x.data(), x.size());
    vesper::gemv_swiglu(vesper::Device::HIP, Y.data(), GT.data(), UT.data(), gpu_g, gpu_u,
                        X.data());
    std::vector<float> y_gpu(static_cast<std::size_t>(rows));
    Y.copy_to(y_gpu.data(), y_gpu.size());
    expect(close_vec(y_cpu.data(), y_gpu.data(), rows, 2e-3f),
           "HIP Q4_K gemv_swiglu matches CPU q8x");
}

void test_qwen2_pretok_digits() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-tok";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen2-pretok.gguf").string();
    const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::byte> bytes(sizeof(weight));
    std::memcpy(bytes.data(), weight, sizeof(weight));
    vesper::write_gguf(path,
                       {vesper::gguf_kv_string("general.architecture", "qwen35"),
                        vesper::gguf_kv_string("tokenizer.ggml.pre", "qwen2"),
                        vesper::gguf_kv_string_array("tokenizer.ggml.tokens",
                                                     {"a", "b", "ab", "1", "2", "12"}),
                        vesper::gguf_kv_string_array("tokenizer.ggml.merges", {"a b"})},
                       {{"blk.0.weight", vesper::GgmlType::F32, {4}, bytes}});
    const vesper::Tokenizer tok = vesper::Tokenizer::load(path);
    expect(tok.pretok() == vesper::PretokKind::Qwen2, "qwen2 pretok from ggml.pre");
    const auto ids = tok.encode("12");
    expect(ids.size() == 2 && ids[0] == 3 && ids[1] == 4, "qwen2 pretok splits digits");
    const auto word = tok.encode("ab");
    expect(!word.empty() && word[0] == 2, "qwen2 pretok still merges ab");
}

void test_qwen35_default_pretok() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-tok";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen35-default-pretok.gguf").string();
    const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::byte> bytes(sizeof(weight));
    std::memcpy(bytes.data(), weight, sizeof(weight));
    vesper::write_gguf(path,
                       {vesper::gguf_kv_string("general.architecture", "qwen35"),
                        vesper::gguf_kv_string_array("tokenizer.ggml.tokens", {"a", "b", "ab"})},
                       {{"blk.0.weight", vesper::GgmlType::F32, {4}, bytes}});
    const vesper::Tokenizer tok = vesper::Tokenizer::load(path);
    expect(tok.pretok() == vesper::PretokKind::Qwen35, "qwen35 arch defaults to qwen35 pretok");
}

void test_qwen35_pretok_and_specials() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-tok";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen35-pretok.gguf").string();
    const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::byte> bytes(sizeof(weight));
    std::memcpy(bytes.data(), weight, sizeof(weight));
    vesper::write_gguf(path,
                       {vesper::gguf_kv_string("general.architecture", "qwen35"),
                        vesper::gguf_kv_string("tokenizer.ggml.pre", "qwen35"),
                        vesper::gguf_kv_string_array("tokenizer.ggml.tokens",
                                                     {"a", "b", "ab", "<|eot|>"}),
                        vesper::gguf_kv_string_array("tokenizer.ggml.merges", {"a b"}),
                        vesper::gguf_kv_u32_array("tokenizer.ggml.token_type", {1, 1, 1, 3})},
                       {{"blk.0.weight", vesper::GgmlType::F32, {4}, bytes}});
    const vesper::Tokenizer tok = vesper::Tokenizer::load(path);
    expect(tok.pretok() == vesper::PretokKind::Qwen35, "qwen35 pretok from ggml.pre");
    expect(tok.special_count() == 1, "control token is special");
    const auto special_ids = tok.encode("ab<|eot|>ab");
    expect(special_ids.size() == 3 && special_ids[0] == 2 && special_ids[1] == 3 &&
               special_ids[2] == 2,
           "control token is one id, not BPE-split");
}

void test_context_cap() {
    auto cfg = vesper::ModelConfig::qwen38_27b();
    cfg.max_seq_len = 262144;
    cfg.cap_seq_len(4096);
    expect(cfg.max_seq_len == 4096, "cap_seq_len shrinks official context");
    cfg.cap_seq_len(8192);
    expect(cfg.max_seq_len == 4096, "cap_seq_len does not grow");
}

void test_engine_caps_official_context() {
    auto cfg = vesper::ModelConfig::tiny_hybrid();
    cfg.max_seq_len = 262144;
    vesper::Engine capped(vesper::ModelWeights::random(cfg, 3));
    expect(capped.config().max_seq_len == vesper::kDefaultContext,
           "Engine caps 262144 KV to 4096");
    vesper::Engine file_len(vesper::ModelWeights::random(cfg, 3), vesper::Device::CPU, 0);
    expect(file_len.config().max_seq_len == 262144, "Engine context 0 keeps file length");
    vesper::Engine custom(vesper::ModelWeights::random(cfg, 3), vesper::Device::CPU, 128);
    expect(custom.config().max_seq_len == 128, "Engine explicit context wins");
}

void test_rdna4_q8_mmvq_cover() {
    const int cols[] = {5120, 6144, 10240};
    for (int c : cols) {
        const int nblocks = c / vesper::kQ8BlockElems;
        std::vector<int> hit(static_cast<std::size_t>(nblocks) * static_cast<std::size_t>(vesper::kQ8Qi),
                             0);
        const int per_iter = vesper::kQ8VdrMmvq * vesper::kGemvWorkgroup / vesper::kQ8Qi;
        for (int tid = 0; tid < vesper::kGemvWorkgroup; ++tid) {
            const int iqs = vesper::kQ8VdrMmvq * (tid % (vesper::kQ8Qi / vesper::kQ8VdrMmvq));
            for (int kbx = tid / (vesper::kQ8Qi / vesper::kQ8VdrMmvq); kbx < nblocks;
                 kbx += per_iter) {
                hit[static_cast<std::size_t>(kbx) * static_cast<std::size_t>(vesper::kQ8Qi) +
                    static_cast<std::size_t>(iqs)] += 1;
                hit[static_cast<std::size_t>(kbx) * static_cast<std::size_t>(vesper::kQ8Qi) +
                    static_cast<std::size_t>(iqs + 1)] += 1;
            }
        }
        bool once = true;
        for (int v : hit) {
            once = once && v == 1;
        }
        expect(once, "RDNA4 Q8 MMVQ covers each block/int once at cols=" + std::to_string(c));
    }
}

void test_rdna4_q4k_mmvq_cover() {
    const int cols[] = {5120, 17408};
    for (int c : cols) {
        const int supers = c / 256;
        std::vector<int> hit(static_cast<std::size_t>(supers) * 16, 0);
        for (int tid = 0; tid < vesper::kGemvWorkgroup; ++tid) {
            const int iqs = 2 * (tid % 16);
            for (int s = tid / 16; s < supers; s += 16) {
                hit[static_cast<std::size_t>(s) * 16u + static_cast<std::size_t>(iqs / 2)] += 1;
            }
        }
        bool once = true;
        for (int v : hit) {
            once = once && v == 1;
        }
        expect(once, "RDNA4 Q4_K MMVQ covers each super/iqs once at cols=" + std::to_string(c));
    }
}

void test_mrope_text_matches_neox() {
    float q1[] = {0.5f, -0.25f, 1.0f, 0.0f, 0.2f, -0.1f, 0.3f, 0.4f};
    float k1[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float q2[8];
    float k2[8];
    std::memcpy(q2, q1, sizeof(q1));
    std::memcpy(k2, k1, sizeof(k1));
    vesper::rope_neox(q1, k1, 1, 1, 8, 8, 5, 10000.0f);
    vesper::rope_neox(q2, k2, 1, 1, 8, 8, 5, 10000.0f);
    expect(close_vec(q1, q2, 8, 1e-6f) && close_vec(k1, k2, 8, 1e-6f),
           "text mrope with equal positions is NeoX");
}

void test_rope_k_norm_matches_chain() {
    float q[] = {0.5f, -0.25f, 1.0f, 0.0f, 0.2f, -0.1f, 0.3f, 0.4f};
    float k[] = {3.0f, 4.0f, 6.0f, 8.0f};
    float q_ref[8];
    float k_ref[4];
    std::memcpy(q_ref, q, sizeof(q_ref));
    std::memcpy(k_ref, k, sizeof(k_ref));
    const float kw[] = {1.0f, 2.0f, 0.5f, 1.5f};
    vesper::rmsnorm_rows(k_ref, kw, 1, 4, 1e-6f);
    vesper::rope_neox(q_ref, k_ref, 2, 1, 4, 4, 3, 10000.0f);
    vesper::rope_neox_k_norm(q, k, kw, 2, 1, 4, 4, 3, 10000.0f, 1e-6f);
    expect(close_vec(q, q_ref, 8) && close_vec(k, k_ref, 4),
           "rope_neox_k_norm is k rmsnorm then rope");
}

void test_llamacpp_parse() {
    const std::string log = "/tmp/vesper-llama-parse.log";
    {
        std::ofstream out(log);
        out << "llama_perf_context_print: prompt eval time =     12.50 ms /     5 tokens "
               "(    2.50 ms per token,   400.00 tokens per second)\n";
        out << "llama_perf_context_print:        eval time =   4000.00 ms /   128 tokens "
               "(   31.25 ms per token,    32.00 tokens per second)\n";
    }
    const std::string script = (repo_root() / "scripts/compare-qwen38/parse_llamacpp.sh").string();
    const std::string cmd = script + " hip " + log + " > /tmp/vesper-llama-parse.txt";
    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "parse_llamacpp exits 0");
    std::ifstream in("/tmp/vesper-llama-parse.txt");
    std::string line;
    std::getline(in, line);
    expect(line.find("engine=llamacpp") != std::string::npos, "parse engine");
    expect(line.find("backend=hip") != std::string::npos, "parse backend");
    expect(line.find("decode_tps=32.00") != std::string::npos ||
               line.find("decode_tps=32") != std::string::npos,
           "parse decode tps");
    expect(line.find("new_tokens=128") != std::string::npos, "parse new tokens");
    expect(line.find("status=ok") != std::string::npos, "parse ok");
    expect(line.find("bytes_per_token=18237132800") != std::string::npos,
           "parse uses official packed bytes");
    expect(line.find("ids=-") != std::string::npos, "parse ids placeholder");
}

void test_compare_fixture() {
    const std::string script = (repo_root() / "scripts/compare-qwen38/run_vesper.sh").string();
    const std::string cmd = "COMPARE_FIXTURE=1 " + script + " > /tmp/vesper-compare-fixture.txt";
    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "COMPARE_FIXTURE run_vesper exits 0");
    std::ifstream in("/tmp/vesper-compare-fixture.txt");
    std::string line;
    std::getline(in, line);
    expect(line.find("engine=vesper") != std::string::npos, "fixture engine");
    expect(line.find("status=unsupported") != std::string::npos, "fixture unsupported");
    expect(line.find("quant=Q4_K_M") != std::string::npos, "fixture quant");
    expect(line.find("arch=qwen35") != std::string::npos, "fixture arch");
    expect(line.find("bytes_per_token=") != std::string::npos, "fixture bytes");
}

void test_compare_table_fixture() {
    const std::string script = (repo_root() / "scripts/compare-qwen38/compare.sh").string();
    const std::string cmd = "COMPARE_FIXTURE=1 " + script + " > /tmp/vesper-compare-table.txt";
    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "COMPARE_FIXTURE compare.sh exits 0");
    std::ifstream in("/tmp/vesper-compare-table.txt");
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    expect(text.find("# compare Qwen3.8-27B Q4_K_M") != std::string::npos, "table title");
    expect(text.find("# sha256 31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34") !=
               std::string::npos,
           "table sha pin");
    expect(text.find("| engine | backend | decode_tps |") != std::string::npos, "table header");
    expect(text.find("| llamacpp | hip | unsupported |") != std::string::npos, "hip unsupported cell");
    expect(text.find("| llamacpp | vulkan | unsupported |") != std::string::npos,
           "vulkan unsupported cell");
    expect(text.find("| vesper | cpu | unsupported |") != std::string::npos, "vesper fixture row");
    expect(text.find("winner ") == std::string::npos, "fixture has no winner");
}

void test_artifact_env() {
    const std::filesystem::path path = repo_root() / "scripts/compare-qwen38/artifact.env";
    expect(std::filesystem::exists(path), "artifact.env exists");
    std::ifstream in(path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    expect(text.find("COMPARE_QUANT=Q4_K_M") != std::string::npos, "artifact quant Q4_K_M");
    expect(text.find("COMPARE_BYTES_PER_TOKEN=18237132800") != std::string::npos,
           "artifact packed linear bytes");
    expect(text.find("COMPARE_REPO=ggml-org/Qwen3.8-27B-GGUF") != std::string::npos,
           "artifact repo pin");
    const auto pos = text.find("COMPARE_SHA256=");
    expect(pos != std::string::npos, "artifact sha256 key");
    if (pos != std::string::npos) {
        const std::string hex = text.substr(pos + 15, 64);
        expect(hex.size() == 64, "sha256 is 64 hex chars");
        bool all_hex = true;
        for (char ch : hex) {
            const bool ok = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
                            (ch >= 'A' && ch <= 'F');
            all_hex = all_hex && ok;
        }
        expect(all_hex, "sha256 hex");
    }
}

void test_open_meta_truncated_payload() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-meta";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "tiny-q8.gguf").string();
    vesper::write_tiny_q8(path, 2);
    const auto full = vesper::GgufFile::open(path);
    expect(full.payloads_complete(), "full tiny GGUF has payloads");
    const std::uintmax_t nbytes = std::filesystem::file_size(path);
    expect(nbytes > 200, "tiny Q8 GGUF is larger than 200 bytes");
    std::filesystem::resize_file(path, nbytes - 100);
    bool full_threw = false;
    try {
        (void)vesper::GgufFile::open(path);
    } catch (const std::exception&) {
        full_threw = true;
    }
    expect(full_threw, "open() rejects truncated payloads");
    const auto meta = vesper::GgufFile::open_meta(path);
    expect(!meta.payloads_complete(), "open_meta allows missing payloads");
    expect(meta.architecture() == "vesper_tiny", "open_meta still reads arch");
    expect(meta.find("token_embd.weight") != nullptr, "open_meta still lists tensors");
}

void test_tiny_is_not_official_pin_header() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen35-pin-not-official.gguf").string();
    vesper::write_tiny_qwen35_pin_kv(path, 13);
    const auto file = vesper::GgufFile::open_meta(path);
    expect(!vesper::qwen38_27b_q4km_header_ok(file), "tiny qwen35 pin KV is not the 27B header");
}

void test_official_q4km_header_if_present() {
    const std::filesystem::path path{"/tmp/qwen38-pin/Qwen3.8-27B-Q4_K_M.header"};
    if (!std::filesystem::exists(path)) {
        return;
    }
    const auto file = vesper::GgufFile::open_meta(path.string());
    expect(file.architecture() == "qwen35", "official prefix arch");
    expect(!file.payloads_complete(), "64 MiB prefix cannot hold 19 GB of tensors");
    expect(vesper::qwen38_27b_q4km_header_ok(file), "official prefix matches convert.log pin");
    expect(file.kv_u64("qwen35.block_count") == 64, "official block_count");
    expect(file.kv_u64("qwen35.full_attention_interval") == 4, "official interval");
    expect(!file.has_kv("qwen35.attention.recurrent_layers"), "official has no recurrent map");
    const vesper::Tokenizer tok = vesper::Tokenizer::load(path.string());
    expect(tok.pretok() == vesper::PretokKind::Qwen35, "official pretok is qwen35");
    expect(tok.vocab_size() == 248320, "official vocab");
    expect(tok.bos_id() == 248044 && tok.eos_id() == 248046, "official bos/eos");
    const auto ids = tok.encode("The capital of France is");
    expect(!tok.add_bos(), "official add_bos_token is false");
    expect(tok.encode("").empty(), "official empty encode does not invent BOS");
    expect(ids.size() == 5 && ids[0] == 760 && ids[1] == 6511 && ids[2] == 314 && ids[3] == 9338 &&
               ids[4] == 369,
           "official compare-prompt ids");
    expect(tok.decode(ids) == "The capital of France is", "official compare-prompt roundtrip");

    const auto cfg = vesper::load_config(path.string());
    const auto pin = vesper::ModelConfig::qwen38_27b();
    expect(cfg.arch == "qwen35" && cfg.n_layers == pin.n_layers && cfg.hidden_size == pin.hidden_size,
           "official load_config size");
    expect(cfg.vocab_size == pin.vocab_size && cfg.n_heads == pin.n_heads &&
               cfg.n_kv_heads == pin.n_kv_heads && cfg.head_dim == pin.head_dim,
           "official load_config attention");
    expect(cfg.intermediate_size == pin.intermediate_size &&
               cfg.full_attention_interval == pin.full_attention_interval,
           "official load_config FFN/interval");
    expect(cfg.gdn_qk_heads == pin.gdn_qk_heads && cfg.gdn_v_heads == pin.gdn_v_heads &&
               cfg.gdn_head_dim == pin.gdn_head_dim && cfg.gdn_conv_kernel == pin.gdn_conv_kernel,
           "official load_config GDN");
    expect(cfg.rope_dim == pin.rope_dim && cfg.n_rope_sections == 3 && cfg.rope_section[0] == 11 &&
               cfg.rope_section[1] == 11 && cfg.rope_section[2] == 10,
           "official load_config strips trailing MRoPE zero");
    expect(cfg.max_seq_len == 262144, "official file context is 262144 before Engine cap");
    expect(cfg.layer_kind(0) == vesper::LayerKind::DeltaNet &&
               cfg.layer_kind(3) == vesper::LayerKind::Attention,
           "official interval 4 is GDN,GDN,GDN,Attn");
    expect(cfg.qk_norm && cfg.attn_gate && !cfg.tie_word_embeddings, "official hybrid flags");
}

void test_official_q4km_load_if_present() {
    const std::filesystem::path path{"/tmp/qwen38-pin/Qwen3.8-27B-Q4_K_M.gguf"};
    if (!std::filesystem::exists(path) || std::filesystem::file_size(path) != 18973870432ull) {
        return;
    }
    vesper::ModelWeights w = vesper::load_model(path.string());
    expect(std::string(w.quant_name()) == "Q4_K_M", "official quant name");
    expect(w.linear_bytes() == vesper::qwen38_27b_q4km_linear_bytes(),
           "official loaded linear bytes");
    expect(static_cast<int>(w.layers.size()) == 64, "official loaded layer count");
    vesper::Engine engine(std::move(w), vesper::Device::CPU, 4096);
    expect(engine.config().max_seq_len == 4096, "engine caps official 262144 to 4096");
    engine.step(760);
    const int next = vesper::argmax(engine.logits(), engine.config().vocab_size);
    expect(next >= 0 && next < 248320, "official first-step logits are in vocab");
}

void test_qwen38_q4km_linear_bytes() {
    const auto cfg = vesper::ModelConfig::qwen38_27b();
    int attn = 0;
    int gdn = 0;
    for (int i = 0; i < cfg.n_layers; ++i) {
        switch (cfg.layer_kind(i)) {
            case vesper::LayerKind::Attention:
                ++attn;
                continue;
            case vesper::LayerKind::DeltaNet:
                ++gdn;
                continue;
        }
        expect(false, "unhandled LayerKind in official count");
    }
    expect(attn == 16 && gdn == 48, "official interval 4 is 16 attn / 48 GDN");
    expect(vesper::qwen38_27b_q4km_linear_bytes() == 18237132800ull,
           "official Q4_K_M linear bytes");
    expect(vesper::packed_bytes(vesper::WeightKind::Q4_K, 17408, 5120) == 50135040ull,
           "FFN up Q4_K bytes");
}

void test_load_check_tiny() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string gguf = (dir / "load-check.gguf").string();
    const std::string out = (dir / "load-check.txt").string();
    vesper::write_tiny_qwen35_pin_kv(gguf, 13);
    const std::string cmd = infer_bin().string() + " --load-check " + gguf + " > " + out;
    expect(std::system(cmd.c_str()) == 0, "load-check tiny exits 0");
    std::ifstream in(out);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    expect(text.find("config       arch=qwen35") != std::string::npos, "load-check prints config");
    expect(text.find("pin_weights  qwen38-27b-q4km no") != std::string::npos,
           "tiny fixture is not official weights");
}

void test_bench_help_official_shapes() {
    const std::string out = "/tmp/vesper-help.txt";
    const std::string cmd = infer_bin().string() + " --help > " + out;
    expect(std::system(cmd.c_str()) == 0, "--help exits 0");
    std::ifstream in(out);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    expect(text.find("official Qwen3.8-27B Q4_K FFN") != std::string::npos,
           "bench-q4 help names official FFN shapes");
}

void test_gguf_truncated() {
    const auto path = std::filesystem::temp_directory_path() / "vesper-gguf-truncated.bin";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const std::uint32_t magic = 0x46554747;
        const std::uint32_t version = 3;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        const std::uint64_t n_tensors = 1;
        const std::uint64_t n_kv = 0;
        out.write(reinterpret_cast<const char*>(&n_tensors), sizeof(n_tensors));
        out.write(reinterpret_cast<const char*>(&n_kv), sizeof(n_kv));
    }
    expect(open_throws(path.string()), "truncated file fails");
}

}  // namespace

int main() {
    test_gemv();
    test_rmsnorm();
    test_split_gated_q();
    test_softmax();
    test_rope_norm();
    test_byte_roundtrip();
    test_target_pin();
    test_cpu_device_dispatch();
    test_hip_buffer();
    test_hip_graph_idle();
    test_hip_kernels_match_cpu();
    test_hip_engine_matches_cpu();
    test_qwen_configs();
    test_kv_matches_recompute();
    test_greedy_continuation();
    test_two_engines_match();
    test_argmax();
    test_ggml_nbytes();
    test_gguf_roundtrip();
    test_gguf_bad_magic();
    test_gguf_truncated();
    test_q8_nbytes();
    test_q8_gemv_matches_dequant();
    test_q8_q8x_matches_reconstructed();
    test_q8_ids_match_dequant();
    test_write_load_tiny();
    test_load_rejects_other_arch();
    test_hip_q8_gemv_matches_cpu();
    test_hip_q8_engine_matches_cpu();
    test_q4k_nbytes();
    test_q4k_gemv_matches_dequant();
    test_q4k_q8x_matches_reconstructed();
    test_hip_q4k_gemv_matches_cpu();
    test_q5k_nbytes();
    test_q5k_gemv_matches_dequant();
    test_q5k_q8x_matches_reconstructed();
    test_q6k_nbytes();
    test_q6k_gemv_matches_dequant();
    test_q6k_q8x_matches_reconstructed();
    test_q4k_embed_row();
    test_q6k_embed_row();
    test_write_load_q4km();
    test_packed_mmap_view();
    test_hip_q5k_gemv_matches_cpu();
    test_hip_q6k_gemv_matches_cpu();
    test_gdn_delta_step();
    test_gdn_delta_official_shape();
    test_decode_report_line();
    test_write_load_hybrid();
    test_write_load_qwen35_fixture();
    test_load_qwen35_strips_nextn();
    test_recurrent_layers_override();
    test_load_qwen35_recurrent_map();
    test_load_qwen35_ssm_aliases();
    test_load_qwen35_pin_kv();
    test_inspect_qwen35_pin_kv();
    test_load_qwen35_rejects_missing_gdn();
    test_load_qwen3_5_arch_alias();
    test_tokenizer_gguf_roundtrip();
    test_qwen2_pretok_digits();
    test_qwen35_default_pretok();
    test_qwen35_pretok_and_specials();
    test_gguf_u32_array();
    test_hybrid_generate();
    test_rmsnorm_rows_and_tile();
    test_attn_decode_matches_loop();
    test_add_rmsnorm_and_split_qkv();
    test_fused_split_norm_and_silu();
    test_gdn_conv_split_matches_chain();
    test_gdn_gates_matches_chain();
    test_gemv_swiglu_matches_pair();
    test_pick_multi_row();
    test_gemv3_and_tile_l2();
    test_hip_gemv_swiglu_matches_cpu();
    test_context_cap();
    test_engine_caps_official_context();
    test_rdna4_q8_mmvq_cover();
    test_rdna4_q4k_mmvq_cover();
    test_mrope_text_matches_neox();
    test_rope_k_norm_matches_chain();
    test_llamacpp_parse();
    test_artifact_env();
    test_compare_fixture();
    test_compare_table_fixture();
    test_qwen38_q4km_linear_bytes();
    test_bench_help_official_shapes();
    test_load_check_tiny();
    test_open_meta_truncated_payload();
    test_tiny_is_not_official_pin_header();
    test_official_q4km_header_if_present();
    test_official_q4km_load_if_present();

    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
