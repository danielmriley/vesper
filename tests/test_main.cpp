#include "vesper/buffer.h"
#include "vesper/config.h"
#include "vesper/engine.h"
#include "vesper/gdn.h"
#include "vesper/gguf.h"
#include "vesper/gguf_write.h"
#include "vesper/hip.h"
#include "vesper/kernels.h"
#include "vesper/model_io.h"
#include "vesper/q4k.h"
#include "vesper/q8.h"
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

void test_target_pin() {
    expect(std::string(vesper::kHipArch) == "gfx1201", "v1 HIP arch is gfx1201");
    expect(vesper::kWavefront == 32, "RDNA4 wave32");
    expect(vesper::kCachelineBytes == 256, "RDNA4 256B cacheline");
    expect(vesper::kComputeUnits == 64, "R9700 64 CUs");
    expect(vesper::kGemvWorkgroup == 256, "GEMV workgroup 256");
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
    const auto q27 = vesper::ModelConfig::qwen38_27b();
    expect(q27.arch == "qwen35", "qwen38 arch qwen35");
    expect(q27.n_layers == 64 && q27.hidden_size == 5120, "qwen38 64x5120");
    expect(q27.gdn_v_heads == 48 && q27.gdn_qk_heads == 16, "qwen38 GDN heads");
    expect(q27.attn_gate && q27.rope_dim == 64, "qwen38 gated partial rope");
    ++g_passed;  // validate() did not throw
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
    std::vector<float> y_cpu(static_cast<std::size_t>(rows));
    vesper::gemv_q8(y_cpu.data(), packed.packed(), x.data(), rows, cols);

    auto gpu_w = packed.to(vesper::Device::HIP);
    vesper::Buffer X(static_cast<std::size_t>(cols), vesper::Device::HIP);
    vesper::Buffer Y(static_cast<std::size_t>(rows), vesper::Device::HIP);
    X.copy_from(x.data(), x.size());
    vesper::gemv(vesper::Device::HIP, Y.data(), gpu_w, X.data());
    std::vector<float> y_gpu(static_cast<std::size_t>(rows));
    Y.copy_to(y_gpu.data(), y_gpu.size());
    expect(close_vec(y_cpu.data(), y_gpu.data(), rows, 2e-4f), "HIP Q8 GEMV matches CPU");
}

void test_hip_q8_engine_matches_cpu() {
    if (!vesper::hip_available()) {
        return;
    }
    const auto cfg = vesper::ModelConfig::tiny_demo();
    const auto q8 = vesper::ModelWeights::random(cfg, 3).to_q8();
    vesper::Engine cpu(q8, vesper::Device::CPU);
    vesper::Engine gpu(q8, vesper::Device::HIP);
    expect(cpu.generate({9, 8, 7}, 8) == gpu.generate({9, 8, 7}, 8),
           "HIP Q8 engine greedy tokens match CPU");
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
    std::vector<float> y_cpu(static_cast<std::size_t>(rows));
    vesper::gemv_q4k(y_cpu.data(), packed.packed(), x.data(), rows, cols);
    auto gpu_w = packed.to(vesper::Device::HIP);
    vesper::Buffer X(static_cast<std::size_t>(cols), vesper::Device::HIP);
    vesper::Buffer Y(static_cast<std::size_t>(rows), vesper::Device::HIP);
    X.copy_from(x.data(), x.size());
    vesper::gemv(vesper::Device::HIP, Y.data(), gpu_w, X.data());
    std::vector<float> y_gpu(static_cast<std::size_t>(rows));
    Y.copy_to(y_gpu.data(), y_gpu.size());
    expect(close_vec(y_cpu.data(), y_gpu.data(), rows, 2e-3f), "HIP Q4_K GEMV matches CPU");
}

void test_gdn_delta_step() {
    const int heads = 2;
    const int dim = 16;
    std::vector<float> rec(static_cast<std::size_t>(heads * dim * dim), 0.1f);
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
    for (int h = 0; h < heads; ++h) {
        float* S = rec_ref.data() + h * dim * dim;
        for (int i = 0; i < dim * dim; ++i) {
            S[i] *= decay[static_cast<std::size_t>(h)];
        }
        for (int j = 0; j < dim; ++j) {
            float retrieved = 0.0f;
            for (int i = 0; i < dim; ++i) {
                retrieved += S[i * dim + j] * k[static_cast<std::size_t>(h * dim + i)];
            }
            const float delta = beta[static_cast<std::size_t>(h)] *
                                (v[static_cast<std::size_t>(h * dim + j)] - retrieved);
            for (int i = 0; i < dim; ++i) {
                S[i * dim + j] += k[static_cast<std::size_t>(h * dim + i)] * delta;
            }
            float acc = 0.0f;
            for (int i = 0; i < dim; ++i) {
                acc += S[i * dim + j] * q[static_cast<std::size_t>(h * dim + i)];
            }
            y_ref[static_cast<std::size_t>(h * dim + j)] = acc;
        }
    }
    expect(close_vec(y.data(), y_ref.data(), heads * dim, 1e-5f), "GDN delta matches reference");
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
}

void test_write_load_qwen35_fixture() {
    const auto dir = std::filesystem::temp_directory_path() / "vesper-gguf-hybrid";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "qwen35.gguf").string();
    vesper::write_tiny_qwen35(path, 9);
    const auto loaded = vesper::load_model(path);
    expect(loaded.config.arch == "qwen35", "loaded qwen35 arch");
    expect(loaded.config.is_hybrid(), "qwen35 is hybrid");
    vesper::Engine engine(loaded);
    const auto ids = engine.generate({1, 2}, 4);
    expect(ids.size() == 6, "qwen35 fixture generate length");
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
    expect(a.generate({9, 8, 7}, 8) == b.generate({9, 8, 7}, 8),
           "two hybrid engines same tokens");
    const vesper::DecodeReport report = a.last_report();
    expect(report.model == "tiny_hybrid", "hybrid report model");
    expect(report.new_tokens == 8, "hybrid report tokens");
}

void test_compare_fixture() {
    const int rc = std::system(
        "COMPARE_FIXTURE=1 scripts/compare-qwen38/run_vesper.sh > /tmp/vesper-compare-fixture.txt");
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

void test_artifact_env() {
    const std::filesystem::path path = std::filesystem::path("scripts/compare-qwen38/artifact.env");
    expect(std::filesystem::exists(path), "artifact.env exists");
    std::ifstream in(path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    expect(text.find("COMPARE_QUANT=Q4_K_M") != std::string::npos, "artifact quant Q4_K_M");
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
    test_softmax();
    test_rope_norm();
    test_byte_roundtrip();
    test_target_pin();
    test_cpu_device_dispatch();
    test_hip_buffer();
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
    test_q8_ids_match_dequant();
    test_write_load_tiny();
    test_load_rejects_other_arch();
    test_hip_q8_gemv_matches_cpu();
    test_hip_q8_engine_matches_cpu();
    test_q4k_nbytes();
    test_q4k_gemv_matches_dequant();
    test_hip_q4k_gemv_matches_cpu();
    test_gdn_delta_step();
    test_decode_report_line();
    test_write_load_hybrid();
    test_write_load_qwen35_fixture();
    test_load_qwen35_rejects_missing_gdn();
    test_tokenizer_gguf_roundtrip();
    test_gguf_u32_array();
    test_hybrid_generate();
    test_artifact_env();
    test_compare_fixture();

    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
