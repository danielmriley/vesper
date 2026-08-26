#include "vesper/model_io.h"

#include "vesper/gguf.h"
#include "vesper/gguf_write.h"
#include "vesper/q4k.h"
#include "vesper/q5k.h"
#include "vesper/q6k.h"
#include "vesper/q8.h"
#include "vesper/types.h"
#include "vesper/weight.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

namespace vesper {
namespace {

constexpr const char* kTinyArch = "vesper_tiny";
constexpr const char* kHybridArch = "vesper_hybrid";
constexpr const char* kQwen35Arch = "qwen35";
constexpr const char* kQwen35HfArch = "qwen3_5";

std::string blk_name(int layer, const char* suffix);

int require_u32(const GgufFile& file, const char* key) {
    const std::uint64_t value = file.kv_u64(key);
    check(value <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
          std::string("GGUF key overflow: ") + key);
    return static_cast<int>(value);
}

int optional_u32(const GgufFile& file, const char* key, int fallback) {
    if (!file.has_kv(key)) {
        return fallback;
    }
    return require_u32(file, key);
}

float optional_f32(const GgufFile& file, const char* key, float fallback) {
    if (!file.has_kv(key)) {
        return fallback;
    }
    return static_cast<float>(file.kv_f64(key));
}

bool optional_bool(const GgufFile& file, const char* key, bool fallback) {
    if (!file.has_kv(key)) {
        return fallback;
    }
    return file.kv_bool(key);
}

const GgufTensor& require_tensor(const GgufFile& file, const std::string& name) {
    const GgufTensor* tensor = file.find(name);
    check(tensor != nullptr, "missing GGUF tensor: " + name);
    return *tensor;
}

const GgufTensor& require_blk(const GgufFile& file, int layer,
                              std::initializer_list<const char*> suffixes) {
    std::string tried;
    for (const char* suffix : suffixes) {
        const std::string name = blk_name(layer, suffix);
        if (const GgufTensor* tensor = file.find(name)) {
            return *tensor;
        }
        if (!tried.empty()) {
            tried += " | ";
        }
        tried += name;
    }
    fail("missing GGUF tensor: " + tried);
}

void expect_dims1(const GgufTensor& tensor, int n) {
    check(tensor.dims.size() == 1 && tensor.dims[0] == static_cast<std::uint64_t>(n),
          "bad 1D shape: " + tensor.name);
}

void expect_dims2(const GgufTensor& tensor, int rows, int cols) {
    check(tensor.dims.size() == 2 && tensor.dims[0] == static_cast<std::uint64_t>(cols) &&
              tensor.dims[1] == static_cast<std::uint64_t>(rows),
          "bad 2D shape: " + tensor.name + " (want [" + std::to_string(cols) + ", " +
              std::to_string(rows) + "])");
}

Buffer load_f32_vec(const GgufTensor& tensor, int n) {
    check(tensor.type == GgmlType::F32, "expected F32: " + tensor.name);
    expect_dims1(tensor, n);
    Buffer out(static_cast<std::size_t>(n), Device::CPU);
    std::memcpy(out.data(), tensor.data, static_cast<std::size_t>(n) * sizeof(float));
    return out;
}

Buffer load_f32_mat_buf(const GgufTensor& tensor, int rows, int cols) {
    check(tensor.type == GgmlType::F32, "expected F32: " + tensor.name);
    expect_dims2(tensor, rows, cols);
    Buffer out(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), Device::CPU);
    std::memcpy(out.data(), tensor.data, out.size() * sizeof(float));
    return out;
}

WeightMatrix load_mat(const GgufTensor& tensor, int rows, int cols) {
    expect_dims2(tensor, rows, cols);
    switch (tensor.type) {
        case GgmlType::F32:
            return WeightMatrix::from_f32(
                load_f32_mat_buf(tensor, rows, cols).data(), rows, cols);
        case GgmlType::Q8_0:
            check(tensor.nbytes == q8_packed_bytes(rows, cols), "Q8_0 nbytes: " + tensor.name);
            return WeightMatrix::q8_from_bytes(tensor.data, rows, cols);
        case GgmlType::Q4_K:
            check(tensor.nbytes == q4k_packed_bytes(rows, cols), "Q4_K nbytes: " + tensor.name);
            return WeightMatrix::q4_from_bytes(tensor.data, rows, cols);
        case GgmlType::Q5_K:
            check(tensor.nbytes == q5k_packed_bytes(rows, cols), "Q5_K nbytes: " + tensor.name);
            return WeightMatrix::q5_from_bytes(tensor.data, rows, cols);
        case GgmlType::Q6_K:
            check(tensor.nbytes == q6k_packed_bytes(rows, cols), "Q6_K nbytes: " + tensor.name);
            return WeightMatrix::q6_from_bytes(tensor.data, rows, cols);
        case GgmlType::F16:
        case GgmlType::Q4_0:
        case GgmlType::Q4_1:
        case GgmlType::Q5_0:
        case GgmlType::Q5_1:
        case GgmlType::Q8_1:
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q8_K:
        case GgmlType::IQ2_XXS:
        case GgmlType::IQ2_XS:
        case GgmlType::IQ3_XXS:
        case GgmlType::IQ1_S:
        case GgmlType::IQ4_NL:
        case GgmlType::IQ3_S:
        case GgmlType::IQ2_S:
        case GgmlType::IQ4_XS:
        case GgmlType::I8:
        case GgmlType::I16:
        case GgmlType::I32:
        case GgmlType::I64:
        case GgmlType::F64:
        case GgmlType::IQ1_M:
        case GgmlType::BF16:
            fail("unsupported weight type " + std::string(ggml_type_name(tensor.type)) +
                 " for " + tensor.name);
    }
    throw std::logic_error("unhandled GgmlType");
}

WeightMatrix load_emb(const GgufTensor& tensor, int rows, int cols) {
    return load_mat(tensor, rows, cols);
}

std::vector<std::byte> bytes_of(const float* data, std::size_t n) {
    std::vector<std::byte> out(n * sizeof(float));
    std::memcpy(out.data(), data, out.size());
    return out;
}

GgufTensorWrite f32_vec(std::string name, const Buffer& buf) {
    return GgufTensorWrite{std::move(name),
                           GgmlType::F32,
                           {static_cast<std::uint64_t>(buf.size())},
                           bytes_of(buf.data(), buf.size())};
}

GgufTensorWrite f32_mat(std::string name, const float* data, int rows, int cols) {
    return GgufTensorWrite{
        std::move(name),
        GgmlType::F32,
        {static_cast<std::uint64_t>(cols), static_cast<std::uint64_t>(rows)},
        bytes_of(data, static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols))};
}

GgufTensorWrite packed_mat(std::string name, const WeightMatrix& w) {
    check(w.device() == Device::CPU, "writer needs CPU weights: " + name);
    switch (w.kind()) {
        case WeightKind::F32:
            return f32_mat(std::move(name), w.f32_data(), w.rows(), w.cols());
        case WeightKind::Q8_0: {
            std::vector<std::byte> packed(w.packed(), w.packed() + w.bytes());
            return GgufTensorWrite{
                std::move(name),
                GgmlType::Q8_0,
                {static_cast<std::uint64_t>(w.cols()), static_cast<std::uint64_t>(w.rows())},
                std::move(packed)};
        }
        case WeightKind::Q4_K: {
            std::vector<std::byte> packed(w.packed(), w.packed() + w.bytes());
            return GgufTensorWrite{
                std::move(name),
                GgmlType::Q4_K,
                {static_cast<std::uint64_t>(w.cols()), static_cast<std::uint64_t>(w.rows())},
                std::move(packed)};
        }
        case WeightKind::Q5_K: {
            std::vector<std::byte> packed(w.packed(), w.packed() + w.bytes());
            return GgufTensorWrite{
                std::move(name),
                GgmlType::Q5_K,
                {static_cast<std::uint64_t>(w.cols()), static_cast<std::uint64_t>(w.rows())},
                std::move(packed)};
        }
        case WeightKind::Q6_K: {
            std::vector<std::byte> packed(w.packed(), w.packed() + w.bytes());
            return GgufTensorWrite{
                std::move(name),
                GgmlType::Q6_K,
                {static_cast<std::uint64_t>(w.cols()), static_cast<std::uint64_t>(w.rows())},
                std::move(packed)};
        }
    }
    throw std::logic_error("unhandled WeightKind");
}

std::string blk_name(int layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

void append_ffn(std::vector<GgufTensorWrite>* tensors, int i, const LayerWeights& layer,
                const char* norm_name) {
    tensors->push_back(f32_vec(blk_name(i, norm_name), layer.rms_mlp));
    tensors->push_back(packed_mat(blk_name(i, "ffn_gate.weight"), layer.gate_proj));
    tensors->push_back(packed_mat(blk_name(i, "ffn_up.weight"), layer.up_proj));
    tensors->push_back(packed_mat(blk_name(i, "ffn_down.weight"), layer.down_proj));
}

void write_hybrid_file(const std::string& path, std::uint32_t seed, const char* arch,
                       int nextn_layers) {
    ModelConfig cfg = ModelConfig::tiny_hybrid();
    cfg.arch = arch;
    cfg.nextn_predict_layers = nextn_layers;
    const ModelWeights w = ModelWeights::random(cfg, seed).to_q8();
    const ModelConfig& c = w.config;
    const std::string p = std::string(arch) + ".";
    std::vector<GgufKvWrite> kvs = {
        gguf_kv_string("general.architecture", arch),
        gguf_kv_u32("general.alignment", 32),
        gguf_kv_u32(p + "vocab_size", static_cast<std::uint32_t>(c.vocab_size)),
        gguf_kv_u32(p + "block_count",
                    static_cast<std::uint32_t>(c.n_layers + c.nextn_predict_layers)),
        gguf_kv_u32(p + "nextn_predict_layers",
                    static_cast<std::uint32_t>(c.nextn_predict_layers)),
        gguf_kv_u32(p + "embedding_length", static_cast<std::uint32_t>(c.hidden_size)),
        gguf_kv_u32(p + "feed_forward_length", static_cast<std::uint32_t>(c.intermediate_size)),
        gguf_kv_u32(p + "attention.head_count", static_cast<std::uint32_t>(c.n_heads)),
        gguf_kv_u32(p + "attention.head_count_kv", static_cast<std::uint32_t>(c.n_kv_heads)),
        gguf_kv_u32(p + "attention.key_length", static_cast<std::uint32_t>(c.head_dim)),
        gguf_kv_f32(p + "attention.layer_norm_rms_epsilon", c.rms_eps),
        gguf_kv_f32(p + "rope.freq_base", c.rope_theta),
        gguf_kv_u32(p + "rope.dimension_count", static_cast<std::uint32_t>(c.rotary_dim())),
        gguf_kv_u32(p + "context_length", static_cast<std::uint32_t>(c.max_seq_len)),
        gguf_kv_u32(p + "full_attention_interval",
                    static_cast<std::uint32_t>(c.full_attention_interval)),
        gguf_kv_u32(p + "ssm.conv_kernel", static_cast<std::uint32_t>(c.gdn_conv_kernel)),
        gguf_kv_u32(p + "ssm.inner_size", static_cast<std::uint32_t>(c.gdn_value_dim())),
        gguf_kv_u32(p + "ssm.state_size", static_cast<std::uint32_t>(c.gdn_head_dim)),
        gguf_kv_u32(p + "ssm.group_count", static_cast<std::uint32_t>(c.gdn_qk_heads)),
        gguf_kv_u32(p + "ssm.time_step_rank", static_cast<std::uint32_t>(c.gdn_v_heads)),
        gguf_kv_bool(p + "attention.qk_norm", c.qk_norm),
        gguf_kv_bool(p + "tie_word_embeddings", c.tie_word_embeddings),
    };
    if (std::string(arch) == kQwen35Arch) {
        kvs.push_back(gguf_kv_u32_array(p + "rope.dimension_sections", {3, 3, 2, 0}));
    }

    std::vector<GgufTensorWrite> tensors;
    tensors.push_back(f32_mat("token_embd.weight", w.tok_emb.f32_data(), c.vocab_size, c.hidden_size));
    tensors.push_back(f32_vec("output_norm.weight", w.final_norm));
    tensors.push_back(packed_mat("output.weight", w.lm_head));
    for (int i = 0; i < c.n_layers; ++i) {
        const LayerWeights& layer = w.layers[static_cast<std::size_t>(i)];
        tensors.push_back(f32_vec(blk_name(i, "attn_norm.weight"), layer.rms_attn));
        switch (c.layer_kind(i)) {
            case LayerKind::Attention:
                tensors.push_back(packed_mat(blk_name(i, "attn_q.weight"), layer.q_proj));
                tensors.push_back(packed_mat(blk_name(i, "attn_k.weight"), layer.k_proj));
                tensors.push_back(packed_mat(blk_name(i, "attn_v.weight"), layer.v_proj));
                tensors.push_back(packed_mat(blk_name(i, "attn_output.weight"), layer.o_proj));
                tensors.push_back(f32_vec(blk_name(i, "attn_q_norm.weight"), layer.q_norm));
                tensors.push_back(f32_vec(blk_name(i, "attn_k_norm.weight"), layer.k_norm));
                break;
            case LayerKind::DeltaNet:
                tensors.push_back(packed_mat(blk_name(i, "attn_qkv.weight"), layer.qkv_proj));
                tensors.push_back(packed_mat(blk_name(i, "attn_gate.weight"), layer.z_proj));
                tensors.push_back(f32_mat(blk_name(i, "ssm_conv1d.weight"), layer.conv1d.data(),
                                          c.gdn_qkv_dim(), c.gdn_conv_kernel));
                tensors.push_back(f32_vec(blk_name(i, "ssm_dt.bias"), layer.ssm_dt));
                tensors.push_back(f32_vec(blk_name(i, "ssm_a"), layer.ssm_a));
                tensors.push_back(packed_mat(blk_name(i, "ssm_beta.weight"), layer.beta_proj));
                tensors.push_back(packed_mat(blk_name(i, "ssm_alpha.weight"), layer.alpha_proj));
                tensors.push_back(f32_vec(blk_name(i, "ssm_norm.weight"), layer.ssm_norm));
                tensors.push_back(packed_mat(blk_name(i, "ssm_out.weight"), layer.ssm_out));
                break;
        }
        append_ffn(&tensors, i, layer, "post_attention_norm.weight");
    }
    write_gguf(path, kvs, tensors);
}

LayerWeights load_attn_layer(const GgufFile& file, int i, const ModelConfig& cfg,
                             const char* ffn_norm) {
    const int h = cfg.hidden_size;
    LayerWeights layer;
    layer.rms_attn = load_f32_vec(require_tensor(file, blk_name(i, "attn_norm.weight")), h);
    layer.q_proj = load_mat(require_tensor(file, blk_name(i, "attn_q.weight")), cfg.q_proj_rows(), h);
    layer.k_proj = load_mat(require_tensor(file, blk_name(i, "attn_k.weight")), cfg.kv_dim(), h);
    layer.v_proj = load_mat(require_tensor(file, blk_name(i, "attn_v.weight")), cfg.kv_dim(), h);
    layer.o_proj = load_mat(require_tensor(file, blk_name(i, "attn_output.weight")), h, cfg.q_dim());
    layer.q_norm = load_f32_vec(require_tensor(file, blk_name(i, "attn_q_norm.weight")),
                                cfg.head_dim);
    layer.k_norm = load_f32_vec(require_tensor(file, blk_name(i, "attn_k_norm.weight")),
                                cfg.head_dim);
    layer.rms_mlp = load_f32_vec(require_tensor(file, blk_name(i, ffn_norm)), h);
    layer.gate_proj = load_mat(require_tensor(file, blk_name(i, "ffn_gate.weight")),
                               cfg.intermediate_size, h);
    layer.up_proj = load_mat(require_tensor(file, blk_name(i, "ffn_up.weight")),
                             cfg.intermediate_size, h);
    layer.down_proj = load_mat(require_tensor(file, blk_name(i, "ffn_down.weight")), h,
                               cfg.intermediate_size);
    return layer;
}

LayerWeights load_gdn_layer(const GgufFile& file, int i, const ModelConfig& cfg) {
    const int h = cfg.hidden_size;
    LayerWeights layer;
    layer.rms_attn = load_f32_vec(require_tensor(file, blk_name(i, "attn_norm.weight")), h);
    layer.qkv_proj = load_mat(require_blk(file, i, {"attn_qkv.weight", "ssm.in_proj.weight"}),
                              cfg.gdn_qkv_dim(), h);
    layer.z_proj = load_mat(require_blk(file, i, {"attn_gate.weight"}), cfg.gdn_value_dim(), h);
    const GgufTensor& conv = require_blk(file, i, {"ssm_conv1d.weight", "ssm.conv1d.weight"});
    check(conv.type == GgmlType::F32, "ssm_conv1d must be F32");
    expect_dims2(conv, cfg.gdn_qkv_dim(), cfg.gdn_conv_kernel);
    layer.conv1d = Buffer(static_cast<std::size_t>(cfg.gdn_qkv_dim()) * cfg.gdn_conv_kernel,
                          Device::CPU);
    std::memcpy(layer.conv1d.data(), conv.data, layer.conv1d.size() * sizeof(float));
    layer.ssm_dt = load_f32_vec(require_blk(file, i, {"ssm_dt.bias", "ssm.dt_bias", "dt_bias"}),
                                cfg.gdn_v_heads);
    layer.ssm_a = load_f32_vec(require_blk(file, i, {"ssm_a", "ssm.A_log", "A_log"}),
                               cfg.gdn_v_heads);
    layer.beta_proj = load_mat(require_blk(file, i, {"ssm_beta.weight", "ssm.beta.weight"}),
                               cfg.gdn_v_heads, h);
    layer.alpha_proj = load_mat(require_blk(file, i, {"ssm_alpha.weight", "ssm.alpha.weight"}),
                                cfg.gdn_v_heads, h);
    layer.ssm_norm = load_f32_vec(require_blk(file, i, {"ssm_norm.weight", "ssm.norm.weight"}),
                                  cfg.gdn_head_dim);
    layer.ssm_out = load_mat(require_blk(file, i, {"ssm_out.weight", "ssm.out.weight"}), h,
                             cfg.gdn_value_dim());
    layer.rms_mlp =
        load_f32_vec(require_blk(file, i, {"post_attention_norm.weight", "ffn_norm.weight"}), h);
    layer.gate_proj = load_mat(require_tensor(file, blk_name(i, "ffn_gate.weight")),
                               cfg.intermediate_size, h);
    layer.up_proj = load_mat(require_tensor(file, blk_name(i, "ffn_up.weight")),
                             cfg.intermediate_size, h);
    layer.down_proj = load_mat(require_tensor(file, blk_name(i, "ffn_down.weight")), h,
                               cfg.intermediate_size);
    return layer;
}

ModelConfig load_tiny_config(const GgufFile& file) {
    ModelConfig cfg;
    cfg.arch = kTinyArch;
    cfg.vocab_size = require_u32(file, "vesper_tiny.vocab_size");
    cfg.hidden_size = require_u32(file, "vesper_tiny.hidden_size");
    cfg.n_layers = require_u32(file, "vesper_tiny.n_layers");
    cfg.n_heads = require_u32(file, "vesper_tiny.n_heads");
    cfg.n_kv_heads = require_u32(file, "vesper_tiny.n_kv_heads");
    cfg.head_dim = require_u32(file, "vesper_tiny.head_dim");
    cfg.intermediate_size = require_u32(file, "vesper_tiny.intermediate_size");
    cfg.rms_eps = static_cast<float>(file.kv_f64("vesper_tiny.rms_eps"));
    cfg.rope_theta = static_cast<float>(file.kv_f64("vesper_tiny.rope_theta"));
    cfg.qk_norm = file.kv_bool("vesper_tiny.qk_norm");
    cfg.tie_word_embeddings = file.kv_bool("vesper_tiny.tie_word_embeddings");
    cfg.max_seq_len = require_u32(file, "vesper_tiny.max_seq_len");
    cfg.validate();
    return cfg;
}

ModelConfig load_hybrid_config(const GgufFile& file, const std::string& prefix) {
    ModelConfig cfg;
    cfg.arch = file.architecture();
    const GgufTensor* emb = file.find("token_embd.weight");
    check(emb != nullptr && emb->dims.size() == 2, "token_embd.weight must be 2D");
    cfg.vocab_size = optional_u32(file, (prefix + "vocab_size").c_str(),
                                  static_cast<int>(emb->dims[1]));
    cfg.hidden_size = require_u32(file, (prefix + "embedding_length").c_str());
    const int block_count = require_u32(file, (prefix + "block_count").c_str());
    cfg.nextn_predict_layers = optional_u32(file, (prefix + "nextn_predict_layers").c_str(), 0);
    check(cfg.nextn_predict_layers >= 0 && cfg.nextn_predict_layers < block_count,
          "nextn_predict_layers must be < block_count");
    cfg.n_layers = block_count - cfg.nextn_predict_layers;
    cfg.n_heads = require_u32(file, (prefix + "attention.head_count").c_str());
    cfg.n_kv_heads = require_u32(file, (prefix + "attention.head_count_kv").c_str());
    cfg.head_dim = require_u32(file, (prefix + "attention.key_length").c_str());
    cfg.intermediate_size = require_u32(file, (prefix + "feed_forward_length").c_str());
    cfg.rms_eps = optional_f32(file, (prefix + "attention.layer_norm_rms_epsilon").c_str(), 1e-6f);
    cfg.rope_theta = optional_f32(file, (prefix + "rope.freq_base").c_str(), 10000.0f);
    cfg.rope_dim = optional_u32(file, (prefix + "rope.dimension_count").c_str(), 0);
    cfg.max_seq_len = optional_u32(file, (prefix + "context_length").c_str(), 4096);
    cfg.full_attention_interval =
        optional_u32(file, (prefix + "full_attention_interval").c_str(), 4);
    cfg.gdn_conv_kernel = optional_u32(file, (prefix + "ssm.conv_kernel").c_str(), 4);
    cfg.gdn_qk_heads = require_u32(file, (prefix + "ssm.group_count").c_str());
    cfg.gdn_v_heads = require_u32(file, (prefix + "ssm.time_step_rank").c_str());
    cfg.gdn_head_dim = require_u32(file, (prefix + "ssm.state_size").c_str());
    cfg.qk_norm = optional_bool(file, (prefix + "attention.qk_norm").c_str(), true);
    cfg.attn_gate = true;
    cfg.tie_word_embeddings = optional_bool(file, (prefix + "tie_word_embeddings").c_str(), false);
    if (file.find("output.weight") == nullptr) {
        cfg.tie_word_embeddings = true;
    }
    const std::string sec_key = prefix + "rope.dimension_sections";
    if (file.has_kv(sec_key)) {
        const std::vector<std::uint64_t> secs = file.kv_u64_array(sec_key);
        check(!secs.empty() && secs.size() <= 4, "rope.dimension_sections must have 1..4 entries");
        int nsec = static_cast<int>(secs.size());
        while (nsec > 0 && secs[static_cast<std::size_t>(nsec - 1)] == 0) {
            --nsec;
        }
        check(nsec > 0, "rope.dimension_sections has no positive entries");
        cfg.n_rope_sections = nsec;
        for (int i = 0; i < nsec; ++i) {
            check(secs[static_cast<std::size_t>(i)] > 0 && secs[static_cast<std::size_t>(i)] <= 256,
                  "rope section out of range");
            cfg.rope_section[i] = static_cast<int>(secs[static_cast<std::size_t>(i)]);
        }
    }
    cfg.validate();
    return cfg;
}

}  // namespace

void write_tiny_q8(const std::string& path, std::uint32_t seed) {
    const ModelWeights w = ModelWeights::random(ModelConfig::tiny_demo(), seed).to_q8();
    const ModelConfig& c = w.config;
    const std::vector<GgufKvWrite> kvs = {
        gguf_kv_string("general.architecture", kTinyArch),
        gguf_kv_u32("general.alignment", 32),
        gguf_kv_u32("vesper_tiny.vocab_size", static_cast<std::uint32_t>(c.vocab_size)),
        gguf_kv_u32("vesper_tiny.hidden_size", static_cast<std::uint32_t>(c.hidden_size)),
        gguf_kv_u32("vesper_tiny.n_layers", static_cast<std::uint32_t>(c.n_layers)),
        gguf_kv_u32("vesper_tiny.n_heads", static_cast<std::uint32_t>(c.n_heads)),
        gguf_kv_u32("vesper_tiny.n_kv_heads", static_cast<std::uint32_t>(c.n_kv_heads)),
        gguf_kv_u32("vesper_tiny.head_dim", static_cast<std::uint32_t>(c.head_dim)),
        gguf_kv_u32("vesper_tiny.intermediate_size",
                    static_cast<std::uint32_t>(c.intermediate_size)),
        gguf_kv_f32("vesper_tiny.rms_eps", c.rms_eps),
        gguf_kv_f32("vesper_tiny.rope_theta", c.rope_theta),
        gguf_kv_bool("vesper_tiny.qk_norm", c.qk_norm),
        gguf_kv_bool("vesper_tiny.tie_word_embeddings", c.tie_word_embeddings),
        gguf_kv_u32("vesper_tiny.max_seq_len", static_cast<std::uint32_t>(c.max_seq_len)),
    };

    std::vector<GgufTensorWrite> tensors;
    tensors.push_back(
        f32_mat("token_embd.weight", w.tok_emb.f32_data(), c.vocab_size, c.hidden_size));
    tensors.push_back(f32_vec("output_norm.weight", w.final_norm));
    tensors.push_back(packed_mat("output.weight", w.lm_head));
    for (int i = 0; i < c.n_layers; ++i) {
        const LayerWeights& layer = w.layers[static_cast<std::size_t>(i)];
        tensors.push_back(f32_vec(blk_name(i, "attn_norm.weight"), layer.rms_attn));
        tensors.push_back(packed_mat(blk_name(i, "attn_q.weight"), layer.q_proj));
        tensors.push_back(packed_mat(blk_name(i, "attn_k.weight"), layer.k_proj));
        tensors.push_back(packed_mat(blk_name(i, "attn_v.weight"), layer.v_proj));
        tensors.push_back(packed_mat(blk_name(i, "attn_output.weight"), layer.o_proj));
        tensors.push_back(f32_vec(blk_name(i, "attn_q_norm.weight"), layer.q_norm));
        tensors.push_back(f32_vec(blk_name(i, "attn_k_norm.weight"), layer.k_norm));
        append_ffn(&tensors, i, layer, "ffn_norm.weight");
    }
    write_gguf(path, kvs, tensors);
}

namespace {

WeightMatrix quantize_kind(const WeightMatrix& w, WeightKind kind) {
    check(w.kind() == WeightKind::F32, "quantize_kind expects F32");
    check(w.device() == Device::CPU, "quantize_kind expects CPU");
    switch (kind) {
        case WeightKind::F32:
            return w;
        case WeightKind::Q8_0:
            return WeightMatrix::q8_from_f32(w.f32_data(), w.rows(), w.cols());
        case WeightKind::Q4_K:
            return WeightMatrix::q4_from_f32(w.f32_data(), w.rows(), w.cols());
        case WeightKind::Q5_K:
            return WeightMatrix::q5_from_f32(w.f32_data(), w.rows(), w.cols());
        case WeightKind::Q6_K:
            return WeightMatrix::q6_from_f32(w.f32_data(), w.rows(), w.cols());
    }
    throw std::logic_error("unhandled WeightKind");
}

void write_tiny_file(const std::string& path, const ModelWeights& w) {
    const ModelConfig& c = w.config;
    check(c.arch == kTinyArch, "write_tiny_file expects vesper_tiny");
    const std::vector<GgufKvWrite> kvs = {
        gguf_kv_string("general.architecture", kTinyArch),
        gguf_kv_u32("general.alignment", 32),
        gguf_kv_u32("vesper_tiny.vocab_size", static_cast<std::uint32_t>(c.vocab_size)),
        gguf_kv_u32("vesper_tiny.hidden_size", static_cast<std::uint32_t>(c.hidden_size)),
        gguf_kv_u32("vesper_tiny.n_layers", static_cast<std::uint32_t>(c.n_layers)),
        gguf_kv_u32("vesper_tiny.n_heads", static_cast<std::uint32_t>(c.n_heads)),
        gguf_kv_u32("vesper_tiny.n_kv_heads", static_cast<std::uint32_t>(c.n_kv_heads)),
        gguf_kv_u32("vesper_tiny.head_dim", static_cast<std::uint32_t>(c.head_dim)),
        gguf_kv_u32("vesper_tiny.intermediate_size",
                    static_cast<std::uint32_t>(c.intermediate_size)),
        gguf_kv_f32("vesper_tiny.rms_eps", c.rms_eps),
        gguf_kv_f32("vesper_tiny.rope_theta", c.rope_theta),
        gguf_kv_bool("vesper_tiny.qk_norm", c.qk_norm),
        gguf_kv_bool("vesper_tiny.tie_word_embeddings", c.tie_word_embeddings),
        gguf_kv_u32("vesper_tiny.max_seq_len", static_cast<std::uint32_t>(c.max_seq_len)),
    };

    std::vector<GgufTensorWrite> tensors;
    tensors.push_back(packed_mat("token_embd.weight", w.tok_emb));
    tensors.push_back(f32_vec("output_norm.weight", w.final_norm));
    tensors.push_back(packed_mat("output.weight", w.lm_head));
    for (int i = 0; i < c.n_layers; ++i) {
        const LayerWeights& layer = w.layers[static_cast<std::size_t>(i)];
        tensors.push_back(f32_vec(blk_name(i, "attn_norm.weight"), layer.rms_attn));
        tensors.push_back(packed_mat(blk_name(i, "attn_q.weight"), layer.q_proj));
        tensors.push_back(packed_mat(blk_name(i, "attn_k.weight"), layer.k_proj));
        tensors.push_back(packed_mat(blk_name(i, "attn_v.weight"), layer.v_proj));
        tensors.push_back(packed_mat(blk_name(i, "attn_output.weight"), layer.o_proj));
        tensors.push_back(f32_vec(blk_name(i, "attn_q_norm.weight"), layer.q_norm));
        tensors.push_back(f32_vec(blk_name(i, "attn_k_norm.weight"), layer.k_norm));
        append_ffn(&tensors, i, layer, "ffn_norm.weight");
    }
    write_gguf(path, kvs, tensors);
}

}  // namespace

void write_tiny_q4km(const std::string& path, std::uint32_t seed) {
    ModelWeights w = ModelWeights::random(ModelConfig::tiny_q4km(), seed);
    w.tok_emb = quantize_kind(w.tok_emb, WeightKind::Q6_K);
    w.lm_head = quantize_kind(w.lm_head, WeightKind::Q6_K);
    for (LayerWeights& layer : w.layers) {
        layer.q_proj = quantize_kind(layer.q_proj, WeightKind::Q5_K);
        layer.k_proj = quantize_kind(layer.k_proj, WeightKind::Q4_K);
        layer.v_proj = quantize_kind(layer.v_proj, WeightKind::Q6_K);
        layer.o_proj = quantize_kind(layer.o_proj, WeightKind::Q4_K);
        layer.gate_proj = quantize_kind(layer.gate_proj, WeightKind::Q4_K);
        layer.up_proj = quantize_kind(layer.up_proj, WeightKind::Q4_K);
        layer.down_proj = quantize_kind(layer.down_proj, WeightKind::Q6_K);
    }
    write_tiny_file(path, w);
}

void write_tiny_hybrid(const std::string& path, std::uint32_t seed) {
    write_hybrid_file(path, seed, kHybridArch, 0);
}

void write_tiny_qwen35(const std::string& path, std::uint32_t seed) {
    write_tiny_qwen35(path, seed, 0);
}

void write_tiny_qwen35(const std::string& path, std::uint32_t seed, int nextn_layers) {
    write_hybrid_file(path, seed, kQwen35Arch, nextn_layers);
}

ModelWeights load_model(const std::string& path) {
    const GgufFile file = GgufFile::open(path);
    const std::string arch = file.architecture();

    ModelConfig cfg;
    const char* ffn_norm = "ffn_norm.weight";
    if (arch == kTinyArch) {
        cfg = load_tiny_config(file);
    } else if (arch == kHybridArch) {
        cfg = load_hybrid_config(file, "vesper_hybrid.");
        ffn_norm = "post_attention_norm.weight";
    } else if (arch == kQwen35Arch || arch == kQwen35HfArch) {
        const char* prefix = "qwen35.";
        if (file.has_kv("qwen3_5.block_count")) {
            prefix = "qwen3_5.";
        } else if (!file.has_kv("qwen35.block_count") && arch == kQwen35HfArch) {
            prefix = "qwen3_5.";
        }
        cfg = load_hybrid_config(file, prefix);
        ffn_norm = "post_attention_norm.weight";
    } else {
        fail("unsupported GGUF architecture '" + arch +
             "' (this build loads vesper_tiny, vesper_hybrid, qwen35, qwen3_5)");
    }

    const int h = cfg.hidden_size;
    const int v = cfg.vocab_size;
    ModelWeights w;
    w.config = cfg;
    w.tok_emb = load_emb(require_tensor(file, "token_embd.weight"), v, h);
    w.final_norm = load_f32_vec(require_tensor(file, "output_norm.weight"), h);
    if (const GgufTensor* out = file.find("output.weight")) {
        w.lm_head = load_mat(*out, v, h);
    } else {
        check(cfg.tie_word_embeddings, "missing output.weight");
        w.lm_head = w.tok_emb;
    }
    w.layers.reserve(static_cast<std::size_t>(cfg.n_layers));
    for (int i = 0; i < cfg.n_layers; ++i) {
        switch (cfg.layer_kind(i)) {
            case LayerKind::Attention:
                w.layers.push_back(load_attn_layer(file, i, cfg, ffn_norm));
                break;
            case LayerKind::DeltaNet:
                w.layers.push_back(load_gdn_layer(file, i, cfg));
                break;
        }
    }
    return w;
}

}  // namespace vesper
