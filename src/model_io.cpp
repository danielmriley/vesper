#include "vesper/model_io.h"

#include "vesper/gguf.h"
#include "vesper/gguf_write.h"
#include "vesper/q8.h"
#include "vesper/types.h"
#include "vesper/weight.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace vesper {
namespace {

constexpr const char* kTinyArch = "vesper_tiny";

int require_u32(const GgufFile& file, const char* key) {
    const std::uint64_t value = file.kv_u64(key);
    check(value <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
          std::string("GGUF key overflow: ") + key);
    return static_cast<int>(value);
}

const GgufTensor& require_tensor(const GgufFile& file, const std::string& name) {
    const GgufTensor* tensor = file.find(name);
    check(tensor != nullptr, "missing GGUF tensor: " + name);
    return *tensor;
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

WeightMatrix load_q8_mat(const GgufTensor& tensor, int rows, int cols) {
    check(tensor.type == GgmlType::Q8_0, "expected Q8_0: " + tensor.name);
    expect_dims2(tensor, rows, cols);
    check(tensor.nbytes == q8_packed_bytes(rows, cols), "Q8_0 nbytes: " + tensor.name);
    return WeightMatrix::q8_from_bytes(tensor.data, rows, cols);
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

GgufTensorWrite q8_mat(std::string name, const WeightMatrix& w) {
    check(w.kind() == WeightKind::Q8_0, "q8_mat needs Q8_0: " + name);
    check(w.device() == Device::CPU, "q8_mat needs CPU weights: " + name);
    std::vector<std::byte> packed(w.packed(), w.packed() + w.bytes());
    return GgufTensorWrite{
        std::move(name),
        GgmlType::Q8_0,
        {static_cast<std::uint64_t>(w.cols()), static_cast<std::uint64_t>(w.rows())},
        std::move(packed)};
}

std::string blk_name(int layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
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
    tensors.reserve(static_cast<std::size_t>(3 + c.n_layers * 11));
    tensors.push_back(f32_mat("token_embd.weight", w.tok_emb.data(), c.vocab_size, c.hidden_size));
    tensors.push_back(f32_vec("output_norm.weight", w.final_norm));
    tensors.push_back(q8_mat("output.weight", w.lm_head));
    for (int i = 0; i < c.n_layers; ++i) {
        const LayerWeights& layer = w.layers[static_cast<std::size_t>(i)];
        tensors.push_back(f32_vec(blk_name(i, "attn_norm.weight"), layer.rms_attn));
        tensors.push_back(q8_mat(blk_name(i, "attn_q.weight"), layer.q_proj));
        tensors.push_back(q8_mat(blk_name(i, "attn_k.weight"), layer.k_proj));
        tensors.push_back(q8_mat(blk_name(i, "attn_v.weight"), layer.v_proj));
        tensors.push_back(q8_mat(blk_name(i, "attn_output.weight"), layer.o_proj));
        tensors.push_back(f32_vec(blk_name(i, "attn_q_norm.weight"), layer.q_norm));
        tensors.push_back(f32_vec(blk_name(i, "attn_k_norm.weight"), layer.k_norm));
        tensors.push_back(f32_vec(blk_name(i, "ffn_norm.weight"), layer.rms_mlp));
        tensors.push_back(q8_mat(blk_name(i, "ffn_gate.weight"), layer.gate_proj));
        tensors.push_back(q8_mat(blk_name(i, "ffn_up.weight"), layer.up_proj));
        tensors.push_back(q8_mat(blk_name(i, "ffn_down.weight"), layer.down_proj));
    }
    write_gguf(path, kvs, tensors);
}

ModelWeights load_model(const std::string& path) {
    const GgufFile file = GgufFile::open(path);
    const std::string arch = file.architecture();
    if (arch != kTinyArch) {
        fail("this build only loads general.architecture=vesper_tiny (got '" + arch + "')");
    }

    ModelConfig cfg;
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

    const int h = cfg.hidden_size;
    const int q = cfg.q_dim();
    const int kv = cfg.kv_dim();
    const int inter = cfg.intermediate_size;
    const int v = cfg.vocab_size;

    ModelWeights w;
    w.config = cfg;
    w.tok_emb = load_f32_mat_buf(require_tensor(file, "token_embd.weight"), v, h);
    w.final_norm = load_f32_vec(require_tensor(file, "output_norm.weight"), h);
    w.lm_head = load_q8_mat(require_tensor(file, "output.weight"), v, h);
    w.layers.reserve(static_cast<std::size_t>(cfg.n_layers));
    for (int i = 0; i < cfg.n_layers; ++i) {
        LayerWeights layer;
        layer.rms_attn = load_f32_vec(require_tensor(file, blk_name(i, "attn_norm.weight")), h);
        layer.q_proj = load_q8_mat(require_tensor(file, blk_name(i, "attn_q.weight")), q, h);
        layer.k_proj = load_q8_mat(require_tensor(file, blk_name(i, "attn_k.weight")), kv, h);
        layer.v_proj = load_q8_mat(require_tensor(file, blk_name(i, "attn_v.weight")), kv, h);
        layer.o_proj = load_q8_mat(require_tensor(file, blk_name(i, "attn_output.weight")), h, q);
        layer.q_norm = load_f32_vec(require_tensor(file, blk_name(i, "attn_q_norm.weight")),
                                    cfg.head_dim);
        layer.k_norm = load_f32_vec(require_tensor(file, blk_name(i, "attn_k_norm.weight")),
                                    cfg.head_dim);
        layer.rms_mlp = load_f32_vec(require_tensor(file, blk_name(i, "ffn_norm.weight")), h);
        layer.gate_proj = load_q8_mat(require_tensor(file, blk_name(i, "ffn_gate.weight")), inter, h);
        layer.up_proj = load_q8_mat(require_tensor(file, blk_name(i, "ffn_up.weight")), inter, h);
        layer.down_proj = load_q8_mat(require_tensor(file, blk_name(i, "ffn_down.weight")), h, inter);
        w.layers.push_back(std::move(layer));
    }
    return w;
}

}  // namespace vesper
