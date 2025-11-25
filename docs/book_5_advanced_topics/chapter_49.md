```markdown
# Chapter 49: GGUF Format Support

## 1. Introduction

**GGUF** (GPT-Generated Unified Format) is the model format used by **llama.cpp** and the broader ecosystem (Ollama, LM Studio, etc.). Supporting GGUF enables:

- **Interoperability**: Use models from the llama.cpp community
- **Quantized weights**: GGUF includes many quantization formats (Q4_0, Q4_K_M, etc.)
- **Single-file distribution**: All model components in one file
- **Metadata**: Model config, tokenizer, and more embedded in the file

This chapter covers:
1. **GGUF file format**: Header, metadata, tensors
2. **Quantization formats**: Q4, Q5, Q6, K-quants
3. **Loading GGUF models**: Parsing and dequantization
4. **Tokenizer extraction**: Using embedded tokenizer data

## 2. GGUF File Structure

```
┌─────────────────────────────────────────────┐
│ Magic Number (4 bytes): "GGUF"              │
├─────────────────────────────────────────────┤
│ Version (4 bytes): uint32                   │
├─────────────────────────────────────────────┤
│ Tensor Count (8 bytes): uint64              │
├─────────────────────────────────────────────┤
│ Metadata KV Count (8 bytes): uint64         │
├─────────────────────────────────────────────┤
│ Metadata Key-Value Pairs                    │
│   ┌─────────────────────────────────────┐   │
│   │ Key (string)                        │   │
│   │ Value Type (uint32)                 │   │
│   │ Value (varies)                      │   │
│   └─────────────────────────────────────┘   │
│   ... repeated for each KV pair             │
├─────────────────────────────────────────────┤
│ Tensor Infos                                │
│   ┌─────────────────────────────────────┐   │
│   │ Name (string)                       │   │
│   │ N Dimensions (uint32)               │   │
│   │ Dimensions (uint64[])               │   │
│   │ Type (uint32)                       │   │
│   │ Offset (uint64)                     │   │
│   └─────────────────────────────────────┘   │
│   ... repeated for each tensor              │
├─────────────────────────────────────────────┤
│ Padding (to alignment)                      │
├─────────────────────────────────────────────┤
│ Tensor Data (binary blob)                   │
│   Data for tensor 0                         │
│   Data for tensor 1                         │
│   ...                                       │
└─────────────────────────────────────────────┘
```

## 3. Data Types and Quantization Formats

### 3.1 GGML Types

```cpp
// include/vesper/io/gguf_types.h

namespace vesper::io::gguf {

enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 29,
};

// Block sizes for quantized types
constexpr int block_size(GGMLType type) {
    switch (type) {
        case GGMLType::Q4_0: return 32;
        case GGMLType::Q4_1: return 32;
        case GGMLType::Q5_0: return 32;
        case GGMLType::Q5_1: return 32;
        case GGMLType::Q8_0: return 32;
        case GGMLType::Q2_K: return 256;
        case GGMLType::Q3_K: return 256;
        case GGMLType::Q4_K: return 256;
        case GGMLType::Q5_K: return 256;
        case GGMLType::Q6_K: return 256;
        default: return 1;
    }
}

// Bytes per block
constexpr size_t type_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32: return 4;
        case GGMLType::F16: return 2;
        case GGMLType::Q4_0: return 2 + 16;  // scale (f16) + 32 nibbles
        case GGMLType::Q4_1: return 2 + 2 + 16;  // scale, min + data
        case GGMLType::Q8_0: return 2 + 32;  // scale + 32 int8s
        case GGMLType::Q4_K: return 144;  // K-quant block
        case GGMLType::Q6_K: return 210;
        // ... other types
        default: return 0;
    }
}

} // namespace vesper::io::gguf
```

### 3.2 Quantization Block Structures

```cpp
// Q4_0: 4-bit quantization with single scale
struct BlockQ4_0 {
    uint16_t d;      // Scale (FP16)
    uint8_t qs[16];  // 32 nibbles packed into 16 bytes
};

// Q4_K: K-quant with super-blocks
struct BlockQ4_K {
    uint16_t d;           // Super-block scale (FP16)
    uint16_t dmin;        // Super-block min (FP16)
    uint8_t scales[12];   // Scales for sub-blocks
    uint8_t qs[128];      // Quantized values
};

// Q6_K: 6-bit K-quant
struct BlockQ6_K {
    uint8_t ql[128];      // Lower 4 bits
    uint8_t qh[64];       // Upper 2 bits
    int8_t scales[16];    // Sub-block scales
    uint16_t d;           // Super-block scale
};
```

## 4. GGUF Reader Implementation

```cpp
// include/vesper/io/gguf_reader.h

namespace vesper::io::gguf {

struct TensorInfo {
    std::string name;
    GGMLType type;
    std::vector<int64_t> shape;
    size_t offset;
};

class GGUFReader {
public:
    explicit GGUFReader(const std::string& path);
    ~GGUFReader();
    
    // Metadata access
    uint32_t version() const { return version_; }
    
    template<typename T>
    T get_metadata(const std::string& key) const;
    
    std::vector<std::string> metadata_keys() const;
    
    // Model architecture (from metadata)
    std::string architecture() const;
    int64_t context_length() const;
    int64_t embedding_length() const;
    int64_t block_count() const;
    int64_t head_count() const;
    int64_t head_count_kv() const;
    
    // Tensor access
    std::vector<std::string> tensor_names() const;
    TensorInfo get_tensor_info(const std::string& name) const;
    
    // Load and dequantize a tensor
    Tensor load_tensor(const std::string& name, Device device = Device::CPU);
    
    // Load tensor as-is (quantized)
    Tensor load_tensor_raw(const std::string& name, Device device = Device::CPU);
    
private:
    std::string path_;
    int fd_;
    const uint8_t* mapped_data_;
    size_t file_size_;
    
    uint32_t version_;
    uint64_t n_tensors_;
    uint64_t n_kv_;
    
    std::unordered_map<std::string, std::pair<uint32_t, std::vector<uint8_t>>> metadata_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    size_t data_offset_;
    
    void parse_header();
    void parse_metadata();
    void parse_tensor_infos();
};

} // namespace vesper::io::gguf
```

```cpp
// src/io/gguf_reader.cpp

GGUFReader::GGUFReader(const std::string& path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Cannot open GGUF file: " + path);
    }
    
    file_size_ = lseek(fd_, 0, SEEK_END);
    lseek(fd_, 0, SEEK_SET);
    
    mapped_data_ = static_cast<const uint8_t*>(
        mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    
    parse_header();
    parse_metadata();
    parse_tensor_infos();
}

void GGUFReader::parse_header() {
    const uint8_t* ptr = mapped_data_;
    
    // Magic number
    uint32_t magic = *reinterpret_cast<const uint32_t*>(ptr);
    if (magic != 0x46554747) {  // "GGUF" little-endian
        throw std::runtime_error("Invalid GGUF magic number");
    }
    ptr += 4;
    
    // Version
    version_ = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += 4;
    
    if (version_ < 2 || version_ > 3) {
        throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version_));
    }
    
    // Counts
    n_tensors_ = *reinterpret_cast<const uint64_t*>(ptr);
    ptr += 8;
    
    n_kv_ = *reinterpret_cast<const uint64_t*>(ptr);
    ptr += 8;
}

void GGUFReader::parse_metadata() {
    const uint8_t* ptr = mapped_data_ + 24;  // After header
    
    for (uint64_t i = 0; i < n_kv_; ++i) {
        // Key (length-prefixed string)
        uint64_t key_len = *reinterpret_cast<const uint64_t*>(ptr);
        ptr += 8;
        std::string key(reinterpret_cast<const char*>(ptr), key_len);
        ptr += key_len;
        
        // Value type
        uint32_t value_type = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;
        
        // Value (varies by type)
        std::vector<uint8_t> value_data;
        size_t value_size = read_value(ptr, value_type, value_data);
        ptr += value_size;
        
        metadata_[key] = {value_type, value_data};
    }
}

void GGUFReader::parse_tensor_infos() {
    // Find where tensor info starts (after metadata)
    const uint8_t* ptr = /* ... after metadata ... */;
    
    for (uint64_t i = 0; i < n_tensors_; ++i) {
        TensorInfo info;
        
        // Name
        uint64_t name_len = *reinterpret_cast<const uint64_t*>(ptr);
        ptr += 8;
        info.name = std::string(reinterpret_cast<const char*>(ptr), name_len);
        ptr += name_len;
        
        // Dimensions
        uint32_t n_dims = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;
        
        info.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            info.shape[d] = *reinterpret_cast<const uint64_t*>(ptr);
            ptr += 8;
        }
        
        // Type
        info.type = static_cast<GGMLType>(*reinterpret_cast<const uint32_t*>(ptr));
        ptr += 4;
        
        // Offset into data section
        info.offset = *reinterpret_cast<const uint64_t*>(ptr);
        ptr += 8;
        
        tensors_[info.name] = info;
    }
    
    // Calculate data section start (with alignment)
    size_t header_end = ptr - mapped_data_;
    data_offset_ = (header_end + 31) & ~31;  // Align to 32 bytes
}

Tensor GGUFReader::load_tensor(const std::string& name, Device device) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    const TensorInfo& info = it->second;
    const uint8_t* data_ptr = mapped_data_ + data_offset_ + info.offset;
    
    // Dequantize if needed
    if (info.type == GGMLType::F32) {
        Tensor t = empty(info.shape, DType::Float32, Device::CPU);
        std::memcpy(t.data_ptr(), data_ptr, t.numel() * 4);
        return device != Device::CPU ? t.to(device) : t;
    }
    else if (info.type == GGMLType::F16) {
        Tensor t = empty(info.shape, DType::Float16, Device::CPU);
        std::memcpy(t.data_ptr(), data_ptr, t.numel() * 2);
        return device != Device::CPU ? t.to(device) : t;
    }
    else {
        // Quantized - dequantize to float
        return dequantize_tensor(info, data_ptr, device);
    }
}
```

## 5. Dequantization Kernels

### 5.1 Q4_0 Dequantization

```cpp
// src/io/gguf_dequant.cpp

Tensor dequantize_q4_0(const uint8_t* data, const std::vector<int64_t>& shape) {
    int64_t n_elements = 1;
    for (auto s : shape) n_elements *= s;
    
    int n_blocks = n_elements / 32;  // 32 elements per block
    
    Tensor output = empty(shape, DType::Float32, Device::CPU);
    float* out_ptr = output.data_ptr<float>();
    
    const BlockQ4_0* blocks = reinterpret_cast<const BlockQ4_0*>(data);
    
    for (int i = 0; i < n_blocks; ++i) {
        float d = fp16_to_fp32(blocks[i].d);
        
        for (int j = 0; j < 16; ++j) {
            uint8_t byte = blocks[i].qs[j];
            
            // Low nibble
            int8_t v0 = (byte & 0x0F) - 8;
            out_ptr[i * 32 + j * 2] = v0 * d;
            
            // High nibble
            int8_t v1 = (byte >> 4) - 8;
            out_ptr[i * 32 + j * 2 + 1] = v1 * d;
        }
    }
    
    return output;
}
```

### 5.2 Q4_K Dequantization (K-quant)

```cpp
Tensor dequantize_q4_k(const uint8_t* data, const std::vector<int64_t>& shape) {
    int64_t n_elements = 1;
    for (auto s : shape) n_elements *= s;
    
    int n_blocks = n_elements / 256;  // 256 elements per super-block
    
    Tensor output = empty(shape, DType::Float32, Device::CPU);
    float* out_ptr = output.data_ptr<float>();
    
    const BlockQ4_K* blocks = reinterpret_cast<const BlockQ4_K*>(data);
    
    for (int i = 0; i < n_blocks; ++i) {
        float d = fp16_to_fp32(blocks[i].d);
        float dmin = fp16_to_fp32(blocks[i].dmin);
        
        // Decode sub-block scales
        uint8_t scales_l[8], scales_h[8];
        for (int j = 0; j < 8; ++j) {
            scales_l[j] = blocks[i].scales[j] & 0x3F;
            scales_h[j] = (blocks[i].scales[j + 4] >> 4) & 0x03;
        }
        
        // Dequantize 256 values
        for (int j = 0; j < 256; ++j) {
            int sub_block = j / 32;
            int scale_idx = sub_block / 2;
            
            uint8_t scale = scales_l[scale_idx] | (scales_h[scale_idx] << 6);
            float sc = d * scale;
            float mn = dmin * (scales_l[scale_idx + 4] | (scales_h[scale_idx + 4] << 6));
            
            // Get 4-bit value
            int byte_idx = j / 2;
            uint8_t byte = blocks[i].qs[byte_idx];
            int8_t q = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            
            out_ptr[i * 256 + j] = q * sc - mn;
        }
    }
    
    return output;
}
```

### 5.3 GPU Dequantization

```cpp
// src/ops/hip/gguf_dequant.hip

__global__ void dequant_q4_0_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int n_blocks)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= n_blocks) return;
    
    // Read block
    const BlockQ4_0* block = reinterpret_cast<const BlockQ4_0*>(
        input + block_idx * sizeof(BlockQ4_0));
    
    float d = __half2float(*reinterpret_cast<const __half*>(&block->d));
    
    float* out = output + block_idx * 32;
    
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        uint8_t byte = block->qs[j];
        out[j * 2] = ((byte & 0x0F) - 8) * d;
        out[j * 2 + 1] = ((byte >> 4) - 8) * d;
    }
}

Tensor dequantize_q4_0_gpu(const Tensor& quantized, const std::vector<int64_t>& shape) {
    int64_t n_elements = 1;
    for (auto s : shape) n_elements *= s;
    int n_blocks = n_elements / 32;
    
    Tensor output = empty(shape, DType::Float32, quantized.device());
    
    int threads = 256;
    int blocks = (n_blocks + threads - 1) / threads;
    
    hipLaunchKernelGGL(
        dequant_q4_0_kernel,
        dim3(blocks), dim3(threads), 0, 0,
        quantized.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        n_blocks);
    
    return output;
}
```

## 6. Model Loading from GGUF

```cpp
// include/vesper/io/gguf_loader.h

namespace vesper::io::gguf {

struct GGUFLoadConfig {
    std::string path;
    Device device = Device::HIP;
    DType compute_dtype = DType::Float16;  // Compute precision
    bool use_mmap = true;
    bool dequantize_on_load = true;  // vs on-the-fly dequant
};

class GGUFLoader {
public:
    // Load model from GGUF
    static std::unique_ptr<models::Transformer> load(const GGUFLoadConfig& config);
    
    // Extract tokenizer from GGUF
    static std::unique_ptr<Tokenizer> load_tokenizer(const std::string& path);
    
    // Get model config from GGUF metadata
    static models::TransformerConfig get_config(const std::string& path);
};

} // namespace vesper::io::gguf
```

```cpp
// src/io/gguf_loader.cpp

models::TransformerConfig GGUFLoader::get_config(const std::string& path) {
    GGUFReader reader(path);
    
    models::TransformerConfig config;
    
    // Map GGUF metadata to TransformerConfig
    config.dim = reader.embedding_length();
    config.n_layers = reader.block_count();
    config.n_heads = reader.head_count();
    config.n_kv_heads = reader.head_count_kv();
    config.vocab_size = reader.get_metadata<int64_t>("tokenizer.ggml.tokens.size");
    config.max_seq_len = reader.context_length();
    
    // FFN dimension
    if (reader.has_metadata("llama.feed_forward_length")) {
        config.ffn_hidden_dim = reader.get_metadata<int64_t>("llama.feed_forward_length");
    } else {
        config.ffn_hidden_dim = (config.dim * 4 * 2) / 3;  // SwiGLU default
        config.ffn_hidden_dim = ((config.ffn_hidden_dim + 255) / 256) * 256;  // Align
    }
    
    // RoPE parameters
    if (reader.has_metadata("llama.rope.freq_base")) {
        config.rope_base = reader.get_metadata<float>("llama.rope.freq_base");
    }
    
    config.norm_eps = reader.get_metadata<float>("llama.attention.layer_norm_rms_epsilon");
    
    return config;
}

std::unique_ptr<models::Transformer> GGUFLoader::load(const GGUFLoadConfig& config) {
    GGUFReader reader(config.path);
    
    // Create model
    auto model_config = get_config(config.path);
    auto model = std::make_unique<models::Transformer>(model_config);
    
    // Weight name mapping: GGUF -> Vesper
    std::unordered_map<std::string, std::string> weight_map = {
        {"token_embd.weight", "tok_embeddings.weight"},
        {"output_norm.weight", "norm.weight"},
        {"output.weight", "output.weight"},
    };
    
    // Layer weights
    for (int i = 0; i < model_config.n_layers; ++i) {
        std::string prefix = "blk." + std::to_string(i);
        std::string vesper_prefix = "layers." + std::to_string(i);
        
        weight_map[prefix + ".attn_q.weight"] = vesper_prefix + ".attention.wq.weight";
        weight_map[prefix + ".attn_k.weight"] = vesper_prefix + ".attention.wk.weight";
        weight_map[prefix + ".attn_v.weight"] = vesper_prefix + ".attention.wv.weight";
        weight_map[prefix + ".attn_output.weight"] = vesper_prefix + ".attention.wo.weight";
        weight_map[prefix + ".ffn_gate.weight"] = vesper_prefix + ".feed_forward.w1.weight";
        weight_map[prefix + ".ffn_down.weight"] = vesper_prefix + ".feed_forward.w2.weight";
        weight_map[prefix + ".ffn_up.weight"] = vesper_prefix + ".feed_forward.w3.weight";
        weight_map[prefix + ".attn_norm.weight"] = vesper_prefix + ".attention_norm.weight";
        weight_map[prefix + ".ffn_norm.weight"] = vesper_prefix + ".ffn_norm.weight";
    }
    
    // Load weights
    for (auto& [gguf_name, vesper_name] : weight_map) {
        if (!reader.has_tensor(gguf_name)) {
            std::cerr << "Warning: tensor not found: " << gguf_name << std::endl;
            continue;
        }
        
        // Load and dequantize
        Tensor weight = reader.load_tensor(gguf_name, Device::CPU);
        
        // Convert to compute dtype
        if (weight.dtype() != config.compute_dtype) {
            weight = weight.to(config.compute_dtype);
        }
        
        // Move to device
        weight = weight.to(config.device);
        
        // Copy to model parameter
        auto& param = model->get_parameter(vesper_name);
        param.copy_(weight);
    }
    
    return model;
}
```

## 7. Tokenizer from GGUF

```cpp
std::unique_ptr<Tokenizer> GGUFLoader::load_tokenizer(const std::string& path) {
    GGUFReader reader(path);
    
    // Extract tokenizer data from metadata
    auto tokens = reader.get_metadata<std::vector<std::string>>("tokenizer.ggml.tokens");
    auto scores = reader.get_metadata<std::vector<float>>("tokenizer.ggml.scores");
    auto token_types = reader.get_metadata<std::vector<int32_t>>("tokenizer.ggml.token_type");
    
    std::string model_type = reader.get_metadata<std::string>("tokenizer.ggml.model");
    
    // Get special tokens
    int64_t bos_id = reader.get_metadata<int64_t>("tokenizer.ggml.bos_token_id");
    int64_t eos_id = reader.get_metadata<int64_t>("tokenizer.ggml.eos_token_id");
    int64_t unk_id = reader.get_metadata<int64_t>("tokenizer.ggml.unknown_token_id");
    
    // Build tokenizer
    if (model_type == "llama" || model_type == "gpt2") {
        return std::make_unique<BPETokenizer>(tokens, scores, bos_id, eos_id, unk_id);
    } else {
        throw std::runtime_error("Unsupported tokenizer model: " + model_type);
    }
}
```

## 8. On-the-fly Dequantization (Memory Efficient)

For very large models, avoid dequantizing all weights at once:

```cpp
class QuantizedLinear : public nn::Module {
public:
    QuantizedLinear(
        Tensor quantized_weight,  // Stays quantized
        GGMLType quant_type,
        std::vector<int64_t> shape)
        : quantized_weight_(std::move(quantized_weight)),
          quant_type_(quant_type),
          shape_(std::move(shape)) {}
    
    Tensor forward(const Tensor& x) override {
        // Dequantize on the fly
        Tensor weight = dequantize(quantized_weight_, quant_type_, shape_);
        
        // Matmul
        Tensor output = matmul(x, weight.t());
        
        // Weight can be freed after use
        return output;
    }
    
private:
    Tensor quantized_weight_;
    GGMLType quant_type_;
    std::vector<int64_t> shape_;
};
```

## 9. Testing Strategy

### 9.1 Unit Tests

```cpp
// tests/io/test_gguf.cpp

TEST(GGUF, ParseHeader) {
    GGUFReader reader("test_models/tiny.gguf");
    
    EXPECT_GE(reader.version(), 2);
    EXPECT_GT(reader.tensor_names().size(), 0);
}

TEST(GGUF, Metadata) {
    GGUFReader reader("test_models/llama-7b.Q4_K_M.gguf");
    
    std::string arch = reader.architecture();
    EXPECT_EQ(arch, "llama");
    
    int64_t n_layers = reader.block_count();
    EXPECT_EQ(n_layers, 32);
    
    int64_t n_heads = reader.head_count();
    EXPECT_EQ(n_heads, 32);
}

TEST(GGUF, DequantQ4_0) {
    // Create known quantized data
    std::vector<BlockQ4_0> blocks(1);
    blocks[0].d = fp32_to_fp16(0.5f);
    for (int i = 0; i < 16; ++i) {
        blocks[0].qs[i] = 0x88;  // Both nibbles = 0
    }
    
    Tensor result = dequantize_q4_0(
        reinterpret_cast<uint8_t*>(blocks.data()),
        {32});
    
    // All values should be 0
    EXPECT_TRUE(allclose(result, zeros({32})));
}

TEST(GGUF, DequantQ4_K) {
    // Load a known Q4_K tensor
    GGUFReader reader("test_models/tiny.Q4_K.gguf");
    
    Tensor q_tensor = reader.load_tensor("token_embd.weight");
    
    // Check shape matches
    auto info = reader.get_tensor_info("token_embd.weight");
    EXPECT_EQ(q_tensor.shape(), info.shape);
    
    // Check values are reasonable (not NaN or Inf)
    EXPECT_FALSE(q_tensor.isnan().any().item<bool>());
    EXPECT_FALSE(q_tensor.isinf().any().item<bool>());
}

TEST(GGUF, LoadModel) {
    GGUFLoadConfig config;
    config.path = "test_models/tiny-llama.gguf";
    config.device = Device::CPU;
    
    auto model = GGUFLoader::load(config);
    ASSERT_NE(model, nullptr);
    
    // Run forward pass
    Tensor input = tensor({{1, 2, 3, 4}}, DType::Int64);
    Tensor output = model->forward(input);
    
    EXPECT_EQ(output.dim(), 3);
}

TEST(GGUF, LoadTokenizer) {
    auto tokenizer = GGUFLoader::load_tokenizer("test_models/llama-7b.Q4_K_M.gguf");
    
    auto tokens = tokenizer->encode("Hello, world!");
    EXPECT_GT(tokens.size(), 0);
    
    std::string decoded = tokenizer->decode(tokens);
    EXPECT_EQ(decoded, "Hello, world!");
}
```

### 9.2 Accuracy Tests

```cpp
TEST(GGUF, DequantAccuracy) {
    // Compare against reference implementation (llama.cpp)
    GGUFReader reader("test_models/llama-7b.Q4_K_M.gguf");
    
    // Load one tensor
    Tensor our_result = reader.load_tensor("blk.0.attn_q.weight");
    
    // Load reference (from llama.cpp or Python gguf library)
    Tensor reference = load_reference_tensor("blk.0.attn_q.weight");
    
    // Should match exactly
    EXPECT_TRUE(allclose(our_result, reference, /*rtol=*/1e-5, /*atol=*/1e-5));
}

TEST(GGUF, ModelOutputAccuracy) {
    // Compare model output to reference
    GGUFLoadConfig config;
    config.path = "test_models/llama-7b.Q4_K_M.gguf";
    
    auto model = GGUFLoader::load(config);
    
    // Known input
    Tensor input = tensor({{1, 2, 3, 4, 5}}, DType::Int64);
    
    // Our output
    Tensor our_output = model->forward(input);
    
    // Reference output (from llama.cpp)
    Tensor ref_output = load_reference_output("test_case_1");
    
    // Check logits match
    float max_diff = (our_output - ref_output).abs().max().item<float>();
    std::cout << "Max diff: " << max_diff << std::endl;
    
    EXPECT_LT(max_diff, 0.1f);  // Allow small numerical differences
}
```

### 9.3 Stress Tests

```cpp
TEST(GGUF, StressTest_LargeModel) {
    // Load a large quantized model
    GGUFLoadConfig config;
    config.path = "test_models/llama-70b.Q4_K_M.gguf";
    config.device = Device::HIP;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto model = GGUFLoader::load(config);
    auto end = std::chrono::high_resolution_clock::now();
    
    double load_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Load time: " << load_time << "s" << std::endl;
    
    // Check memory usage
    size_t gpu_mem = get_gpu_memory_used();
    std::cout << "GPU memory: " << (gpu_mem / 1e9) << " GB" << std::endl;
    
    // Q4_K 70B should fit in ~40GB
    EXPECT_LT(gpu_mem, 45'000'000'000LL);
}

TEST(GGUF, StressTest_DequantSpeed) {
    GGUFReader reader("test_models/llama-7b.Q4_K_M.gguf");
    
    // Benchmark dequantization speed
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& name : reader.tensor_names()) {
        Tensor t = reader.load_tensor(name, Device::CPU);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "Total dequant time: " << ms << " ms" << std::endl;
    
    // Should be fast (< 10 seconds for 7B model)
    EXPECT_LT(ms, 10000);
}

TEST(GGUF, StressTest_AllQuantTypes) {
    std::vector<std::string> test_files = {
        "test_models/tiny.Q4_0.gguf",
        "test_models/tiny.Q4_1.gguf",
        "test_models/tiny.Q5_0.gguf",
        "test_models/tiny.Q8_0.gguf",
        "test_models/tiny.Q4_K_M.gguf",
        "test_models/tiny.Q5_K_M.gguf",
        "test_models/tiny.Q6_K.gguf",
    };
    
    for (const auto& file : test_files) {
        std::cout << "Testing: " << file << std::endl;
        
        GGUFLoadConfig config;
        config.path = file;
        
        EXPECT_NO_THROW({
            auto model = GGUFLoader::load(config);
            Tensor input = tensor({{1, 2, 3}}, DType::Int64);
            model->forward(input);
        });
    }
}
```

## 10. Summary

This chapter covered:

1. **GGUF format**: Header, metadata, tensor structure
2. **Quantization types**: Q4_0, Q4_K, Q6_K, and more
3. **Dequantization**: CPU and GPU kernels
4. **Model loading**: Weight mapping from GGUF to Vesper
5. **Tokenizer extraction**: Using embedded tokenizer data

Key benefits of GGUF support:
- **Ecosystem compatibility**: Use models from llama.cpp, Ollama, etc.
- **Efficient storage**: 4-bit models are ~4GB for 7B params
- **Single-file distribution**: Model + tokenizer in one file

With GGUF support, Vesper can load and run the vast library of community models.
```
