```markdown
# Chapter 33.8: Loading Pre-trained Weights

## 1. Introduction

Training large language models from scratch requires enormous compute resources. Instead, we typically load **pre-trained weights** and either use them directly for inference or fine-tune them on specific tasks.

This chapter covers:
1. **Safetensors**: The modern, safe, and fast format for storing model weights
2. **PyTorch .pt/.bin**: Legacy format compatibility
3. **Weight mapping**: Translating between different naming conventions
4. **Sharded weights**: Handling models split across multiple files

## 2. File Format Overview

### 2.1 Safetensors

Safetensors is the preferred format for modern models (HuggingFace, Llama, Mistral):

**Advantages**:
- **Safe**: No arbitrary code execution (unlike pickle-based formats)
- **Fast**: Memory-mapped loading, zero-copy when possible
- **Simple**: JSON header + raw tensor data

**File Structure**:
```
[8 bytes: header_size (little endian u64)]
[header_size bytes: JSON header]
[remaining bytes: raw tensor data]
```

**JSON Header Example**:
```json
{
  "__metadata__": {"format": "pt"},
  "model.embed_tokens.weight": {
    "dtype": "F32",
    "shape": [32000, 4096],
    "data_offsets": [0, 524288000]
  },
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "F16",
    "shape": [4096, 4096],
    "data_offsets": [524288000, 557842432]
  }
}
```

### 2.2 PyTorch Pickle (.pt, .bin)

Legacy format using Python pickle. **Risks**: Arbitrary code execution.

We support this for compatibility but recommend safetensors.

## 3. Implementation Plan

### 3.1 Safetensors Reader

```cpp
// include/vesper/io/safetensors.h

namespace vesper::io {

struct TensorInfo {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    size_t data_offset;
    size_t data_size;
};

class SafetensorsReader {
public:
    explicit SafetensorsReader(const std::string& path);
    ~SafetensorsReader();
    
    // Get list of all tensors
    std::vector<std::string> tensor_names() const;
    
    // Get info for a specific tensor
    TensorInfo get_info(const std::string& name) const;
    
    // Load a tensor to device
    Tensor load_tensor(const std::string& name, Device device = Device::CPU);
    
    // Load all tensors as a map
    std::unordered_map<std::string, Tensor> load_all(Device device = Device::CPU);
    
    // Get metadata
    std::unordered_map<std::string, std::string> metadata() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Helper: load sharded safetensors (multiple files)
std::unordered_map<std::string, Tensor> load_sharded_safetensors(
    const std::string& model_dir,
    Device device = Device::CPU);

} // namespace vesper::io
```

### 3.2 Safetensors Implementation

```cpp
// src/io/safetensors.cpp

#include <vesper/io/safetensors.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace vesper::io {

using json = nlohmann::json;

// DType string mapping
DType parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "F32" || dtype_str == "float32") return DType::Float32;
    if (dtype_str == "F16" || dtype_str == "float16") return DType::Float16;
    if (dtype_str == "BF16" || dtype_str == "bfloat16") return DType::BFloat16;
    if (dtype_str == "I64" || dtype_str == "int64") return DType::Int64;
    if (dtype_str == "I32" || dtype_str == "int32") return DType::Int32;
    if (dtype_str == "I16" || dtype_str == "int16") return DType::Int16;
    if (dtype_str == "I8" || dtype_str == "int8") return DType::Int8;
    if (dtype_str == "U8" || dtype_str == "uint8") return DType::UInt8;
    if (dtype_str == "BOOL" || dtype_str == "bool") return DType::Bool;
    throw std::runtime_error("Unknown dtype: " + dtype_str);
}

class SafetensorsReader::Impl {
public:
    Impl(const std::string& path) : path_(path) {
        // Open file
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        
        // Get file size
        file_size_ = lseek(fd_, 0, SEEK_END);
        lseek(fd_, 0, SEEK_SET);
        
        // Memory map the file
        mapped_data_ = static_cast<const uint8_t*>(
            mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mapped_data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Failed to mmap file: " + path);
        }
        
        // Read header size (first 8 bytes, little endian)
        header_size_ = *reinterpret_cast<const uint64_t*>(mapped_data_);
        
        // Parse JSON header
        std::string header_str(
            reinterpret_cast<const char*>(mapped_data_ + 8),
            header_size_);
        header_ = json::parse(header_str);
        
        // Data starts after header
        data_start_ = 8 + header_size_;
        
        // Parse tensor info
        for (auto& [name, info] : header_.items()) {
            if (name == "__metadata__") {
                for (auto& [k, v] : info.items()) {
                    metadata_[k] = v.get<std::string>();
                }
                continue;
            }
            
            TensorInfo ti;
            ti.name = name;
            ti.dtype = parse_dtype(info["dtype"].get<std::string>());
            ti.shape = info["shape"].get<std::vector<int64_t>>();
            
            auto offsets = info["data_offsets"].get<std::vector<size_t>>();
            ti.data_offset = offsets[0];
            ti.data_size = offsets[1] - offsets[0];
            
            tensors_[name] = ti;
        }
    }
    
    ~Impl() {
        if (mapped_data_ && mapped_data_ != MAP_FAILED) {
            munmap(const_cast<uint8_t*>(mapped_data_), file_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }
    
    std::vector<std::string> tensor_names() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : tensors_) {
            names.push_back(name);
        }
        return names;
    }
    
    TensorInfo get_info(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return it->second;
    }
    
    Tensor load_tensor(const std::string& name, Device device) {
        TensorInfo info = get_info(name);
        
        // Create tensor
        Tensor tensor = empty(info.shape, info.dtype, Device::CPU);
        
        // Copy data
        const uint8_t* src = mapped_data_ + data_start_ + info.data_offset;
        std::memcpy(tensor.data_ptr(), src, info.data_size);
        
        // Move to target device if needed
        if (device != Device::CPU) {
            tensor = tensor.to(device);
        }
        
        return tensor;
    }
    
    std::unordered_map<std::string, Tensor> load_all(Device device) {
        std::unordered_map<std::string, Tensor> result;
        for (const auto& [name, _] : tensors_) {
            result[name] = load_tensor(name, device);
        }
        return result;
    }
    
    const std::unordered_map<std::string, std::string>& metadata() const {
        return metadata_;
    }
    
private:
    std::string path_;
    int fd_ = -1;
    const uint8_t* mapped_data_ = nullptr;
    size_t file_size_ = 0;
    size_t header_size_ = 0;
    size_t data_start_ = 0;
    json header_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    std::unordered_map<std::string, std::string> metadata_;
};

// Public interface
SafetensorsReader::SafetensorsReader(const std::string& path)
    : impl_(std::make_unique<Impl>(path)) {}

SafetensorsReader::~SafetensorsReader() = default;

std::vector<std::string> SafetensorsReader::tensor_names() const {
    return impl_->tensor_names();
}

TensorInfo SafetensorsReader::get_info(const std::string& name) const {
    return impl_->get_info(name);
}

Tensor SafetensorsReader::load_tensor(const std::string& name, Device device) {
    return impl_->load_tensor(name, device);
}

std::unordered_map<std::string, Tensor> SafetensorsReader::load_all(Device device) {
    return impl_->load_all(device);
}

std::unordered_map<std::string, std::string> SafetensorsReader::metadata() const {
    return impl_->metadata();
}

} // namespace vesper::io
```

### 3.3 Sharded Model Loading

Large models are split across multiple files:

```cpp
// src/io/sharded_loader.cpp

std::unordered_map<std::string, Tensor> load_sharded_safetensors(
    const std::string& model_dir, Device device) 
{
    namespace fs = std::filesystem;
    
    std::unordered_map<std::string, Tensor> all_tensors;
    
    // Find all .safetensors files
    std::vector<std::string> shard_files;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".safetensors") {
            shard_files.push_back(entry.path().string());
        }
    }
    
    // Sort for deterministic order
    std::sort(shard_files.begin(), shard_files.end());
    
    // Load each shard
    for (const auto& shard_path : shard_files) {
        SafetensorsReader reader(shard_path);
        auto tensors = reader.load_all(device);
        
        for (auto& [name, tensor] : tensors) {
            if (all_tensors.count(name)) {
                throw std::runtime_error("Duplicate tensor in shards: " + name);
            }
            all_tensors[name] = std::move(tensor);
        }
    }
    
    return all_tensors;
}
```

## 4. Weight Mapping

### 4.1 The Mapping Problem

Different frameworks use different naming conventions:

| Vesper Name | HuggingFace Llama | Meta Llama |
|-------------|-------------------|------------|
| `layers.0.attention.wq.weight` | `model.layers.0.self_attn.q_proj.weight` | `layers.0.attention.wq.weight` |
| `tok_embeddings.weight` | `model.embed_tokens.weight` | `tok_embeddings.weight` |
| `norm.weight` | `model.norm.weight` | `norm.weight` |
| `output.weight` | `lm_head.weight` | `output.weight` |

### 4.2 Weight Mapper

```cpp
// include/vesper/io/weight_mapper.h

namespace vesper::io {

class WeightMapper {
public:
    // Add a mapping rule
    void add_rule(const std::string& vesper_pattern, 
                  const std::string& source_pattern);
    
    // Map a Vesper parameter name to source tensor name
    std::string map_to_source(const std::string& vesper_name) const;
    
    // Map a source tensor name to Vesper parameter name
    std::string map_from_source(const std::string& source_name) const;
    
    // Predefined mappers
    static WeightMapper huggingface_llama();
    static WeightMapper meta_llama();
    static WeightMapper huggingface_gpt2();
    static WeightMapper huggingface_mistral();
    
private:
    std::vector<std::pair<std::regex, std::string>> to_source_rules_;
    std::vector<std::pair<std::regex, std::string>> from_source_rules_;
};

} // namespace vesper::io
```

### 4.3 Mapper Implementation

```cpp
// src/io/weight_mapper.cpp

namespace vesper::io {

WeightMapper WeightMapper::huggingface_llama() {
    WeightMapper mapper;
    
    // Embeddings
    mapper.add_rule("tok_embeddings.weight", "model.embed_tokens.weight");
    
    // Layers
    mapper.add_rule(
        R"(layers\.(\d+)\.attention\.wq\.weight)",
        R"(model.layers.$1.self_attn.q_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.attention\.wk\.weight)",
        R"(model.layers.$1.self_attn.k_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.attention\.wv\.weight)",
        R"(model.layers.$1.self_attn.v_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.attention\.wo\.weight)",
        R"(model.layers.$1.self_attn.o_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.feed_forward\.w1\.weight)",
        R"(model.layers.$1.mlp.gate_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.feed_forward\.w2\.weight)",
        R"(model.layers.$1.mlp.down_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.feed_forward\.w3\.weight)",
        R"(model.layers.$1.mlp.up_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.attention_norm\.weight)",
        R"(model.layers.$1.input_layernorm.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.ffn_norm\.weight)",
        R"(model.layers.$1.post_attention_layernorm.weight)");
    
    // Final layers
    mapper.add_rule("norm.weight", "model.norm.weight");
    mapper.add_rule("output.weight", "lm_head.weight");
    
    return mapper;
}

std::string WeightMapper::map_from_source(const std::string& source_name) const {
    for (const auto& [pattern, replacement] : from_source_rules_) {
        std::smatch match;
        if (std::regex_match(source_name, match, pattern)) {
            std::string result = replacement;
            for (size_t i = 1; i < match.size(); ++i) {
                std::string placeholder = "$" + std::to_string(i);
                size_t pos = result.find(placeholder);
                if (pos != std::string::npos) {
                    result.replace(pos, placeholder.length(), match[i].str());
                }
            }
            return result;
        }
    }
    return source_name;  // No mapping, use as-is
}

} // namespace vesper::io
```

## 5. Model Loading Interface

### 5.1 High-Level Loader

```cpp
// include/vesper/io/model_loader.h

namespace vesper::io {

struct LoadConfig {
    std::string model_path;           // Directory or single file
    std::string format = "auto";      // "safetensors", "pytorch", "auto"
    std::string source_format = "";   // "huggingface", "meta", "auto"
    Device device = Device::CPU;
    DType dtype = DType::Float32;     // Target dtype (for conversion)
    bool strict = true;               // Fail on missing/extra weights
};

class ModelLoader {
public:
    // Load weights into a Transformer model
    static void load(models::Transformer& model, const LoadConfig& config);
    
    // Load raw tensors as a map
    static std::unordered_map<std::string, Tensor> load_tensors(
        const LoadConfig& config);
    
    // Get model config from a HuggingFace model directory
    static models::TransformerConfig load_config(const std::string& model_path);
};

} // namespace vesper::io
```

### 5.2 Loader Implementation

```cpp
// src/io/model_loader.cpp

namespace vesper::io {

void ModelLoader::load(models::Transformer& model, const LoadConfig& config) {
    // 1. Detect format
    std::string format = config.format;
    if (format == "auto") {
        if (std::filesystem::is_directory(config.model_path)) {
            // Check for .safetensors files
            for (const auto& f : std::filesystem::directory_iterator(config.model_path)) {
                if (f.path().extension() == ".safetensors") {
                    format = "safetensors";
                    break;
                }
            }
            if (format == "auto") format = "pytorch";
        } else {
            if (config.model_path.ends_with(".safetensors")) {
                format = "safetensors";
            } else {
                format = "pytorch";
            }
        }
    }
    
    // 2. Load tensors
    std::unordered_map<std::string, Tensor> tensors;
    if (format == "safetensors") {
        if (std::filesystem::is_directory(config.model_path)) {
            tensors = load_sharded_safetensors(config.model_path, config.device);
        } else {
            SafetensorsReader reader(config.model_path);
            tensors = reader.load_all(config.device);
        }
    } else {
        throw std::runtime_error("PyTorch format not yet supported");
    }
    
    // 3. Detect source format and create mapper
    WeightMapper mapper;
    std::string source = config.source_format;
    if (source == "" || source == "auto") {
        // Try to detect from tensor names
        if (tensors.count("model.embed_tokens.weight")) {
            source = "huggingface";
        } else if (tensors.count("tok_embeddings.weight")) {
            source = "meta";
        } else {
            throw std::runtime_error("Cannot detect weight format");
        }
    }
    
    if (source == "huggingface") {
        mapper = WeightMapper::huggingface_llama();
    } else if (source == "meta") {
        mapper = WeightMapper::meta_llama();
    }
    
    // 4. Load weights into model
    std::unordered_set<std::string> loaded;
    std::vector<std::string> missing;
    std::vector<std::string> unexpected;
    
    for (auto& [param_name, param] : model.named_parameters()) {
        std::string source_name = mapper.map_to_source(param_name);
        
        auto it = tensors.find(source_name);
        if (it == tensors.end()) {
            missing.push_back(param_name);
            continue;
        }
        
        Tensor& source_tensor = it->second;
        
        // Shape check
        if (source_tensor.shape() != param.shape()) {
            throw std::runtime_error(
                "Shape mismatch for " + param_name + ": expected " +
                shape_to_string(param.shape()) + ", got " +
                shape_to_string(source_tensor.shape()));
        }
        
        // Type conversion if needed
        if (source_tensor.dtype() != config.dtype) {
            source_tensor = source_tensor.to(config.dtype);
        }
        
        // Device transfer if needed
        if (source_tensor.device() != param.device()) {
            source_tensor = source_tensor.to(param.device());
        }
        
        // Copy weights
        param.copy_(source_tensor);
        loaded.insert(source_name);
    }
    
    // Check for unexpected weights
    for (const auto& [name, _] : tensors) {
        if (!loaded.count(name)) {
            unexpected.push_back(name);
        }
    }
    
    // Report
    if (!missing.empty()) {
        std::string msg = "Missing weights: ";
        for (const auto& n : missing) msg += n + ", ";
        if (config.strict) {
            throw std::runtime_error(msg);
        } else {
            std::cerr << "Warning: " << msg << std::endl;
        }
    }
    
    if (!unexpected.empty() && config.strict) {
        std::string msg = "Unexpected weights: ";
        for (const auto& n : unexpected) msg += n + ", ";
        throw std::runtime_error(msg);
    }
    
    std::cout << "Loaded " << loaded.size() << " tensors" << std::endl;
}

models::TransformerConfig ModelLoader::load_config(const std::string& model_path) {
    // Load config.json from HuggingFace model
    std::string config_path = model_path + "/config.json";
    std::ifstream f(config_path);
    if (!f) {
        throw std::runtime_error("Cannot find config.json in " + model_path);
    }
    
    json config = json::parse(f);
    
    models::TransformerConfig tc;
    tc.vocab_size = config.value("vocab_size", 32000);
    tc.dim = config.value("hidden_size", 4096);
    tc.n_layers = config.value("num_hidden_layers", 32);
    tc.n_heads = config.value("num_attention_heads", 32);
    tc.n_kv_heads = config.value("num_key_value_heads", tc.n_heads);
    tc.max_seq_len = config.value("max_position_embeddings", 4096);
    tc.ffn_hidden_dim = config.value("intermediate_size", 11008);
    tc.norm_eps = config.value("rms_norm_eps", 1e-5f);
    tc.rope_base = config.value("rope_theta", 10000.0f);
    tc.use_rms_norm = true;  // All Llama models use RMSNorm
    tc.use_bias = false;
    
    return tc;
}

} // namespace vesper::io
```

## 6. Usage Examples

### 6.1 Loading a HuggingFace Model

```cpp
#include <vesper/io/model_loader.h>
#include <vesper/models/transformer.h>

int main() {
    // Load config
    auto config = io::ModelLoader::load_config("models/llama-2-7b-hf");
    
    // Create model
    auto model = std::make_unique<models::Transformer>(config);
    model->to(Device::HIP);
    
    // Load weights
    io::LoadConfig load_config;
    load_config.model_path = "models/llama-2-7b-hf";
    load_config.device = Device::HIP;
    load_config.dtype = DType::Float16;
    
    io::ModelLoader::load(*model, load_config);
    
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "Parameters: " << model->num_parameters() << std::endl;
    
    return 0;
}
```

### 6.2 Loading Sharded Model

```cpp
// For very large models split across multiple files
io::LoadConfig config;
config.model_path = "models/llama-2-70b-hf";  // Contains multiple .safetensors
config.device = Device::HIP;
config.dtype = DType::BFloat16;

io::ModelLoader::load(*model, config);
```

### 6.3 Inspecting Weights Before Loading

```cpp
io::SafetensorsReader reader("model.safetensors");

std::cout << "Tensors in file:" << std::endl;
for (const auto& name : reader.tensor_names()) {
    auto info = reader.get_info(name);
    std::cout << "  " << name << ": " 
              << dtype_to_string(info.dtype) << " "
              << shape_to_string(info.shape) << std::endl;
}
```

## 7. Safetensors Writer

For saving fine-tuned models:

```cpp
// include/vesper/io/safetensors_writer.h

namespace vesper::io {

class SafetensorsWriter {
public:
    explicit SafetensorsWriter(const std::string& path);
    
    // Add tensor to be written
    void add_tensor(const std::string& name, const Tensor& tensor);
    
    // Set metadata
    void set_metadata(const std::string& key, const std::string& value);
    
    // Write to file
    void write();
    
private:
    std::string path_;
    std::unordered_map<std::string, Tensor> tensors_;
    std::unordered_map<std::string, std::string> metadata_;
};

// Save model weights to safetensors
void save_model(const models::Transformer& model, const std::string& path);

} // namespace vesper::io
```

### 7.1 Writer Implementation

```cpp
// src/io/safetensors_writer.cpp

void SafetensorsWriter::write() {
    json header;
    
    // Metadata
    if (!metadata_.empty()) {
        header["__metadata__"] = metadata_;
    }
    
    // Calculate offsets and add tensor info
    size_t current_offset = 0;
    std::vector<std::pair<std::string, Tensor>> ordered_tensors;
    
    for (const auto& [name, tensor] : tensors_) {
        size_t size = tensor.numel() * dtype_size(tensor.dtype());
        
        header[name] = {
            {"dtype", dtype_to_safetensors_string(tensor.dtype())},
            {"shape", tensor.shape()},
            {"data_offsets", {current_offset, current_offset + size}}
        };
        
        ordered_tensors.emplace_back(name, tensor);
        current_offset += size;
    }
    
    // Serialize header
    std::string header_str = header.dump();
    
    // Pad header to 8-byte alignment
    while ((8 + header_str.size()) % 8 != 0) {
        header_str += ' ';
    }
    
    // Write file
    std::ofstream f(path_, std::ios::binary);
    
    // Header size
    uint64_t header_size = header_str.size();
    f.write(reinterpret_cast<char*>(&header_size), 8);
    
    // Header
    f.write(header_str.data(), header_str.size());
    
    // Tensor data
    for (const auto& [name, tensor] : ordered_tensors) {
        Tensor cpu_tensor = tensor.to(Device::CPU).contiguous();
        size_t size = cpu_tensor.numel() * dtype_size(cpu_tensor.dtype());
        f.write(static_cast<const char*>(cpu_tensor.data_ptr()), size);
    }
}

void save_model(const models::Transformer& model, const std::string& path) {
    SafetensorsWriter writer(path);
    
    for (const auto& [name, param] : model.named_parameters()) {
        writer.add_tensor(name, param);
    }
    
    writer.set_metadata("format", "vesper");
    writer.set_metadata("version", "1.0");
    
    writer.write();
}
```

## 8. Testing Strategy

### 8.1 Unit Tests

```cpp
// tests/io/test_safetensors.cpp

TEST(Safetensors, ReadSingleTensor) {
    // Create a test safetensors file
    std::string test_file = "/tmp/test_single.safetensors";
    
    // Write test file
    {
        SafetensorsWriter writer(test_file);
        writer.add_tensor("test", randn({3, 4}));
        writer.write();
    }
    
    // Read it back
    SafetensorsReader reader(test_file);
    
    auto names = reader.tensor_names();
    EXPECT_EQ(names.size(), 1);
    EXPECT_EQ(names[0], "test");
    
    auto info = reader.get_info("test");
    EXPECT_EQ(info.shape, std::vector<int64_t>({3, 4}));
    EXPECT_EQ(info.dtype, DType::Float32);
    
    Tensor t = reader.load_tensor("test");
    EXPECT_EQ(t.shape(), std::vector<int64_t>({3, 4}));
}

TEST(Safetensors, RoundTrip) {
    std::string test_file = "/tmp/test_roundtrip.safetensors";
    
    Tensor original = randn({16, 32});
    
    // Write
    {
        SafetensorsWriter writer(test_file);
        writer.add_tensor("data", original);
        writer.write();
    }
    
    // Read
    SafetensorsReader reader(test_file);
    Tensor loaded = reader.load_tensor("data");
    
    EXPECT_TRUE(allclose(original, loaded));
}

TEST(Safetensors, MultipleDtypes) {
    std::string test_file = "/tmp/test_dtypes.safetensors";
    
    {
        SafetensorsWriter writer(test_file);
        writer.add_tensor("f32", randn({2, 3}, DType::Float32));
        writer.add_tensor("i64", randint(0, 100, {2, 3}, DType::Int64));
        writer.write();
    }
    
    SafetensorsReader reader(test_file);
    
    EXPECT_EQ(reader.get_info("f32").dtype, DType::Float32);
    EXPECT_EQ(reader.get_info("i64").dtype, DType::Int64);
}

TEST(Safetensors, Metadata) {
    std::string test_file = "/tmp/test_metadata.safetensors";
    
    {
        SafetensorsWriter writer(test_file);
        writer.add_tensor("x", randn({2, 2}));
        writer.set_metadata("author", "vesper");
        writer.set_metadata("version", "1.0");
        writer.write();
    }
    
    SafetensorsReader reader(test_file);
    auto meta = reader.metadata();
    
    EXPECT_EQ(meta["author"], "vesper");
    EXPECT_EQ(meta["version"], "1.0");
}

TEST(ModelLoader, WeightMapping) {
    WeightMapper mapper = WeightMapper::huggingface_llama();
    
    EXPECT_EQ(
        mapper.map_from_source("model.embed_tokens.weight"),
        "tok_embeddings.weight");
    
    EXPECT_EQ(
        mapper.map_from_source("model.layers.5.self_attn.q_proj.weight"),
        "layers.5.attention.wq.weight");
    
    EXPECT_EQ(
        mapper.map_from_source("model.layers.10.mlp.gate_proj.weight"),
        "layers.10.feed_forward.w1.weight");
}
```

### 8.2 Stress Tests

```cpp
TEST(Safetensors, StressTest_LargeTensor) {
    std::string test_file = "/tmp/test_large.safetensors";
    
    // 1GB tensor (256M floats)
    Tensor large = randn({16384, 16384});
    
    auto start = std::chrono::high_resolution_clock::now();
    
    {
        SafetensorsWriter writer(test_file);
        writer.add_tensor("large", large);
        writer.write();
    }
    
    auto write_end = std::chrono::high_resolution_clock::now();
    
    SafetensorsReader reader(test_file);
    Tensor loaded = reader.load_tensor("large");
    
    auto read_end = std::chrono::high_resolution_clock::now();
    
    auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        write_end - start).count();
    auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        read_end - write_end).count();
    
    std::cout << "Write 1GB: " << write_ms << " ms" << std::endl;
    std::cout << "Read 1GB: " << read_ms << " ms" << std::endl;
    
    // Should be reasonably fast (> 100 MB/s)
    EXPECT_LT(write_ms, 10000);
    EXPECT_LT(read_ms, 10000);
    
    EXPECT_TRUE(allclose(large, loaded));
}

TEST(Safetensors, StressTest_ManyTensors) {
    std::string test_file = "/tmp/test_many.safetensors";
    
    // Simulate a model with 500 tensors
    std::unordered_map<std::string, Tensor> original;
    
    {
        SafetensorsWriter writer(test_file);
        for (int i = 0; i < 500; ++i) {
            std::string name = "layer" + std::to_string(i) + ".weight";
            Tensor t = randn({512, 512});
            original[name] = t;
            writer.add_tensor(name, t);
        }
        writer.write();
    }
    
    SafetensorsReader reader(test_file);
    EXPECT_EQ(reader.tensor_names().size(), 500);
    
    // Verify a few random tensors
    for (int i : {0, 100, 250, 499}) {
        std::string name = "layer" + std::to_string(i) + ".weight";
        Tensor loaded = reader.load_tensor(name);
        EXPECT_TRUE(allclose(original[name], loaded));
    }
}

TEST(ModelLoader, StressTest_FullModelLoad) {
    // Skip if model doesn't exist
    std::string model_path = "test_models/tiny-llama";
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Test model not available";
    }
    
    auto config = io::ModelLoader::load_config(model_path);
    auto model = std::make_unique<models::Transformer>(config);
    
    io::LoadConfig load_config;
    load_config.model_path = model_path;
    load_config.strict = true;
    
    // Should not throw
    EXPECT_NO_THROW(io::ModelLoader::load(*model, load_config));
    
    // Model should be usable
    Tensor input = tensor({{1, 2, 3, 4}}, DType::Int64);
    EXPECT_NO_THROW(model->forward(input));
}
```

### 8.3 GPU Transfer Tests

```cpp
TEST(Safetensors, GPUDirectLoad) {
    std::string test_file = "/tmp/test_gpu.safetensors";
    
    Tensor original = randn({1024, 1024});
    
    {
        SafetensorsWriter writer(test_file);
        writer.add_tensor("data", original);
        writer.write();
    }
    
    // Load directly to GPU
    SafetensorsReader reader(test_file);
    Tensor gpu_tensor = reader.load_tensor("data", Device::HIP);
    
    EXPECT_EQ(gpu_tensor.device(), Device::HIP);
    EXPECT_TRUE(allclose(original, gpu_tensor.to(Device::CPU)));
}
```

## 9. Performance Considerations

### 9.1 Memory-Mapped I/O

We use mmap for reading, which provides:
- **Zero-copy access** to file contents
- **OS-level caching** of frequently accessed regions
- **Lazy loading** - only pages accessed are read from disk

### 9.2 Parallel Loading

For sharded models, load tensors in parallel:

```cpp
std::unordered_map<std::string, Tensor> load_parallel(
    const std::vector<std::string>& files, Device device, int num_threads = 4) 
{
    std::unordered_map<std::string, Tensor> result;
    std::mutex result_mutex;
    
    ThreadPool pool(num_threads);
    
    for (const auto& file : files) {
        pool.submit([&, file]() {
            SafetensorsReader reader(file);
            auto tensors = reader.load_all(device);
            
            std::lock_guard<std::mutex> lock(result_mutex);
            for (auto& [name, tensor] : tensors) {
                result[name] = std::move(tensor);
            }
        });
    }
    
    pool.wait();
    return result;
}
```

## 10. Summary

This chapter covered:

1. **Safetensors Format**: Fast, safe, and simple tensor storage
2. **Weight Mapping**: Translating between naming conventions
3. **Model Loading**: High-level API for loading pre-trained weights
4. **Saving**: Writing models in safetensors format

Key implementation points:
- Use memory-mapped I/O for large files
- Support sharded models (multiple files)
- Handle dtype conversions gracefully
- Provide clear error messages for mismatches

With weight loading implemented, Vesper can now run pre-trained models from HuggingFace and other sources.

```
