/// @file model_loader.cpp
/// @brief Implementation of high-level model loading interface.

#include <vesper/io/model_loader.h>
#include <vesper/io/safetensors.h>
#include <vesper/io/weight_mapper.h>

#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace vesper::io {

namespace fs = std::filesystem;

namespace {

// ============================================================================
// Simple JSON parser for config.json (reusing pattern from safetensors.cpp)
// ============================================================================

struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    
    std::string string_value;
    int64_t int_value = 0;
    double float_value = 0.0;
    bool bool_value = false;
    std::vector<JsonValue> array_values;
    std::unordered_map<std::string, JsonValue> object_values;
    
    template<typename T>
    T value(T default_val) const {
        if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int>) {
            return type == Number ? static_cast<T>(int_value) : default_val;
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return type == Number ? static_cast<T>(float_value) : default_val;
        } else if constexpr (std::is_same_v<T, bool>) {
            return type == Bool ? bool_value : default_val;
        } else if constexpr (std::is_same_v<T, std::string>) {
            return type == String ? string_value : default_val;
        }
        return default_val;
    }
    
    bool has(const std::string& key) const {
        return type == Object && object_values.count(key) > 0;
    }
    
    const JsonValue& operator[](const std::string& key) const {
        static JsonValue null_value;
        if (type != Object) return null_value;
        auto it = object_values.find(key);
        return it != object_values.end() ? it->second : null_value;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& json) : json_(json), pos_(0) {}
    
    JsonValue parse() {
        skip_whitespace();
        return parse_value();
    }
    
private:
    const std::string& json_;
    size_t pos_;
    
    void skip_whitespace() {
        while (pos_ < json_.size() && std::isspace(json_[pos_])) ++pos_;
    }
    
    char peek() const { return pos_ < json_.size() ? json_[pos_] : '\0'; }
    char get() { return pos_ < json_.size() ? json_[pos_++] : '\0'; }
    
    void expect(char c) {
        skip_whitespace();
        if (get() != c) throw std::runtime_error("JSON parse error");
    }
    
    JsonValue parse_value() {
        skip_whitespace();
        char c = peek();
        if (c == '"') return parse_string();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit(c)) return parse_number();
        throw std::runtime_error("JSON parse error at " + std::to_string(pos_));
    }
    
    JsonValue parse_string() {
        expect('"');
        std::string result;
        while (peek() != '"') {
            char c = get();
            if (c == '\\') {
                c = get();
                switch (c) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'u': for (int i = 0; i < 4 && pos_ < json_.size(); ++i) get(); result += '?'; break;
                    default: result += c;
                }
            } else {
                result += c;
            }
        }
        expect('"');
        JsonValue v; v.type = JsonValue::String; v.string_value = std::move(result);
        return v;
    }
    
    JsonValue parse_number() {
        size_t start = pos_;
        bool is_float = false;
        if (peek() == '-') get();
        while (std::isdigit(peek())) get();
        if (peek() == '.') { is_float = true; get(); while (std::isdigit(peek())) get(); }
        if (peek() == 'e' || peek() == 'E') { is_float = true; get(); if (peek() == '+' || peek() == '-') get(); while (std::isdigit(peek())) get(); }
        std::string num = json_.substr(start, pos_ - start);
        JsonValue v; v.type = JsonValue::Number;
        if (is_float) { v.float_value = std::stod(num); v.int_value = static_cast<int64_t>(v.float_value); }
        else { v.int_value = std::stoll(num); v.float_value = static_cast<double>(v.int_value); }
        return v;
    }
    
    JsonValue parse_object() {
        expect('{');
        JsonValue v; v.type = JsonValue::Object;
        skip_whitespace();
        if (peek() != '}') {
            do {
                skip_whitespace(); if (peek() == ',') get(); skip_whitespace();
                auto key = parse_string();
                expect(':');
                v.object_values[key.string_value] = parse_value();
                skip_whitespace();
            } while (peek() == ',');
        }
        expect('}');
        return v;
    }
    
    JsonValue parse_array() {
        expect('[');
        JsonValue v; v.type = JsonValue::Array;
        skip_whitespace();
        if (peek() != ']') {
            do {
                skip_whitespace(); if (peek() == ',') get(); skip_whitespace();
                v.array_values.push_back(parse_value());
                skip_whitespace();
            } while (peek() == ',');
        }
        expect(']');
        return v;
    }
    
    JsonValue parse_bool() {
        JsonValue v; v.type = JsonValue::Bool;
        if (json_.substr(pos_, 4) == "true") { pos_ += 4; v.bool_value = true; }
        else if (json_.substr(pos_, 5) == "false") { pos_ += 5; v.bool_value = false; }
        else throw std::runtime_error("Invalid bool");
        return v;
    }
    
    JsonValue parse_null() {
        if (json_.substr(pos_, 4) == "null") { pos_ += 4; return JsonValue{}; }
        throw std::runtime_error("Invalid null");
    }
};

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

} // anonymous namespace

// ============================================================================
// ModelLoader Implementation
// ============================================================================

std::string ModelLoader::detect_format(const std::string& model_path) {
    if (!fs::exists(model_path)) {
        throw std::runtime_error("Model path does not exist: " + model_path);
    }
    
    if (fs::is_directory(model_path)) {
        // Check for safetensors files
        for (const auto& entry : fs::directory_iterator(model_path)) {
            if (entry.path().extension() == ".safetensors") {
                return "safetensors";
            }
        }
        // Check for pytorch files
        for (const auto& entry : fs::directory_iterator(model_path)) {
            std::string ext = entry.path().extension().string();
            if (ext == ".pt" || ext == ".bin" || ext == ".pth") {
                return "pytorch";
            }
        }
        return "unknown";
    } else {
        std::string ext = fs::path(model_path).extension().string();
        if (ext == ".safetensors") return "safetensors";
        if (ext == ".pt" || ext == ".bin" || ext == ".pth") return "pytorch";
        
        // Try to detect by content
        if (is_safetensors_file(model_path)) return "safetensors";
        
        return "unknown";
    }
}

std::unordered_map<std::string, Tensor> ModelLoader::load_safetensors(const LoadConfig& config) {
    if (fs::is_directory(config.model_path)) {
        return load_sharded_safetensors(config.model_path, config.device);
    } else {
        SafetensorsReader reader(config.model_path);
        return reader.load_all(config.device);
    }
}

std::unordered_map<std::string, Tensor> ModelLoader::load_tensors(const LoadConfig& config) {
    std::string format = config.format;
    if (format == "auto") {
        format = detect_format(config.model_path);
    }
    
    if (format == "safetensors") {
        return load_safetensors(config);
    } else if (format == "pytorch") {
        throw std::runtime_error("PyTorch format (.pt/.bin) not yet supported. "
                                "Convert to safetensors using: "
                                "python -m safetensors.convert " + config.model_path);
    } else {
        throw std::runtime_error("Unknown model format: " + format);
    }
}

std::vector<std::string> ModelLoader::list_tensors(const std::string& model_path) {
    std::vector<std::string> names;
    
    std::string format = detect_format(model_path);
    if (format != "safetensors") {
        throw std::runtime_error("list_tensors only supports safetensors format");
    }
    
    if (fs::is_directory(model_path)) {
        std::vector<std::string> shard_files;
        for (const auto& entry : fs::directory_iterator(model_path)) {
            if (entry.path().extension() == ".safetensors") {
                shard_files.push_back(entry.path().string());
            }
        }
        std::sort(shard_files.begin(), shard_files.end());
        
        for (const auto& shard : shard_files) {
            SafetensorsReader reader(shard);
            auto shard_names = reader.tensor_names();
            names.insert(names.end(), shard_names.begin(), shard_names.end());
        }
    } else {
        SafetensorsReader reader(model_path);
        names = reader.tensor_names();
    }
    
    std::sort(names.begin(), names.end());
    return names;
}

int64_t ModelLoader::count_parameters(const std::string& model_path) {
    int64_t total = 0;
    
    std::string format = detect_format(model_path);
    if (format != "safetensors") {
        throw std::runtime_error("count_parameters only supports safetensors format");
    }
    
    if (fs::is_directory(model_path)) {
        for (const auto& entry : fs::directory_iterator(model_path)) {
            if (entry.path().extension() == ".safetensors") {
                SafetensorsReader reader(entry.path().string());
                for (const auto& name : reader.tensor_names()) {
                    auto info = reader.get_info(name);
                    int64_t numel = 1;
                    for (auto dim : info.shape) numel *= dim;
                    total += numel;
                }
            }
        }
    } else {
        SafetensorsReader reader(model_path);
        for (const auto& name : reader.tensor_names()) {
            auto info = reader.get_info(name);
            int64_t numel = 1;
            for (auto dim : info.shape) numel *= dim;
            total += numel;
        }
    }
    
    return total;
}

LoadResult ModelLoader::load(nn::Module& module, const LoadConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    LoadResult result;
    
    // 1. Load tensors from file
    auto tensors = load_tensors(config);
    
    if (config.verbose) {
        std::cout << "Loaded " << tensors.size() << " tensors from " << config.model_path << std::endl;
    }
    
    // 2. Detect source format and create mapper
    std::string source = config.source_format;
    if (source == "auto") {
        std::vector<std::string> names;
        for (const auto& [name, _] : tensors) {
            names.push_back(name);
        }
        source = detect_weight_format(names);
        if (config.verbose) {
            std::cout << "Detected weight format: " << source << std::endl;
        }
    }
    
    WeightMapper mapper;
    if (source == "huggingface_llama" || source == "huggingface") {
        mapper = WeightMapper::huggingface_llama();
    } else if (source == "meta_llama" || source == "meta") {
        mapper = WeightMapper::meta_llama();
    } else if (source == "huggingface_gpt2") {
        mapper = WeightMapper::huggingface_gpt2();
    } else if (source == "huggingface_mistral") {
        mapper = WeightMapper::huggingface_mistral();
    }
    // else: empty mapper, use names as-is
    
    // 3. Get module's parameters
    auto params = module.named_parameters_ptrs();
    
    // 4. Load weights into module
    std::unordered_set<std::string> loaded_source_names;
    
    for (auto& [param_name, param_ptr] : params) {
        // Map vesper param name to source tensor name
        std::string source_name = mapper.map_to_source(param_name);
        
        // Try direct match first, then mapped name
        auto it = tensors.find(source_name);
        if (it == tensors.end()) {
            // Try without mapping
            it = tensors.find(param_name);
        }
        
        if (it == tensors.end()) {
            result.missing.push_back(param_name);
            continue;
        }
        
        Tensor& source_tensor = it->second;
        Tensor& target_param = *param_ptr;
        
        // Shape check
        if (source_tensor.shape() != target_param.shape()) {
            std::string msg = "Shape mismatch for '" + param_name + "': "
                            "model expects " + shape_to_string(target_param.shape()) + 
                            ", got " + shape_to_string(source_tensor.shape());
            if (config.strict) {
                throw std::runtime_error(msg);
            } else {
                std::cerr << "Warning: " << msg << std::endl;
                result.missing.push_back(param_name);
                continue;
            }
        }
        
        // Type conversion if requested
        Tensor tensor_to_copy = source_tensor;
        if (config.dtype.has_value() && source_tensor.dtype() != config.dtype.value()) {
            tensor_to_copy = source_tensor.to(config.dtype.value());
        }
        
        // Device transfer if needed
        if (tensor_to_copy.device() != target_param.device()) {
            tensor_to_copy = tensor_to_copy.to(target_param.device());
        }
        
        // Copy weights
        target_param.copy_(tensor_to_copy);
        
        loaded_source_names.insert(it->first);
        result.loaded_count++;
        result.total_parameters += target_param.numel();
        result.total_bytes += target_param.numel() * GetDTypeSize(target_param.dtype());
    }
    
    // 5. Check for unexpected weights
    for (const auto& [name, _] : tensors) {
        if (loaded_source_names.find(name) == loaded_source_names.end()) {
            result.unexpected.push_back(name);
        }
    }
    
    // 6. Report issues
    if (!result.missing.empty()) {
        std::string msg = "Missing weights (" + std::to_string(result.missing.size()) + "): ";
        for (size_t i = 0; i < std::min(result.missing.size(), size_t(5)); ++i) {
            if (i > 0) msg += ", ";
            msg += result.missing[i];
        }
        if (result.missing.size() > 5) msg += ", ...";
        
        if (config.strict) {
            throw std::runtime_error(msg);
        } else if (config.verbose) {
            std::cerr << "Warning: " << msg << std::endl;
        }
    }
    
    if (!result.unexpected.empty() && config.verbose) {
        std::cout << "Note: " << result.unexpected.size() << " weights in file not used by model" << std::endl;
    }
    
    // 7. Timing
    auto end_time = std::chrono::high_resolution_clock::now();
    result.load_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    if (config.verbose) {
        std::cout << "Loaded " << result.loaded_count << " parameters ("
                  << (result.total_bytes / 1024 / 1024) << " MB) in "
                  << static_cast<int>(result.load_time_ms) << " ms" << std::endl;
    }
    
    return result;
}

models::TransformerConfig ModelLoader::load_config(const std::string& model_path) {
    std::string config_path;
    
    if (fs::is_directory(model_path)) {
        config_path = model_path + "/config.json";
    } else {
        // Assume model_path is a file, look for config.json in same directory
        config_path = fs::path(model_path).parent_path().string() + "/config.json";
    }
    
    if (!fs::exists(config_path)) {
        throw std::runtime_error("Cannot find config.json at: " + config_path);
    }
    
    // Read file
    std::ifstream f(config_path);
    if (!f) {
        throw std::runtime_error("Cannot open config.json: " + config_path);
    }
    
    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string json_str = buffer.str();
    
    // Parse JSON
    JsonParser parser(json_str);
    JsonValue config = parser.parse();
    
    // Extract config values
    models::TransformerConfig tc;
    
    // Standard HuggingFace config keys
    tc.vocab_size = config["vocab_size"].value<int64_t>(32000);
    tc.dim = config["hidden_size"].value<int64_t>(4096);
    tc.n_layers = config["num_hidden_layers"].value<int64_t>(32);
    tc.n_heads = config["num_attention_heads"].value<int64_t>(32);
    tc.n_kv_heads = config["num_key_value_heads"].value<int64_t>(tc.n_heads);
    tc.max_seq_len = config["max_position_embeddings"].value<int64_t>(4096);
    tc.ffn_hidden_dim = config["intermediate_size"].value<int64_t>(0);
    tc.norm_eps = static_cast<float>(config["rms_norm_eps"].value<double>(1e-5));
    tc.rope_base = static_cast<float>(config["rope_theta"].value<double>(10000.0));
    
    // Detect model type from architecture
    if (config.has("model_type")) {
        std::string model_type = config["model_type"].value<std::string>("");
        if (model_type == "llama" || model_type == "mistral") {
            tc.use_rms_norm = true;
            tc.use_bias = false;
        } else if (model_type == "gpt2") {
            tc.use_rms_norm = false;
            tc.use_bias = true;
            tc.rope_base = 0;  // GPT-2 uses learned positions
        }
    }
    
    // Tie word embeddings
    tc.tie_word_embeddings = config["tie_word_embeddings"].value<bool>(false);
    
    return tc;
}

} // namespace vesper::io
