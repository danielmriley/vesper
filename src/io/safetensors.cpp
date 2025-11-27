/// @file safetensors.cpp
/// @brief Implementation of safetensors format reader.

#include <vesper/io/safetensors.h>
#include <vesper/core/factories.h>

#include <fstream>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <cctype>

// Platform-specific memory mapping
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace vesper::io {

namespace {

// ============================================================================
// Simple JSON parser for safetensors header
// ============================================================================

// A minimal JSON value type for parsing safetensors headers.
// Supports: objects, arrays, strings, numbers (as strings or int64).
struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    
    std::string string_value;
    int64_t int_value = 0;
    double float_value = 0.0;
    bool bool_value = false;
    std::vector<JsonValue> array_values;
    std::unordered_map<std::string, JsonValue> object_values;
    
    bool is_null() const { return type == Null; }
    bool is_string() const { return type == String; }
    bool is_number() const { return type == Number; }
    bool is_array() const { return type == Array; }
    bool is_object() const { return type == Object; }
    
    const std::string& as_string() const {
        if (type != String) throw std::runtime_error("JSON value is not a string");
        return string_value;
    }
    
    int64_t as_int() const {
        if (type != Number) throw std::runtime_error("JSON value is not a number");
        return int_value;
    }
    
    const std::vector<JsonValue>& as_array() const {
        if (type != Array) throw std::runtime_error("JSON value is not an array");
        return array_values;
    }
    
    const std::unordered_map<std::string, JsonValue>& as_object() const {
        if (type != Object) throw std::runtime_error("JSON value is not an object");
        return object_values;
    }
    
    bool has(const std::string& key) const {
        return type == Object && object_values.count(key) > 0;
    }
    
    const JsonValue& operator[](const std::string& key) const {
        if (type != Object) throw std::runtime_error("JSON value is not an object");
        auto it = object_values.find(key);
        if (it == object_values.end()) throw std::runtime_error("Key not found: " + key);
        return it->second;
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
        while (pos_ < json_.size() && std::isspace(json_[pos_])) {
            ++pos_;
        }
    }
    
    char peek() const {
        return pos_ < json_.size() ? json_[pos_] : '\0';
    }
    
    char get() {
        return pos_ < json_.size() ? json_[pos_++] : '\0';
    }
    
    void expect(char c) {
        skip_whitespace();
        if (get() != c) {
            throw std::runtime_error(std::string("Expected '") + c + "' at position " + std::to_string(pos_-1));
        }
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
        
        throw std::runtime_error("Unexpected character in JSON at position " + std::to_string(pos_));
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
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'u': {
                        // Skip unicode escapes for simplicity
                        for (int i = 0; i < 4 && pos_ < json_.size(); ++i) get();
                        result += '?';
                        break;
                    }
                    default: result += c; break;
                }
            } else {
                result += c;
            }
        }
        expect('"');
        JsonValue v;
        v.type = JsonValue::String;
        v.string_value = std::move(result);
        return v;
    }
    
    JsonValue parse_number() {
        size_t start = pos_;
        bool is_float = false;
        
        if (peek() == '-') get();
        while (std::isdigit(peek())) get();
        if (peek() == '.') {
            is_float = true;
            get();
            while (std::isdigit(peek())) get();
        }
        if (peek() == 'e' || peek() == 'E') {
            is_float = true;
            get();
            if (peek() == '+' || peek() == '-') get();
            while (std::isdigit(peek())) get();
        }
        
        std::string num_str = json_.substr(start, pos_ - start);
        JsonValue v;
        v.type = JsonValue::Number;
        if (is_float) {
            v.float_value = std::stod(num_str);
            v.int_value = static_cast<int64_t>(v.float_value);
        } else {
            v.int_value = std::stoll(num_str);
            v.float_value = static_cast<double>(v.int_value);
        }
        return v;
    }
    
    JsonValue parse_object() {
        expect('{');
        JsonValue v;
        v.type = JsonValue::Object;
        skip_whitespace();
        
        if (peek() != '}') {
            do {
                skip_whitespace();
                if (peek() == ',') get();
                skip_whitespace();
                
                JsonValue key = parse_string();
                expect(':');
                JsonValue value = parse_value();
                v.object_values[key.string_value] = std::move(value);
                
                skip_whitespace();
            } while (peek() == ',');
        }
        
        expect('}');
        return v;
    }
    
    JsonValue parse_array() {
        expect('[');
        JsonValue v;
        v.type = JsonValue::Array;
        skip_whitespace();
        
        if (peek() != ']') {
            do {
                skip_whitespace();
                if (peek() == ',') get();
                skip_whitespace();
                
                v.array_values.push_back(parse_value());
                skip_whitespace();
            } while (peek() == ',');
        }
        
        expect(']');
        return v;
    }
    
    JsonValue parse_bool() {
        JsonValue v;
        v.type = JsonValue::Bool;
        if (json_.substr(pos_, 4) == "true") {
            pos_ += 4;
            v.bool_value = true;
        } else if (json_.substr(pos_, 5) == "false") {
            pos_ += 5;
            v.bool_value = false;
        } else {
            throw std::runtime_error("Invalid boolean at position " + std::to_string(pos_));
        }
        return v;
    }
    
    JsonValue parse_null() {
        if (json_.substr(pos_, 4) == "null") {
            pos_ += 4;
            return JsonValue{};
        }
        throw std::runtime_error("Invalid null at position " + std::to_string(pos_));
    }
};

// ============================================================================
// DType parsing
// ============================================================================

DType parse_dtype(const std::string& dtype_str) {
    // Safetensors uses uppercase type names
    if (dtype_str == "F32" || dtype_str == "float32") return DType::Float32;
    if (dtype_str == "F64" || dtype_str == "float64") return DType::Float64;
    if (dtype_str == "F16" || dtype_str == "float16") return DType::Float16;
    if (dtype_str == "BF16" || dtype_str == "bfloat16") return DType::BFloat16;
    if (dtype_str == "I64" || dtype_str == "int64") return DType::Int64;
    if (dtype_str == "I32" || dtype_str == "int32") return DType::Int32;
    // Note: We don't support all types - add more as needed
    throw std::runtime_error("Unsupported dtype in safetensors: " + dtype_str);
}

std::string dtype_to_safetensors_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "F32";
        case DType::Float64: return "F64";
        case DType::Float16: return "F16";
        case DType::BFloat16: return "BF16";
        case DType::Int64: return "I64";
        case DType::Int32: return "I32";
    }
    throw std::runtime_error("Unsupported dtype for safetensors");
}

} // anonymous namespace

// ============================================================================
// SafetensorsReader Implementation
// ============================================================================

class SafetensorsReader::Impl {
public:
    Impl(const std::string& path) : path_(path) {
#ifdef _WIN32
        // Windows memory mapping
        file_handle_ = CreateFileA(
            path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        
        LARGE_INTEGER size;
        if (!GetFileSizeEx(file_handle_, &size)) {
            CloseHandle(file_handle_);
            throw std::runtime_error("Cannot get file size: " + path);
        }
        file_size_ = static_cast<size_t>(size.QuadPart);
        
        mapping_handle_ = CreateFileMappingA(
            file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_handle_) {
            CloseHandle(file_handle_);
            throw std::runtime_error("Failed to create file mapping: " + path);
        }
        
        mapped_data_ = static_cast<const uint8_t*>(
            MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
        if (!mapped_data_) {
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            throw std::runtime_error("Failed to map file: " + path);
        }
#else
        // POSIX memory mapping
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open file: " + path + " - " + strerror(errno));
        }
        
        struct stat st;
        if (fstat(fd_, &st) < 0) {
            close(fd_);
            throw std::runtime_error("Cannot stat file: " + path);
        }
        file_size_ = static_cast<size_t>(st.st_size);
        
        if (file_size_ < 8) {
            close(fd_);
            throw std::runtime_error("File too small to be safetensors: " + path);
        }
        
        mapped_data_ = static_cast<const uint8_t*>(
            mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mapped_data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Failed to mmap file: " + path + " - " + strerror(errno));
        }
#endif
        
        parse_header();
    }
    
    ~Impl() {
#ifdef _WIN32
        if (mapped_data_) UnmapViewOfFile(mapped_data_);
        if (mapping_handle_) CloseHandle(mapping_handle_);
        if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
#else
        if (mapped_data_ && mapped_data_ != MAP_FAILED) {
            munmap(const_cast<uint8_t*>(mapped_data_), file_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
#endif
    }
    
    // Move operations
    Impl(Impl&& other) noexcept
        : path_(std::move(other.path_)),
          file_size_(other.file_size_),
          header_size_(other.header_size_),
          data_start_(other.data_start_),
          tensors_(std::move(other.tensors_)),
          metadata_(std::move(other.metadata_)),
          mapped_data_(other.mapped_data_)
#ifdef _WIN32
          , file_handle_(other.file_handle_)
          , mapping_handle_(other.mapping_handle_)
#else
          , fd_(other.fd_)
#endif
    {
        other.mapped_data_ = nullptr;
#ifdef _WIN32
        other.file_handle_ = INVALID_HANDLE_VALUE;
        other.mapping_handle_ = nullptr;
#else
        other.fd_ = -1;
#endif
    }
    
    std::vector<std::string> tensor_names() const {
        std::vector<std::string> names;
        names.reserve(tensors_.size());
        for (const auto& [name, _] : tensors_) {
            names.push_back(name);
        }
        std::sort(names.begin(), names.end());
        return names;
    }
    
    size_t num_tensors() const {
        return tensors_.size();
    }
    
    TensorInfo get_info(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Tensor not found in safetensors: " + name);
        }
        return it->second;
    }
    
    bool has_tensor(const std::string& name) const {
        return tensors_.count(name) > 0;
    }
    
    Tensor load_tensor(const std::string& name, Device device) {
        TensorInfo info = get_info(name);
        
        // Create tensor on CPU first
        Tensor tensor = empty(info.shape, info.dtype, Device::CPU);
        
        // Copy data from memory-mapped region
        const uint8_t* src = mapped_data_ + data_start_ + info.data_offset;
        
        // Get raw pointer - we need to access storage directly
        // Since Tensor has data_ptr<T> template, we use memcpy with size
        size_t dtype_size = GetDTypeSize(info.dtype);
        size_t num_elements = tensor.numel();
        
        // Copy based on dtype
        switch (info.dtype) {
            case DType::Float32:
                std::memcpy(tensor.data_ptr<float>(), src, num_elements * sizeof(float));
                break;
            case DType::Float64:
                std::memcpy(tensor.data_ptr<double>(), src, num_elements * sizeof(double));
                break;
            case DType::Float16:
                // Float16 uses uint16_t storage
                std::memcpy(tensor.data_ptr<uint16_t>(), src, num_elements * sizeof(uint16_t));
                break;
            case DType::BFloat16:
                // BFloat16 uses uint16_t storage
                std::memcpy(tensor.data_ptr<uint16_t>(), src, num_elements * sizeof(uint16_t));
                break;
            case DType::Int32:
                std::memcpy(tensor.data_ptr<int32_t>(), src, num_elements * sizeof(int32_t));
                break;
            case DType::Int64:
                std::memcpy(tensor.data_ptr<int64_t>(), src, num_elements * sizeof(int64_t));
                break;
        }
        
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
    
    const std::string& path() const {
        return path_;
    }
    
    size_t file_size() const {
        return file_size_;
    }
    
private:
    void parse_header() {
        // Read header size (first 8 bytes, little endian)
        uint64_t header_size;
        std::memcpy(&header_size, mapped_data_, sizeof(uint64_t));
        header_size_ = static_cast<size_t>(header_size);
        
        if (8 + header_size_ > file_size_) {
            throw std::runtime_error("Invalid safetensors: header size exceeds file size");
        }
        
        // Parse JSON header
        std::string header_str(
            reinterpret_cast<const char*>(mapped_data_ + 8),
            header_size_);
        
        JsonParser parser(header_str);
        JsonValue header = parser.parse();
        
        if (!header.is_object()) {
            throw std::runtime_error("Invalid safetensors: header is not an object");
        }
        
        // Data starts after header
        data_start_ = 8 + header_size_;
        
        // Parse tensor info and metadata
        for (const auto& [name, info_value] : header.as_object()) {
            if (name == "__metadata__") {
                // Parse metadata
                if (info_value.is_object()) {
                    for (const auto& [k, v] : info_value.as_object()) {
                        if (v.is_string()) {
                            metadata_[k] = v.as_string();
                        }
                    }
                }
                continue;
            }
            
            // Parse tensor info
            if (!info_value.is_object()) {
                throw std::runtime_error("Invalid tensor info for: " + name);
            }
            
            TensorInfo ti;
            ti.name = name;
            ti.dtype = parse_dtype(info_value["dtype"].as_string());
            
            // Parse shape
            const auto& shape_arr = info_value["shape"].as_array();
            ti.shape.reserve(shape_arr.size());
            for (const auto& dim : shape_arr) {
                ti.shape.push_back(dim.as_int());
            }
            
            // Parse data offsets
            const auto& offsets = info_value["data_offsets"].as_array();
            if (offsets.size() != 2) {
                throw std::runtime_error("Invalid data_offsets for: " + name);
            }
            ti.data_offset = static_cast<size_t>(offsets[0].as_int());
            size_t end_offset = static_cast<size_t>(offsets[1].as_int());
            ti.data_size = end_offset - ti.data_offset;
            
            tensors_[name] = ti;
        }
    }
    
    std::string path_;
    size_t file_size_ = 0;
    size_t header_size_ = 0;
    size_t data_start_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
    std::unordered_map<std::string, std::string> metadata_;
    const uint8_t* mapped_data_ = nullptr;
    
#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ============================================================================
// SafetensorsReader public interface
// ============================================================================

SafetensorsReader::SafetensorsReader(const std::string& path)
    : impl_(std::make_unique<Impl>(path)) {}

SafetensorsReader::~SafetensorsReader() = default;

SafetensorsReader::SafetensorsReader(SafetensorsReader&&) noexcept = default;
SafetensorsReader& SafetensorsReader::operator=(SafetensorsReader&&) noexcept = default;

std::vector<std::string> SafetensorsReader::tensor_names() const {
    return impl_->tensor_names();
}

size_t SafetensorsReader::num_tensors() const {
    return impl_->num_tensors();
}

TensorInfo SafetensorsReader::get_info(const std::string& name) const {
    return impl_->get_info(name);
}

bool SafetensorsReader::has_tensor(const std::string& name) const {
    return impl_->has_tensor(name);
}

Tensor SafetensorsReader::load_tensor(const std::string& name, Device device) {
    return impl_->load_tensor(name, device);
}

std::unordered_map<std::string, Tensor> SafetensorsReader::load_all(Device device) {
    return impl_->load_all(device);
}

const std::unordered_map<std::string, std::string>& SafetensorsReader::metadata() const {
    return impl_->metadata();
}

const std::string& SafetensorsReader::path() const {
    return impl_->path();
}

size_t SafetensorsReader::file_size() const {
    return impl_->file_size();
}

// ============================================================================
// Helper functions
// ============================================================================

std::unordered_map<std::string, Tensor> load_sharded_safetensors(
    const std::string& model_dir, Device device) 
{
    namespace fs = std::filesystem;
    
    if (!fs::exists(model_dir)) {
        throw std::runtime_error("Model directory not found: " + model_dir);
    }
    
    if (!fs::is_directory(model_dir)) {
        throw std::runtime_error("Not a directory: " + model_dir);
    }
    
    std::unordered_map<std::string, Tensor> all_tensors;
    
    // Find all .safetensors files
    std::vector<std::string> shard_files;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".safetensors") {
            shard_files.push_back(entry.path().string());
        }
    }
    
    if (shard_files.empty()) {
        throw std::runtime_error("No .safetensors files found in: " + model_dir);
    }
    
    // Sort for deterministic order (important for sharded files like model-00001-of-00005)
    std::sort(shard_files.begin(), shard_files.end());
    
    // Load each shard
    for (const auto& shard_path : shard_files) {
        SafetensorsReader reader(shard_path);
        auto tensors = reader.load_all(device);
        
        for (auto& [name, tensor] : tensors) {
            if (all_tensors.count(name)) {
                throw std::runtime_error("Duplicate tensor name across shards: " + name + 
                                       " (found in " + shard_path + ")");
            }
            all_tensors[name] = std::move(tensor);
        }
    }
    
    return all_tensors;
}

bool is_safetensors_file(const std::string& path) {
    namespace fs = std::filesystem;
    
    if (!fs::exists(path) || !fs::is_regular_file(path)) {
        return false;
    }
    
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    
    // Read header size
    uint64_t header_size;
    if (!file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size))) {
        return false;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(file.tellg());
    
    // Basic sanity check: header size should be reasonable
    return header_size > 0 && header_size < file_size && (8 + header_size) <= file_size;
}

} // namespace vesper::io
