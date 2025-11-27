/// @file safetensors_writer.cpp
/// @brief Implementation of safetensors format writer.

#include <vesper/io/safetensors_writer.h>
#include <vesper/core/dtype.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace vesper::io {

namespace {

/// Converts a DType to safetensors string representation.
std::string dtype_to_safetensors_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "F32";
        case DType::Float64: return "F64";
        case DType::Float16: return "F16";
        case DType::BFloat16: return "BF16";
        case DType::Int32: return "I32";
        case DType::Int64: return "I64";
    }
    throw std::runtime_error("Unsupported dtype for safetensors");
}

/// Escapes a string for JSON output.
std::string json_escape(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 10);
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // Control character - skip or encode
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

} // anonymous namespace

// ============================================================================
// SafetensorsWriter Implementation
// ============================================================================

SafetensorsWriter::SafetensorsWriter(const std::string& path) : path_(path) {}

SafetensorsWriter::~SafetensorsWriter() = default;

SafetensorsWriter::SafetensorsWriter(SafetensorsWriter&&) noexcept = default;
SafetensorsWriter& SafetensorsWriter::operator=(SafetensorsWriter&&) noexcept = default;

void SafetensorsWriter::add_tensor(const std::string& name, const Tensor& tensor) {
    // Store a CPU copy of the tensor
    Tensor cpu_tensor = (tensor.device() != Device::CPU) ? tensor.to(Device::CPU) : tensor;
    
    // Make contiguous if not already
    if (!cpu_tensor.is_contiguous()) {
        cpu_tensor = cpu_tensor.contiguous();
    }
    
    tensors_[name] = std::move(cpu_tensor);
}

void SafetensorsWriter::add_tensors(const std::unordered_map<std::string, Tensor>& tensors) {
    for (const auto& [name, tensor] : tensors) {
        add_tensor(name, tensor);
    }
}

void SafetensorsWriter::set_metadata(const std::string& key, const std::string& value) {
    metadata_[key] = value;
}

void SafetensorsWriter::clear() {
    tensors_.clear();
    metadata_.clear();
}

void SafetensorsWriter::write() {
    if (tensors_.empty()) {
        throw std::runtime_error("No tensors to write");
    }
    
    // Sort tensor names for deterministic output
    std::vector<std::string> sorted_names;
    sorted_names.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        sorted_names.push_back(name);
    }
    std::sort(sorted_names.begin(), sorted_names.end());
    
    // Calculate data offsets
    size_t current_offset = 0;
    std::vector<std::pair<std::string, size_t>> tensor_offsets;  // name -> (offset, size)
    std::vector<size_t> tensor_sizes;
    
    for (const auto& name : sorted_names) {
        const Tensor& tensor = tensors_[name];
        size_t size = tensor.numel() * GetDTypeSize(tensor.dtype());
        tensor_offsets.emplace_back(name, current_offset);
        tensor_sizes.push_back(size);
        current_offset += size;
    }
    
    // Build JSON header
    std::ostringstream header;
    header << "{";
    
    bool first = true;
    
    // Add metadata if present
    if (!metadata_.empty()) {
        header << "\"__metadata__\":{";
        bool meta_first = true;
        for (const auto& [k, v] : metadata_) {
            if (!meta_first) header << ",";
            header << "\"" << json_escape(k) << "\":\"" << json_escape(v) << "\"";
            meta_first = false;
        }
        header << "}";
        first = false;
    }
    
    // Add tensor info
    for (size_t i = 0; i < sorted_names.size(); ++i) {
        const std::string& name = sorted_names[i];
        const Tensor& tensor = tensors_[name];
        size_t offset = tensor_offsets[i].second;
        size_t size = tensor_sizes[i];
        
        if (!first) header << ",";
        first = false;
        
        header << "\"" << json_escape(name) << "\":{";
        header << "\"dtype\":\"" << dtype_to_safetensors_string(tensor.dtype()) << "\",";
        header << "\"shape\":[";
        for (size_t j = 0; j < tensor.shape().size(); ++j) {
            if (j > 0) header << ",";
            header << tensor.shape()[j];
        }
        header << "],";
        header << "\"data_offsets\":[" << offset << "," << (offset + size) << "]";
        header << "}";
    }
    
    header << "}";
    
    std::string header_str = header.str();
    
    // Pad header to 8-byte alignment
    while ((8 + header_str.size()) % 8 != 0) {
        header_str += ' ';
    }
    
    // Write file
    std::ofstream file(path_, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + path_);
    }
    
    // Write header size (little endian uint64)
    uint64_t header_size = header_str.size();
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    
    // Write header
    file.write(header_str.data(), header_str.size());
    
    // Write tensor data (in sorted order)
    for (const auto& name : sorted_names) {
        const Tensor& tensor = tensors_[name];
        size_t size = tensor.numel() * GetDTypeSize(tensor.dtype());
        
        // Write raw data based on dtype
        switch (tensor.dtype()) {
            case DType::Float32:
                file.write(reinterpret_cast<const char*>(tensor.data_ptr<float>()), size);
                break;
            case DType::Float64:
                file.write(reinterpret_cast<const char*>(tensor.data_ptr<double>()), size);
                break;
            case DType::Float16:
                file.write(reinterpret_cast<const char*>(tensor.data_ptr<uint16_t>()), size);
                break;
            case DType::BFloat16:
                file.write(reinterpret_cast<const char*>(tensor.data_ptr<uint16_t>()), size);
                break;
            case DType::Int32:
                file.write(reinterpret_cast<const char*>(tensor.data_ptr<int32_t>()), size);
                break;
            case DType::Int64:
                file.write(reinterpret_cast<const char*>(tensor.data_ptr<int64_t>()), size);
                break;
        }
    }
    
    if (!file) {
        throw std::runtime_error("Error writing to file: " + path_);
    }
}

// ============================================================================
// Convenience functions
// ============================================================================

void save_model(const nn::Module& module, const std::string& path) {
    SafetensorsWriter writer(path);
    
    // Get state dict and add all tensors
    auto state_dict = module.state_dict();
    for (const auto& [name, tensor] : state_dict) {
        writer.add_tensor(name, tensor);
    }
    
    writer.set_metadata("format", "vesper");
    writer.set_metadata("version", "1.0");
    
    writer.write();
}

void save_state_dict(const std::unordered_map<std::string, Tensor>& state_dict,
                     const std::string& path) {
    SafetensorsWriter writer(path);
    
    for (const auto& [name, tensor] : state_dict) {
        writer.add_tensor(name, tensor);
    }
    
    writer.write();
}

} // namespace vesper::io
