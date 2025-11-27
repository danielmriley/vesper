#pragma once
/// @file safetensors.h
/// @brief Safetensors format reader for loading pre-trained model weights.
///
/// Safetensors is a modern, safe, and fast format for storing model weights.
/// Advantages:
/// - Safe: No arbitrary code execution (unlike pickle-based formats)
/// - Fast: Memory-mapped loading, zero-copy when possible  
/// - Simple: JSON header + raw tensor data
///
/// File structure:
///   [8 bytes: header_size (little endian u64)]
///   [header_size bytes: JSON header]
///   [remaining bytes: raw tensor data]

#include <vesper/core/tensor.h>
#include <vesper/core/dtype.h>
#include <vesper/core/device.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace vesper::io {

/// Information about a tensor stored in a safetensors file.
struct TensorInfo {
    std::string name;              ///< Name of the tensor
    DType dtype;                   ///< Data type
    std::vector<int64_t> shape;    ///< Shape of the tensor
    size_t data_offset;            ///< Byte offset in the data section
    size_t data_size;              ///< Size in bytes
};

/// Reader for safetensors format files.
///
/// Uses memory-mapped I/O for efficient reading of large model files.
/// Example usage:
/// @code
///   SafetensorsReader reader("model.safetensors");
///   auto names = reader.tensor_names();
///   Tensor weights = reader.load_tensor("model.embed_tokens.weight", Device::HIP);
/// @endcode
class SafetensorsReader {
public:
    /// Opens a safetensors file for reading.
    /// @param path Path to the .safetensors file
    /// @throws std::runtime_error if file cannot be opened or is invalid
    explicit SafetensorsReader(const std::string& path);
    
    /// Destructor - unmaps file and closes handle.
    ~SafetensorsReader();
    
    // Non-copyable
    SafetensorsReader(const SafetensorsReader&) = delete;
    SafetensorsReader& operator=(const SafetensorsReader&) = delete;
    
    // Movable
    SafetensorsReader(SafetensorsReader&&) noexcept;
    SafetensorsReader& operator=(SafetensorsReader&&) noexcept;
    
    /// Returns the list of all tensor names in the file.
    std::vector<std::string> tensor_names() const;
    
    /// Returns the number of tensors in the file.
    size_t num_tensors() const;
    
    /// Gets detailed information about a specific tensor.
    /// @param name Name of the tensor
    /// @throws std::runtime_error if tensor not found
    TensorInfo get_info(const std::string& name) const;
    
    /// Checks if a tensor with the given name exists.
    bool has_tensor(const std::string& name) const;
    
    /// Loads a single tensor from the file.
    /// @param name Name of the tensor to load
    /// @param device Target device (default: CPU)
    /// @return The loaded tensor
    /// @throws std::runtime_error if tensor not found
    Tensor load_tensor(const std::string& name, Device device = Device::CPU);
    
    /// Loads all tensors from the file.
    /// @param device Target device (default: CPU)
    /// @return Map from tensor names to tensors
    std::unordered_map<std::string, Tensor> load_all(Device device = Device::CPU);
    
    /// Returns metadata stored in the file.
    /// Common metadata keys: "format", "version", etc.
    const std::unordered_map<std::string, std::string>& metadata() const;
    
    /// Returns the file path.
    const std::string& path() const;
    
    /// Returns the total file size in bytes.
    size_t file_size() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// Loads tensors from multiple sharded safetensors files.
///
/// Large models are often split across multiple files (e.g., model-00001-of-00005.safetensors).
/// This function loads all shards from a directory.
///
/// @param model_dir Directory containing .safetensors files
/// @param device Target device (default: CPU)
/// @return Combined map of all tensors from all shards
/// @throws std::runtime_error if duplicate tensor names found across shards
std::unordered_map<std::string, Tensor> load_sharded_safetensors(
    const std::string& model_dir,
    Device device = Device::CPU);

/// Checks if a file is a valid safetensors file.
/// @param path Path to check
/// @return true if file exists and has valid safetensors header
bool is_safetensors_file(const std::string& path);

} // namespace vesper::io
