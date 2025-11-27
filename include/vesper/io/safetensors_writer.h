#pragma once
/// @file safetensors_writer.h
/// @brief Safetensors format writer for saving model weights.
///
/// Writes tensors in the safetensors format, which is safe, fast, and portable.
/// Used for saving fine-tuned models, checkpoints, etc.
///
/// Example usage:
/// @code
///   SafetensorsWriter writer("model.safetensors");
///   writer.add_tensor("layer.0.weight", tensor1);
///   writer.add_tensor("layer.0.bias", tensor2);
///   writer.set_metadata("format", "vesper");
///   writer.write();
/// @endcode

#include <vesper/core/tensor.h>
#include <vesper/nn/module.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace vesper::io {

/// Writer for safetensors format files.
class SafetensorsWriter {
public:
    /// Creates a writer targeting the specified path.
    /// @param path Output file path (will be overwritten if exists)
    explicit SafetensorsWriter(const std::string& path);
    
    /// Destructor.
    ~SafetensorsWriter();
    
    // Non-copyable
    SafetensorsWriter(const SafetensorsWriter&) = delete;
    SafetensorsWriter& operator=(const SafetensorsWriter&) = delete;
    
    // Movable
    SafetensorsWriter(SafetensorsWriter&&) noexcept;
    SafetensorsWriter& operator=(SafetensorsWriter&&) noexcept;
    
    /// Adds a tensor to be written.
    /// @param name Name for the tensor (e.g., "model.layers.0.weight")
    /// @param tensor The tensor to save (will be moved to CPU if on GPU)
    void add_tensor(const std::string& name, const Tensor& tensor);
    
    /// Adds multiple tensors at once.
    /// @param tensors Map of tensor names to tensors
    void add_tensors(const std::unordered_map<std::string, Tensor>& tensors);
    
    /// Sets a metadata key-value pair.
    /// @param key Metadata key
    /// @param value Metadata value (string)
    void set_metadata(const std::string& key, const std::string& value);
    
    /// Returns the output path.
    const std::string& path() const { return path_; }
    
    /// Returns the number of tensors added so far.
    size_t num_tensors() const { return tensors_.size(); }
    
    /// Writes all tensors to the file.
    /// @throws std::runtime_error if file cannot be written
    void write();
    
    /// Clears all added tensors and metadata (for reuse).
    void clear();
    
private:
    std::string path_;
    std::unordered_map<std::string, Tensor> tensors_;
    std::unordered_map<std::string, std::string> metadata_;
};

/// Saves a module's state_dict to a safetensors file.
/// @param module The module to save
/// @param path Output file path
void save_model(const nn::Module& module, const std::string& path);

/// Saves a state_dict to a safetensors file.
/// @param state_dict Map of parameter names to tensors
/// @param path Output file path
void save_state_dict(const std::unordered_map<std::string, Tensor>& state_dict, 
                     const std::string& path);

} // namespace vesper::io
