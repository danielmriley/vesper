#pragma once
/// @file model_loader.h
/// @brief High-level interface for loading pre-trained model weights.
///
/// Provides a unified interface for loading model weights from various formats:
/// - Safetensors (preferred, safe, fast)
/// - PyTorch .pt/.bin (legacy support, requires pickle)
///
/// Example usage:
/// @code
///   // Load HuggingFace Llama model
///   io::LoadConfig config;
///   config.model_path = "models/llama-2-7b-hf";
///   config.device = Device::HIP;
///   config.dtype = DType::Float16;
///
///   auto model_config = io::ModelLoader::load_config(config.model_path);
///   auto model = std::make_unique<models::TransformerLM>(model_config);
///   io::ModelLoader::load(*model, config);
/// @endcode

#include <vesper/core/tensor.h>
#include <vesper/core/dtype.h>
#include <vesper/core/device.h>
#include <vesper/models/config.h>
#include <vesper/nn/module.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>
#include <optional>

namespace vesper::io {

/// Configuration for model loading.
struct LoadConfig {
    /// Path to model file or directory.
    /// - Single file: path/to/model.safetensors
    /// - Directory: path/to/model/ (will scan for .safetensors files)
    std::string model_path;
    
    /// File format: "safetensors", "pytorch", "auto" (default)
    /// "auto" will detect based on file extension and contents.
    std::string format = "auto";
    
    /// Source model format for weight mapping:
    /// "huggingface", "meta", "auto" (default)
    /// "auto" will try to detect from tensor names.
    std::string source_format = "auto";
    
    /// Target device for loaded tensors.
    Device device = Device::CPU;
    
    /// Target dtype (tensors will be converted if different).
    /// Use DType::Float32 by default to avoid precision loss.
    /// Set to DType::Float16 or DType::BFloat16 for inference.
    std::optional<DType> dtype = std::nullopt;  // nullptr = keep original dtype
    
    /// If true, fail on missing or unexpected weights.
    /// If false, print warnings but continue.
    bool strict = true;
    
    /// Verbose output during loading.
    bool verbose = true;
};

/// Result of a model loading operation.
struct LoadResult {
    /// Number of tensors successfully loaded
    size_t loaded_count = 0;
    
    /// Parameter names that were not found in the weight file
    std::vector<std::string> missing;
    
    /// Weight names in the file that didn't match any model parameter
    std::vector<std::string> unexpected;
    
    /// Total number of parameters loaded
    int64_t total_parameters = 0;
    
    /// Total bytes loaded
    size_t total_bytes = 0;
    
    /// Loading time in milliseconds
    double load_time_ms = 0.0;
    
    /// Check if loading was successful (no missing in strict mode)
    bool success() const { return missing.empty(); }
};

/// High-level interface for loading pre-trained models.
class ModelLoader {
public:
    /// Loads weights into a Module.
    ///
    /// @param module Target module (e.g., TransformerLM)
    /// @param config Loading configuration
    /// @return LoadResult with statistics and any issues
    /// @throws std::runtime_error on fatal errors
    static LoadResult load(nn::Module& module, const LoadConfig& config);
    
    /// Loads raw tensors from model files.
    ///
    /// @param config Loading configuration (only model_path, format, device used)
    /// @return Map of tensor names to tensors
    static std::unordered_map<std::string, Tensor> load_tensors(const LoadConfig& config);
    
    /// Loads model configuration from a HuggingFace model directory.
    ///
    /// Parses config.json to extract TransformerConfig.
    ///
    /// @param model_path Path to model directory containing config.json
    /// @return Parsed TransformerConfig
    /// @throws std::runtime_error if config.json not found or invalid
    static models::TransformerConfig load_config(const std::string& model_path);
    
    /// Checks what format a model file/directory contains.
    ///
    /// @param model_path Path to check
    /// @return "safetensors", "pytorch", or "unknown"
    static std::string detect_format(const std::string& model_path);
    
    /// Lists all tensor names in a model file/directory.
    ///
    /// Useful for debugging weight mapping issues.
    ///
    /// @param model_path Path to model file or directory
    /// @return Sorted list of tensor names
    static std::vector<std::string> list_tensors(const std::string& model_path);
    
    /// Computes total parameter count from model files.
    ///
    /// @param model_path Path to model file or directory
    /// @return Total number of parameters (sum of all tensor elements)
    static int64_t count_parameters(const std::string& model_path);
    
private:
    // Implementation helpers
    static std::unordered_map<std::string, Tensor> load_safetensors(const LoadConfig& config);
};

/// Convenience function to load a complete model.
///
/// Combines load_config and load into a single call.
///
/// @tparam ModelType Type of model to create (must take TransformerConfig)
/// @param model_path Path to model directory
/// @param device Target device
/// @param dtype Target dtype (nullopt = keep original)
/// @return Loaded model
template<typename ModelType>
std::unique_ptr<ModelType> load_model(
    const std::string& model_path,
    Device device = Device::CPU,
    std::optional<DType> dtype = std::nullopt);

} // namespace vesper::io

// Template implementation
namespace vesper::io {

template<typename ModelType>
std::unique_ptr<ModelType> load_model(
    const std::string& model_path,
    Device device,
    std::optional<DType> dtype) 
{
    // Load config
    auto config = ModelLoader::load_config(model_path);
    
    // Create model
    auto model = std::make_unique<ModelType>(config);
    
    // Move to device before loading (avoids copying)
    model->to(device);
    
    // Load weights
    LoadConfig load_config;
    load_config.model_path = model_path;
    load_config.device = device;
    load_config.dtype = dtype;
    
    ModelLoader::load(*model, load_config);
    
    return model;
}

} // namespace vesper::io
