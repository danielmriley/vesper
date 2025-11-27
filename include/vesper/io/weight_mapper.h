#pragma once
/// @file weight_mapper.h
/// @brief Maps weight names between different model formats.
///
/// Different frameworks use different naming conventions for model parameters:
/// - Vesper: layers.0.attention.wq.weight
/// - HuggingFace: model.layers.0.self_attn.q_proj.weight
/// - Meta: layers.0.attention.wq.weight
///
/// This class provides bidirectional mapping between these formats.

#include <string>
#include <vector>
#include <regex>
#include <functional>

namespace vesper::io {

/// Maps parameter names between Vesper and source model formats.
///
/// Example usage:
/// @code
///   WeightMapper mapper = WeightMapper::huggingface_llama();
///   std::string vesper_name = mapper.map_from_source("model.layers.0.self_attn.q_proj.weight");
///   // Returns: "layers.0.attention.wq.weight"
/// @endcode
class WeightMapper {
public:
    /// Creates an empty mapper (identity mapping).
    WeightMapper() = default;
    
    /// Adds a bidirectional mapping rule.
    ///
    /// @param vesper_pattern Regex pattern matching Vesper parameter names.
    ///        Capture groups ($1, $2, etc.) can be used in replacement.
    /// @param source_pattern Corresponding pattern in source format.
    ///        Can use $1, $2 for captured groups from vesper_pattern.
    ///
    /// Example:
    /// @code
    ///   mapper.add_rule(
    ///       R"(layers\.(\d+)\.attention\.wq\.weight)",
    ///       R"(model.layers.$1.self_attn.q_proj.weight)");
    /// @endcode
    void add_rule(const std::string& vesper_pattern, 
                  const std::string& source_pattern);
    
    /// Adds a simple direct mapping (no regex).
    void add_direct(const std::string& vesper_name,
                    const std::string& source_name);
    
    /// Maps a Vesper parameter name to the source model's tensor name.
    /// @param vesper_name Parameter name in Vesper format
    /// @return Corresponding name in source format, or vesper_name if no mapping
    std::string map_to_source(const std::string& vesper_name) const;
    
    /// Maps a source tensor name to Vesper parameter name.
    /// @param source_name Tensor name from source model
    /// @return Corresponding name in Vesper format, or source_name if no mapping
    std::string map_from_source(const std::string& source_name) const;
    
    /// Returns whether this mapper has any rules defined.
    bool empty() const { return to_source_rules_.empty() && direct_mappings_.empty(); }
    
    /// Returns the number of mapping rules.
    size_t num_rules() const { return to_source_rules_.size() + direct_mappings_.size(); }
    
    // ========================================================================
    // Predefined mappers for common model formats
    // ========================================================================
    
    /// Creates mapper for HuggingFace Llama format.
    /// Maps: Vesper ↔ HuggingFace Transformers Llama naming
    static WeightMapper huggingface_llama();
    
    /// Creates mapper for Meta's original Llama format.
    /// Note: Meta format is close to Vesper's internal naming.
    static WeightMapper meta_llama();
    
    /// Creates mapper for HuggingFace GPT-2 format.
    static WeightMapper huggingface_gpt2();
    
    /// Creates mapper for HuggingFace Mistral format.
    /// Similar to Llama but may have minor differences.
    static WeightMapper huggingface_mistral();
    
private:
    // Rule: (vesper_pattern_regex, source_template)
    std::vector<std::pair<std::regex, std::string>> to_source_rules_;
    // Rule: (source_pattern_regex, vesper_template)
    std::vector<std::pair<std::regex, std::string>> from_source_rules_;
    // Direct mappings (exact string match)
    std::vector<std::pair<std::string, std::string>> direct_mappings_;
};

/// Attempts to detect the weight format from tensor names.
/// @param tensor_names List of tensor names from a model file
/// @return Format string: "huggingface_llama", "meta_llama", "unknown"
std::string detect_weight_format(const std::vector<std::string>& tensor_names);

} // namespace vesper::io
