/// @file weight_mapper.cpp
/// @brief Implementation of weight name mapping between model formats.

#include <vesper/io/weight_mapper.h>
#include <stdexcept>
#include <algorithm>

namespace vesper::io {

namespace {

/// Applies regex substitution with capture groups.
/// Replaces $1, $2, etc. in template with captured groups from match.
std::string apply_substitution(const std::smatch& match, const std::string& template_str) {
    std::string result = template_str;
    
    // Replace $N with captured groups (up to 9 groups)
    for (size_t i = 1; i < match.size() && i <= 9; ++i) {
        std::string placeholder = "$" + std::to_string(i);
        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), match[i].str());
            pos += match[i].length();
        }
    }
    
    return result;
}

/// Creates the reverse pattern from a mapping rule.
/// Converts "model.layers.$1.self_attn.q_proj.weight" to regex matching pattern.
std::pair<std::regex, std::string> create_reverse_rule(
    const std::string& vesper_pattern,
    const std::string& source_pattern) 
{
    // Convert source pattern to regex by escaping regex chars and converting $N to (\d+) or (.+)
    std::string regex_str;
    std::string vesper_template;
    
    // Simple conversion: escape dots, replace $N with capture groups
    for (size_t i = 0; i < source_pattern.size(); ++i) {
        char c = source_pattern[i];
        if (c == '$' && i + 1 < source_pattern.size() && std::isdigit(source_pattern[i + 1])) {
            // $N -> capture group for numbers (commonly layer indices)
            regex_str += R"((\d+))";
            ++i; // skip the digit
        } else if (c == '.') {
            regex_str += R"(\.)";
        } else if (c == '(' || c == ')' || c == '[' || c == ']' || 
                   c == '{' || c == '}' || c == '+' || c == '*' ||
                   c == '?' || c == '^' || c == '|' || c == '\\') {
            regex_str += '\\';
            regex_str += c;
        } else {
            regex_str += c;
        }
    }
    
    // The vesper template just uses the same capture group references
    // Extract from the original vesper_pattern
    vesper_template = vesper_pattern;
    // Convert regex patterns to simple $N references
    // e.g., R"(layers\.(\d+)\.attention)" -> "layers.$1.attention"
    std::string simple_vesper;
    size_t group_num = 1;
    for (size_t i = 0; i < vesper_pattern.size(); ++i) {
        if (i + 1 < vesper_pattern.size() && vesper_pattern[i] == '\\' && vesper_pattern[i+1] == '.') {
            simple_vesper += '.';
            ++i;
        } else if (vesper_pattern[i] == '(' && vesper_pattern.find(")?", i) == std::string::npos) {
            // Start of capture group - find end
            size_t end = vesper_pattern.find(')', i);
            if (end != std::string::npos) {
                simple_vesper += "$" + std::to_string(group_num++);
                i = end;
            }
        } else if (vesper_pattern[i] != '\\') {
            simple_vesper += vesper_pattern[i];
        }
    }
    
    return {std::regex(regex_str), simple_vesper};
}

} // anonymous namespace

void WeightMapper::add_rule(const std::string& vesper_pattern, 
                            const std::string& source_pattern) {
    // Forward rule: vesper -> source
    to_source_rules_.emplace_back(std::regex(vesper_pattern), source_pattern);
    
    // Reverse rule: source -> vesper
    auto [source_regex, vesper_template] = create_reverse_rule(vesper_pattern, source_pattern);
    from_source_rules_.emplace_back(std::move(source_regex), vesper_template);
}

void WeightMapper::add_direct(const std::string& vesper_name,
                              const std::string& source_name) {
    direct_mappings_.emplace_back(vesper_name, source_name);
}

std::string WeightMapper::map_to_source(const std::string& vesper_name) const {
    // Check direct mappings first
    for (const auto& [vesper, source] : direct_mappings_) {
        if (vesper_name == vesper) {
            return source;
        }
    }
    
    // Try regex rules
    for (const auto& [pattern, replacement] : to_source_rules_) {
        std::smatch match;
        if (std::regex_match(vesper_name, match, pattern)) {
            return apply_substitution(match, replacement);
        }
    }
    
    // No mapping found, return as-is
    return vesper_name;
}

std::string WeightMapper::map_from_source(const std::string& source_name) const {
    // Check direct mappings first
    for (const auto& [vesper, source] : direct_mappings_) {
        if (source_name == source) {
            return vesper;
        }
    }
    
    // Try regex rules
    for (const auto& [pattern, replacement] : from_source_rules_) {
        std::smatch match;
        if (std::regex_match(source_name, match, pattern)) {
            return apply_substitution(match, replacement);
        }
    }
    
    // No mapping found, return as-is
    return source_name;
}

// ============================================================================
// Predefined mappers
// ============================================================================

WeightMapper WeightMapper::huggingface_llama() {
    WeightMapper mapper;
    
    // Embeddings
    mapper.add_direct("tok_embeddings.weight", "model.embed_tokens.weight");
    
    // Attention layers
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
    
    // Feed-forward layers
    mapper.add_rule(
        R"(layers\.(\d+)\.feed_forward\.w1\.weight)",
        R"(model.layers.$1.mlp.gate_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.feed_forward\.w2\.weight)",
        R"(model.layers.$1.mlp.down_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.feed_forward\.w3\.weight)",
        R"(model.layers.$1.mlp.up_proj.weight)");
    
    // Layer norms
    mapper.add_rule(
        R"(layers\.(\d+)\.attention_norm\.weight)",
        R"(model.layers.$1.input_layernorm.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.ffn_norm\.weight)",
        R"(model.layers.$1.post_attention_layernorm.weight)");
    
    // Final layers
    mapper.add_direct("norm.weight", "model.norm.weight");
    mapper.add_direct("output.weight", "lm_head.weight");
    
    return mapper;
}

WeightMapper WeightMapper::meta_llama() {
    // Meta's format is very close to Vesper's internal naming
    // Most names match directly, but there might be minor differences
    WeightMapper mapper;
    
    // Meta format uses nearly identical names to Vesper
    // No mapping needed for most parameters
    // Add any specific differences here if discovered
    
    return mapper;
}

WeightMapper WeightMapper::huggingface_gpt2() {
    WeightMapper mapper;
    
    // Embeddings
    mapper.add_direct("tok_embeddings.weight", "transformer.wte.weight");
    mapper.add_direct("pos_embeddings.weight", "transformer.wpe.weight");
    
    // Attention layers (GPT-2 uses combined qkv projection)
    mapper.add_rule(
        R"(layers\.(\d+)\.attention\.c_attn\.weight)",
        R"(transformer.h.$1.attn.c_attn.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.attention\.c_attn\.bias)",
        R"(transformer.h.$1.attn.c_attn.bias)");
    mapper.add_rule(
        R"(layers\.(\d+)\.attention\.c_proj\.weight)",
        R"(transformer.h.$1.attn.c_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.attention\.c_proj\.bias)",
        R"(transformer.h.$1.attn.c_proj.bias)");
    
    // Feed-forward layers
    mapper.add_rule(
        R"(layers\.(\d+)\.mlp\.c_fc\.weight)",
        R"(transformer.h.$1.mlp.c_fc.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.mlp\.c_fc\.bias)",
        R"(transformer.h.$1.mlp.c_fc.bias)");
    mapper.add_rule(
        R"(layers\.(\d+)\.mlp\.c_proj\.weight)",
        R"(transformer.h.$1.mlp.c_proj.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.mlp\.c_proj\.bias)",
        R"(transformer.h.$1.mlp.c_proj.bias)");
    
    // Layer norms
    mapper.add_rule(
        R"(layers\.(\d+)\.ln_1\.weight)",
        R"(transformer.h.$1.ln_1.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.ln_1\.bias)",
        R"(transformer.h.$1.ln_1.bias)");
    mapper.add_rule(
        R"(layers\.(\d+)\.ln_2\.weight)",
        R"(transformer.h.$1.ln_2.weight)");
    mapper.add_rule(
        R"(layers\.(\d+)\.ln_2\.bias)",
        R"(transformer.h.$1.ln_2.bias)");
    
    // Final layer norm
    mapper.add_direct("ln_f.weight", "transformer.ln_f.weight");
    mapper.add_direct("ln_f.bias", "transformer.ln_f.bias");
    
    // LM head (GPT-2 ties weights with embeddings, but some checkpoints have separate)
    mapper.add_direct("lm_head.weight", "lm_head.weight");
    
    return mapper;
}

WeightMapper WeightMapper::huggingface_mistral() {
    // Mistral uses nearly identical naming to Llama
    WeightMapper mapper = huggingface_llama();
    
    // Add any Mistral-specific differences here
    // (Currently Mistral uses same naming as Llama)
    
    return mapper;
}

// ============================================================================
// Format detection
// ============================================================================

std::string detect_weight_format(const std::vector<std::string>& tensor_names) {
    // Check for HuggingFace Llama indicators
    for (const auto& name : tensor_names) {
        if (name.find("model.embed_tokens.weight") != std::string::npos ||
            name.find("model.layers.") != std::string::npos) {
            return "huggingface_llama";
        }
    }
    
    // Check for Meta Llama indicators
    for (const auto& name : tensor_names) {
        if (name.find("tok_embeddings.weight") != std::string::npos) {
            return "meta_llama";
        }
    }
    
    // Check for GPT-2 indicators
    for (const auto& name : tensor_names) {
        if (name.find("transformer.wte.weight") != std::string::npos ||
            name.find("transformer.h.") != std::string::npos) {
            return "huggingface_gpt2";
        }
    }
    
    return "unknown";
}

} // namespace vesper::io
