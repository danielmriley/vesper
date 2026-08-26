#pragma once

#include "vesper/gguf.h"
#include "vesper/weights.h"

#include <cstdint>
#include <string>

namespace vesper {

void write_tiny_q8(const std::string& path, std::uint32_t seed);
void write_tiny_hybrid(const std::string& path, std::uint32_t seed);
void write_tiny_qwen35(const std::string& path, std::uint32_t seed);
void write_tiny_qwen35(const std::string& path, std::uint32_t seed, int nextn_layers);
void write_tiny_qwen35(const std::string& path, std::uint32_t seed, const ModelConfig& cfg);
void write_tiny_qwen35_ssm_aliases(const std::string& path, std::uint32_t seed);
void write_tiny_qwen35_pin_kv(const std::string& path, std::uint32_t seed);
void write_tiny_q4km(const std::string& path, std::uint32_t seed);
ModelConfig load_config(const GgufFile& file);
ModelConfig load_config(const std::string& path);
ModelWeights load_model(const std::string& path);

// True when the GGUF header matches ggml-org Qwen3.8-27B-Q4_K_M (convert.log).
// Tensor payloads are not required.
bool qwen38_27b_q4km_header_ok(const GgufFile& file);

}  // namespace vesper
