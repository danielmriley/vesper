#pragma once

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
void write_tiny_q4km(const std::string& path, std::uint32_t seed);
ModelWeights load_model(const std::string& path);

}  // namespace vesper
