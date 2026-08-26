#pragma once

#include "vesper/weights.h"

#include <cstdint>
#include <string>

namespace vesper {

void write_tiny_q8(const std::string& path, std::uint32_t seed);
ModelWeights load_model(const std::string& path);

}  // namespace vesper
