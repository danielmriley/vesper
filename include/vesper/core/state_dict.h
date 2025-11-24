#pragma once
#include <vesper/core/tensor.h>
#include <map>
#include <string>

namespace vesper {

// A dictionary mapping parameter names to Tensors.
// Used for saving/loading models and optimizers.
using StateDict = std::map<std::string, Tensor>;

} // namespace vesper
