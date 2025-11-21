#pragma once
#include <vesper/core/tensor.h>
#include <vector>

namespace vesper::ops {
    // Stacks a vector of tensors along a new dimension `dim`.
    Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);
}
