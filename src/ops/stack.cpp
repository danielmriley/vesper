#include <vesper/ops/stack.h>
#include <vesper/core/factories.h>
#include <stdexcept>

namespace vesper::ops {

Tensor stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot stack an empty list of tensors.");
    }
    
    if (dim != 0) {
        throw std::runtime_error("stack currently only supports dim=0.");
    }

    // 1. All tensors must have the same shape and device
    const auto& first_shape = tensors[0].shape();
    const auto& device = tensors[0].device();
    const auto& dtype = tensors[0].dtype();
    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].shape() != first_shape || tensors[i].device() != device) {
            throw std::runtime_error("All tensors in stack must have the same shape and device.");
        }
    }

    // 2. Determine the output shape
    auto output_shape = first_shape;
    output_shape.insert(output_shape.begin() + dim, tensors.size());
    Tensor output = empty(output_shape, dtype, device);

    // 3. Copy data
    // Optimization: Use Tensor::copy_from which handles DeviceToDevice copy
    for (size_t i = 0; i < tensors.size(); ++i) {
        // output.slice(i) creates a view of the i-th slice
        // Since output is contiguous, slice(i) (where dim=0) is also contiguous in memory
        // (just offset). So copy_from will work efficiently.
        // slice(i) returns a Tensor view.
        output.slice(i).copy_from(tensors[i]);
    }
    
    // NOTE: This implementation does not support autograd for simplicity.
    return output;
}

} // namespace vesper::ops
