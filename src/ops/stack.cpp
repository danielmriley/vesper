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

    // 3. Copy data (slow, CPU-based implementation)
    size_t single_tensor_size = tensors[0].numel();
    size_t single_tensor_bytes = single_tensor_size * GetDTypeSize(dtype);
    std::vector<char> host_buffer(output.numel() * GetDTypeSize(dtype));

    for (size_t i = 0; i < tensors.size(); ++i) {
        // Copy each tensor's data into the correct slice of the host buffer
        tensors[i].copy_to_host(host_buffer.data() + i * single_tensor_bytes);
    }
    
    // Copy the entire collated buffer to the output tensor
    output.copy_from_host(host_buffer.data());

    // NOTE: This implementation does not support autograd for simplicity.
    return output;
}

} // namespace vesper::ops
