#include <vesper/ops/comparison.h>
#include <stdexcept>

namespace vesper::ops {

void greater_than_cuda_dispatch(const Tensor& a, float b, Tensor& out) {
    throw std::runtime_error("greater_than not implemented for CUDA backend yet.");
}

}
