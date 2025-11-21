#include <vesper/ops/reduction.h>
#include <stdexcept>

namespace vesper::ops {

void sum_cuda_dispatch(const Tensor& input, Tensor& output) {
    throw std::runtime_error("sum not implemented for CUDA backend yet.");
}

}
