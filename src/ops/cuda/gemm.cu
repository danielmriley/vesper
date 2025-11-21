#include <vesper/ops/gemm.h>
#include <stdexcept>

namespace vesper::ops {

void gemm_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& c, bool transA, bool transB) {
    throw std::runtime_error("GEMM not implemented for CUDA backend yet.");
}

}
