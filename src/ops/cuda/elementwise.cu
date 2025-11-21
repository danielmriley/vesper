#include <vesper/ops/elementwise.h>
#include <stdexcept>

namespace vesper::ops {

void add_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    throw std::runtime_error("add not implemented for CUDA backend yet.");
}

void sub_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    throw std::runtime_error("sub not implemented for CUDA backend yet.");
}

void mul_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    throw std::runtime_error("mul not implemented for CUDA backend yet.");
}

void div_cuda_dispatch(const Tensor& a, const Tensor& b, Tensor& out) {
    throw std::runtime_error("div not implemented for CUDA backend yet.");
}

void sum_rows_cuda_dispatch(const Tensor& input, Tensor& output) {
    throw std::runtime_error("sum_rows not implemented for CUDA backend yet.");
}

}
