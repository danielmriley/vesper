#include <vesper/nn/functional.h>
#include <stdexcept>

namespace vesper::nn::functional {

void sigmoid_cuda_dispatch(const Tensor& input, Tensor& output) {
    throw std::runtime_error("sigmoid not implemented for CUDA backend yet.");
}

void relu_cuda_dispatch(const Tensor& input, Tensor& output) {
    throw std::runtime_error("relu not implemented for CUDA backend yet.");
}

}
