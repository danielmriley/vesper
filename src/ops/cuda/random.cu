#include <vesper/ops/random.h>
#include <stdexcept>

namespace vesper::ops {

void uniform_cuda_dispatch(Tensor& tensor, float min, float max) {
    throw std::runtime_error("uniform_ not implemented for CUDA backend yet.");
}

}
