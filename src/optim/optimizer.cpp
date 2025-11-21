#include <vesper/optim/optimizer.h>
#include <vesper/core/factories.h> // For zeros()

namespace vesper::optim {

Optimizer::Optimizer(std::vector<Tensor*> params) : params_(std::move(params)) {}

void Optimizer::zero_grad() {
    for (auto* param : params_) {
        if (param && param->requires_grad()) {
            // This re-creates the gradient tensor, effectively zeroing it.
            // A more efficient `fill_(0)` method is a future optimization.
            param->grad() = zeros(param->shape(), param->dtype(), param->device());
        }
    }
}

} // namespace vesper::optim
