#include <vesper/nn/module.h>
#include <vesper/core/factories.h>

namespace vesper::nn {

void Module::register_parameter(const std::string& name, Tensor param) {
    param.set_requires_grad(true); // All registered parameters are trainable
    _parameters[name] = param;
}

void Module::register_module(const std::string& name, std::shared_ptr<Module> module) {
    _modules[name] = std::move(module);
}

std::vector<Tensor*> Module::parameters() {
    std::vector<Tensor*> params;
    // Add this module's own parameters
    for (auto& [name, param] : _parameters) {
        params.push_back(&param);
    }
    // Recursively add parameters from sub-modules
    for (auto const& [name, module] : _modules) {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
}

void Module::zero_grad() {
    for (auto* param : this->parameters()) {
        if (param->requires_grad()) {
            // Re-create the gradient tensor, effectively zeroing it.
            // A more efficient `fill_(0)` method is a future optimization.
            param->grad() = zeros(param->shape(), param->dtype(), param->device());
        }
    }
}

} // namespace vesper::nn
