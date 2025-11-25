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

std::vector<Tensor> Module::parameters() {
    std::vector<Tensor> params;
    // Add this module's own parameters
    for (auto& [name, param] : _parameters) {
        params.push_back(param);
    }
    // Recursively add parameters from sub-modules
    for (auto const& [name, module] : _modules) {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
}

std::map<std::string, Tensor> Module::named_parameters() const {
    std::map<std::string, Tensor> result;
    // Add this module's own parameters
    for (const auto& [name, param] : _parameters) {
        result[name] = param;
    }
    // Recursively add parameters from sub-modules with prefixes
    for (const auto& [mod_name, module] : _modules) {
        auto sub_params = module->named_parameters();
        for (const auto& [param_name, param] : sub_params) {
            result[mod_name + "." + param_name] = param;
        }
    }
    return result;
}

void Module::zero_grad() {
    for (auto param : this->parameters()) {
        if (param.requires_grad()) {
            // Re-create the gradient tensor, effectively zeroing it.
            // A more efficient `fill_(0)` method is a future optimization.
            param.grad() = zeros(param.shape(), param.dtype(), param.device());
        }
    }
}

void Module::train(bool mode) {
    training_ = mode;
    // Recursively set training mode for all submodules
    for (auto& [name, module] : _modules) {
        module->train(mode);
    }
}

void Module::eval() {
    train(false);
}

void Module::to(Device device) {
    // Move all parameters to the specified device
    for (auto& [name, param] : _parameters) {
        _parameters[name] = param.to(device);
    }
    // Recursively move submodules
    for (auto& [name, module] : _modules) {
        module->to(device);
    }
}

StateDict Module::state_dict() const {
    StateDict out;
    _gather_state_dict(out, "");
    return out;
}

void Module::_gather_state_dict(StateDict& out, const std::string& prefix) const {
    for (const auto& [name, param] : _parameters) {
        out[prefix + name] = param;
    }
    for (const auto& [name, module] : _modules) {
        module->_gather_state_dict(out, prefix + name + ".");
    }
}

void Module::load_state_dict(const StateDict& state_dict) {
    _load_from_state_dict(state_dict, "");
}

void Module::_load_from_state_dict(const StateDict& state_dict, const std::string& prefix) {
    for (auto& [name, param] : _parameters) {
        std::string key = prefix + name;
        if (state_dict.count(key)) {
            // Copy data from state_dict tensor to parameter tensor
            // This preserves the parameter object (and its grad link), just updates data.
            param.copy_from(state_dict.at(key));
        }
        // Else: warning? For now silent skip (non-strict).
    }
    for (auto& [name, module] : _modules) {
        module->_load_from_state_dict(state_dict, prefix + name + ".");
    }
}

} // namespace vesper::nn
