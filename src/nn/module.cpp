#include <vesper/nn/module.h>
#include <vesper/core/factories.h>

namespace vesper::nn {

void Module::register_parameter(const std::string& name, Tensor* param) {
    // Check for duplicate name registration
    if (_parameters.count(name)) {
        throw std::runtime_error(
            "register_parameter: parameter with name '" + name + 
            "' is already registered. Use a unique name for each parameter."
        );
    }
    
    param->set_requires_grad(true); // All registered parameters are trainable
    _parameters[name] = param;      // Store pointer to actual member variable
}

std::vector<Tensor*> Module::parameters_ptrs() {
    std::vector<Tensor*> params;
    // Add this module's own parameters (pointers, not copies)
    for (auto& [name, param_ptr] : _parameters) {
        params.push_back(param_ptr);
    }
    // Recursively add parameters from sub-modules
    for (auto const& [name, module_ptr] : _module_ptrs) {
        auto sub_params = module_ptr->parameters_ptrs();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    // Add parameters from ModuleLists
    for (auto& [name, list_ptr] : _module_lists) {
        auto& ops = _module_list_ops[name];
        auto sub_params = ops.parameters_ptrs(list_ptr);
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
}

std::vector<Tensor> Module::parameters() {
    std::vector<Tensor> params;
    // Add this module's own parameters (dereference pointers)
    for (auto& [name, param_ptr] : _parameters) {
        params.push_back(*param_ptr);
    }
    // Recursively add parameters from sub-modules
    for (auto const& [name, module_ptr] : _module_ptrs) {
        auto sub_params = module_ptr->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    // Add parameters from ModuleLists
    for (auto& [name, list_ptr] : _module_lists) {
        auto& ops = _module_list_ops[name];
        auto sub_params = ops.parameters(list_ptr);
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
}

std::map<std::string, Tensor*> Module::named_parameters_ptrs() {
    std::map<std::string, Tensor*> result;
    // Add this module's own parameters
    for (auto& [name, param_ptr] : _parameters) {
        result[name] = param_ptr;
    }
    // Recursively add parameters from sub-modules with prefixes
    for (auto& [mod_name, module_ptr] : _module_ptrs) {
        auto sub_params = module_ptr->named_parameters_ptrs();
        for (auto& [param_name, param_ptr] : sub_params) {
            result[mod_name + "." + param_name] = param_ptr;
        }
    }
    return result;
}

std::map<std::string, Tensor> Module::named_parameters() const {
    std::map<std::string, Tensor> result;
    // Add this module's own parameters (dereference pointers for const access)
    for (const auto& [name, param_ptr] : _parameters) {
        result[name] = *param_ptr;
    }
    // Recursively add parameters from sub-modules with prefixes
    for (const auto& [mod_name, module_ptr] : _module_ptrs) {
        auto sub_params = module_ptr->named_parameters();
        for (const auto& [param_name, param] : sub_params) {
            result[mod_name + "." + param_name] = param;
        }
    }
    return result;
}

void Module::zero_grad() {
    for (auto param_ptr : this->parameters_ptrs()) {
        if (param_ptr->requires_grad()) {
            // Use zero_() for efficient in-place zeroing
            if (param_ptr->grad().defined()) {
                param_ptr->grad().zero_();
            }
        }
    }
}

void Module::train(bool mode) {
    training_ = mode;
    // Recursively set training mode for all submodules
    for (auto& [name, module_ptr] : _module_ptrs) {
        module_ptr->train(mode);
    }
    // Set training mode for ModuleLists
    for (auto& [name, list_ptr] : _module_lists) {
        _module_list_ops[name].train(list_ptr, mode);
    }
}

void Module::eval() {
    train(false);
}

void Module::to(Device device, bool non_blocking) {
    // Move all parameters to the specified device using in-place modification
    for (auto& [name, param_ptr] : _parameters) {
        param_ptr->to_(device, non_blocking);  // In-place device transfer
    }
    // Recursively move submodules (which will update their member variables)
    for (auto& [name, module_ptr] : _module_ptrs) {
        module_ptr->to(device, non_blocking);
    }
    // Move ModuleLists
    for (auto& [name, list_ptr] : _module_lists) {
        _module_list_ops[name].to_device(list_ptr, device, non_blocking);
    }
}

StateDict Module::state_dict() const {
    StateDict out;
    _gather_state_dict(out, "");
    return out;
}

void Module::_gather_state_dict(StateDict& out, const std::string& prefix) const {
    for (const auto& [name, param_ptr] : _parameters) {
        out[prefix + name] = *param_ptr;  // Dereference pointer
    }
    for (const auto& [name, module_ptr] : _module_ptrs) {
        module_ptr->_gather_state_dict(out, prefix + name + ".");
    }
    // Handle ModuleLists - each element gets indexed like "layers.0.", "layers.1.", etc.
    for (const auto& [name, list_ptr] : _module_lists) {
        const auto& ops = _module_list_ops.at(name);
        size_t count = ops.size(const_cast<void*>(list_ptr));
        for (size_t i = 0; i < count; ++i) {
            Module* mod = ops.module_ptr_at(const_cast<void*>(list_ptr), i);
            mod->_gather_state_dict(out, prefix + name + "." + std::to_string(i) + ".");
        }
    }
}

void Module::load_state_dict(const StateDict& state_dict) {
    _load_from_state_dict(state_dict, "");
}

void Module::_load_from_state_dict(const StateDict& state_dict, const std::string& prefix) {
    for (auto& [name, param_ptr] : _parameters) {
        std::string key = prefix + name;
        if (state_dict.count(key)) {
            // Use copy_() for efficient in-place data update
            // This preserves the parameter object (and its grad link), just updates data.
            param_ptr->copy_(state_dict.at(key));
        }
        // Else: warning? For now silent skip (non-strict).
    }
    for (auto& [name, module_ptr] : _module_ptrs) {
        module_ptr->_load_from_state_dict(state_dict, prefix + name + ".");
    }
    // Handle ModuleLists
    for (auto& [name, list_ptr] : _module_lists) {
        auto& ops = _module_list_ops[name];
        size_t count = ops.size(list_ptr);
        for (size_t i = 0; i < count; ++i) {
            Module* mod = ops.module_ptr_at(list_ptr, i);
            mod->_load_from_state_dict(state_dict, prefix + name + "." + std::to_string(i) + ".");
        }
    }
}

} // namespace vesper::nn
