#pragma once

#include <vesper/core/tensor.h>
#include <vesper/core/state_dict.h>
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace vesper::nn {

class Module : public std::enable_shared_from_this<Module> {
public:
    virtual ~Module() = default;

    // The forward pass must be implemented by all subclasses
    virtual Tensor forward(const Tensor& input) {
        throw std::runtime_error("Forward not implemented.");
    };

    // Syntactic sugar to make modules callable: `model(input)`
    Tensor operator()(const Tensor& input) {
        return this->forward(input);
    }

    // Gathers all parameters from this module and its sub-modules
    std::vector<Tensor> parameters();

    // Zeros the gradients of all parameters
    void zero_grad();

    // Returns a dictionary containing a whole state of the module.
    // Recursively includes parameters from submodules with dot notation (e.g. "sub.bias").
    StateDict state_dict() const;

    // Copies parameters and buffers from state_dict into this module and its descendants.
    void load_state_dict(const StateDict& state_dict);

protected:
    // Methods for subclasses to register their components
    void register_parameter(const std::string& name, Tensor param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

private:
    // Helper for recursive state_dict generation
    void _gather_state_dict(StateDict& out, const std::string& prefix) const;
    
    // Helper for recursive loading
    void _load_from_state_dict(const StateDict& state_dict, const std::string& prefix);

    std::map<std::string, Tensor> _parameters;
    std::map<std::string, std::shared_ptr<Module>> _modules;
};

} // namespace vesper::nn
