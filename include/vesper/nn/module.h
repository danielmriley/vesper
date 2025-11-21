#pragma once

#include <vesper/core/tensor.h>
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
    std::vector<Tensor*> parameters();

    // Zeros the gradients of all parameters
    void zero_grad();

protected:
    // Methods for subclasses to register their components
    void register_parameter(const std::string& name, Tensor& param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

private:
    std::map<std::string, Tensor*> _parameters;
    std::map<std::string, std::shared_ptr<Module>> _modules;
};

} // namespace vesper::nn
