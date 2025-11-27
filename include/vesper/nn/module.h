#pragma once

#include <vesper/core/tensor.h>
#include <vesper/core/state_dict.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <functional>

namespace vesper::nn {

// Forward declaration for ModuleList (defined in module_list.h)
template<typename T> class ModuleList;

/**
 * Base class for all neural network modules.
 * 
 * IMPORTANT NOTES:
 * 
 * 1. POINTER-BASED REGISTRATION: This class uses pointer-based registration for proper
 *    `to(device)` behavior. Because of this, modules should NOT be stored in containers
 *    that may reallocate (like std::vector without reserve()). When using std::vector<Module>,
 *    always call reserve() beforehand, or use std::vector<std::unique_ptr<Module>> instead.
 * 
 * 2. INHERITANCE: If you derive from a module that already registers parameters (e.g.,
 *    class MyLinear : Linear), the base class parameters are automatically registered
 *    via the base constructor. Derived-class-specific parameters need explicit registration.
 * 
 * 3. THREADING: Module operations (to(), parameters(), etc.) are NOT thread-safe.
 *    Do not call these methods concurrently on the same module.
 * 
 * Example safe usage:
 *   std::vector<MyModule> modules;
 *   modules.reserve(num_modules);  // IMPORTANT: prevents reallocation
 *   for (int i = 0; i < num_modules; ++i) modules.emplace_back(...);
 * 
 * Example using unique_ptr (also safe):
 *   std::vector<std::unique_ptr<MyModule>> modules;
 *   for (int i = 0; i < num_modules; ++i) modules.push_back(std::make_unique<MyModule>(...));
 */
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
    // Returns copies of tensors (for backward compatibility with optimizers)
    std::vector<Tensor> parameters();
    
    // Gathers pointers to all parameters (more efficient, no copies)
    // Use this when you need to modify parameters in-place
    std::vector<Tensor*> parameters_ptrs();

    // Zeros the gradients of all parameters
    void zero_grad();

    // Returns a dictionary containing a whole state of the module.
    // Recursively includes parameters from submodules with dot notation (e.g. "sub.bias").
    StateDict state_dict() const;

    // Copies parameters and buffers from state_dict into this module and its descendants.
    void load_state_dict(const StateDict& state_dict);

    // Training mode control
    // Sets the module in training mode (affects dropout, batch norm, etc.)
    void train(bool mode = true);
    
    // Sets the module in evaluation mode (equivalent to train(false))
    void eval();
    
    // Returns whether the module is in training mode
    bool is_training() const { return training_; }

    // Move all parameters to a specific device
    // non_blocking: if true, may return before copy completes (for async streams)
    void to(Device device, bool non_blocking = false);

    // Get named parameters (pointers) for modification
    // Keys use dot notation for nested modules (e.g., "layer1.weight")
    std::map<std::string, Tensor*> named_parameters_ptrs();
    
    // Get named parameters (copies) for read-only access
    std::map<std::string, Tensor> named_parameters() const;

protected:
    // Methods for subclasses to register their components
    
    // Register a parameter using a raw pointer (basic version)
    // IMPORTANT: Pass a pointer to the actual member variable, not a copy!
    // This allows to() to update the member variable directly.
    // Throws if a parameter with the same name is already registered.
    void register_parameter(const std::string& name, Tensor* param);
    
    // Register a parameter using pointer-to-member syntax (type-safe version)
    // Example: register_parameter<Linear>("weight", &Linear::weight);
    // This provides compile-time type checking and IDE autocomplete.
    template<typename Derived>
    void register_parameter(const std::string& name, Tensor Derived::*member_ptr) {
        static_assert(std::is_base_of_v<Module, Derived>, 
            "Derived must inherit from Module");
        // Get the actual member from this object
        Derived* self = static_cast<Derived*>(this);
        Tensor* param = &(self->*member_ptr);
        register_parameter(name, param);
    }
    
    // For submodules using raw pointer (basic version)
    template<typename T>
    void register_module(const std::string& name, T* module_ptr) {
        static_assert(std::is_base_of_v<Module, T>, "T must derive from Module");
        _module_ptrs[name] = static_cast<Module*>(module_ptr);
        _modules[name] = std::shared_ptr<Module>(module_ptr, [](Module*){});  // Non-owning shared_ptr for compatibility
    }
    
    // Register a submodule using pointer-to-member syntax (type-safe version)
    // Example: register_module<TransformerBlock>("attn", &TransformerBlock::attention);
    template<typename Derived, typename ModuleType>
    void register_module(const std::string& name, ModuleType Derived::*member_ptr) {
        static_assert(std::is_base_of_v<Module, Derived>, 
            "Derived must inherit from Module");
        static_assert(std::is_base_of_v<Module, ModuleType>, 
            "ModuleType must inherit from Module");
        Derived* self = static_cast<Derived*>(this);
        ModuleType* module = &(self->*member_ptr);
        _module_ptrs[name] = static_cast<Module*>(module);
        _modules[name] = std::shared_ptr<Module>(module, [](Module*){});
    }
    
    // Register a ModuleList with the parent module
    // Each element in the list is registered as "name.0", "name.1", etc.
    // The ModuleList pointer is stored for to()/train()/eval() propagation.
    template<typename T>
    void register_module_list(const std::string& name, ModuleList<T>* list) {
        static_assert(std::is_base_of_v<Module, T>, "T must derive from Module");
        _module_lists[name] = reinterpret_cast<void*>(list);
        _module_list_ops[name] = {
            // to() operation
            [](void* ptr, Device device, bool non_blocking) {
                static_cast<ModuleList<T>*>(ptr)->to(device, non_blocking);
            },
            // train() operation
            [](void* ptr, bool mode) {
                static_cast<ModuleList<T>*>(ptr)->train(mode);
            },
            // parameters() operation
            [](void* ptr) -> std::vector<Tensor> {
                return static_cast<ModuleList<T>*>(ptr)->parameters();
            },
            // parameters_ptrs() operation
            [](void* ptr) -> std::vector<Tensor*> {
                return static_cast<ModuleList<T>*>(ptr)->parameters_ptrs();
            },
            // zero_grad() operation
            [](void* ptr) {
                static_cast<ModuleList<T>*>(ptr)->zero_grad();
            },
            // size() operation
            [](void* ptr) -> size_t {
                return static_cast<ModuleList<T>*>(ptr)->size();
            },
            // module_ptr_at() operation - get Module* at index
            [](void* ptr, size_t idx) -> Module* {
                return static_cast<Module*>(static_cast<ModuleList<T>*>(ptr)->get_ptr(idx));
            }
        };
    }
    
    // Training mode flag
    bool training_ = true;

private:
    // Helper for recursive state_dict generation
    void _gather_state_dict(StateDict& out, const std::string& prefix) const;
    
    // Helper for recursive loading
    void _load_from_state_dict(const StateDict& state_dict, const std::string& prefix);

    // Store pointers to actual member tensors (not copies!)
    std::map<std::string, Tensor*> _parameters;
    
    // For submodules: raw pointers to member variables for device movement
    std::map<std::string, Module*> _module_ptrs;
    
    // Keep shared_ptr map for compatibility with existing code that expects it
    std::map<std::string, std::shared_ptr<Module>> _modules;
    
    // Type-erased operations for ModuleList
    struct ModuleListOps {
        std::function<void(void*, Device, bool)> to_device;
        std::function<void(void*, bool)> train;
        std::function<std::vector<Tensor>(void*)> parameters;
        std::function<std::vector<Tensor*>(void*)> parameters_ptrs;
        std::function<void(void*)> zero_grad;
        std::function<size_t(void*)> size;
        std::function<Module*(void*, size_t)> module_ptr_at;
    };
    
    // Storage for registered ModuleLists (type-erased)
    std::map<std::string, void*> _module_lists;
    std::map<std::string, ModuleListOps> _module_list_ops;
};

} // namespace vesper::nn
