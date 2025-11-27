#include <vesper/nn/module.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <memory>

// --- Test Case 1: Nested Modules ---
class LeafModule : public vesper::nn::Module {
public:
    LeafModule() 
        : p(vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f))
    {
        register_parameter("p", &p);  // Pass pointer to member
    }
    vesper::Tensor p;
};

class ContainerModule : public vesper::nn::Module {
public:
    ContainerModule() {
        register_module("leaf", &leaf);  // Pass pointer to member
    }
    LeafModule leaf;  // Direct member, not shared_ptr
};

void test_nested_modules() {
    std::cout << "Testing nested modules..." << std::endl;
    ContainerModule container;
    auto params = container.parameters();
    
    assert(params.size() == 1);
    // Check if we got the leaf's parameter
    std::vector<float> data(1);
    params[0].copy_to_host(data.data());
    assert(data[0] == 1.0f);
    std::cout << "Nested modules test passed!" << std::endl;
}

// --- Test Case 2: Shared Parameters ---
// Note: Shared parameters now require both to point to the same member variable
class SharedParamModule : public vesper::nn::Module {
public:
    SharedParamModule() 
        : shared_weight(vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f))
    {
        register_parameter("w1", &shared_weight);
        register_parameter("w2", &shared_weight); // Register same tensor again - both point to same member
    }
    vesper::Tensor shared_weight;
};

void test_shared_parameters() {
    std::cout << "Testing shared parameters..." << std::endl;
    SharedParamModule model;
    auto params = model.parameters();
    
    // Should return both registrations, but they point to the same data
    assert(params.size() == 2);
    // Wrappers are copies, but point to same storage
    
    // Modify one, check other
    std::vector<float> new_val = {5.0f};
    params[0].copy_from_host(new_val.data());
    
    std::vector<float> check_val(1);
    params[1].copy_to_host(check_val.data());
    assert(check_val[0] == 5.0f);
    
    std::cout << "Shared parameters test passed!" << std::endl;
}

int main() {
    test_nested_modules();
    test_shared_parameters();
    return 0;
}
