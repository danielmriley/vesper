/**
 * @file test_module_list.cpp
 * @brief Tests for ModuleList<T> container
 */

#include <vesper/nn/module_list.h>
#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <vesper/core/factories.h>
#include <vesper/ops/gemm.h>
#include <iostream>
#include <cassert>

using namespace vesper;
using namespace vesper::nn;

// Simple test module
class SimpleLayer : public Module {
public:
    SimpleLayer(int64_t in_features, int64_t out_features)
        : weight(randn({out_features, in_features}, DType::Float32, Device::CPU, true))
    {
        register_parameter("weight", &weight);
    }
    
    Tensor forward(const Tensor& input) override {
        return ops::matmul(input, weight.transpose(0, 1));
    }
    
    Tensor weight;
};

// Model using ModuleList
class StackedLayers : public Module {
public:
    StackedLayers(int num_layers, int64_t features) {
        for (int i = 0; i < num_layers; ++i) {
            layers.emplace_back(features, features);
        }
        register_module_list("layers", &layers);
    }
    
    Tensor forward(const Tensor& input) override {
        Tensor x = input;
        for (auto& layer : layers) {
            x = layer.forward(x);
        }
        return x;
    }
    
    ModuleList<SimpleLayer> layers;
};

void test_module_list_basic() {
    std::cout << "Testing ModuleList basic operations..." << std::endl;
    
    ModuleList<SimpleLayer> layers;
    assert(layers.empty());
    assert(layers.size() == 0);
    
    // Append modules
    layers.emplace_back(4, 4);
    layers.emplace_back(4, 4);
    layers.emplace_back(4, 4);
    
    assert(layers.size() == 3);
    assert(!layers.empty());
    
    // Indexing
    SimpleLayer& first = layers[0];
    SimpleLayer& last = layers.back();
    assert(&first == &layers.front());
    assert(last.weight.shape()[0] == 4);
    
    std::cout << "  PASSED: Basic operations" << std::endl;
}

void test_module_list_iteration() {
    std::cout << "Testing ModuleList iteration..." << std::endl;
    
    ModuleList<SimpleLayer> layers;
    for (int i = 0; i < 5; ++i) {
        layers.emplace_back(3, 3);
    }
    
    // Range-based for loop
    int count = 0;
    for (auto& layer : layers) {
        assert(layer.weight.shape()[0] == 3);
        count++;
    }
    assert(count == 5);
    
    // Const iteration
    const ModuleList<SimpleLayer>& const_layers = layers;
    count = 0;
    for (const auto& layer : const_layers) {
        assert(layer.weight.defined());
        count++;
    }
    assert(count == 5);
    
    std::cout << "  PASSED: Iteration" << std::endl;
}

void test_module_list_parameters() {
    std::cout << "Testing ModuleList parameter collection..." << std::endl;
    
    ModuleList<SimpleLayer> layers;
    layers.emplace_back(2, 2);
    layers.emplace_back(2, 2);
    
    auto params = layers.parameters();
    assert(params.size() == 2);  // One weight per layer
    
    auto param_ptrs = layers.parameters_ptrs();
    assert(param_ptrs.size() == 2);
    
    std::cout << "  PASSED: Parameter collection" << std::endl;
}

void test_module_list_to_device() {
    std::cout << "Testing ModuleList device transfer..." << std::endl;
    
    ModuleList<SimpleLayer> layers;
    layers.emplace_back(2, 2);
    layers.emplace_back(2, 2);
    
    // All start on CPU
    assert(layers[0].weight.device() == Device::CPU);
    assert(layers[1].weight.device() == Device::CPU);
    
#if USE_HIP_BACKEND
    layers.to(Device::HIP);
    assert(layers[0].weight.device() == Device::HIP);
    assert(layers[1].weight.device() == Device::HIP);
    
    layers.to(Device::CPU);
    assert(layers[0].weight.device() == Device::CPU);
    
    std::cout << "  PASSED: Device transfer (HIP)" << std::endl;
#else
    std::cout << "  SKIPPED: HIP backend not available" << std::endl;
#endif
}

void test_stacked_model_registration() {
    std::cout << "Testing StackedLayers model with ModuleList..." << std::endl;
    
    StackedLayers model(3, 4);  // 3 layers, 4 features each
    
    // Check parameters are collected from all layers
    auto params = model.parameters();
    assert(params.size() == 3);  // One weight per layer
    
    // Check state dict has correct keys
    auto sd = model.state_dict();
    assert(sd.count("layers.0.weight") == 1);
    assert(sd.count("layers.1.weight") == 1);
    assert(sd.count("layers.2.weight") == 1);
    
    std::cout << "  PASSED: StackedLayers model" << std::endl;
}

void test_stacked_model_to_device() {
    std::cout << "Testing StackedLayers to(device)..." << std::endl;
    
    StackedLayers model(2, 4);
    
    // Initial state
    assert(model.layers[0].weight.device() == Device::CPU);
    assert(model.layers[1].weight.device() == Device::CPU);
    
#if USE_HIP_BACKEND
    // Move entire model to HIP
    model.to(Device::HIP);
    
    assert(model.layers[0].weight.device() == Device::HIP);
    assert(model.layers[1].weight.device() == Device::HIP);
    
    // Forward should work
    Tensor x = randn({2, 4}, DType::Float32, Device::HIP, false);
    Tensor y = model.forward(x);
    assert(y.device() == Device::HIP);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 4);
    
    std::cout << "  PASSED: StackedLayers to(device) (HIP)" << std::endl;
#else
    std::cout << "  SKIPPED: HIP backend not available" << std::endl;
#endif
}

void test_stacked_model_state_dict() {
    std::cout << "Testing StackedLayers state_dict save/load..." << std::endl;
    
    // Create and modify a model
    StackedLayers model1(2, 3);
    
    // Set known values in weights
    std::vector<float> known_values(9, 1.5f);  // 3x3 = 9
    model1.layers[0].weight.copy_from_host(known_values.data());
    
    // Save state dict
    auto sd = model1.state_dict();
    
    // Create new model and load
    StackedLayers model2(2, 3);
    model2.load_state_dict(sd);
    
    // Verify weights match
    std::vector<float> loaded(9);
    model2.layers[0].weight.copy_to_host(loaded.data());
    for (int i = 0; i < 9; ++i) {
        assert(std::abs(loaded[i] - 1.5f) < 1e-5f);
    }
    
    std::cout << "  PASSED: State dict save/load" << std::endl;
}

void test_module_list_train_eval() {
    std::cout << "Testing ModuleList train/eval mode..." << std::endl;
    
    StackedLayers model(3, 4);
    
    // Default is training mode
    assert(model.is_training());
    
    model.eval();
    assert(!model.is_training());
    // Note: SimpleLayer doesn't have is_training() check, but layers inherit training_ flag
    
    model.train();
    assert(model.is_training());
    
    std::cout << "  PASSED: Train/eval mode" << std::endl;
}

void test_module_list_with_linear() {
    std::cout << "Testing ModuleList with nn::Linear..." << std::endl;
    
    // Create MLP using ModuleList<Linear>
    class MLP : public Module {
    public:
        MLP(int64_t in_features, int64_t hidden, int64_t out_features) {
            layers.emplace_back(in_features, hidden);
            layers.emplace_back(hidden, hidden);
            layers.emplace_back(hidden, out_features);
            register_module_list("layers", &layers);
        }
        
        Tensor forward(const Tensor& x) override {
            Tensor out = x;
            for (size_t i = 0; i < layers.size(); ++i) {
                out = layers[i].forward(out);
                if (i < layers.size() - 1) {
                    // Could add activation here
                }
            }
            return out;
        }
        
        ModuleList<Linear> layers;
    };
    
    MLP mlp(10, 20, 5);
    
    // Check parameters: each Linear has weight + bias = 2 params, 3 layers = 6 params
    auto params = mlp.parameters();
    assert(params.size() == 6);
    
    // Check state dict keys
    auto sd = mlp.state_dict();
    assert(sd.count("layers.0.weight") == 1);
    assert(sd.count("layers.0.bias") == 1);
    assert(sd.count("layers.1.weight") == 1);
    assert(sd.count("layers.1.bias") == 1);
    assert(sd.count("layers.2.weight") == 1);
    assert(sd.count("layers.2.bias") == 1);
    
    // Forward pass
    Tensor x = randn({4, 10}, DType::Float32, Device::CPU, false);
    Tensor y = mlp.forward(x);
    assert(y.shape()[0] == 4);
    assert(y.shape()[1] == 5);
    
    std::cout << "  PASSED: ModuleList<Linear> MLP" << std::endl;
}

void test_module_list_named_parameters() {
    std::cout << "Testing ModuleList parameters appear in named_parameters()..." << std::endl;

    StackedLayers model(3, 4);  // 3 layers, each SimpleLayer registers "weight"

    // named_parameters() must recurse into registered ModuleLists and key each
    // element as "<list>.<index>.<subkey>".
    auto named = model.named_parameters();
    assert(named.count("layers.0.weight") == 1);
    assert(named.count("layers.1.weight") == 1);
    assert(named.count("layers.2.weight") == 1);
    assert(named.size() == 3);

    // The pointer variant must expose the same keys.
    auto named_ptrs = model.named_parameters_ptrs();
    assert(named_ptrs.count("layers.0.weight") == 1);
    assert(named_ptrs.count("layers.1.weight") == 1);
    assert(named_ptrs.count("layers.2.weight") == 1);
    assert(named_ptrs.size() == 3);
    // Pointers must reference the actual module parameters, not copies.
    assert(named_ptrs["layers.0.weight"] == &model.layers[0].weight);

    std::cout << "  PASSED: ModuleList named_parameters" << std::endl;
}

int main() {
    std::cout << "=== Testing ModuleList<T> ===" << std::endl;

    test_module_list_basic();
    test_module_list_iteration();
    test_module_list_parameters();
    test_module_list_to_device();
    test_stacked_model_registration();
    test_stacked_model_to_device();
    test_stacked_model_state_dict();
    test_module_list_train_eval();
    test_module_list_with_linear();
    test_module_list_named_parameters();
    
    std::cout << "\n=== All ModuleList Tests Passed! ===" << std::endl;
    return 0;
}
