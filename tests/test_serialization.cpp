#include <vesper/serialization.h>
#include <vesper/nn/linear.h>
#include <vesper/optim/adam.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <cstdio>

void test_save_load_tensor() {
    std::cout << "Testing save/load raw StateDict..." << std::endl;
    vesper::StateDict dict;
    dict["t1"] = vesper::full({2, 2}, vesper::DType::Float32, vesper::Device::CPU, 1.5f);
    // Create Int32 tensor via cast
    dict["t2"] = vesper::full({5}, vesper::DType::Float32, vesper::Device::CPU, 42.0f).to(vesper::DType::Int32);
    
    std::string filename = "test_dict.vsp";
    vesper::save(dict, filename);
    
    vesper::StateDict loaded = vesper::load(filename);
    assert(loaded.count("t1"));
    assert(loaded.count("t2"));
    
    assert(loaded["t1"].shape() == std::vector<int64_t>({2, 2}));
    assert(loaded["t1"].item<float>() == 1.5f); // Check first element
    
    assert(loaded["t2"].dtype() == vesper::DType::Int32);
    // We don't have item<int> yet? item<T> is template.
    // If item implementation supports int, it works.
    // Let's check item().
    // item implementation: T val; copy_to_host(&val); return val;
    // copy_to_host uses byte copy.
    // So item<int>() should work if tensor is Int32.
    assert(loaded["t2"].item<int>() == 42);
    
    std::remove(filename.c_str());
    std::cout << "Save/Load StateDict passed!" << std::endl;
}

void test_save_load_module() {
    std::cout << "Testing save/load Module..." << std::endl;
    auto model = vesper::nn::Linear(10, 5);
    // Modify weights
    model.weight.copy_from(vesper::full(model.weight.shape(), vesper::DType::Float32, vesper::Device::CPU, 0.123f));
    
    std::string filename = "test_model.vsp";
    vesper::save(model, filename);
    
    auto loaded_model = vesper::nn::Linear(10, 5);
    vesper::load(loaded_model, filename);
    
    std::vector<float> w_data(model.weight.numel());
    loaded_model.weight.copy_to_host(w_data.data());
    assert(std::abs(w_data[0] - 0.123f) < 1e-5);
    
    std::remove(filename.c_str());
    std::cout << "Save/Load Module passed!" << std::endl;
}

void test_save_load_optimizer() {
    std::cout << "Testing save/load Optimizer..." << std::endl;
    auto x = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f, true);
    auto optimizer = vesper::optim::Adam({x}, 0.1f);
    
    // Step to create state
    x.grad() = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 0.5f);
    optimizer.step(); // t=1, m updated
    
    std::string filename = "test_optim.vsp";
    vesper::save(optimizer.state_dict(), filename);
    
    // New optimizer
    auto x2 = vesper::full({1}, vesper::DType::Float32, vesper::Device::CPU, 1.0f, true); // Same param structure
    auto optimizer2 = vesper::optim::Adam({x2}, 0.01f); // Different LR initially
    
    optimizer2.load_state_dict(vesper::load(filename));
    
    // Check LR restored
    assert(std::abs(optimizer2.get_lr() - 0.1f) < 1e-5);
    
    // Check state restored (t=1)
    // We can't access t_ directly unless we friend or check side effects.
    // If we step, correction should use t=2.
    // We can verify if next step produces same result as continuing first optimizer.
    
    // Optimizer 1 next step
    optimizer.step();
    std::vector<float> x_data(1);
    x.copy_to_host(x_data.data());
    
    // Optimizer 2 next step
    optimizer2.step();
    std::vector<float> x2_data(1);
    x2.copy_to_host(x2_data.data());
    
    assert(std::abs(x_data[0] - x2_data[0]) < 1e-5);
    
    std::remove(filename.c_str());
    std::cout << "Save/Load Optimizer passed!" << std::endl;
}

int main() {
    test_save_load_tensor();
    test_save_load_module();
    test_save_load_optimizer();
    return 0;
}
