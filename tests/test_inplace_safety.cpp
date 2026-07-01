#include <vesper/core/factories.h>
#include <vesper/core/tensor.h>
#include <vesper/ops/elementwise.h>
#include <vesper/autograd/guard.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>

#if defined(USE_HIP_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CUDA;
#elif defined(USE_CPU_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CPU;
#else
    #error "No backend enabled for testing"
#endif

void test_inplace_safety() {
    std::cout << "Testing in-place safety check..." << std::endl;
    
    auto a = vesper::full({2, 2}, vesper::DType::Float32, TEST_DEVICE, 1.0f, true); // requires_grad = true
    auto b = vesper::full({2, 2}, vesper::DType::Float32, TEST_DEVICE, 2.0f);
    
    // 1. Should throw because grad mode is enabled by default
    bool threw = false;
    try {
        a.add_(b);
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        if (msg.find("In-place operations") != std::string::npos) {
            threw = true;
        } else {
            std::cerr << "Unexpected error: " << msg << std::endl;
        }
    }
    assert(threw && "In-place op should throw when requires_grad=true and grad_mode=true");
    
    // 2. Should NOT throw if NoGradGuard is active
    {
        vesper::autograd::NoGradGuard guard;
        try {
            a.add_(b);
        } catch (...) {
            assert(false && "In-place op should NOT throw inside NoGradGuard");
        }
    }
    
    // 3. Should NOT throw if requires_grad=false
    auto c = vesper::full({2, 2}, vesper::DType::Float32, TEST_DEVICE, 1.0f, false);
    try {
        c.add_(b);
    } catch (...) {
        assert(false && "In-place op should NOT throw if requires_grad=false");
    }

    std::cout << "In-place safety test passed!" << std::endl;
}

void test_inplace_grad_guard() {
    std::cout << "Testing in-place grad guard for zero_/copy_..." << std::endl;

    auto a = vesper::full({2, 2}, vesper::DType::Float32, TEST_DEVICE, 1.0f, true); // requires_grad = true
    auto b = vesper::full({2, 2}, vesper::DType::Float32, TEST_DEVICE, 2.0f);

    // 1. Under default grad mode, zero_() on a leaf requiring grad must throw.
    bool zero_threw = false;
    try {
        a.zero_();
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        if (msg.find("In-place") != std::string::npos) {
            zero_threw = true;
        } else {
            std::cerr << "Unexpected error: " << msg << std::endl;
        }
    }
    assert(zero_threw && "zero_() should throw when requires_grad=true and grad_mode=true");

    // copy_() on a leaf requiring grad must throw.
    bool copy_threw = false;
    try {
        a.copy_(b);
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        if (msg.find("In-place") != std::string::npos) {
            copy_threw = true;
        } else {
            std::cerr << "Unexpected error: " << msg << std::endl;
        }
    }
    assert(copy_threw && "copy_() should throw when requires_grad=true and grad_mode=true");

    // 2. Inside a NoGradGuard, both should succeed.
    {
        vesper::autograd::NoGradGuard guard;
        try {
            a.zero_();
            a.copy_(b);
        } catch (...) {
            assert(false && "zero_()/copy_() should NOT throw inside NoGradGuard");
        }
    }

    std::cout << "In-place grad guard test passed!" << std::endl;
}

int main() {
    test_inplace_safety();
    test_inplace_grad_guard();
    return 0;
}
