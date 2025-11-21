#include <vesper/ops/stack.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <cmath>

void test_stack_op() {
    std::cout << "Testing stack operation..." << std::endl;
    auto device = vesper::Device::CPU;

    // 1. Create tensors to stack
    auto t1 = vesper::full({2}, vesper::DType::Float32, device, 1.0f); // {1, 1}
    auto t2 = vesper::full({2}, vesper::DType::Float32, device, 2.0f); // {2, 2}
    auto t3 = vesper::full({2}, vesper::DType::Float32, device, 3.0f); // {3, 3}

    std::vector<vesper::Tensor> tensors = {t1, t2, t3};
    
    // 2. Stack them
    auto stacked = vesper::ops::stack(tensors, 0);

    // 3. Verify shape
    assert(stacked.shape() == std::vector<int64_t>({3, 2}));

    // 4. Verify data
    std::vector<float> data(stacked.numel());
    stacked.copy_to_host(data.data());

    assert(std::fabs(data[0] - 1.0f) < 1e-6); assert(std::fabs(data[1] - 1.0f) < 1e-6);
    assert(std::fabs(data[2] - 2.0f) < 1e-6); assert(std::fabs(data[3] - 2.0f) < 1e-6);
    assert(std::fabs(data[4] - 3.0f) < 1e-6); assert(std::fabs(data[5] - 3.0f) < 1e-6);
    
    std::cout << "Stack operation test passed!" << std::endl;
}

int main() {
    test_stack_op();
    return 0;
}
