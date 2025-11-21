#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_transpose_and_contiguous() {
    std::cout << "Testing transpose and contiguous..." << std::endl;
    
    auto device = vesper::Device::CPU;
    auto a = vesper::empty({2, 3}, vesper::DType::Float32, device);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    a.copy_from_host(data.data());

    // Original strides should be {3, 1}
    assert(a.strides()[0] == 3 && a.strides()[1] == 1);
    assert(a.is_contiguous());

    // 1. Test transpose
    auto b = a.transpose(0, 1);
    assert(b.shape() == std::vector<int64_t>({3, 2}));
    // New strides should be {1, 3}
    assert(b.strides()[0] == 1 && b.strides()[1] == 3);
    assert(!b.is_contiguous()); // Transposed tensor is not contiguous

    // 2. Test contiguous
    auto c = b.contiguous();
    assert(c.is_contiguous());
    assert(c.shape() == std::vector<int64_t>({3, 2}));
    // New contiguous strides should be {2, 1}
    assert(c.strides()[0] == 2 && c.strides()[1] == 1);

    // 3. Verify data is preserved
    std::vector<float> c_data(c.numel());
    c.copy_to_host(c_data.data());
    
    // Expected data in c: {1, 4, 2, 5, 3, 6} (transposed layout)
    assert(std::fabs(c_data[0] - 1) < 1e-6); assert(std::fabs(c_data[1] - 4) < 1e-6);
    assert(std::fabs(c_data[2] - 2) < 1e-6); assert(std::fabs(c_data[3] - 5) < 1e-6);
    assert(std::fabs(c_data[4] - 3) < 1e-6); assert(std::fabs(c_data[5] - 6) < 1e-6);

    std::cout << "Transpose and contiguous test passed!" << std::endl;
}

int main() {
    test_transpose_and_contiguous();
    return 0;
}
