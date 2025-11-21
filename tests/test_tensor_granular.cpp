#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>
#include <vector>

void test_scalar_tensor() {
    std::cout << "Testing scalar tensor..." << std::endl;
#if USE_HIP_BACKEND
    {
        vesper::Tensor t = vesper::empty({}, vesper::DType::Float32, vesper::Device::HIP);
        assert(t.shape().empty());
        assert(t.numel() == 1);
        assert(t.strides().empty()); // Strides for scalar are usually empty
        assert(t.is_contiguous());
    }
#endif
}

void test_1d_tensor() {
    std::cout << "Testing 1D tensor..." << std::endl;
#if USE_HIP_BACKEND
    {
        vesper::Tensor t = vesper::empty({10}, vesper::DType::Int32, vesper::Device::HIP);
        assert(t.shape().size() == 1);
        assert(t.shape()[0] == 10);
        assert(t.strides().size() == 1);
        assert(t.strides()[0] == 1);
        assert(t.numel() == 10);
        assert(t.is_contiguous());
    }
#endif
}

void test_3d_tensor() {
    std::cout << "Testing 3D tensor..." << std::endl;
#if USE_HIP_BACKEND
    {
        // Shape: (2, 3, 4)
        // Strides should be: (12, 4, 1)
        vesper::Tensor t = vesper::empty({2, 3, 4}, vesper::DType::Float64, vesper::Device::HIP);
        assert(t.numel() == 24);
        assert(t.strides()[0] == 12);
        assert(t.strides()[1] == 4);
        assert(t.strides()[2] == 1);
    }
#endif
}

int main() {
    test_scalar_tensor();
    test_1d_tensor();
    test_3d_tensor();
    std::cout << "Tensor granular tests passed!" << std::endl;
    return 0;
}
