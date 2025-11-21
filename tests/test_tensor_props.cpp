#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

void test_empty_distinct() {
    std::cout << "Testing distinct allocations..." << std::endl;
#if USE_HIP_BACKEND
    vesper::Tensor t1 = vesper::empty({10}, vesper::DType::Float32, vesper::Device::HIP);
    vesper::Tensor t2 = vesper::empty({10}, vesper::DType::Float32, vesper::Device::HIP);
    
    assert(t1.data_ptr<float>() != t2.data_ptr<float>());
#endif
}

void test_tensor_properties_0d() {
    std::cout << "Testing 0D properties..." << std::endl;
#if USE_HIP_BACKEND
    vesper::Tensor t = vesper::empty({}, vesper::DType::Float32, vesper::Device::HIP);
    assert(t.shape().size() == 0);
    assert(t.numel() == 1);
    assert(t.strides().size() == 0);
#endif
}

void test_tensor_properties_1d() {
    std::cout << "Testing 1D properties..." << std::endl;
#if USE_HIP_BACKEND
    vesper::Tensor t = vesper::empty({5}, vesper::DType::Float32, vesper::Device::HIP);
    assert(t.shape().size() == 1);
    assert(t.numel() == 5);
    assert(t.strides()[0] == 1);
#endif
}

void test_tensor_properties_2d() {
    std::cout << "Testing 2D properties..." << std::endl;
#if USE_HIP_BACKEND
    vesper::Tensor t = vesper::empty({5, 2}, vesper::DType::Float32, vesper::Device::HIP);
    assert(t.shape().size() == 2);
    assert(t.numel() == 10);
    assert(t.strides()[0] == 2);
    assert(t.strides()[1] == 1);
#endif
}

void test_tensor_properties_4d() {
    std::cout << "Testing 4D properties..." << std::endl;
#if USE_HIP_BACKEND
    // NCHW format often used in vision: Batch=2, Channels=3, Height=4, Width=5
    vesper::Tensor t = vesper::empty({2, 3, 4, 5}, vesper::DType::Float32, vesper::Device::HIP);
    assert(t.shape().size() == 4);
    assert(t.numel() == 2 * 3 * 4 * 5); // 120
    assert(t.strides()[0] == 3 * 4 * 5); // 60
    assert(t.strides()[1] == 4 * 5);     // 20
    assert(t.strides()[2] == 5);         // 5
    assert(t.strides()[3] == 1);         // 1
#endif
}

int main() {
    test_empty_distinct();
    test_tensor_properties_0d();
    test_tensor_properties_1d();
    test_tensor_properties_2d();
    test_tensor_properties_4d();
    std::cout << "Tensor properties tests passed!" << std::endl;
    return 0;
}
