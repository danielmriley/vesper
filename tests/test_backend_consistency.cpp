#include "vesper/core/factories.h"
#include "vesper/ops/elementwise.h"
#include "vesper/ops/reduction.h"
#include "vesper/ops/gemm.h"
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <cassert>

using namespace vesper;

// Helper to fill a tensor with random data
void fill_random(Tensor& t) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(t.numel());
    for (auto& x : data) x = dist(rng);
    t.copy_from_host(data.data());
}

// Helper to compare CPU and HIP tensors
void check_tensors_close(const Tensor& cpu_tensor, const Tensor& hip_tensor, float tol = 1e-4) {
    assert(cpu_tensor.shape() == hip_tensor.shape());
    assert(cpu_tensor.numel() == hip_tensor.numel());

    std::vector<float> cpu_data(cpu_tensor.numel());
    std::vector<float> hip_data(hip_tensor.numel());

    cpu_tensor.copy_to_host(cpu_data.data());
    hip_tensor.copy_to_host(hip_data.data());

    for (size_t i = 0; i < cpu_data.size(); ++i) {
        if (std::fabs(cpu_data[i] - hip_data[i]) > tol) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << cpu_data[i] << ", HIP=" << hip_data[i] << std::endl;
            exit(1);
        }
    }
}

void test_add() {
    std::cout << "Testing Add Consistency..." << std::endl;
    std::vector<int64_t> shape = {100, 100};
    auto a_cpu = vesper::empty(shape, DType::Float32, Device::CPU);
    auto b_cpu = vesper::empty(shape, DType::Float32, Device::CPU);
    fill_random(a_cpu);
    fill_random(b_cpu);

    auto a_hip = vesper::empty(shape, DType::Float32, Device::HIP);
    auto b_hip = vesper::empty(shape, DType::Float32, Device::HIP);
    a_hip.copy_from_host(a_cpu.data_ptr<float>());
    b_hip.copy_from_host(b_cpu.data_ptr<float>());

    auto res_cpu = ops::add(a_cpu, b_cpu);
    auto res_hip = ops::add(a_hip, b_hip);

    check_tensors_close(res_cpu, res_hip);
    std::cout << "PASSED" << std::endl;
}

void test_sub() {
    std::cout << "Testing Sub Consistency..." << std::endl;
    std::vector<int64_t> shape = {100, 100};
    auto a_cpu = vesper::empty(shape, DType::Float32, Device::CPU);
    auto b_cpu = vesper::empty(shape, DType::Float32, Device::CPU);
    fill_random(a_cpu);
    fill_random(b_cpu);

    auto a_hip = vesper::empty(shape, DType::Float32, Device::HIP);
    auto b_hip = vesper::empty(shape, DType::Float32, Device::HIP);
    a_hip.copy_from_host(a_cpu.data_ptr<float>());
    b_hip.copy_from_host(b_cpu.data_ptr<float>());

    auto res_cpu = ops::sub(a_cpu, b_cpu);
    auto res_hip = ops::sub(a_hip, b_hip);

    check_tensors_close(res_cpu, res_hip);
    std::cout << "PASSED" << std::endl;
}

void test_mul() {
    std::cout << "Testing Mul Consistency..." << std::endl;
    std::vector<int64_t> shape = {100, 100};
    auto a_cpu = vesper::empty(shape, DType::Float32, Device::CPU);
    auto b_cpu = vesper::empty(shape, DType::Float32, Device::CPU);
    fill_random(a_cpu);
    fill_random(b_cpu);

    auto a_hip = vesper::empty(shape, DType::Float32, Device::HIP);
    auto b_hip = vesper::empty(shape, DType::Float32, Device::HIP);
    a_hip.copy_from_host(a_cpu.data_ptr<float>());
    b_hip.copy_from_host(b_cpu.data_ptr<float>());

    auto res_cpu = ops::mul(a_cpu, b_cpu);
    auto res_hip = ops::mul(a_hip, b_hip);

    check_tensors_close(res_cpu, res_hip);
    std::cout << "PASSED" << std::endl;
}

void test_sum() {
    std::cout << "Testing Sum Consistency..." << std::endl;
    std::vector<int64_t> shape = {1024, 1024}; // Large enough to trigger multi-block reduction
    auto a_cpu = vesper::empty(shape, DType::Float32, Device::CPU);
    fill_random(a_cpu);

    auto a_hip = vesper::empty(shape, DType::Float32, Device::HIP);
    a_hip.copy_from_host(a_cpu.data_ptr<float>());

    auto res_cpu = ops::sum(a_cpu);
    auto res_hip = ops::sum(a_hip);

    // Sum accumulation can have larger errors due to order of operations
    check_tensors_close(res_cpu, res_hip, 1e-2);
    std::cout << "PASSED" << std::endl;
}

void test_matmul() {
    std::cout << "Testing Matmul Consistency..." << std::endl;
    int M = 64, K = 64, N = 64;
    auto a_cpu = vesper::empty({M, K}, DType::Float32, Device::CPU);
    auto b_cpu = vesper::empty({K, N}, DType::Float32, Device::CPU);
    fill_random(a_cpu);
    fill_random(b_cpu);

    auto a_hip = vesper::empty({M, K}, DType::Float32, Device::HIP);
    auto b_hip = vesper::empty({K, N}, DType::Float32, Device::HIP);
    a_hip.copy_from_host(a_cpu.data_ptr<float>());
    b_hip.copy_from_host(b_cpu.data_ptr<float>());

    auto res_cpu = ops::matmul(a_cpu, b_cpu);
    auto res_hip = ops::matmul(a_hip, b_hip);

    check_tensors_close(res_cpu, res_hip, 1e-3);
    std::cout << "PASSED" << std::endl;
}

int main() {
#ifdef USE_HIP_BACKEND
    test_add();
    test_sub();
    test_mul();
    test_sum();
    test_matmul();
#else
    std::cout << "HIP backend not enabled. Skipping consistency tests." << std::endl;
#endif
    return 0;
}
