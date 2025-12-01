// Softmax micro-benchmark
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/normalization.h>
#include <vesper/ops/random.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

using namespace vesper;
using namespace vesper::ops;

int main() {
    std::cout << "=== Softmax Benchmark ===" << std::endl;
    
    Device device = Device::HIP;
    
    // Typical transformer dimensions
    // batch=4, seq=256, heads=16, head_dim=64 -> attention softmax is [4*16, 256, 256]
    // For vocabulary softmax: [batch*seq, vocab] -> [1024, 32000]
    
    std::vector<std::pair<std::string, std::vector<int64_t>>> test_cases = {
        {"Attention softmax (B=4, H=16, S=256)", {64, 256, 256}},  // 64*256*256 = 4M elements
        {"Vocab softmax (batch*seq=1024, vocab=32000)", {1024, 32000}},  // Large vocabulary
        {"Small softmax (1024, 64)", {1024, 64}},  // Small inner dim
        {"Very wide (256, 4096)", {256, 4096}},  // Wide rows
    };
    
    const int warmup_iters = 10;
    const int bench_iters = 100;
    
    for (const auto& [name, shape] : test_cases) {
        std::cout << "\n" << name << " - shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Calculate total elements
        int64_t total = 1;
        for (auto s : shape) total *= s;
        
        Tensor input = randn(shape, DType::Float32, device);
        Tensor output;
        
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            output = softmax(input, -1);
        }
        
        // Synchronize before timing
        hipDeviceSynchronize();
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
            output = softmax(input, -1);
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double time_per_iter_us = (time_ms * 1000.0) / bench_iters;
        
        // Calculate bandwidth (read input + write output)
        double bytes = 2.0 * total * sizeof(float);  // Read + write
        double bandwidth_gbps = (bytes * bench_iters / 1e9) / (time_ms / 1000.0);
        
        std::cout << "  Time per softmax: " << time_per_iter_us << " us" << std::endl;
        std::cout << "  Effective bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
    }
    
    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    return 0;
}
