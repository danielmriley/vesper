#include <vesper/ops/elementwise.h>
#include <vesper/core/tensor.h>
#include <vector>
#include <thread>
#include <algorithm>

namespace vesper::ops {

// Get number of threads for parallelization
inline unsigned int get_num_threads() {
    return std::max(1u, std::thread::hardware_concurrency());
}

// CPU implementation logic with broadcasting support - parallelized with std::thread
template <typename Op>
void cpu_broadcast_op(const Tensor& a, const std::vector<int64_t>& strides_a, 
                      const Tensor& b, const std::vector<int64_t>& strides_b, 
                      Tensor& out, Op op) {
    
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    size_t n = out.numel();
    int dims = out.shape().size();
    const auto& shape = out.shape();
    
    // Only parallelize if large enough
    if (n < 10000) {
        for (size_t i = 0; i < n; ++i) {
            size_t offset_a = 0;
            size_t offset_b = 0;
            size_t remaining = i;
            
            for (int d = dims - 1; d >= 0; --d) {
                size_t coord = remaining % shape[d];
                remaining /= shape[d];
                offset_a += coord * strides_a[d];
                offset_b += coord * strides_b[d];
            }
            out_ptr[i] = op(a_ptr[offset_a], b_ptr[offset_b]);
        }
        return;
    }
    
    unsigned int num_threads = get_num_threads();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    size_t chunk_size = (n + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        
        if (start < end) {
            threads.emplace_back([=]() {
                for (size_t i = start; i < end; ++i) {
                    size_t offset_a = 0;
                    size_t offset_b = 0;
                    size_t remaining = i;
                    
                    for (int d = dims - 1; d >= 0; --d) {
                        size_t coord = remaining % shape[d];
                        remaining /= shape[d];
                        offset_a += coord * strides_a[d];
                        offset_b += coord * strides_b[d];
                    }
                    out_ptr[i] = op(a_ptr[offset_a], b_ptr[offset_b]);
                }
            });
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename Op>
void cpu_scalar_op(const Tensor& a, float scalar, Tensor& out, Op op) {
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    size_t n = out.numel();
    int dims = out.shape().size();
    const auto& shape = out.shape();
    const auto& strides_a = a.strides();
    
    // Only parallelize if large enough
    if (n < 10000) {
        for (size_t i = 0; i < n; ++i) {
            size_t offset_a = 0;
            size_t remaining = i;
            
            for (int d = dims - 1; d >= 0; --d) {
                size_t coord = remaining % shape[d];
                remaining /= shape[d];
                offset_a += coord * strides_a[d];
            }
            out_ptr[i] = op(a_ptr[offset_a], scalar);
        }
        return;
    }
    
    unsigned int num_threads = get_num_threads();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    size_t chunk_size = (n + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        
        if (start < end) {
            threads.emplace_back([=, &strides_a]() {
                for (size_t i = start; i < end; ++i) {
                    size_t offset_a = 0;
                    size_t remaining = i;
                    
                    for (int d = dims - 1; d >= 0; --d) {
                        size_t coord = remaining % shape[d];
                        remaining /= shape[d];
                        offset_a += coord * strides_a[d];
                    }
                    out_ptr[i] = op(a_ptr[offset_a], scalar);
                }
            });
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

void add_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    cpu_broadcast_op(a, strides_a, b, strides_b, out, [](float x, float y) { return x + y; });
}

void add_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out) {
    cpu_scalar_op(a, b, out, [](float x, float y) { return x + y; });
}

void sub_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    cpu_broadcast_op(a, strides_a, b, strides_b, out, [](float x, float y) { return x - y; });
}

void sub_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out) {
    cpu_scalar_op(a, b, out, [](float x, float y) { return x - y; });
}

void mul_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    cpu_broadcast_op(a, strides_a, b, strides_b, out, [](float x, float y) { return x * y; });
}

void mul_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out) {
    cpu_scalar_op(a, b, out, [](float x, float y) { return x * y; });
}

void div_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    cpu_broadcast_op(a, strides_a, b, strides_b, out, [](float x, float y) { return x / y; });
}

void div_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out) {
    cpu_scalar_op(a, b, out, [](float x, float y) { return x / y; });
}

} // namespace vesper::ops