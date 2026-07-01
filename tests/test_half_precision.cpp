/**
 * @file test_half_precision.cpp
 * @brief Tests for FP16 and BF16 half-precision types and casting
 * 
 * Chapter 33.9: Memory-Efficient Training
 */

#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/core/half.h>
#include <vesper/ops/cast.h>
#include <vesper/ops/random.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/elementwise.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <limits>

using namespace vesper;

// ============================================================================
// Test Utilities
// ============================================================================

template<typename T>
bool approx_equal(T a, T b, T rel_tol = 0.01, T abs_tol = 1e-5) {
    T diff = std::abs(a - b);
    return diff <= abs_tol || diff <= rel_tol * std::max(std::abs(a), std::abs(b));
}

bool allclose(const Tensor& a, const Tensor& b, float rtol = 0.01f, float atol = 1e-5f) {
    Tensor a_cpu = a.to(Device::CPU);
    Tensor b_cpu = b.to(Device::CPU);
    
    if (a_cpu.dtype() != b_cpu.dtype()) {
        // Cast both to float for comparison
        a_cpu = ops::cast(a_cpu, DType::Float32);
        b_cpu = ops::cast(b_cpu, DType::Float32);
    }
    
    const float* pa = a_cpu.data_ptr<float>();
    const float* pb = b_cpu.data_ptr<float>();
    
    for (size_t i = 0; i < a_cpu.numel(); ++i) {
        if (!approx_equal(pa[i], pb[i], rtol, atol)) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Float16 Tests
// ============================================================================

void test_float16_basic_conversion() {
    std::cout << "Testing Float16 basic conversion..." << std::endl;
    
    // Test positive numbers
    Float16 h1(1.0f);
    assert(approx_equal(float(h1), 1.0f, 0.001f));
    
    Float16 h2(3.14159f);
    assert(approx_equal(float(h2), 3.14159f, 0.001f));
    
    // Test negative numbers
    Float16 h3(-42.5f);
    assert(approx_equal(float(h3), -42.5f, 0.001f));
    
    // Test zero
    Float16 h4(0.0f);
    assert(float(h4) == 0.0f);
    
    // Test small numbers
    Float16 h5(0.0001f);
    assert(approx_equal(float(h5), 0.0001f, 0.1f, 1e-6f));  // FP16 has limited precision
    
    std::cout << "Float16 basic conversion: PASSED" << std::endl;
}

void test_float16_range() {
    std::cout << "Testing Float16 range limits..." << std::endl;
    
    // Max positive value in FP16 is ~65504
    Float16 h_max(65504.0f);
    assert(float(h_max) > 60000.0f);
    
    // Overflow should result in infinity
    Float16 h_overflow(100000.0f);
    float val = float(h_overflow);
    assert(std::isinf(val));
    
    // Underflow to zero for very small numbers
    Float16 h_tiny(1e-10f);
    assert(float(h_tiny) == 0.0f);  // Underflows to zero
    
    std::cout << "Float16 range limits: PASSED" << std::endl;
}

void test_float16_arithmetic() {
    std::cout << "Testing Float16 arithmetic..." << std::endl;
    
    Float16 a(5.0f);
    Float16 b(3.0f);
    
    // Addition
    Float16 sum = a + b;
    assert(approx_equal(float(sum), 8.0f, 0.01f));
    
    // Subtraction
    Float16 diff = a - b;
    assert(approx_equal(float(diff), 2.0f, 0.01f));
    
    // Multiplication
    Float16 prod = a * b;
    assert(approx_equal(float(prod), 15.0f, 0.01f));
    
    // Division
    Float16 quot = a / b;
    assert(approx_equal(float(quot), 5.0f/3.0f, 0.01f));
    
    std::cout << "Float16 arithmetic: PASSED" << std::endl;
}

// ============================================================================
// BFloat16 Tests
// ============================================================================

void test_bfloat16_basic_conversion() {
    std::cout << "Testing BFloat16 basic conversion..." << std::endl;
    
    // Test positive numbers
    BFloat16 b1(1.0f);
    assert(approx_equal(float(b1), 1.0f, 0.01f));
    
    BFloat16 b2(3.14159f);
    assert(approx_equal(float(b2), 3.14159f, 0.01f));  // BF16 has less precision
    
    // Test negative numbers
    BFloat16 b3(-42.5f);
    assert(approx_equal(float(b3), -42.5f, 0.01f));
    
    // Test zero
    BFloat16 b4(0.0f);
    assert(float(b4) == 0.0f);
    
    std::cout << "BFloat16 basic conversion: PASSED" << std::endl;
}

void test_bfloat16_range() {
    std::cout << "Testing BFloat16 range (same as FP32)..." << std::endl;
    
    // BF16 has the same range as FP32
    BFloat16 b_large(1e30f);
    assert(std::abs(float(b_large) - 1e30f) / 1e30f < 0.01f);  // ~1% error
    
    BFloat16 b_small(1e-30f);
    assert(std::abs(float(b_small) - 1e-30f) / 1e-30f < 0.1f);
    
    // No overflow for large numbers (unlike FP16)
    BFloat16 b_hundred_k(100000.0f);
    assert(!std::isinf(float(b_hundred_k)));
    assert(approx_equal(float(b_hundred_k), 100000.0f, 0.01f));
    
    std::cout << "BFloat16 range: PASSED" << std::endl;
}

void test_bfloat16_arithmetic() {
    std::cout << "Testing BFloat16 arithmetic..." << std::endl;
    
    BFloat16 a(5.0f);
    BFloat16 b(3.0f);
    
    // Addition
    BFloat16 sum = a + b;
    assert(approx_equal(float(sum), 8.0f, 0.01f));
    
    // Subtraction
    BFloat16 diff = a - b;
    assert(approx_equal(float(diff), 2.0f, 0.01f));
    
    // Multiplication
    BFloat16 prod = a * b;
    assert(approx_equal(float(prod), 15.0f, 0.01f));
    
    // Division
    BFloat16 quot = a / b;
    assert(approx_equal(float(quot), 5.0f/3.0f, 0.02f));  // BF16 has less precision
    
    std::cout << "BFloat16 arithmetic: PASSED" << std::endl;
}

// ============================================================================
// Tensor Cast Tests (CPU)
// ============================================================================

void test_tensor_cast_fp32_to_fp16_cpu() {
    std::cout << "Testing Tensor cast FP32 -> FP16 (CPU)..." << std::endl;
    
    std::vector<float> data = {1.0f, 2.5f, -3.7f, 0.0f, 100.0f};
    Tensor fp32 = vesper::empty({5}, DType::Float32, Device::CPU);
    fp32.copy_from_host(data.data());
    
    Tensor fp16 = ops::cast(fp32, DType::Float16);
    
    assert(fp16.dtype() == DType::Float16);
    assert(fp16.shape()[0] == 5);
    
    // Cast back to check values
    Tensor back = ops::cast(fp16, DType::Float32);
    
    const float* result = back.data_ptr<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        assert(approx_equal(result[i], data[i], 0.001f, 1e-4f));
    }
    
    std::cout << "Tensor cast FP32 -> FP16 (CPU): PASSED" << std::endl;
}

void test_tensor_cast_fp32_to_bf16_cpu() {
    std::cout << "Testing Tensor cast FP32 -> BF16 (CPU)..." << std::endl;
    
    std::vector<float> data = {1.0f, 2.5f, -3.7f, 0.0f, 100000.0f};  // Note: 100000 fits in BF16
    Tensor fp32 = vesper::empty({5}, DType::Float32, Device::CPU);
    fp32.copy_from_host(data.data());
    
    Tensor bf16 = ops::cast(fp32, DType::BFloat16);
    
    assert(bf16.dtype() == DType::BFloat16);
    
    // Cast back
    Tensor back = ops::cast(bf16, DType::Float32);
    
    const float* result = back.data_ptr<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        // BF16 has ~1% relative error
        assert(approx_equal(result[i], data[i], 0.01f, 1e-4f));
    }
    
    std::cout << "Tensor cast FP32 -> BF16 (CPU): PASSED" << std::endl;
}

void test_tensor_cast_roundtrip_cpu() {
    std::cout << "Testing Tensor cast roundtrip (CPU)..." << std::endl;
    
    // Create random tensor
    Tensor original = vesper::randn({100, 100}, DType::Float32, Device::CPU);
    
    // FP32 -> FP16 -> FP32
    Tensor fp16 = ops::cast(original, DType::Float16);
    Tensor back_fp16 = ops::cast(fp16, DType::Float32);
    assert(allclose(original, back_fp16, 0.001f, 1e-3f));
    
    // FP32 -> BF16 -> FP32
    Tensor bf16 = ops::cast(original, DType::BFloat16);
    Tensor back_bf16 = ops::cast(bf16, DType::Float32);
    assert(allclose(original, back_bf16, 0.01f, 1e-3f));
    
    // FP16 -> BF16 -> FP16
    Tensor fp16_to_bf16 = ops::cast(fp16, DType::BFloat16);
    Tensor bf16_back_to_fp16 = ops::cast(fp16_to_bf16, DType::Float16);
    Tensor both_to_fp32 = ops::cast(bf16_back_to_fp16, DType::Float32);
    assert(allclose(back_fp16, both_to_fp32, 0.02f, 1e-2f));
    
    std::cout << "Tensor cast roundtrip (CPU): PASSED" << std::endl;
}

// ============================================================================
// Tensor Cast Tests (GPU)
// ============================================================================

void test_tensor_cast_fp32_to_fp16_gpu() {
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
    std::cout << "Testing Tensor cast FP32 -> FP16 (GPU)..." << std::endl;
    
    Device device = Device::CPU;
#ifdef USE_HIP_BACKEND
    device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    device = Device::CUDA;
#endif
    
    std::vector<float> data = {1.0f, 2.5f, -3.7f, 0.0f, 100.0f};
    Tensor fp32_cpu = vesper::empty({5}, DType::Float32, Device::CPU);
    fp32_cpu.copy_from_host(data.data());
    
    // Move to GPU
    Tensor fp32_gpu = fp32_cpu.to(device);
    
    // Cast on GPU
    Tensor fp16_gpu = ops::cast(fp32_gpu, DType::Float16);
    
    assert(fp16_gpu.dtype() == DType::Float16);
    assert(fp16_gpu.device() == device);
    
    // Cast back and move to CPU
    Tensor back_gpu = ops::cast(fp16_gpu, DType::Float32);
    Tensor back_cpu = back_gpu.to(Device::CPU);
    
    const float* result = back_cpu.data_ptr<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        assert(approx_equal(result[i], data[i], 0.001f, 1e-4f));
    }
    
    std::cout << "Tensor cast FP32 -> FP16 (GPU): PASSED" << std::endl;
#else
    std::cout << "Tensor cast FP32 -> FP16 (GPU): SKIPPED (no GPU backend)" << std::endl;
#endif
}

void test_tensor_cast_fp32_to_bf16_gpu() {
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
    std::cout << "Testing Tensor cast FP32 -> BF16 (GPU)..." << std::endl;
    
    Device device = Device::CPU;
#ifdef USE_HIP_BACKEND
    device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    device = Device::CUDA;
#endif
    
    std::vector<float> data = {1.0f, 2.5f, -3.7f, 0.0f, 100000.0f};
    Tensor fp32_cpu = vesper::empty({5}, DType::Float32, Device::CPU);
    fp32_cpu.copy_from_host(data.data());
    
    Tensor fp32_gpu = fp32_cpu.to(device);
    Tensor bf16_gpu = ops::cast(fp32_gpu, DType::BFloat16);
    
    assert(bf16_gpu.dtype() == DType::BFloat16);
    
    Tensor back_gpu = ops::cast(bf16_gpu, DType::Float32);
    Tensor back_cpu = back_gpu.to(Device::CPU);
    
    const float* result = back_cpu.data_ptr<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        assert(approx_equal(result[i], data[i], 0.01f, 1e-3f));
    }
    
    std::cout << "Tensor cast FP32 -> BF16 (GPU): PASSED" << std::endl;
#else
    std::cout << "Tensor cast FP32 -> BF16 (GPU): SKIPPED (no GPU backend)" << std::endl;
#endif
}

void test_tensor_cast_large_gpu() {
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
    std::cout << "Testing Tensor cast large tensor (GPU)..." << std::endl;
    
    Device device = Device::CPU;
#ifdef USE_HIP_BACKEND
    device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    device = Device::CUDA;
#endif
    
    // Create a large tensor
    Tensor fp32_gpu = vesper::randn({1024, 1024}, DType::Float32, device);
    
    // Cast to FP16
    Tensor fp16_gpu = ops::cast(fp32_gpu, DType::Float16);
    Tensor back_fp16 = ops::cast(fp16_gpu, DType::Float32);
    
    // Should be close - use mean squared error since we don't have abs()
    Tensor diff = ops::sub(fp32_gpu, back_fp16);
    Tensor sq_diff = ops::mul(diff, diff);
    Tensor mse = ops::mean(sq_diff);
    
    float mse_val = mse.to(Device::CPU).item<float>();
    std::cout << "  FP16 roundtrip MSE: " << mse_val << std::endl;
    assert(mse_val < 0.01f);  // FP16 can have some error
    
    // Cast to BF16
    Tensor bf16_gpu = ops::cast(fp32_gpu, DType::BFloat16);
    Tensor back_bf16 = ops::cast(bf16_gpu, DType::Float32);
    
    Tensor diff_bf16 = ops::sub(fp32_gpu, back_bf16);
    Tensor sq_diff_bf16 = ops::mul(diff_bf16, diff_bf16);
    Tensor mse_bf16 = ops::mean(sq_diff_bf16);
    
    float mse_bf16_val = mse_bf16.to(Device::CPU).item<float>();
    std::cout << "  BF16 roundtrip MSE: " << mse_bf16_val << std::endl;
    assert(mse_bf16_val < 0.01f);
    
    std::cout << "Tensor cast large tensor (GPU): PASSED" << std::endl;
#else
    std::cout << "Tensor cast large tensor (GPU): SKIPPED (no GPU backend)" << std::endl;
#endif
}

// ============================================================================
// Autograd Cast Tests
// ============================================================================

void test_cast_autograd() {
    std::cout << "Testing cast with autograd..." << std::endl;
    
    Device device = Device::CPU;
#ifdef USE_HIP_BACKEND
    device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    device = Device::CUDA;
#endif
    
    // Create tensor with gradient
    Tensor x = vesper::randn({4, 4}, DType::Float32, device, true);
    
    // Cast to BF16 (or FP16)
    Tensor y = ops::cast(x, DType::BFloat16);
    
    // Cast back and compute loss
    Tensor z = ops::cast(y, DType::Float32);
    Tensor loss = ops::sum(z);
    
    // Backward
    loss.backward();
    
    // Check gradient exists and has correct shape/dtype
    assert(x.grad().defined());
    assert(x.grad().dtype() == DType::Float32);
    assert(x.grad().shape() == x.shape());
    
    // Gradient should be all ones (since loss = sum(z) and z = cast(cast(x)))
    Tensor grad_cpu = x.grad().to(Device::CPU);
    const float* grad_ptr = grad_cpu.data_ptr<float>();
    for (size_t i = 0; i < grad_cpu.numel(); ++i) {
        assert(approx_equal(grad_ptr[i], 1.0f, 0.01f, 0.01f));
    }
    
    std::cout << "Cast with autograd: PASSED" << std::endl;
}

// M1: data_ptr<T>() must reject mismatched element widths, allow raw-byte and same-width.
void test_data_ptr_dtype_guard() {
    std::cout << "Testing data_ptr<T>() dtype-width guard..." << std::endl;

    Tensor h = vesper::empty({4}, DType::Float16, Device::CPU);
    bool threw = false;
    try { (void)h.data_ptr<float>(); } catch (const std::exception&) { threw = true; }
    assert(threw && "data_ptr<float> on a Float16 tensor must throw");

    // Same element width (2 bytes) is permitted.
    bool ok = true;
    try { (void)h.data_ptr<Float16>(); } catch (...) { ok = false; }
    assert(ok && "data_ptr<Float16> on a Float16 tensor must not throw");

    // Matched float path and raw-byte access must not throw.
    Tensor f = vesper::empty({4}, DType::Float32, Device::CPU);
    ok = true;
    try { (void)f.data_ptr<float>(); (void)f.data_ptr<void>(); (void)f.data_ptr<char>(); }
    catch (...) { ok = false; }
    assert(ok && "matched-width and raw-byte data_ptr must not throw");

    std::cout << "data_ptr dtype-width guard: PASSED" << std::endl;
}

// D5: float -> Float16 must round to nearest (ties to even), not truncate.

// Reference: the previous truncating conversion, used to prove that the new
// round-to-nearest behavior is never worse than plain mantissa truncation.
static Float16 truncate_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;
    uint16_t out;
    if (exponent <= 0) {
        if (exponent < -10) {
            out = static_cast<uint16_t>(sign);
        } else {
            mantissa |= 0x800000;
            int shift = 1 - exponent + 13;
            mantissa >>= shift;
            out = static_cast<uint16_t>(sign | mantissa);
        }
    } else if (exponent >= 31) {
        if (exponent == 31 && mantissa != 0) {
            out = static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
        } else {
            out = static_cast<uint16_t>(sign | 0x7C00);
        }
    } else {
        out = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
    }
    return Float16::from_bits(out);
}

void test_float16_round_to_nearest_even() {
    std::cout << "Testing Float16 round-to-nearest-even..." << std::endl;

    // Round-to-nearest is never worse than truncation for finite normal-range
    // values (it picks the closest representable fp16, truncation merely floors).
    const float samples[] = {
        1.0f, 1.0015f, 1.000732421875f, 3.14159f, 2.71828f, 0.1f, 0.2f,
        1.3f, 100.5f, 1000.1f, 1234.567f, 42.42f, 7.7f, 0.333333f,
        -1.0015f, -3.14159f, -1234.567f, 60000.0f, 65000.0f, 65503.0f,
    };
    for (float x : samples) {
        float rounded = float(Float16(x));
        float truncated = float(truncate_to_fp16(x));
        assert(std::abs(rounded - x) <= std::abs(truncated - x) &&
               "round-to-nearest must never be worse than truncation");
    }

    // A value whose discarded mantissa bits exceed half must round UP to the
    // nearest fp16, where truncation would floor. x = 1 + 0x1800/2^23 sits 3/4
    // of the way from 1.0 to the next fp16 step (1.0009765625).
    float x = 1.000732421875f;
    Float16 r(x);
    assert(r.to_bits() == 0x3C01 && "must round up to the nearest fp16");
    assert(truncate_to_fp16(x).to_bits() == 0x3C00 && "truncation floors here");
    assert(std::abs(float(r) - x) < std::abs(float(truncate_to_fp16(x)) - x) &&
           "rounded result is strictly closer than the truncated one");

    // Values just below the max normal (65504) stay finite and map nearby.
    for (float below : {60000.0f, 64000.0f, 65000.0f, 65503.0f, 65504.0f}) {
        float v = float(Float16(below));
        assert(std::isfinite(v) && "value at/below max-normal must stay finite");
        assert(std::abs(v - below) <= 32.0f && "should map to a nearby fp16 step");
    }
    // The largest finite fp16 must not be pushed to infinity by rounding.
    assert(float(Float16(65504.0f)) == 65504.0f);

    // Rounding overflow at the top of the normal range yields infinity: 65520
    // is the midpoint of [65504, 65536]; the mantissa carry must bump the
    // exponent (30 -> 31) via integer addition rather than wrap silently.
    assert(std::isinf(float(Float16(65520.0f))) &&
           "rounding carry past max-normal must produce infinity");

    std::cout << "Float16 round-to-nearest-even: PASSED" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "Half Precision (FP16/BF16) Tests" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Float16 type tests
        test_float16_basic_conversion();
        test_float16_range();
        test_float16_arithmetic();
        test_float16_round_to_nearest_even();
        
        // BFloat16 type tests
        test_bfloat16_basic_conversion();
        test_bfloat16_range();
        test_bfloat16_arithmetic();
        
        // CPU tensor cast tests
        test_tensor_cast_fp32_to_fp16_cpu();
        test_tensor_cast_fp32_to_bf16_cpu();
        test_tensor_cast_roundtrip_cpu();
        
        // GPU tensor cast tests
        test_tensor_cast_fp32_to_fp16_gpu();
        test_tensor_cast_fp32_to_bf16_gpu();
        test_tensor_cast_large_gpu();
        
        // Autograd tests
        test_cast_autograd();

        // data_ptr dtype-width guard (M1)
        test_data_ptr_dtype_guard();

        std::cout << "========================================" << std::endl;
        std::cout << "All half precision tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
