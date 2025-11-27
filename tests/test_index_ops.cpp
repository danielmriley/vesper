/**
 * @file test_index_ops.cpp
 * @brief Tests for index operations: gather, scatter, where, masked_fill
 */

#include <vesper/vesper.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>

using namespace vesper;
using namespace vesper::ops;

// Test helper
#define TEST_CASE(name) \
    std::cout << "\n[TEST] " << name << std::endl;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1; \
        } \
    } while(0)

#define CHECK_CLOSE(a, b, eps, msg) \
    CHECK(std::abs((a) - (b)) < (eps), msg << " (got " << (a) << ", expected " << (b) << ")")

// ============================================================================
// gather() Tests
// ============================================================================

int test_gather_1d_cpu() {
    TEST_CASE("gather() 1D on CPU");
    
    // Input: [10, 20, 30, 40, 50]
    // Index: [1, 3, 0, 4]
    // Expected: [20, 40, 10, 50]
    auto input = zeros({5}, DType::Float32, Device::CPU);
    float* in_ptr = input.data_ptr<float>();
    in_ptr[0] = 10; in_ptr[1] = 20; in_ptr[2] = 30; in_ptr[3] = 40; in_ptr[4] = 50;
    
    auto index = zeros({4}, DType::Int64, Device::CPU);
    int64_t* idx_ptr = index.data_ptr<int64_t>();
    idx_ptr[0] = 1; idx_ptr[1] = 3; idx_ptr[2] = 0; idx_ptr[3] = 4;
    
    auto result = gather(input, 0, index);
    
    CHECK(result.shape() == std::vector<int64_t>({4}), "Shape mismatch");
    
    float* out_ptr = result.data_ptr<float>();
    CHECK_CLOSE(out_ptr[0], 20.0f, 1e-5f, "Value at [0]");
    CHECK_CLOSE(out_ptr[1], 40.0f, 1e-5f, "Value at [1]");
    CHECK_CLOSE(out_ptr[2], 10.0f, 1e-5f, "Value at [2]");
    CHECK_CLOSE(out_ptr[3], 50.0f, 1e-5f, "Value at [3]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

int test_gather_2d_dim0_cpu() {
    TEST_CASE("gather() 2D dim=0 on CPU");
    
    // Input: [[1, 2, 3],
    //         [4, 5, 6],
    //         [7, 8, 9]]
    // Index: [[0, 2, 1],
    //         [1, 0, 2]]
    // Expected: [[1, 8, 6],
    //            [4, 2, 9]]
    auto input = zeros({3, 3}, DType::Float32, Device::CPU);
    float* in_ptr = input.data_ptr<float>();
    for (int i = 0; i < 9; i++) in_ptr[i] = static_cast<float>(i + 1);
    
    auto index = zeros({2, 3}, DType::Int64, Device::CPU);
    int64_t* idx_ptr = index.data_ptr<int64_t>();
    idx_ptr[0] = 0; idx_ptr[1] = 2; idx_ptr[2] = 1;  // row 0
    idx_ptr[3] = 1; idx_ptr[4] = 0; idx_ptr[5] = 2;  // row 1
    
    auto result = gather(input, 0, index);
    
    CHECK(result.shape() == std::vector<int64_t>({2, 3}), "Shape mismatch");
    
    float* out_ptr = result.data_ptr<float>();
    // Row 0: gather from rows [0,2,1] → [1, 8, 6]
    CHECK_CLOSE(out_ptr[0], 1.0f, 1e-5f, "Value at [0,0]");
    CHECK_CLOSE(out_ptr[1], 8.0f, 1e-5f, "Value at [0,1]");
    CHECK_CLOSE(out_ptr[2], 6.0f, 1e-5f, "Value at [0,2]");
    // Row 1: gather from rows [1,0,2] → [4, 2, 9]
    CHECK_CLOSE(out_ptr[3], 4.0f, 1e-5f, "Value at [1,0]");
    CHECK_CLOSE(out_ptr[4], 2.0f, 1e-5f, "Value at [1,1]");
    CHECK_CLOSE(out_ptr[5], 9.0f, 1e-5f, "Value at [1,2]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

int test_gather_2d_dim1_cpu() {
    TEST_CASE("gather() 2D dim=1 on CPU (column selection)");
    
    // Input: [[10, 20, 30, 40],
    //         [50, 60, 70, 80]]
    // Index: [[1, 3],
    //         [0, 2]]
    // Expected: [[20, 40],
    //            [50, 70]]
    auto input = zeros({2, 4}, DType::Float32, Device::CPU);
    float* in_ptr = input.data_ptr<float>();
    in_ptr[0] = 10; in_ptr[1] = 20; in_ptr[2] = 30; in_ptr[3] = 40;
    in_ptr[4] = 50; in_ptr[5] = 60; in_ptr[6] = 70; in_ptr[7] = 80;
    
    auto index = zeros({2, 2}, DType::Int64, Device::CPU);
    int64_t* idx_ptr = index.data_ptr<int64_t>();
    idx_ptr[0] = 1; idx_ptr[1] = 3;  // row 0
    idx_ptr[2] = 0; idx_ptr[3] = 2;  // row 1
    
    auto result = gather(input, 1, index);
    
    CHECK(result.shape() == std::vector<int64_t>({2, 2}), "Shape mismatch");
    
    float* out_ptr = result.data_ptr<float>();
    CHECK_CLOSE(out_ptr[0], 20.0f, 1e-5f, "Value at [0,0]");
    CHECK_CLOSE(out_ptr[1], 40.0f, 1e-5f, "Value at [0,1]");
    CHECK_CLOSE(out_ptr[2], 50.0f, 1e-5f, "Value at [1,0]");
    CHECK_CLOSE(out_ptr[3], 70.0f, 1e-5f, "Value at [1,1]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

#ifdef USE_HIP_BACKEND
int test_gather_gpu() {
    TEST_CASE("gather() on GPU");
    
    auto input = zeros({4, 8}, DType::Float32, Device::CPU);
    float* in_ptr = input.data_ptr<float>();
    for (int i = 0; i < 32; i++) in_ptr[i] = static_cast<float>(i);
    input = input.to(Device::HIP);
    
    auto index = zeros({4, 3}, DType::Int64, Device::CPU);
    int64_t* idx_ptr = index.data_ptr<int64_t>();
    // Each row selects columns [0, 4, 7]
    for (int i = 0; i < 4; i++) {
        idx_ptr[i*3 + 0] = 0;
        idx_ptr[i*3 + 1] = 4;
        idx_ptr[i*3 + 2] = 7;
    }
    index = index.to(Device::HIP);
    
    auto result = gather(input, 1, index).to(Device::CPU);
    
    CHECK(result.shape() == std::vector<int64_t>({4, 3}), "Shape mismatch");
    
    float* out_ptr = result.data_ptr<float>();
    // Row 0: [0, 4, 7], Row 1: [8, 12, 15], etc.
    CHECK_CLOSE(out_ptr[0], 0.0f, 1e-5f, "Value at [0,0]");
    CHECK_CLOSE(out_ptr[1], 4.0f, 1e-5f, "Value at [0,1]");
    CHECK_CLOSE(out_ptr[2], 7.0f, 1e-5f, "Value at [0,2]");
    CHECK_CLOSE(out_ptr[3], 8.0f, 1e-5f, "Value at [1,0]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}
#endif

// ============================================================================
// scatter() Tests
// ============================================================================

int test_scatter_cpu() {
    TEST_CASE("scatter() on CPU");
    
    // Scatter values [100, 200, 300, 400] into positions [1, 3, 0, 2]
    auto src = zeros({4}, DType::Float32, Device::CPU);
    float* src_ptr = src.data_ptr<float>();
    src_ptr[0] = 100; src_ptr[1] = 200; src_ptr[2] = 300; src_ptr[3] = 400;
    
    auto index = zeros({4}, DType::Int64, Device::CPU);
    int64_t* idx_ptr = index.data_ptr<int64_t>();
    idx_ptr[0] = 1; idx_ptr[1] = 3; idx_ptr[2] = 0; idx_ptr[3] = 2;
    
    auto base = zeros({5}, DType::Float32, Device::CPU);
    auto result = scatter(base, 0, index, src);  // scatter returns a new tensor
    
    float* out_ptr = result.data_ptr<float>();
    CHECK_CLOSE(out_ptr[0], 300.0f, 1e-5f, "Value at [0]");  // from idx 2
    CHECK_CLOSE(out_ptr[1], 100.0f, 1e-5f, "Value at [1]");  // from idx 0
    CHECK_CLOSE(out_ptr[2], 400.0f, 1e-5f, "Value at [2]");  // from idx 3
    CHECK_CLOSE(out_ptr[3], 200.0f, 1e-5f, "Value at [3]");  // from idx 1
    CHECK_CLOSE(out_ptr[4], 0.0f, 1e-5f, "Value at [4]");    // unchanged
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

int test_scatter_2d_cpu() {
    TEST_CASE("scatter() 2D diagonal on CPU");
    
    // Create 4x4 zeros, scatter ones on diagonal
    auto base = zeros({4, 4}, DType::Float32, Device::CPU);
    auto src = ones({4, 1}, DType::Float32, Device::CPU);
    
    auto index = zeros({4, 1}, DType::Int64, Device::CPU);
    int64_t* idx_ptr = index.data_ptr<int64_t>();
    idx_ptr[0] = 0; idx_ptr[1] = 1; idx_ptr[2] = 2; idx_ptr[3] = 3;
    
    auto result = scatter(base, 1, index, src);  // returns new tensor
    
    float* out_ptr = result.data_ptr<float>();
    // Check diagonal is 1, rest is 0
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            CHECK_CLOSE(out_ptr[i*4 + j], expected, 1e-5f, "Diagonal check");
        }
    }
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

#ifdef USE_HIP_BACKEND
int test_scatter_gpu() {
    TEST_CASE("scatter() on GPU");
    
    auto base = zeros({4, 4}, DType::Float32, Device::HIP);
    auto src = ones({4, 1}, DType::Float32, Device::HIP);
    
    auto index = zeros({4, 1}, DType::Int64, Device::CPU);
    int64_t* idx_ptr = index.data_ptr<int64_t>();
    idx_ptr[0] = 0; idx_ptr[1] = 1; idx_ptr[2] = 2; idx_ptr[3] = 3;
    index = index.to(Device::HIP);
    
    auto result = scatter(base, 1, index, src).to(Device::CPU);  // returns new tensor
    
    float* out_ptr = result.data_ptr<float>();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            CHECK_CLOSE(out_ptr[i*4 + j], expected, 1e-5f, "GPU Diagonal check");
        }
    }
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}
#endif

// ============================================================================
// scatter() with scalar Tests
// ============================================================================

int test_scatter_scalar_cpu() {
    TEST_CASE("scatter() scalar on CPU");
    
    auto result = zeros({5}, DType::Float32, Device::CPU);
    auto index = zeros({3}, DType::Int64, Device::CPU);
    int64_t* idx_ptr = index.data_ptr<int64_t>();
    idx_ptr[0] = 0; idx_ptr[1] = 2; idx_ptr[2] = 4;
    
    scatter_(result, 0, index, 7.0f);
    
    float* out_ptr = result.data_ptr<float>();
    CHECK_CLOSE(out_ptr[0], 7.0f, 1e-5f, "Value at [0]");
    CHECK_CLOSE(out_ptr[1], 0.0f, 1e-5f, "Value at [1]");
    CHECK_CLOSE(out_ptr[2], 7.0f, 1e-5f, "Value at [2]");
    CHECK_CLOSE(out_ptr[3], 0.0f, 1e-5f, "Value at [3]");
    CHECK_CLOSE(out_ptr[4], 7.0f, 1e-5f, "Value at [4]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

// ============================================================================
// where() Tests - Using Int32 as mask (0 = false, non-zero = true)
// ============================================================================

int test_where_cpu() {
    TEST_CASE("where() on CPU");
    
    // condition: [1, 0, 1, 0] (true, false, true, false)
    // x: [10, 20, 30, 40]
    // y: [100, 200, 300, 400]
    // expected: [10, 200, 30, 400]
    auto condition = zeros({4}, DType::Int32, Device::CPU);
    int32_t* cond_ptr = condition.data_ptr<int32_t>();
    cond_ptr[0] = 1; cond_ptr[1] = 0; cond_ptr[2] = 1; cond_ptr[3] = 0;
    
    auto x = zeros({4}, DType::Float32, Device::CPU);
    float* x_ptr = x.data_ptr<float>();
    x_ptr[0] = 10; x_ptr[1] = 20; x_ptr[2] = 30; x_ptr[3] = 40;
    
    auto y = zeros({4}, DType::Float32, Device::CPU);
    float* y_ptr = y.data_ptr<float>();
    y_ptr[0] = 100; y_ptr[1] = 200; y_ptr[2] = 300; y_ptr[3] = 400;
    
    auto result = where(condition, x, y);
    
    float* out_ptr = result.data_ptr<float>();
    CHECK_CLOSE(out_ptr[0], 10.0f, 1e-5f, "Value at [0]");
    CHECK_CLOSE(out_ptr[1], 200.0f, 1e-5f, "Value at [1]");
    CHECK_CLOSE(out_ptr[2], 30.0f, 1e-5f, "Value at [2]");
    CHECK_CLOSE(out_ptr[3], 400.0f, 1e-5f, "Value at [3]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

int test_where_2d_cpu() {
    TEST_CASE("where() 2D on CPU");
    
    // Create 2x3 tensors with checkerboard pattern
    auto condition = zeros({2, 3}, DType::Int32, Device::CPU);
    int32_t* cond_ptr = condition.data_ptr<int32_t>();
    // Checkerboard pattern
    cond_ptr[0] = 1;  cond_ptr[1] = 0; cond_ptr[2] = 1;
    cond_ptr[3] = 0; cond_ptr[4] = 1;  cond_ptr[5] = 0;
    
    auto x = ones({2, 3}, DType::Float32, Device::CPU);
    auto y = zeros({2, 3}, DType::Float32, Device::CPU);
    
    auto result = where(condition, x, y);
    
    float* out_ptr = result.data_ptr<float>();
    CHECK_CLOSE(out_ptr[0], 1.0f, 1e-5f, "Value at [0,0]");
    CHECK_CLOSE(out_ptr[1], 0.0f, 1e-5f, "Value at [0,1]");
    CHECK_CLOSE(out_ptr[2], 1.0f, 1e-5f, "Value at [0,2]");
    CHECK_CLOSE(out_ptr[3], 0.0f, 1e-5f, "Value at [1,0]");
    CHECK_CLOSE(out_ptr[4], 1.0f, 1e-5f, "Value at [1,1]");
    CHECK_CLOSE(out_ptr[5], 0.0f, 1e-5f, "Value at [1,2]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

#ifdef USE_HIP_BACKEND
int test_where_gpu() {
    TEST_CASE("where() on GPU");
    
    auto condition = zeros({4}, DType::Int32, Device::CPU);
    int32_t* cond_ptr = condition.data_ptr<int32_t>();
    cond_ptr[0] = 1; cond_ptr[1] = 0; cond_ptr[2] = 1; cond_ptr[3] = 0;
    condition = condition.to(Device::HIP);
    
    auto x = full({4}, DType::Float32, Device::HIP, 10.0f);
    auto y = full({4}, DType::Float32, Device::HIP, 200.0f);
    
    auto result = where(condition, x, y).to(Device::CPU);
    
    float* out_ptr = result.data_ptr<float>();
    CHECK_CLOSE(out_ptr[0], 10.0f, 1e-5f, "Value at [0]");
    CHECK_CLOSE(out_ptr[1], 200.0f, 1e-5f, "Value at [1]");
    CHECK_CLOSE(out_ptr[2], 10.0f, 1e-5f, "Value at [2]");
    CHECK_CLOSE(out_ptr[3], 200.0f, 1e-5f, "Value at [3]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}
#endif

// ============================================================================
// masked_fill() Tests - Using Int32 as mask (0 = false, non-zero = true)
// ============================================================================

int test_masked_fill_cpu() {
    TEST_CASE("masked_fill() on CPU");
    
    auto input = zeros({5}, DType::Float32, Device::CPU);
    float* in_ptr = input.data_ptr<float>();
    in_ptr[0] = 1; in_ptr[1] = 2; in_ptr[2] = 3; in_ptr[3] = 4; in_ptr[4] = 5;
    
    auto mask = zeros({5}, DType::Int32, Device::CPU);
    int32_t* mask_ptr = mask.data_ptr<int32_t>();
    mask_ptr[1] = 1; mask_ptr[3] = 1;  // Mask positions 1 and 3
    
    auto result = masked_fill(input, mask, -999.0f);
    
    float* out_ptr = result.data_ptr<float>();
    CHECK_CLOSE(out_ptr[0], 1.0f, 1e-5f, "Value at [0]");
    CHECK_CLOSE(out_ptr[1], -999.0f, 1e-5f, "Value at [1]");
    CHECK_CLOSE(out_ptr[2], 3.0f, 1e-5f, "Value at [2]");
    CHECK_CLOSE(out_ptr[3], -999.0f, 1e-5f, "Value at [3]");
    CHECK_CLOSE(out_ptr[4], 5.0f, 1e-5f, "Value at [4]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

int test_masked_fill_attention_mask_cpu() {
    TEST_CASE("masked_fill() attention mask pattern on CPU");
    
    // Simulate attention scores with upper triangular mask (causal)
    auto scores = ones({4, 4}, DType::Float32, Device::CPU);
    
    // Create upper triangular mask (positions above diagonal)
    auto mask = zeros({4, 4}, DType::Int32, Device::CPU);
    int32_t* mask_ptr = mask.data_ptr<int32_t>();
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            mask_ptr[i * 4 + j] = 1;
        }
    }
    
    auto result = masked_fill(scores, mask, -std::numeric_limits<float>::infinity());
    
    float* out_ptr = result.data_ptr<float>();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (j > i) {
                CHECK(std::isinf(out_ptr[i*4 + j]) && out_ptr[i*4 + j] < 0, 
                      "Should be -inf at [" << i << "," << j << "]");
            } else {
                CHECK_CLOSE(out_ptr[i*4 + j], 1.0f, 1e-5f, 
                           "Should be 1 at [" << i << "," << j << "]");
            }
        }
    }
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

#ifdef USE_HIP_BACKEND
int test_masked_fill_gpu() {
    TEST_CASE("masked_fill() on GPU");
    
    auto input = zeros({4}, DType::Float32, Device::CPU);
    float* in_ptr = input.data_ptr<float>();
    in_ptr[0] = 1; in_ptr[1] = 2; in_ptr[2] = 3; in_ptr[3] = 4;
    input = input.to(Device::HIP);
    
    auto mask = zeros({4}, DType::Int32, Device::CPU);
    int32_t* mask_ptr = mask.data_ptr<int32_t>();
    mask_ptr[0] = 1; mask_ptr[2] = 1;
    mask = mask.to(Device::HIP);
    
    auto result = masked_fill(input, mask, -std::numeric_limits<float>::infinity()).to(Device::CPU);
    
    float* out_ptr = result.data_ptr<float>();
    CHECK(std::isinf(out_ptr[0]) && out_ptr[0] < 0, "Value at [0] should be -inf");
    CHECK_CLOSE(out_ptr[1], 2.0f, 1e-5f, "Value at [1]");
    CHECK(std::isinf(out_ptr[2]) && out_ptr[2] < 0, "Value at [2] should be -inf");
    CHECK_CLOSE(out_ptr[3], 4.0f, 1e-5f, "Value at [3]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}
#endif

// ============================================================================
// gather/scatter roundtrip test
// ============================================================================

int test_gather_scatter_roundtrip_cpu() {
    TEST_CASE("gather/scatter roundtrip on CPU");
    
    // Create indices for a permutation
    auto index = zeros({4}, DType::Int64, Device::CPU);
    int64_t* idx_ptr = index.data_ptr<int64_t>();
    idx_ptr[0] = 2; idx_ptr[1] = 0; idx_ptr[2] = 3; idx_ptr[3] = 1;  // permutation
    
    auto original = zeros({4}, DType::Float32, Device::CPU);
    float* orig_ptr = original.data_ptr<float>();
    orig_ptr[0] = 10; orig_ptr[1] = 20; orig_ptr[2] = 30; orig_ptr[3] = 40;
    
    // Gather permutes the values: gathered[i] = original[index[i]]
    // gathered[0] = original[2] = 30
    // gathered[1] = original[0] = 10
    // gathered[2] = original[3] = 40
    // gathered[3] = original[1] = 20
    // So gathered = [30, 10, 40, 20]
    auto gathered = gather(original, 0, index);
    
    float* g_ptr = gathered.data_ptr<float>();
    CHECK_CLOSE(g_ptr[0], 30.0f, 1e-5f, "Gathered value at [0]");
    CHECK_CLOSE(g_ptr[1], 10.0f, 1e-5f, "Gathered value at [1]");
    CHECK_CLOSE(g_ptr[2], 40.0f, 1e-5f, "Gathered value at [2]");
    CHECK_CLOSE(g_ptr[3], 20.0f, 1e-5f, "Gathered value at [3]");
    
    // To restore, scatter with same index:
    // restored[index[i]] = gathered[i]
    // restored[2] = gathered[0] = 30
    // restored[0] = gathered[1] = 10
    // restored[3] = gathered[2] = 40
    // restored[1] = gathered[3] = 20
    // So restored = [10, 20, 30, 40] = original
    auto base = zeros({4}, DType::Float32, Device::CPU);
    auto restored = scatter(base, 0, index, gathered);
    
    // Check that scatter undoes the gather
    float* res_ptr = restored.data_ptr<float>();
    CHECK_CLOSE(res_ptr[0], 10.0f, 1e-5f, "Restored value at [0]");
    CHECK_CLOSE(res_ptr[1], 20.0f, 1e-5f, "Restored value at [1]");
    CHECK_CLOSE(res_ptr[2], 30.0f, 1e-5f, "Restored value at [2]");
    CHECK_CLOSE(res_ptr[3], 40.0f, 1e-5f, "Restored value at [3]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

// ============================================================================
// Int32 index tests
// ============================================================================

int test_gather_int32_index_cpu() {
    TEST_CASE("gather() with Int32 indices on CPU");
    
    auto input = zeros({5}, DType::Float32, Device::CPU);
    float* in_ptr = input.data_ptr<float>();
    in_ptr[0] = 10; in_ptr[1] = 20; in_ptr[2] = 30; in_ptr[3] = 40; in_ptr[4] = 50;
    
    auto index = zeros({3}, DType::Int32, Device::CPU);
    int32_t* idx_ptr = index.data_ptr<int32_t>();
    idx_ptr[0] = 4; idx_ptr[1] = 2; idx_ptr[2] = 0;
    
    auto result = gather(input, 0, index);
    
    float* out_ptr = result.data_ptr<float>();
    CHECK_CLOSE(out_ptr[0], 50.0f, 1e-5f, "Value at [0]");
    CHECK_CLOSE(out_ptr[1], 30.0f, 1e-5f, "Value at [1]");
    CHECK_CLOSE(out_ptr[2], 10.0f, 1e-5f, "Value at [2]");
    
    std::cout << "  PASSED!" << std::endl;
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Index Operations Tests ===" << std::endl;
    
    int failures = 0;
    
    // gather tests
    failures += test_gather_1d_cpu();
    failures += test_gather_2d_dim0_cpu();
    failures += test_gather_2d_dim1_cpu();
    failures += test_gather_int32_index_cpu();
    
    // scatter tests
    failures += test_scatter_cpu();
    failures += test_scatter_2d_cpu();
    failures += test_scatter_scalar_cpu();
    
    // where tests
    failures += test_where_cpu();
    failures += test_where_2d_cpu();
    
    // masked_fill tests
    failures += test_masked_fill_cpu();
    failures += test_masked_fill_attention_mask_cpu();
    
    // roundtrip test
    failures += test_gather_scatter_roundtrip_cpu();
    
#ifdef USE_HIP_BACKEND
    std::cout << "\n=== GPU Tests ===" << std::endl;
    failures += test_gather_gpu();
    failures += test_scatter_gpu();
    failures += test_where_gpu();
    failures += test_masked_fill_gpu();
#endif
    
    std::cout << "\n=== Summary ===" << std::endl;
    if (failures == 0) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << failures << " test(s) FAILED!" << std::endl;
        return 1;
    }
}
