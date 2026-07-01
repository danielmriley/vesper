#include <vesper/ops/cat.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include "grad_check_utils.h"
#include <iostream>
#include <cmath>
#include <vector>

using vesper::Tensor;
using vesper::DType;
using vesper::Device;

static Tensor make_tensor(const std::vector<int64_t>& shape,
                          const std::vector<float>& data,
                          bool requires_grad = false) {
    auto t = vesper::empty(shape, DType::Float32, Device::CPU, requires_grad);
    t.copy_from_host(data.data());
    return t;
}

static void expect_close(const Tensor& t, const std::vector<float>& expected, const char* what) {
    std::vector<float> got(t.numel());
    t.copy_to_host(got.data());
    if (got.size() != expected.size()) {
        std::cerr << what << ": size mismatch " << got.size() << " vs " << expected.size() << std::endl;
        std::exit(1);
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-5f) {
            std::cerr << what << ": mismatch at " << i << " got=" << got[i]
                      << " expected=" << expected[i] << std::endl;
            std::exit(1);
        }
    }
}

void test_cat_autograd() {
    std::cout << "Testing cat forward + autograd..." << std::endl;

    // (a) Forward, dim=1: [2,3] + [2,2] -> [2,5]
    {
        auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
        auto b = make_tensor({2, 2}, {7, 8, 9, 10});
        auto y = vesper::ops::cat({a, b}, 1);
        if (y.shape() != std::vector<int64_t>({2, 5})) {
            std::cerr << "cat dim=1 wrong shape" << std::endl;
            std::exit(1);
        }
        // Row 0: a[1,2,3] | b[7,8]; Row 1: a[4,5,6] | b[9,10]
        expect_close(y, {1, 2, 3, 7, 8, 4, 5, 6, 9, 10}, "cat dim=1 values");
    }

    // (a) Forward, dim=0: [2,3] + [1,3] -> [3,3]
    {
        auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
        auto b = make_tensor({1, 3}, {7, 8, 9});
        auto y = vesper::ops::cat({a, b}, 0);
        if (y.shape() != std::vector<int64_t>({3, 3})) {
            std::cerr << "cat dim=0 wrong shape" << std::endl;
            std::exit(1);
        }
        expect_close(y, {1, 2, 3, 4, 5, 6, 7, 8, 9}, "cat dim=0 values");
    }

    // (b) Backward grad-check, dim=1: y = cat({a,b},1), loss = sum(y * coeffs).
    // Gradient flows identity to each slice, so a.grad == coeffs[:, 0:3] and
    // b.grad == coeffs[:, 3:5]. A non-uniform coeffs tensor catches a zero/wrong
    // backward (uniform-1 grads would pass even if the slicing were wrong).
    {
        auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6}, /*requires_grad=*/true);
        auto b = make_tensor({2, 2}, {7, 8, 9, 10}, /*requires_grad=*/true);

        // coeffs laid out over the [2,5] cat output (non-uniform).
        std::vector<float> coeffs_data = {
            10, 20, 30, 40, 50,
            60, 70, 80, 90, 100
        };
        auto coeffs = make_tensor({2, 5}, coeffs_data);

        auto y = vesper::ops::cat({a, b}, 1);
        auto weighted = vesper::ops::mul(y, coeffs);
        auto loss = vesper::ops::sum(weighted);
        loss.backward();

        // d loss / d y == coeffs; a takes columns 0..2, b takes columns 3..4.
        std::vector<float> expected_a = {10, 20, 30, 60, 70, 80};
        std::vector<float> expected_b = {40, 50, 90, 100};

        Tensor num_grad_a = vesper::test::compute_numerical_gradient(
            [&](const Tensor& t) {
                auto yy = vesper::ops::cat({t, b}, 1);
                auto l = vesper::ops::sum(vesper::ops::mul(yy, coeffs));
                float v; l.copy_to_host(&v); return v;
            }, a);
        Tensor num_grad_b = vesper::test::compute_numerical_gradient(
            [&](const Tensor& t) {
                auto yy = vesper::ops::cat({a, t}, 1);
                auto l = vesper::ops::sum(vesper::ops::mul(yy, coeffs));
                float v; l.copy_to_host(&v); return v;
            }, b);

        expect_close(a.grad(), expected_a, "a.grad analytical");
        expect_close(b.grad(), expected_b, "b.grad analytical");

        if (!vesper::test::check_gradients(a.grad(), num_grad_a)) {
            std::cerr << "a.grad finite-diff check FAILED" << std::endl;
            std::exit(1);
        }
        if (!vesper::test::check_gradients(b.grad(), num_grad_b)) {
            std::cerr << "b.grad finite-diff check FAILED" << std::endl;
            std::exit(1);
        }
    }

    std::cout << "cat autograd test passed!" << std::endl;
}

int main() {
    test_cat_autograd();
    return 0;
}
