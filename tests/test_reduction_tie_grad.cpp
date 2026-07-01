#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <vesper/core/tensor.h>
#include "grad_check_utils.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using vesper::Tensor;
using vesper::DType;
using vesper::Device;

namespace {

// Grad-checks at ties must use ANALYTICAL expectations: finite differences are
// invalid at a tie because max/min is non-differentiable there (the +eps and
// -eps perturbations pick different single argmaxes and average to the wrong
// value). The amax/amin convention splits the upstream grad equally among ties.

Tensor leaf(const std::vector<int64_t>& shape, const std::vector<float>& data) {
    Tensor t = vesper::empty(shape, DType::Float32, Device::CPU, false);
    t.copy_from_host(data.data());
    t.set_requires_grad(true);
    return t;
}

bool grad_matches(Tensor& x, const std::vector<float>& expected, const char* label) {
    Tensor& g = x.grad();
    if (g.numel() != expected.size()) {
        std::cerr << label << ": grad numel " << g.numel()
                  << " != expected " << expected.size() << "\n";
        return false;
    }
    std::vector<float> got(g.numel());
    g.copy_to_host(got.data());
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(got[i] - expected[i]) > 1e-5f) {
            std::cerr << label << ": grad mismatch at " << i << " got " << got[i]
                      << " expected " << expected[i] << "\n";
            return false;
        }
    }
    std::cout << label << ": ok\n";
    return true;
}

} // namespace

int test_reduction_tie_grad() {
    bool ok = true;

    // (a) full max with a tie: max([1,3,3,2]) -> grad splits over the two 3s.
    {
        Tensor x = leaf({4}, {1.0f, 3.0f, 3.0f, 2.0f});
        Tensor y = vesper::ops::max(x);
        y.backward();
        ok &= grad_matches(x, {0.0f, 0.5f, 0.5f, 0.0f}, "full-max tie");
    }

    // (b) full min with a tie: min([2,1,1,4]) -> grad splits over the two 1s.
    {
        Tensor x = leaf({4}, {2.0f, 1.0f, 1.0f, 4.0f});
        Tensor y = vesper::ops::min(x);
        y.backward();
        ok &= grad_matches(x, {0.0f, 0.5f, 0.5f, 0.0f}, "full-min tie");
    }

    // (c) dim max over dim=1 on a [2,3] tensor (M=2 > 1). This path used to throw
    //     ("equal: shape mismatch") for M>1; assert it now RUNS and that the grad
    //     splits equally among the tied maxima per row and is zero elsewhere.
    //     Row 0: max=3 tied at cols 1,2 -> 0.5 each. Row 1: max=5 at col 0 -> 1.0.
    {
        Tensor x = leaf({2, 3}, {1.0f, 3.0f, 3.0f, 5.0f, 2.0f, 1.0f});
        bool ran = true;
        try {
            Tensor y = vesper::ops::max(x, /*dim=*/1); // keepdim=false -> shape [2]
            y.backward();
        } catch (const std::exception& e) {
            std::cerr << "dim-max M>1 threw (regression): " << e.what() << "\n";
            ran = false;
        }
        ok &= ran;
        if (ran) {
            ok &= grad_matches(x, {0.0f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f}, "dim-max tie M>1");
        }
    }

    if (ok) {
        std::cout << "test_reduction_tie_grad passed\n";
        return 0;
    }
    std::cerr << "test_reduction_tie_grad FAILED\n";
    return 1;
}

int main() {
    return test_reduction_tie_grad();
}
