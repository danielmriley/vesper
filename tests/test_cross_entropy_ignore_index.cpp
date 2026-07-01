#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/core/tensor.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>

using namespace vesper;

// Manual per-row negative log-likelihood: -log_softmax(logits[row])[target].
static float row_nll(const std::vector<float>& logits, int64_t V, int64_t row, int64_t target) {
    const float* r = logits.data() + row * V;
    float m = r[0];
    for (int64_t j = 1; j < V; ++j) m = std::max(m, r[j]);
    float denom = 0.0f;
    for (int64_t j = 0; j < V; ++j) denom += std::exp(r[j] - m);
    float log_softmax_t = (r[target] - m) - std::log(denom);
    return -log_softmax_t;
}

static void softmax_row(const std::vector<float>& logits, int64_t V, int64_t row, std::vector<float>& out) {
    const float* r = logits.data() + row * V;
    float m = r[0];
    for (int64_t j = 1; j < V; ++j) m = std::max(m, r[j]);
    float denom = 0.0f;
    for (int64_t j = 0; j < V; ++j) denom += std::exp(r[j] - m);
    out.resize(V);
    for (int64_t j = 0; j < V; ++j) out[j] = std::exp(r[j] - m) / denom;
}

static const std::vector<float> kLogits = {
    1.0f,  2.0f,  0.5f, -1.0f,   // row 0
    0.0f, -0.5f,  3.0f,  1.0f,   // row 1
    2.0f,  1.0f,  0.0f, -1.0f    // row 2
};

// One target equals ignore_index: that row must drop out of both loss and grad,
// and the mean must be over the non-ignored rows only.
void test_ignore_index_loss_and_grad() {
    std::cout << "Testing cross_entropy_loss ignore_index (loss + grad)..." << std::endl;

    const int64_t N = 3, V = 4;
    const int64_t IGNORE = -1;

    std::vector<int32_t> tgt_data = {2, 0, static_cast<int32_t>(IGNORE)};  // row 2 ignored

    Tensor logits = vesper::empty({N, V}, DType::Float32, Device::CPU, true);
    logits.copy_from_host(kLogits.data());
    Tensor targets = vesper::empty({N}, DType::Int32, Device::CPU, false);
    targets.copy_from_host(tgt_data.data());

    Tensor loss = nn::functional::cross_entropy_loss(logits, targets, IGNORE);
    float loss_val = loss.item<float>();

    // CE over only the non-ignored rows (0 and 1), averaged over 2 valid rows.
    // Dividing by 2 (valid count) rather than N=3 is the property under test.
    float manual = (row_nll(kLogits, V, 0, 2) + row_nll(kLogits, V, 1, 0)) / 2.0f;
    assert(std::abs(loss_val - manual) < 1e-5f);

    loss.backward();
    std::vector<float> grad(static_cast<size_t>(N * V));
    logits.grad().copy_to_host(grad.data());

    // Ignored row must contribute exactly zero gradient.
    for (int64_t j = 0; j < V; ++j) {
        assert(std::abs(grad[2 * V + j]) < 1e-6f);
    }

    // Valid rows: grad = (softmax - one_hot) / valid_count (=2).
    const int64_t valid_targets[2] = {2, 0};
    for (int64_t row = 0; row < 2; ++row) {
        std::vector<float> sm;
        softmax_row(kLogits, V, row, sm);
        for (int64_t j = 0; j < V; ++j) {
            float onehot = (j == valid_targets[row]) ? 1.0f : 0.0f;
            float expected = (sm[j] - onehot) / 2.0f;
            assert(std::abs(grad[row * V + j] - expected) < 1e-5f);
        }
    }

    std::cout << "  ignore_index loss+grad passed (loss=" << loss_val << ")" << std::endl;
}

// With no target equal to the (default) ignore_index, behavior must match a plain
// mean cross-entropy over all rows (backward compatibility).
void test_default_no_ignore_matches_mean() {
    std::cout << "Testing cross_entropy_loss default path matches mean CE..." << std::endl;

    const int64_t N = 3, V = 4;
    std::vector<int32_t> tgt_data = {2, 0, 3};  // all valid, none == -1

    Tensor logits = vesper::empty({N, V}, DType::Float32, Device::CPU, true);
    logits.copy_from_host(kLogits.data());
    Tensor targets = vesper::empty({N}, DType::Int32, Device::CPU, false);
    targets.copy_from_host(tgt_data.data());

    Tensor loss = nn::functional::cross_entropy_loss(logits, targets);  // default ignore_index = -1
    float loss_val = loss.item<float>();

    float manual = (row_nll(kLogits, V, 0, 2) + row_nll(kLogits, V, 1, 0) + row_nll(kLogits, V, 2, 3)) / 3.0f;
    assert(std::abs(loss_val - manual) < 1e-5f);

    std::cout << "  default path matches mean CE (loss=" << loss_val << ")" << std::endl;
}

// Exercise the Int64 target host-branch as well.
void test_ignore_index_int64() {
    std::cout << "Testing cross_entropy_loss ignore_index with Int64 targets..." << std::endl;

    const int64_t N = 3, V = 4;
    const int64_t IGNORE = -1;
    std::vector<int64_t> tgt_data = {2, 0, IGNORE};  // row 2 ignored

    Tensor logits = vesper::empty({N, V}, DType::Float32, Device::CPU, false);
    logits.copy_from_host(kLogits.data());
    Tensor targets = vesper::empty({N}, DType::Int64, Device::CPU, false);
    targets.copy_from_host(tgt_data.data());

    Tensor loss = nn::functional::cross_entropy_loss(logits, targets, IGNORE);
    float loss_val = loss.item<float>();

    float manual = (row_nll(kLogits, V, 0, 2) + row_nll(kLogits, V, 1, 0)) / 2.0f;
    assert(std::abs(loss_val - manual) < 1e-5f);

    std::cout << "  Int64 ignore_index passed (loss=" << loss_val << ")" << std::endl;
}

int main() {
    test_ignore_index_loss_and_grad();
    test_default_no_ignore_matches_mean();
    test_ignore_index_int64();
    std::cout << "All cross_entropy_loss ignore_index tests passed!" << std::endl;
    return 0;
}
