#include <vesper/nn/transformer.h>
#include <vesper/optim/sgd.h>
#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/ops/random.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <numeric>

using namespace vesper;

Tensor randn(const std::vector<int64_t>& shape) {
    Tensor t = vesper::empty(shape, DType::Float32, Device::CPU, true);
    ops::normal_(t, 0.0f, 1.0f);
    return t;
}

void test_parameter_count() {
    std::cout << "Testing Parameter Count..." << std::endl;
    int embed_dim = 768;
    int num_heads = 12;
    nn::TransformerBlock block(embed_dim, num_heads);

    // Calculate expected parameters
    // Attention:
    // c_attn: (embed_dim, 3 * embed_dim) weight + bias -> 768 * 2304 + 2304
    // c_proj: (embed_dim, embed_dim) weight + bias -> 768 * 768 + 768
    // MLP:
    // c_fc: (embed_dim, 4 * embed_dim) weight + bias -> 768 * 3072 + 3072
    // c_proj: (4 * embed_dim, embed_dim) weight + bias -> 3072 * 768 + 768
    // LayerNorms:
    // ln1: 2 * embed_dim
    // ln2: 2 * embed_dim

    long long expected_params = 0;
    // Attn
    expected_params += (768LL * (3 * 768)) + (3 * 768); // c_attn
    expected_params += (768LL * 768) + 768; // c_proj
    // MLP
    expected_params += (768LL * (4 * 768)) + (4 * 768); // c_fc
    expected_params += ((4 * 768LL) * 768) + 768; // c_proj
    // LNs
    expected_params += 2 * 768; // ln1
    expected_params += 2 * 768; // ln2

    auto params = block.parameters();
    long long actual_params = 0;
    for (auto& p : params) {
        actual_params += p.numel();
    }

    std::cout << "Expected: " << expected_params << ", Actual: " << actual_params << std::endl;
    assert(actual_params == expected_params);
    std::cout << "Parameter Count Passed!" << std::endl;
}

void test_forward_shape() {
    std::cout << "Testing Forward Shape..." << std::endl;
    int embed_dim = 64;
    int num_heads = 4;
    int batch_size = 2;
    int seq_len = 10;

    nn::TransformerBlock block(embed_dim, num_heads);
    Tensor x = randn({batch_size, seq_len, embed_dim});
    Tensor out = block.forward(x);

    assert(out.shape() == std::vector<int64_t>({batch_size, seq_len, embed_dim}));
    std::cout << "Forward Shape Passed!" << std::endl;
}

void test_overfitting() {
    std::cout << "Testing Overfitting..." << std::endl;
    int embed_dim = 32;
    int num_heads = 4;
    int batch_size = 1;
    int seq_len = 5;

    nn::TransformerBlock block(embed_dim, num_heads);
    optim::SGD optimizer(block.parameters(), 0.01);

    Tensor x = randn({batch_size, seq_len, embed_dim});
    Tensor target = randn({batch_size, seq_len, embed_dim});

    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();
        Tensor out = block.forward(x);
        Tensor loss = nn::functional::mse_loss(out, target);
        loss.backward();
        optimizer.step();

        if (i % 20 == 0) {
            std::cout << "Step " << i << ", Loss: " << loss.item<float>() << std::endl;
        }
        if (loss.item<float>() < 0.01) {
            std::cout << "Converged at step " << i << std::endl;
            break;
        }
    }
    
    Tensor out = block.forward(x);
    Tensor loss = nn::functional::mse_loss(out, target);
    std::cout << "Final Loss: " << loss.item<float>() << std::endl;
    assert(loss.item<float>() < 0.1); // Should be small
    std::cout << "Overfitting Passed!" << std::endl;
}

int main() {
    test_parameter_count();
    test_forward_shape();
    test_overfitting();
    return 0;
}
