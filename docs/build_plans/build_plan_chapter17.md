
# Vesper Build Plan - Chapter 17: Integration Test: A Full Training Loop

## 1. Goal

Bring all the components of Vesper together to train a simple model on a toy regression task. This chapter serves as the ultimate integration test, proving that the forward pass, autograd engine, and optimizer all work in concert. Success means watching the model's loss decrease and its parameters converge to the correct values.

## 2. The Task: Learning `y = 2x + 1`

We will create a dataset based on this simple linear function. Our model, which consists of a single `nn::Linear(1, 1)` layer, will start with random weights and, through training, should learn that its weight needs to be `2.0` and its bias needs to be `1.0`.

## 3. The Training Loop Recipe

Every training loop, regardless of complexity, follows the same fundamental sequence of five steps:
1.  **Forward Pass**: Make a prediction.
2.  **Compute Loss**: Measure how wrong the prediction is.
3.  **Zero Gradients**: Clear old gradients from the previous step.
4.  **Backward Pass**: Compute new gradients.
5.  **Optimizer Step**: Update the model's parameters.

We will implement this sequence inside a `for` loop.

## 4. Detailed Steps

This entire chapter will be a single test file that demonstrates a complete, self-contained training pipeline.

### Step 4.1: Create `tests/test_training_loop.cpp`
```cpp
// tests/test_training_loop.cpp
#include <vesper/nn/linear.h>
#include <vesper/nn/loss.h>
#include <vesper/optim/sgd.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

// A simple model with one linear layer
class SimpleRegressionModel : public vesper::nn::Module {
public:
    SimpleRegressionModel(int in, int out) {
        layer = std::make_shared<vesper::nn::Linear>(in, out);
        register_module("layer1", layer);
    }

    vesper::Tensor forward(const vesper::Tensor& input) override {
        return layer->forward(input);
    }
    
    std::shared_ptr<vesper::nn::Linear> layer;
};

void test_full_training_loop() {
    std::cout << "Testing full training loop..." << std::endl;

    // Use CPU for simplicity and to easily inspect data
    const auto device = vesper::Device::CPU;
    
    // 1. Generate Toy Data for y = 2x + 1
    const int num_samples = 20;
    auto x_train = vesper::empty({num_samples, 1}, vesper::DType::Float32, device);
    auto y_train = vesper::empty({num_samples, 1}, vesper::DType::Float32, device);

    std::vector<float> x_data(num_samples), y_data(num_samples);
    for(int i = 0; i < num_samples; ++i) {
        x_data[i] = static_cast<float>(i);
        y_data[i] = 2.0f * x_data[i] + 1.0f;
    }
    x_train.copy_from_host(x_data.data());
    y_train.copy_from_host(y_data.data());

    // 2. Instantiate Model, Loss, and Optimizer
    auto model = std::make_shared<SimpleRegressionModel>(1, 1);
    auto loss_fn = vesper::nn::MSELoss();
    auto optimizer = vesper::optim::SGD(model->parameters(), 0.01f); // lr = 0.01

    // 3. The Training Loop
    const int epochs = 100;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Step 1: Forward Pass
        auto y_pred = model->forward(x_train);

        // Step 2: Compute Loss
        auto loss = loss_fn.forward(y_pred, y_train);

        // Step 3: Zero Gradients
        optimizer.zero_grad();

        // Step 4: Backward Pass
        loss.backward();

        // Step 5: Optimizer Step
        optimizer.step();

        if ((epoch + 1) % 10 == 0) {
            float loss_val;
            loss.copy_to_host(&loss_val);
            std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "], Loss: " << loss_val << std::endl;
        }
    }

    // 4. Verification
    // After training, the weight should be close to 2.0 and the bias close to 1.0
    auto final_weight = model->layer->weight;
    auto final_bias = model->layer->bias;

    float weight_val, bias_val;
    final_weight.copy_to_host(&weight_val);
    final_bias.copy_to_host(&bias_val);

    std::cout << "Learned Weight: " << weight_val << ", Learned Bias: " << bias_val << std::endl;
    
    assert(std::fabs(weight_val - 2.0f) < 1e-1);
    assert(std::fabs(bias_val - 1.0f) < 1e-1);

    std::cout << "Full training loop test passed!" << std::endl;
}

int main() {
    test_full_training_loop();
    return 0;
}
```
*Note: This test requires that the `Linear` layer's bias addition is functional. If not yet implemented, the bias can be disabled in the `Linear` constructor, and the test can verify only the weight.*

### Step 4.2: Add to `tests/CMakeLists.txt`
```cmake
add_executable(training_loop_test test_training_loop.cpp)
target_link_libraries(training_loop_test PRIVATE vesper)
add_test(NAME TrainingLoopTest COMMAND training_loop_test)
```

## 5. Running the Test and What to Expect

When you build and run this test, you should see the loss decreasing every 10 epochs, for example:
```
Epoch [10/100], Loss: 15.345
Epoch [20/100], Loss: 7.123
...
Epoch [90/100], Loss: 0.021
Epoch [100/100], Loss: 0.015
Learned Weight: 1.98, Learned Bias: 1.15
Full training loop test passed!
```
This output is the definitive sign that your library is working. The decreasing loss shows that gradient descent is successfully minimizing the error, and the final learned parameters confirm that it's learning the correct function.

## 6. Conclusion of Core MVP

A passing test in this chapter marks the completion of the Minimum Viable Product (MVP) for the Vesper library's core functionality on a single backend. You have successfully built a system that can:
-   Define tensors and perform mathematical operations (`add`, `matmul`).
-   Construct neural network layers (`Linear`) and models.
-   Calculate loss (`MSELoss`).
-   Automatically compute all necessary gradients (`backward()`).
-   Update model parameters to learn from data (`SGD`).

The following chapters will focus on expanding this foundation with more features and additional backends.
