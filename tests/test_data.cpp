#include <vesper/data/dataset.h>
#include <vesper/data/dataloader.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <cassert>

void test_tensor_dataset() {
    std::cout << "Testing TensorDataset..." << std::endl;
    
    // 1. Create data
    auto x = vesper::empty({10, 5}, vesper::DType::Float32, vesper::Device::CPU);
    auto y = vesper::empty({10, 1}, vesper::DType::Float32, vesper::Device::CPU);
    
    // 2. Create dataset
    auto dataset = std::make_shared<vesper::data::TensorDataset>(x, y);

    // 3. Check size
    assert(dataset->size() == 10);

    // 4. Get an item and check its shape
    auto sample = dataset->get_item(3);
    auto sample_x = sample.first;
    auto sample_y = sample.second;

    assert(sample_x.shape() == std::vector<int64_t>({5}));
    assert(sample_y.shape() == std::vector<int64_t>({1}));

    std::cout << "TensorDataset test passed!" << std::endl;
}

void test_dataloader_batching() {
    std::cout << "Testing DataLoader batching..." << std::endl;
    
    auto x = vesper::zeros({10, 5}, vesper::DType::Float32, vesper::Device::CPU);
    auto y = vesper::zeros({10, 1}, vesper::DType::Float32, vesper::Device::CPU);
    auto dataset = std::make_shared<vesper::data::TensorDataset>(x, y);

    const size_t batch_size = 4;
    auto loader = vesper::data::DataLoader(dataset, batch_size, false);

    size_t num_batches = 0;
    size_t total_samples = 0;
    for (const auto& batch : loader) {
        num_batches++;
        total_samples += batch.first.shape()[0];

        if (num_batches == 1) { // First batch
            assert(batch.first.shape() == std::vector<int64_t>({4, 5}));
            assert(batch.second.shape() == std::vector<int64_t>({4, 1}));
        } else if (num_batches == 3) { // Last batch
            assert(batch.first.shape() == std::vector<int64_t>({2, 5}));
            assert(batch.second.shape() == std::vector<int64_t>({2, 1}));
        }
    }

    assert(num_batches == 3);
    assert(total_samples == 10);

    std::cout << "DataLoader batching test passed!" << std::endl;
}

int main() {
    test_tensor_dataset();
    test_dataloader_batching();
    return 0;
}
