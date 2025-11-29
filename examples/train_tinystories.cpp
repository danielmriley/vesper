#include <vesper/models/transformer.h>
#include <vesper/data/dataset.h>
#include <vesper/data/dataloader.h>
#include <vesper/optim/adam.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/core/device.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <memory>

using namespace vesper;

class TinyStoriesDataset : public data::Dataset {
public:
    TinyStoriesDataset(const std::string& path, int64_t seq_len) 
        : seq_len_(seq_len) {
        
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Could not open file: " + path);
        }
        
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        if (size % sizeof(uint16_t) != 0) {
            throw std::runtime_error("File size is not a multiple of uint16 size");
        }
        
        size_t num_tokens = size / sizeof(uint16_t);
        data_.resize(num_tokens);
        
        if (!file.read(reinterpret_cast<char*>(data_.data()), size)) {
            throw std::runtime_error("Failed to read file");
        }
        
        std::cout << "Loaded " << num_tokens << " tokens from " << path << std::endl;
    }

    data::Sample get_item(size_t index) override {
        // Map index to a position in the data.
        size_t start_idx = index * seq_len_;
        if (start_idx + seq_len_ + 1 > data_.size()) {
            start_idx = 0; 
        }

        // Create input and target tensors
        std::vector<int32_t> input_vec(seq_len_);
        std::vector<int32_t> target_vec(seq_len_);
        
        for (int64_t i = 0; i < seq_len_; ++i) {
            input_vec[i] = static_cast<int32_t>(data_[start_idx + i]);
            target_vec[i] = static_cast<int32_t>(data_[start_idx + i + 1]);
        }
        
        Tensor input = empty({seq_len_}, DType::Int32, Device::CPU, false);
        input.copy_from_host(input_vec.data());
        
        Tensor target = empty({seq_len_}, DType::Int32, Device::CPU, false);
        target.copy_from_host(target_vec.data());
        
        return {input, target};
    }

    size_t size() const override {
        if (data_.size() <= seq_len_ + 1) return 0;
        return (data_.size() - 1) / seq_len_;
    }

private:
    std::vector<uint16_t> data_;
    int64_t seq_len_;
};

int main() {
    try {
        // Configuration
        int64_t seq_len = 2048;
        int64_t batch_size = 1; // Reduced to 1 to debug GPU hang
        int64_t total_steps = 100000;
        
        // Model Config (~250M params)
        models::TransformerConfig config;
        config.vocab_size = 32000;
        config.dim = 768;
        config.n_layers = 32;
        config.n_heads = 12;
        config.max_seq_len = seq_len;
        config.use_rms_norm = true;
        config.rope_base = 10000.0f;
        
        std::cout << "Model Configuration:\n" << config.describe() << std::endl;

        // Device
        Device device = Device::HIP; 
        // if (!device.is_available()) {
        //     std::cout << "CUDA not available, falling back to CPU (will be slow!)" << std::endl;
        //     device = Device::CPU;
        // }
        
        // Dataset
        std::string data_path = "data/tinystories/train.bin";
        auto dataset = std::make_shared<TinyStoriesDataset>(data_path, seq_len);
        data::DataLoader loader(dataset, batch_size, true /* shuffle */);
        
        // Model
        auto model = models::create_model(config);
        model->to(device);
        std::cout << "Model created with " << model->num_parameters() << " parameters." << std::endl;
        
        // Optimizer
        optim::Adam optimizer(model->parameters(), 0.0003f); 
        
        // Training Loop
        std::cout << "Starting training..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int step = 0;
        float total_loss = 0.0f;
        int log_interval = 10;
        
        while (step < total_steps) {
            for (auto batch : loader) {
                if (step >= total_steps) break;
                
                Tensor input = batch.first.to(device);
                Tensor target = batch.second.to(device);
                
                // Forward
                Tensor logits = model->forward(input);
                Tensor loss = model->compute_loss(logits, target);
                
                // Backward
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
                
                total_loss += loss.item<float>();
                
                if ((step + 1) % log_interval == 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                    float avg_loss = total_loss / log_interval;
                    float steps_per_sec = log_interval / (duration / 1000.0f);
                    
                    std::cout << "Step " << step + 1 << "/" << total_steps 
                              << " | Loss: " << avg_loss 
                              << " | Steps/s: " << steps_per_sec << std::endl;
                              
                    total_loss = 0.0f;
                    start_time = now;
                }
                
                step++;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
