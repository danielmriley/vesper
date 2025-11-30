#include <vesper/models/transformer.h>
#include <vesper/data/dataset.h>
#include <vesper/data/dataloader.h>
#include <vesper/optim/adam.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/core/device.h>
#include <vesper/serialization.h>
#include <iostream>
#include <iomanip>
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
        int64_t seq_len = 512;       // Context length
        int64_t batch_size = 2;      // Batch size (adjust based on GPU memory)
        int64_t total_steps = 100000;
        int64_t save_interval = 5000;  // Save checkpoint every N steps
        std::string checkpoint_dir = "../checkpoints";
        
        // Model Config (~110M params - medium model)
        models::TransformerConfig config;
        config.vocab_size = 32000;
        config.dim = 768;            // Larger hidden dimension
        config.n_layers = 12;        // More layers
        config.n_heads = 12;         // More attention heads
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
        
        // Dataset - use path relative to source directory
        // When running from build/, we need to go up one level
        std::string data_path = "../data/tinystories/train.bin";
        auto dataset = std::make_shared<TinyStoriesDataset>(data_path, seq_len);
        data::DataLoader loader(dataset, batch_size, true /* shuffle */);
        
        // Model
        auto model = models::create_model(config);
        model->to(device);
        std::cout << "Model created with " << model->num_parameters() << " parameters." << std::endl;
        
        // Create checkpoint directory
        std::string mkdir_cmd = "mkdir -p " + checkpoint_dir;
        system(mkdir_cmd.c_str());
        
        // Optimizer (lower learning rate for larger model)
        optim::Adam optimizer(model->parameters(), 0.0001f); 
        
        // Training Loop
        std::cout << "Starting training..." << std::endl;
        std::cout << "  Total steps: " << total_steps << std::endl;
        std::cout << "  Batch size: " << batch_size << std::endl;
        std::cout << "  Sequence length: " << seq_len << std::endl;
        std::cout << "  Save interval: every " << save_interval << " steps" << std::endl;
        std::cout << "  Checkpoint dir: " << checkpoint_dir << std::endl;
        std::cout << "  Log interval: every 10 steps" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto training_start = start_time;
        
        int step = 0;
        float total_loss = 0.0f;
        int log_interval = 10;
        
        while (step < total_steps) {
            for (auto batch : loader) {
                if (step >= total_steps) break;
                
                Tensor input = batch.first.to(device);
                Tensor target = batch.second.to(device);
                
                // Forward + loss computation
                // Note: compute_loss takes input tokens and targets, not logits
                Tensor loss = model->compute_loss(input, target);
                
                // Backward
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
                
                float loss_val = loss.item<float>();
                total_loss += loss_val;
                
                // Print first step immediately for feedback
                if (step == 0) {
                    std::cout << "Step 1/" << total_steps 
                              << " | Loss: " << loss_val 
                              << " | (warming up...)" << std::endl;
                }
                else if ((step + 1) % log_interval == 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                    float avg_loss = total_loss / log_interval;
                    float steps_per_sec = (duration > 0) ? (log_interval * 1000.0f / duration) : 0.0f;
                    
                    // Calculate ETA
                    int64_t remaining_steps = total_steps - (step + 1);
                    float eta_seconds = (steps_per_sec > 0) ? (remaining_steps / steps_per_sec) : 0.0f;
                    int eta_hours = static_cast<int>(eta_seconds / 3600);
                    int eta_mins = static_cast<int>((eta_seconds - eta_hours * 3600) / 60);
                    int eta_secs = static_cast<int>(eta_seconds) % 60;
                    
                    // Progress percentage
                    float progress = 100.0f * (step + 1) / total_steps;
                    
                    std::cout << "Step " << step + 1 << "/" << total_steps 
                              << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                              << " | Loss: " << std::setprecision(4) << avg_loss 
                              << " | " << std::setprecision(2) << steps_per_sec << " steps/s"
                              << " | ETA: " << eta_hours << "h " << eta_mins << "m " << eta_secs << "s"
                              << std::endl;
                              
                    total_loss = 0.0f;
                    start_time = now;
                }
                
                // Save checkpoint periodically
                if ((step + 1) % save_interval == 0) {
                    std::string checkpoint_path = checkpoint_dir + "/checkpoint_step_" + std::to_string(step + 1) + ".bin";
                    std::cout << "Saving checkpoint to " << checkpoint_path << "..." << std::endl;
                    save(*model, checkpoint_path);
                    std::cout << "Checkpoint saved." << std::endl;
                }
                
                step++;
            }
        }
        
        // Save final model
        std::string final_path = checkpoint_dir + "/model_final.bin";
        std::cout << "Saving final model to " << final_path << "..." << std::endl;
        save(*model, final_path);
        std::cout << "Final model saved." << std::endl;
        
        // Final summary
        auto training_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(training_end - training_start).count();
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "Training complete!" << std::endl;
        std::cout << "  Total time: " << (total_duration / 3600) << "h " 
                  << ((total_duration % 3600) / 60) << "m " 
                  << (total_duration % 60) << "s" << std::endl;
        std::cout << "  Model saved to: " << final_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
