#include <vesper/optim/optimizer.h>
#include <vesper/core/factories.h>
#include <vesper/nn/init.h> 

namespace vesper::optim {

Optimizer::Optimizer(std::vector<Tensor> params) : params_(std::move(params)) {
    state_.resize(params_.size());
}

void Optimizer::zero_grad() {
    for (auto& param : params_) {
        if (param.requires_grad()) {
            // param.grad() ensures the gradient tensor exists (allocating zeros if needed).
            // We then explicitly fill it with zeros to clear any accumulated gradients.
            // This operates on the shared gradient tensor, so it affects the model's parameters.
            vesper::nn::init::zeros_(param.grad());
        }
    }
}

StateDict Optimizer::state_dict() const {
    StateDict dict;
    // Save LR as a scalar tensor
    dict["defaults.lr"] = full({1}, DType::Float32, Device::CPU, get_lr());
    
    // Save Parameter State
    for (size_t i = 0; i < state_.size(); ++i) {
        for (const auto& [key, tensor] : state_[i]) {
            std::string full_key = "state." + std::to_string(i) + "." + key;
            dict[full_key] = tensor;
        }
    }
    return dict;
}

void Optimizer::load_state_dict(const StateDict& dict) {
    if (dict.count("defaults.lr")) {
        set_lr(dict.at("defaults.lr").item<float>());
    }
    
    // Clear current state? No, usually merge or overwrite.
    // For simplicity, we overwrite/insert.
    
    for (const auto& [key, tensor] : dict) {
        if (key.rfind("state.", 0) == 0) { // Starts with "state."
            // Format: state.<index>.<name>
            size_t first_dot = 5; // "state" len is 5. index starts at 6. "state."
            size_t second_dot = key.find('.', first_dot + 1);
            if (second_dot == std::string::npos) continue;
            
            std::string idx_str = key.substr(first_dot + 1, second_dot - (first_dot + 1));
            std::string name = key.substr(second_dot + 1);
            
            try {
                size_t idx = std::stoul(idx_str);
                if (idx < state_.size()) {
                    state_[idx][name] = tensor;
                }
            } catch (...) {
                // Ignore parse errors
            }
        }
    }
}

} // namespace vesper::optim
