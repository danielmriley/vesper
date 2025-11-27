#pragma once

#include <vesper/nn/module.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>

namespace vesper::nn {

/**
 * A type-safe container for holding submodules in a list.
 * 
 * ModuleList automatically handles:
 * - Registration of submodules with the parent
 * - Proper device movement via to()
 * - Parameter collection for optimizers
 * - State dict serialization
 * 
 * Unlike raw std::vector<T>, ModuleList uses unique_ptr internally to ensure
 * stable addresses, making it safe with the pointer-based parameter registration.
 * 
 * Example usage:
 * @code
 * class TransformerStack : public Module {
 * public:
 *     TransformerStack(int num_layers, int d_model) {
 *         for (int i = 0; i < num_layers; ++i) {
 *             layers.append(TransformerBlock(d_model));
 *         }
 *         register_module_list("layers", &layers);
 *     }
 *     
 *     Tensor forward(const Tensor& x) override {
 *         Tensor out = x;
 *         for (auto& layer : layers) {
 *             out = layer.forward(out);
 *         }
 *         return out;
 *     }
 *     
 *     ModuleList<TransformerBlock> layers;
 * };
 * @endcode
 */
template<typename T>
class ModuleList {
    static_assert(std::is_base_of_v<Module, T>, 
        "ModuleList can only hold types derived from Module");

public:
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;

    // Default constructor
    ModuleList() = default;

    // Constructor with initializer list
    ModuleList(std::initializer_list<T> modules) {
        for (const auto& m : modules) {
            append(m);
        }
    }

    // Disable copy (modules shouldn't be copied due to parameter registration)
    ModuleList(const ModuleList&) = delete;
    ModuleList& operator=(const ModuleList&) = delete;

    // Move constructor and assignment
    ModuleList(ModuleList&& other) noexcept = default;
    ModuleList& operator=(ModuleList&& other) noexcept = default;

    // --- Element Access ---
    
    reference operator[](size_type index) {
        if (index >= modules_.size()) {
            throw std::out_of_range("ModuleList index out of range");
        }
        return *modules_[index];
    }

    const_reference operator[](size_type index) const {
        if (index >= modules_.size()) {
            throw std::out_of_range("ModuleList index out of range");
        }
        return *modules_[index];
    }

    reference at(size_type index) {
        return operator[](index);
    }

    const_reference at(size_type index) const {
        return operator[](index);
    }

    reference front() { return *modules_.front(); }
    const_reference front() const { return *modules_.front(); }
    
    reference back() { return *modules_.back(); }
    const_reference back() const { return *modules_.back(); }

    // --- Capacity ---
    
    bool empty() const noexcept { return modules_.empty(); }
    size_type size() const noexcept { return modules_.size(); }

    // --- Modifiers ---
    
    // Append a module (by copy)
    void append(const T& module) {
        modules_.push_back(std::make_unique<T>(module));
    }

    // Append a module (by move)
    void append(T&& module) {
        modules_.push_back(std::make_unique<T>(std::move(module)));
    }

    // Emplace a module in-place
    template<typename... Args>
    T& emplace_back(Args&&... args) {
        modules_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        return *modules_.back();
    }

    // Insert at a specific position
    void insert(size_type index, const T& module) {
        if (index > modules_.size()) {
            throw std::out_of_range("ModuleList insert index out of range");
        }
        modules_.insert(modules_.begin() + index, std::make_unique<T>(module));
    }

    void insert(size_type index, T&& module) {
        if (index > modules_.size()) {
            throw std::out_of_range("ModuleList insert index out of range");
        }
        modules_.insert(modules_.begin() + index, std::make_unique<T>(std::move(module)));
    }

    // Remove the last element
    void pop_back() {
        if (modules_.empty()) {
            throw std::out_of_range("ModuleList pop_back on empty list");
        }
        modules_.pop_back();
    }

    // Remove element at index
    void erase(size_type index) {
        if (index >= modules_.size()) {
            throw std::out_of_range("ModuleList erase index out of range");
        }
        modules_.erase(modules_.begin() + index);
    }

    // Clear all modules
    void clear() noexcept {
        modules_.clear();
    }

    // --- Iterator Support ---
    
    // Custom iterator that dereferences unique_ptr
    class iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T*;
        using reference = T&;

        iterator(typename std::vector<std::unique_ptr<T>>::iterator it) : it_(it) {}

        reference operator*() const { return **it_; }
        pointer operator->() const { return it_->get(); }

        iterator& operator++() { ++it_; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++it_; return tmp; }
        iterator& operator--() { --it_; return *this; }
        iterator operator--(int) { iterator tmp = *this; --it_; return tmp; }

        iterator& operator+=(difference_type n) { it_ += n; return *this; }
        iterator& operator-=(difference_type n) { it_ -= n; return *this; }
        iterator operator+(difference_type n) const { return iterator(it_ + n); }
        iterator operator-(difference_type n) const { return iterator(it_ - n); }
        difference_type operator-(const iterator& other) const { return it_ - other.it_; }

        reference operator[](difference_type n) const { return **(it_ + n); }

        bool operator==(const iterator& other) const { return it_ == other.it_; }
        bool operator!=(const iterator& other) const { return it_ != other.it_; }
        bool operator<(const iterator& other) const { return it_ < other.it_; }
        bool operator<=(const iterator& other) const { return it_ <= other.it_; }
        bool operator>(const iterator& other) const { return it_ > other.it_; }
        bool operator>=(const iterator& other) const { return it_ >= other.it_; }

    private:
        typename std::vector<std::unique_ptr<T>>::iterator it_;
    };

    class const_iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = const T;
        using pointer = const T*;
        using reference = const T&;

        const_iterator(typename std::vector<std::unique_ptr<T>>::const_iterator it) : it_(it) {}

        reference operator*() const { return **it_; }
        pointer operator->() const { return it_->get(); }

        const_iterator& operator++() { ++it_; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++it_; return tmp; }
        const_iterator& operator--() { --it_; return *this; }
        const_iterator operator--(int) { const_iterator tmp = *this; --it_; return tmp; }

        const_iterator& operator+=(difference_type n) { it_ += n; return *this; }
        const_iterator& operator-=(difference_type n) { it_ -= n; return *this; }
        const_iterator operator+(difference_type n) const { return const_iterator(it_ + n); }
        const_iterator operator-(difference_type n) const { return const_iterator(it_ - n); }
        difference_type operator-(const const_iterator& other) const { return it_ - other.it_; }

        reference operator[](difference_type n) const { return **(it_ + n); }

        bool operator==(const const_iterator& other) const { return it_ == other.it_; }
        bool operator!=(const const_iterator& other) const { return it_ != other.it_; }
        bool operator<(const const_iterator& other) const { return it_ < other.it_; }
        bool operator<=(const const_iterator& other) const { return it_ <= other.it_; }
        bool operator>(const const_iterator& other) const { return it_ > other.it_; }
        bool operator>=(const const_iterator& other) const { return it_ >= other.it_; }

    private:
        typename std::vector<std::unique_ptr<T>>::const_iterator it_;
    };

    iterator begin() { return iterator(modules_.begin()); }
    iterator end() { return iterator(modules_.end()); }
    const_iterator begin() const { return const_iterator(modules_.begin()); }
    const_iterator end() const { return const_iterator(modules_.end()); }
    const_iterator cbegin() const { return const_iterator(modules_.cbegin()); }
    const_iterator cend() const { return const_iterator(modules_.cend()); }

    // --- Module Integration ---
    
    // Get raw pointer to module at index (for registration)
    pointer get_ptr(size_type index) {
        if (index >= modules_.size()) {
            throw std::out_of_range("ModuleList get_ptr index out of range");
        }
        return modules_[index].get();
    }

    // Get all module pointers (for parameter collection)
    std::vector<Module*> module_ptrs() {
        std::vector<Module*> ptrs;
        ptrs.reserve(modules_.size());
        for (auto& m : modules_) {
            ptrs.push_back(static_cast<Module*>(m.get()));
        }
        return ptrs;
    }

    // Collect all parameters from all modules
    std::vector<Tensor> parameters() {
        std::vector<Tensor> params;
        for (auto& m : modules_) {
            auto sub_params = m->parameters();
            params.insert(params.end(), sub_params.begin(), sub_params.end());
        }
        return params;
    }

    // Collect all parameter pointers
    std::vector<Tensor*> parameters_ptrs() {
        std::vector<Tensor*> params;
        for (auto& m : modules_) {
            auto sub_params = m->parameters_ptrs();
            params.insert(params.end(), sub_params.begin(), sub_params.end());
        }
        return params;
    }

    // Move all modules to a device
    void to(Device device, bool non_blocking = false) {
        for (auto& m : modules_) {
            m->to(device, non_blocking);
        }
    }

    // Set training mode
    void train(bool mode = true) {
        for (auto& m : modules_) {
            m->train(mode);
        }
    }

    void eval() {
        train(false);
    }

    // Zero all gradients
    void zero_grad() {
        for (auto& m : modules_) {
            m->zero_grad();
        }
    }

private:
    std::vector<std::unique_ptr<T>> modules_;
};

} // namespace vesper::nn
