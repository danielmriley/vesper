#pragma once
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace vesper {

// Represents a Python-style slice: start:stop:step
// None (std::nullopt) represents ":" (select all)
struct Slice {
    std::optional<int64_t> start;
    std::optional<int64_t> stop;
    std::optional<int64_t> step;

    Slice() : start(std::nullopt), stop(std::nullopt), step(std::nullopt) {} // ":"
    Slice(int64_t stop) : start(0), stop(stop), step(1) {} // "0:stop"
    Slice(int64_t start, int64_t stop, int64_t step = 1) : start(start), stop(stop), step(step) {}
    // Constructor for optionals
    Slice(std::optional<int64_t> start, std::optional<int64_t> stop, std::optional<int64_t> step = 1) 
        : start(start), stop(stop), step(step) {}
};

// A single index selector can be an integer or a slice
using IndexSelector = std::variant<int64_t, Slice>;

} // namespace vesper
