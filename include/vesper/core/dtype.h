#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace vesper {

enum class DType {
    Float32,
    Float64,
    Float16,  // Added for LLM support
    BFloat16, // Added for LLM support
    Int32,
    Int64
};

// Utility to get the size of a DType in bytes
inline size_t GetDTypeSize(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Float64: return sizeof(double);
        case DType::Float16: return 2; // 16-bit
        case DType::BFloat16: return 2; // 16-bit
        case DType::Int32:   return sizeof(int32_t);
        case DType::Int64:   return sizeof(int64_t);
    }
    // This part should ideally be unreachable
    throw std::runtime_error("Unknown DType");
}

// Utility for printing the DType
inline std::ostream& operator<<(std::ostream& os, const DType& dtype) {
    switch (dtype) {
        case DType::Float32: os << "Float32"; break;
        case DType::Float64: os << "Float64"; break;
        case DType::Float16: os << "Float16"; break;
        case DType::BFloat16: os << "BFloat16"; break;
        case DType::Int32:   os << "Int32"; break;
        case DType::Int64:   os << "Int64"; break;
    }
    return os;
}

} // namespace vesper
