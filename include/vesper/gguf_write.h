#pragma once

#include "vesper/gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vesper {

struct GgufKvWrite {
    enum class Kind { U32, U64, F32, Bool, String };
    Kind kind = Kind::U32;
    std::string key;
    std::uint64_t u = 0;
    double f = 0;
    bool b = false;
    std::string s;
};

inline GgufKvWrite gguf_kv_u32(std::string key, std::uint32_t value) {
    return GgufKvWrite{GgufKvWrite::Kind::U32, std::move(key), value, 0, false, {}};
}

inline GgufKvWrite gguf_kv_u64(std::string key, std::uint64_t value) {
    return GgufKvWrite{GgufKvWrite::Kind::U64, std::move(key), value, 0, false, {}};
}

inline GgufKvWrite gguf_kv_f32(std::string key, float value) {
    return GgufKvWrite{GgufKvWrite::Kind::F32, std::move(key), 0, value, false, {}};
}

inline GgufKvWrite gguf_kv_bool(std::string key, bool value) {
    return GgufKvWrite{GgufKvWrite::Kind::Bool, std::move(key), 0, 0, value, {}};
}

inline GgufKvWrite gguf_kv_string(std::string key, std::string value) {
    return GgufKvWrite{GgufKvWrite::Kind::String, std::move(key), 0, 0, false, std::move(value)};
}

struct GgufTensorWrite {
    std::string name;
    GgmlType type = GgmlType::F32;
    std::vector<std::uint64_t> dims;
    std::vector<std::byte> bytes;
};

void write_gguf(const std::string& path, const std::vector<GgufKvWrite>& kvs,
                const std::vector<GgufTensorWrite>& tensors);

}  // namespace vesper
