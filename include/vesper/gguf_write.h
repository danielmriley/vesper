#pragma once

#include "vesper/gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vesper {

struct GgufKvWrite {
    enum class Kind { U32, U64, F32, Bool, String, U32Array, StringArray };
    Kind kind = Kind::U32;
    std::string key;
    std::uint64_t u = 0;
    double f = 0;
    bool b = false;
    std::string s;
    std::vector<std::uint32_t> u32s;
    std::vector<std::string> strings;
};

inline GgufKvWrite gguf_kv_u32(std::string key, std::uint32_t value) {
    GgufKvWrite kv;
    kv.kind = GgufKvWrite::Kind::U32;
    kv.key = std::move(key);
    kv.u = value;
    return kv;
}

inline GgufKvWrite gguf_kv_u64(std::string key, std::uint64_t value) {
    GgufKvWrite kv;
    kv.kind = GgufKvWrite::Kind::U64;
    kv.key = std::move(key);
    kv.u = value;
    return kv;
}

inline GgufKvWrite gguf_kv_f32(std::string key, float value) {
    GgufKvWrite kv;
    kv.kind = GgufKvWrite::Kind::F32;
    kv.key = std::move(key);
    kv.f = value;
    return kv;
}

inline GgufKvWrite gguf_kv_bool(std::string key, bool value) {
    GgufKvWrite kv;
    kv.kind = GgufKvWrite::Kind::Bool;
    kv.key = std::move(key);
    kv.b = value;
    return kv;
}

inline GgufKvWrite gguf_kv_string(std::string key, std::string value) {
    return GgufKvWrite{GgufKvWrite::Kind::String, std::move(key), 0, 0, false, std::move(value), {}, {}};
}

inline GgufKvWrite gguf_kv_u32_array(std::string key, std::vector<std::uint32_t> values) {
    return GgufKvWrite{GgufKvWrite::Kind::U32Array, std::move(key), 0, 0, false, {}, std::move(values), {}};
}

inline GgufKvWrite gguf_kv_string_array(std::string key, std::vector<std::string> values) {
    return GgufKvWrite{GgufKvWrite::Kind::StringArray, std::move(key), 0, 0, false, {}, {},
                       std::move(values)};
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
