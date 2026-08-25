#pragma once

#include "vesper/gguf.h"
#include "vesper/types.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace vesper::gguf_test {

struct Kv {
    enum class Kind { U32, U64, String };
    Kind kind = Kind::U32;
    std::string key;
    std::uint64_t u = 0;
    std::string s;
};

inline Kv kv_u32(std::string key, std::uint32_t value) {
    return Kv{Kv::Kind::U32, std::move(key), value, {}};
}

inline Kv kv_u64(std::string key, std::uint64_t value) {
    return Kv{Kv::Kind::U64, std::move(key), value, {}};
}

inline Kv kv_string(std::string key, std::string value) {
    return Kv{Kv::Kind::String, std::move(key), 0, std::move(value)};
}

struct TensorSpec {
    std::string name;
    GgmlType type = GgmlType::F32;
    std::vector<std::uint64_t> dims;
    std::vector<std::byte> bytes;
};

inline void write_raw(std::ostream& out, const void* data, std::size_t n) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    check(static_cast<bool>(out), "GGUF test writer failed");
}

template <typename T>
void write_le(std::ostream& out, T value) {
    write_raw(out, &value, sizeof(value));
}

inline void write_string(std::ostream& out, const std::string& s) {
    write_le<std::uint64_t>(out, s.size());
    if (!s.empty()) {
        write_raw(out, s.data(), s.size());
    }
}

inline std::uint32_t alignment_of(const std::vector<Kv>& kvs) {
    for (const Kv& kv : kvs) {
        if (kv.key != "general.alignment") {
            continue;
        }
        if (kv.kind != Kv::Kind::U32 && kv.kind != Kv::Kind::U64) {
            fail("general.alignment must be u32 or u64");
        }
        if (kv.u == 0 || kv.u > 0xffffffffu) {
            fail("invalid general.alignment");
        }
        return static_cast<std::uint32_t>(kv.u);
    }
    return 32;
}

inline void write_gguf(const std::string& path, const std::vector<Kv>& kvs,
                       const std::vector<TensorSpec>& tensors) {
    const std::uint32_t alignment = alignment_of(kvs);
    std::vector<std::uint64_t> offsets(tensors.size(), 0);
    std::uint64_t cursor = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const TensorSpec& tensor = tensors[i];
        const std::uint64_t nbytes =
            ggml_nbytes(tensor.type, tensor.dims.data(), static_cast<int>(tensor.dims.size()));
        check(tensor.bytes.size() == nbytes, "GGUF test writer payload size");
        const std::uint64_t rem = cursor % alignment;
        if (rem != 0) {
            cursor += static_cast<std::uint64_t>(alignment) - rem;
        }
        offsets[i] = cursor;
        cursor += nbytes;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    check(static_cast<bool>(out), "open GGUF test path failed: " + path);

    write_le<std::uint32_t>(out, 0x46554747);
    write_le<std::uint32_t>(out, 3);
    write_le<std::uint64_t>(out, tensors.size());
    write_le<std::uint64_t>(out, kvs.size());

    for (const Kv& kv : kvs) {
        write_string(out, kv.key);
        switch (kv.kind) {
            case Kv::Kind::U32:
                write_le<std::uint32_t>(out, static_cast<std::uint32_t>(GgufValType::UINT32));
                write_le<std::uint32_t>(out, static_cast<std::uint32_t>(kv.u));
                break;
            case Kv::Kind::U64:
                write_le<std::uint32_t>(out, static_cast<std::uint32_t>(GgufValType::UINT64));
                write_le<std::uint64_t>(out, kv.u);
                break;
            case Kv::Kind::String:
                write_le<std::uint32_t>(out, static_cast<std::uint32_t>(GgufValType::STRING));
                write_string(out, kv.s);
                break;
        }
    }

    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const TensorSpec& tensor = tensors[i];
        write_string(out, tensor.name);
        write_le<std::uint32_t>(out, static_cast<std::uint32_t>(tensor.dims.size()));
        for (std::uint64_t dim : tensor.dims) {
            write_le<std::uint64_t>(out, dim);
        }
        write_le<std::uint32_t>(out, static_cast<std::uint32_t>(tensor.type));
        write_le<std::uint64_t>(out, offsets[i]);
    }

    const std::uint64_t header_end = static_cast<std::uint64_t>(out.tellp());
    const std::uint64_t rem = header_end % alignment;
    if (rem != 0) {
        const std::vector<char> pad(static_cast<std::size_t>(alignment - rem), 0);
        write_raw(out, pad.data(), pad.size());
    }

    std::uint64_t data_cursor = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (offsets[i] > data_cursor) {
            const std::vector<char> pad(static_cast<std::size_t>(offsets[i] - data_cursor), 0);
            write_raw(out, pad.data(), pad.size());
            data_cursor = offsets[i];
        }
        if (!tensors[i].bytes.empty()) {
            write_raw(out, tensors[i].bytes.data(), tensors[i].bytes.size());
            data_cursor += tensors[i].bytes.size();
        }
    }
    check(static_cast<bool>(out), "GGUF test writer flush failed");
}

}  // namespace vesper::gguf_test
