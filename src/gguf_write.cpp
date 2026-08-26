#include "vesper/gguf_write.h"

#include "vesper/types.h"

#include <cstdint>
#include <fstream>
#include <vector>

namespace vesper {
namespace {

void write_raw(std::ostream& out, const void* data, std::size_t n) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    check(static_cast<bool>(out), "GGUF writer failed");
}

template <typename T>
void write_le(std::ostream& out, T value) {
    write_raw(out, &value, sizeof(value));
}

void write_string(std::ostream& out, const std::string& s) {
    write_le<std::uint64_t>(out, s.size());
    if (!s.empty()) {
        write_raw(out, s.data(), s.size());
    }
}

std::uint32_t alignment_of(const std::vector<GgufKvWrite>& kvs) {
    for (const GgufKvWrite& kv : kvs) {
        if (kv.key != "general.alignment") {
            continue;
        }
        if (kv.kind != GgufKvWrite::Kind::U32 && kv.kind != GgufKvWrite::Kind::U64) {
            fail("general.alignment must be u32 or u64");
        }
        if (kv.u == 0 || kv.u > 0xffffffffu) {
            fail("invalid general.alignment");
        }
        return static_cast<std::uint32_t>(kv.u);
    }
    return 32;
}

}  // namespace

void write_gguf(const std::string& path, const std::vector<GgufKvWrite>& kvs,
                const std::vector<GgufTensorWrite>& tensors) {
    const std::uint32_t alignment = alignment_of(kvs);
    std::vector<std::uint64_t> offsets(tensors.size(), 0);
    std::uint64_t cursor = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const GgufTensorWrite& tensor = tensors[i];
        const std::uint64_t nbytes =
            ggml_nbytes(tensor.type, tensor.dims.data(), static_cast<int>(tensor.dims.size()));
        check(tensor.bytes.size() == nbytes, "GGUF writer payload size");
        const std::uint64_t rem = cursor % alignment;
        if (rem != 0) {
            cursor += static_cast<std::uint64_t>(alignment) - rem;
        }
        offsets[i] = cursor;
        cursor += nbytes;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    check(static_cast<bool>(out), "open GGUF path failed: " + path);

    write_le<std::uint32_t>(out, 0x46554747);
    write_le<std::uint32_t>(out, 3);
    write_le<std::uint64_t>(out, tensors.size());
    write_le<std::uint64_t>(out, kvs.size());

    for (const GgufKvWrite& kv : kvs) {
        write_string(out, kv.key);
        switch (kv.kind) {
            case GgufKvWrite::Kind::U32:
                write_le<std::uint32_t>(out, static_cast<std::uint32_t>(GgufValType::UINT32));
                write_le<std::uint32_t>(out, static_cast<std::uint32_t>(kv.u));
                break;
            case GgufKvWrite::Kind::U64:
                write_le<std::uint32_t>(out, static_cast<std::uint32_t>(GgufValType::UINT64));
                write_le<std::uint64_t>(out, kv.u);
                break;
            case GgufKvWrite::Kind::F32: {
                write_le<std::uint32_t>(out, static_cast<std::uint32_t>(GgufValType::FLOAT32));
                const float fv = static_cast<float>(kv.f);
                write_le<float>(out, fv);
                break;
            }
            case GgufKvWrite::Kind::Bool:
                write_le<std::uint32_t>(out, static_cast<std::uint32_t>(GgufValType::BOOL));
                write_le<std::uint8_t>(out, kv.b ? 1 : 0);
                break;
            case GgufKvWrite::Kind::String:
                write_le<std::uint32_t>(out, static_cast<std::uint32_t>(GgufValType::STRING));
                write_string(out, kv.s);
                break;
        }
    }

    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const GgufTensorWrite& tensor = tensors[i];
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
    check(static_cast<bool>(out), "GGUF writer flush failed");
}

}  // namespace vesper
