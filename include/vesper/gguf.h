#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace vesper {

enum class GgmlType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
};

enum class GgufValType : std::uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

const char* ggml_type_name(GgmlType type);
bool ggml_type_known(GgmlType type);
std::uint64_t ggml_block_elems(GgmlType type);
std::uint64_t ggml_block_bytes(GgmlType type);
std::uint64_t ggml_nbytes(GgmlType type, const std::uint64_t* dims, int n_dims);

struct GgufTensor {
    std::string name;
    GgmlType type = GgmlType::F32;
    std::vector<std::uint64_t> dims;
    std::uint64_t offset = 0;
    std::uint64_t nbytes = 0;
    const std::byte* data = nullptr;
};

class GgufFile {
public:
    static GgufFile open(const std::string& path);
    GgufFile(GgufFile&&) noexcept;
    GgufFile& operator=(GgufFile&&) noexcept;
    ~GgufFile();
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    std::uint32_t version() const;
    std::uint32_t alignment() const;
    std::uint64_t file_size() const;
    std::string architecture() const;
    const std::vector<GgufTensor>& tensors() const;
    const GgufTensor* find(std::string_view name) const;

    bool has_kv(std::string_view key) const;
    std::uint64_t kv_u64(std::string_view key) const;
    std::int64_t kv_i64(std::string_view key) const;
    double kv_f64(std::string_view key) const;
    bool kv_bool(std::string_view key) const;
    std::string kv_string(std::string_view key) const;
    std::vector<std::uint64_t> kv_u64_array(std::string_view key) const;
    std::vector<std::string> kv_string_array(std::string_view key) const;
    std::size_t kv_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit GgufFile(std::unique_ptr<Impl> impl);
};

}  // namespace vesper
