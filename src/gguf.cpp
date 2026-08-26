#include "vesper/gguf.h"

#include "vesper/types.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <type_traits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace vesper {
namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747;
constexpr std::uint32_t kGgufVersion = 3;
constexpr std::uint32_t kDefaultAlignment = 32;
constexpr int kMaxDims = 4;

struct KvValue {
    GgufValType type = GgufValType::UINT8;
    GgufValType elem = GgufValType::UINT8;
    std::uint64_t u = 0;
    std::int64_t i = 0;
    double f = 0;
    bool b = false;
    std::string s;
    std::vector<std::uint64_t> u_arr;
    std::vector<std::int64_t> i_arr;
    std::vector<double> f_arr;
    std::vector<std::string> s_arr;
};

bool gguf_val_type_known(GgufValType type) {
    switch (type) {
        case GgufValType::UINT8:
        case GgufValType::INT8:
        case GgufValType::UINT16:
        case GgufValType::INT16:
        case GgufValType::UINT32:
        case GgufValType::INT32:
        case GgufValType::FLOAT32:
        case GgufValType::BOOL:
        case GgufValType::STRING:
        case GgufValType::ARRAY:
        case GgufValType::UINT64:
        case GgufValType::INT64:
        case GgufValType::FLOAT64:
            return true;
    }
    return false;
}

class Cursor {
public:
    Cursor(const std::byte* base, std::size_t size) : base_(base), size_(size) {}

    std::size_t pos() const { return pos_; }

    void need(std::uint64_t n) const {
        if (n > size_ - pos_) {
            fail("truncated GGUF");
        }
    }

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        need(sizeof(T));
        T value{};
        std::memcpy(&value, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return value;
    }

    std::string read_string() {
        const std::uint64_t len = read<std::uint64_t>();
        if (len > size_ - pos_) {
            fail("truncated GGUF string");
        }
        std::string out(reinterpret_cast<const char*>(base_ + pos_), static_cast<std::size_t>(len));
        pos_ += static_cast<std::size_t>(len);
        return out;
    }

private:
    const std::byte* base_;
    std::size_t size_;
    std::size_t pos_ = 0;
};

KvValue read_value(Cursor& cur, GgufValType type) {
    if (!gguf_val_type_known(type)) {
        fail("unknown GGUF value type " + std::to_string(static_cast<std::uint32_t>(type)));
    }
    KvValue value;
    value.type = type;
    switch (type) {
        case GgufValType::UINT8:
            value.u = cur.read<std::uint8_t>();
            return value;
        case GgufValType::INT8:
            value.i = cur.read<std::int8_t>();
            return value;
        case GgufValType::UINT16:
            value.u = cur.read<std::uint16_t>();
            return value;
        case GgufValType::INT16:
            value.i = cur.read<std::int16_t>();
            return value;
        case GgufValType::UINT32:
            value.u = cur.read<std::uint32_t>();
            return value;
        case GgufValType::INT32:
            value.i = cur.read<std::int32_t>();
            return value;
        case GgufValType::FLOAT32:
            value.f = static_cast<double>(cur.read<float>());
            return value;
        case GgufValType::BOOL: {
            const std::uint8_t raw = cur.read<std::uint8_t>();
            if (raw > 1) {
                fail("GGUF bool must be 0 or 1");
            }
            value.b = raw != 0;
            return value;
        }
        case GgufValType::STRING:
            value.s = cur.read_string();
            return value;
        case GgufValType::ARRAY: {
            const auto elem = static_cast<GgufValType>(cur.read<std::uint32_t>());
            if (elem == GgufValType::ARRAY) {
                fail("nested GGUF array");
            }
            if (!gguf_val_type_known(elem)) {
                fail("unknown GGUF array element type");
            }
            const std::uint64_t count = cur.read<std::uint64_t>();
            value.elem = elem;
            value.u = count;
            switch (elem) {
                case GgufValType::UINT8:
                case GgufValType::UINT16:
                case GgufValType::UINT32:
                case GgufValType::UINT64:
                    value.u_arr.reserve(static_cast<std::size_t>(count));
                    for (std::uint64_t i = 0; i < count; ++i) {
                        value.u_arr.push_back(read_value(cur, elem).u);
                    }
                    return value;
                case GgufValType::INT8:
                case GgufValType::INT16:
                case GgufValType::INT32:
                case GgufValType::INT64:
                    value.i_arr.reserve(static_cast<std::size_t>(count));
                    for (std::uint64_t i = 0; i < count; ++i) {
                        value.i_arr.push_back(read_value(cur, elem).i);
                    }
                    return value;
                case GgufValType::FLOAT32:
                case GgufValType::FLOAT64:
                    value.f_arr.reserve(static_cast<std::size_t>(count));
                    for (std::uint64_t i = 0; i < count; ++i) {
                        value.f_arr.push_back(read_value(cur, elem).f);
                    }
                    return value;
                case GgufValType::BOOL:
                    value.u_arr.reserve(static_cast<std::size_t>(count));
                    for (std::uint64_t i = 0; i < count; ++i) {
                        value.u_arr.push_back(read_value(cur, elem).b ? 1 : 0);
                    }
                    return value;
                case GgufValType::STRING:
                    value.s_arr.reserve(static_cast<std::size_t>(count));
                    for (std::uint64_t i = 0; i < count; ++i) {
                        value.s_arr.push_back(read_value(cur, elem).s);
                    }
                    return value;
                case GgufValType::ARRAY:
                    fail("nested GGUF array");
            }
            fail("unknown GGUF array element type");
        }
        case GgufValType::UINT64:
            value.u = cur.read<std::uint64_t>();
            return value;
        case GgufValType::INT64:
            value.i = cur.read<std::int64_t>();
            return value;
        case GgufValType::FLOAT64:
            value.f = cur.read<double>();
            return value;
    }
    fail("unknown GGUF value type");
}

std::uint64_t align_up(std::uint64_t value, std::uint32_t alignment) {
    if (alignment == 0) {
        fail("GGUF alignment must be non-zero");
    }
    const std::uint64_t rem = value % alignment;
    if (rem == 0) {
        return value;
    }
    const std::uint64_t pad = static_cast<std::uint64_t>(alignment) - rem;
    if (value > std::numeric_limits<std::uint64_t>::max() - pad) {
        fail("GGUF alignment overflow");
    }
    return value + pad;
}

const KvValue* find_kv(const std::unordered_map<std::string, KvValue>& kv, std::string_view key) {
    const auto it = kv.find(std::string(key));
    if (it == kv.end()) {
        return nullptr;
    }
    return &it->second;
}

const KvValue& require_kv(const std::unordered_map<std::string, KvValue>& kv, std::string_view key) {
    const KvValue* value = find_kv(kv, key);
    if (value == nullptr) {
        fail("missing GGUF key: " + std::string(key));
    }
    return *value;
}

}  // namespace

struct GgufFile::Impl {
    int fd = -1;
    const std::byte* map = nullptr;
    std::size_t map_size = 0;
    std::uint32_t version = 0;
    std::uint32_t alignment = kDefaultAlignment;
    std::uint64_t file_size = 0;
    bool payloads_complete = true;
    std::unordered_map<std::string, KvValue> kv;
    std::vector<GgufTensor> tensors;
    std::unordered_map<std::string, std::size_t> by_name;

    ~Impl() {
        if (map != nullptr) {
            munmap(const_cast<std::byte*>(map), map_size);
            map = nullptr;
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }

    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
};

const char* ggml_type_name(GgmlType type) {
    switch (type) {
        case GgmlType::F32:
            return "F32";
        case GgmlType::F16:
            return "F16";
        case GgmlType::Q4_0:
            return "Q4_0";
        case GgmlType::Q4_1:
            return "Q4_1";
        case GgmlType::Q5_0:
            return "Q5_0";
        case GgmlType::Q5_1:
            return "Q5_1";
        case GgmlType::Q8_0:
            return "Q8_0";
        case GgmlType::Q8_1:
            return "Q8_1";
        case GgmlType::Q2_K:
            return "Q2_K";
        case GgmlType::Q3_K:
            return "Q3_K";
        case GgmlType::Q4_K:
            return "Q4_K";
        case GgmlType::Q5_K:
            return "Q5_K";
        case GgmlType::Q6_K:
            return "Q6_K";
        case GgmlType::Q8_K:
            return "Q8_K";
        case GgmlType::IQ2_XXS:
            return "IQ2_XXS";
        case GgmlType::IQ2_XS:
            return "IQ2_XS";
        case GgmlType::IQ3_XXS:
            return "IQ3_XXS";
        case GgmlType::IQ1_S:
            return "IQ1_S";
        case GgmlType::IQ4_NL:
            return "IQ4_NL";
        case GgmlType::IQ3_S:
            return "IQ3_S";
        case GgmlType::IQ2_S:
            return "IQ2_S";
        case GgmlType::IQ4_XS:
            return "IQ4_XS";
        case GgmlType::I8:
            return "I8";
        case GgmlType::I16:
            return "I16";
        case GgmlType::I32:
            return "I32";
        case GgmlType::I64:
            return "I64";
        case GgmlType::F64:
            return "F64";
        case GgmlType::IQ1_M:
            return "IQ1_M";
        case GgmlType::BF16:
            return "BF16";
    }
    fail("unknown ggml type");
}

bool ggml_type_known(GgmlType type) {
    switch (type) {
        case GgmlType::F32:
        case GgmlType::F16:
        case GgmlType::Q4_0:
        case GgmlType::Q4_1:
        case GgmlType::Q5_0:
        case GgmlType::Q5_1:
        case GgmlType::Q8_0:
        case GgmlType::Q8_1:
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K:
        case GgmlType::Q8_K:
        case GgmlType::IQ2_XXS:
        case GgmlType::IQ2_XS:
        case GgmlType::IQ3_XXS:
        case GgmlType::IQ1_S:
        case GgmlType::IQ4_NL:
        case GgmlType::IQ3_S:
        case GgmlType::IQ2_S:
        case GgmlType::IQ4_XS:
        case GgmlType::I8:
        case GgmlType::I16:
        case GgmlType::I32:
        case GgmlType::I64:
        case GgmlType::F64:
        case GgmlType::IQ1_M:
        case GgmlType::BF16:
            return true;
    }
    return false;
}

std::uint64_t ggml_block_elems(GgmlType type) {
    switch (type) {
        case GgmlType::F32:
        case GgmlType::F16:
        case GgmlType::I8:
        case GgmlType::I16:
        case GgmlType::I32:
        case GgmlType::I64:
        case GgmlType::F64:
        case GgmlType::BF16:
            return 1;
        case GgmlType::Q4_0:
        case GgmlType::Q4_1:
        case GgmlType::Q5_0:
        case GgmlType::Q5_1:
        case GgmlType::Q8_0:
        case GgmlType::Q8_1:
        case GgmlType::IQ4_NL:
            return 32;
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K:
        case GgmlType::Q8_K:
        case GgmlType::IQ2_XXS:
        case GgmlType::IQ2_XS:
        case GgmlType::IQ3_XXS:
        case GgmlType::IQ1_S:
        case GgmlType::IQ3_S:
        case GgmlType::IQ2_S:
        case GgmlType::IQ4_XS:
        case GgmlType::IQ1_M:
            return 256;
    }
    fail("unknown ggml type");
}

std::uint64_t ggml_block_bytes(GgmlType type) {
    switch (type) {
        case GgmlType::F32:
            return 4;
        case GgmlType::F16:
            return 2;
        case GgmlType::Q4_0:
            return 18;
        case GgmlType::Q4_1:
            return 20;
        case GgmlType::Q5_0:
            return 22;
        case GgmlType::Q5_1:
            return 24;
        case GgmlType::Q8_0:
            return 34;
        case GgmlType::Q8_1:
            return 36;
        case GgmlType::Q2_K:
            return 84;
        case GgmlType::Q3_K:
            return 110;
        case GgmlType::Q4_K:
            return 144;
        case GgmlType::Q5_K:
            return 176;
        case GgmlType::Q6_K:
            return 210;
        case GgmlType::Q8_K:
            return 292;
        case GgmlType::IQ2_XXS:
            return 66;
        case GgmlType::IQ2_XS:
            return 74;
        case GgmlType::IQ3_XXS:
            return 98;
        case GgmlType::IQ1_S:
            return 50;
        case GgmlType::IQ4_NL:
            return 18;
        case GgmlType::IQ3_S:
            return 110;
        case GgmlType::IQ2_S:
            return 82;
        case GgmlType::IQ4_XS:
            return 136;
        case GgmlType::I8:
            return 1;
        case GgmlType::I16:
            return 2;
        case GgmlType::I32:
            return 4;
        case GgmlType::I64:
            return 8;
        case GgmlType::F64:
            return 8;
        case GgmlType::IQ1_M:
            return 56;
        case GgmlType::BF16:
            return 2;
    }
    fail("unknown ggml type");
}

std::uint64_t ggml_nbytes(GgmlType type, const std::uint64_t* dims, int n_dims) {
    if (n_dims < 1 || n_dims > kMaxDims) {
        fail("ggml n_dims must be 1..4");
    }
    if (dims == nullptr) {
        fail("ggml dims is null");
    }
    std::uint64_t ne = 1;
    for (int i = 0; i < n_dims; ++i) {
        if (dims[i] != 0 && ne > std::numeric_limits<std::uint64_t>::max() / dims[i]) {
            fail("ggml dim overflow");
        }
        ne *= dims[i];
    }
    const std::uint64_t block_elems = ggml_block_elems(type);
    const std::uint64_t block_bytes = ggml_block_bytes(type);
    if (ne % block_elems != 0) {
        fail("tensor element count is not a multiple of the ggml block");
    }
    const std::uint64_t blocks = ne / block_elems;
    if (block_bytes != 0 && blocks > std::numeric_limits<std::uint64_t>::max() / block_bytes) {
        fail("ggml byte size overflow");
    }
    return blocks * block_bytes;
}

GgufFile::GgufFile(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GgufFile::GgufFile(GgufFile&&) noexcept = default;
GgufFile& GgufFile::operator=(GgufFile&&) noexcept = default;
GgufFile::~GgufFile() = default;

GgufFile GgufFile::open(const std::string& path) {
    return open_impl(path, true);
}

GgufFile GgufFile::open_meta(const std::string& path) {
    return open_impl(path, false);
}

GgufFile GgufFile::open_impl(const std::string& path, bool require_payloads) {
    auto impl = std::make_unique<Impl>();
    impl->fd = ::open(path.c_str(), O_RDONLY);
    if (impl->fd < 0) {
        fail("open GGUF failed: " + path + ": " + std::strerror(errno));
    }

    struct stat st {};
    if (fstat(impl->fd, &st) != 0) {
        fail("fstat GGUF failed: " + path + ": " + std::strerror(errno));
    }
    if (st.st_size <= 0) {
        fail("empty GGUF: " + path);
    }
    impl->file_size = static_cast<std::uint64_t>(st.st_size);
    if (impl->file_size > std::numeric_limits<std::size_t>::max()) {
        fail("GGUF is too large to map: " + path);
    }
    impl->map_size = static_cast<std::size_t>(impl->file_size);

    void* mapped = mmap(nullptr, impl->map_size, PROT_READ, MAP_PRIVATE, impl->fd, 0);
    if (mapped == MAP_FAILED) {
        fail("mmap GGUF failed: " + path + ": " + std::strerror(errno));
    }
    impl->map = static_cast<const std::byte*>(mapped);

    Cursor cur(impl->map, impl->map_size);
    const std::uint32_t magic = cur.read<std::uint32_t>();
    if (magic != kGgufMagic) {
        fail("not a GGUF file (bad magic): " + path);
    }
    impl->version = cur.read<std::uint32_t>();
    if (impl->version != kGgufVersion) {
        fail("unsupported GGUF version " + std::to_string(impl->version) + " (need 3)");
    }

    const std::uint64_t n_tensors = cur.read<std::uint64_t>();
    const std::uint64_t n_kv = cur.read<std::uint64_t>();
    if (n_tensors > impl->file_size || n_kv > impl->file_size) {
        fail("implausible GGUF counts");
    }

    impl->kv.reserve(static_cast<std::size_t>(n_kv));
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        std::string key = cur.read_string();
        const auto type = static_cast<GgufValType>(cur.read<std::uint32_t>());
        KvValue value = read_value(cur, type);
        if (impl->kv.find(key) != impl->kv.end()) {
            fail("duplicate GGUF metadata key: " + key);
        }
        impl->kv.emplace(std::move(key), std::move(value));
    }

    if (const KvValue* align = find_kv(impl->kv, "general.alignment")) {
        switch (align->type) {
            case GgufValType::UINT8:
            case GgufValType::UINT16:
            case GgufValType::UINT32:
            case GgufValType::UINT64:
                if (align->u == 0 || align->u > std::numeric_limits<std::uint32_t>::max()) {
                    fail("invalid general.alignment");
                }
                impl->alignment = static_cast<std::uint32_t>(align->u);
                break;
            case GgufValType::INT8:
            case GgufValType::INT16:
            case GgufValType::INT32:
            case GgufValType::INT64:
            case GgufValType::FLOAT32:
            case GgufValType::FLOAT64:
            case GgufValType::BOOL:
            case GgufValType::STRING:
            case GgufValType::ARRAY:
                fail("general.alignment must be an unsigned integer");
        }
    }

    impl->tensors.reserve(static_cast<std::size_t>(n_tensors));
    for (std::uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor tensor;
        tensor.name = cur.read_string();
        const std::uint32_t n_dims = cur.read<std::uint32_t>();
        if (n_dims < 1 || n_dims > static_cast<std::uint32_t>(kMaxDims)) {
            fail("GGUF tensor n_dims must be 1..4: " + tensor.name);
        }
        tensor.dims.resize(n_dims);
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            tensor.dims[d] = cur.read<std::uint64_t>();
        }
        tensor.type = static_cast<GgmlType>(cur.read<std::uint32_t>());
        if (!ggml_type_known(tensor.type)) {
            fail("unknown ggml type in tensor " + tensor.name);
        }
        tensor.offset = cur.read<std::uint64_t>();
        tensor.nbytes = ggml_nbytes(tensor.type, tensor.dims.data(), static_cast<int>(n_dims));
        if (impl->by_name.find(tensor.name) != impl->by_name.end()) {
            fail("duplicate GGUF tensor name: " + tensor.name);
        }
        impl->by_name.emplace(tensor.name, impl->tensors.size());
        impl->tensors.push_back(std::move(tensor));
    }

    const std::uint64_t data_offset = align_up(cur.pos(), impl->alignment);
    if (data_offset > impl->file_size) {
        if (require_payloads) {
            fail("truncated GGUF data section");
        }
        impl->payloads_complete = false;
        return GgufFile(std::move(impl));
    }

    impl->payloads_complete = true;
    for (GgufTensor& tensor : impl->tensors) {
        if (tensor.offset > impl->file_size - data_offset) {
            if (require_payloads) {
                fail("GGUF tensor offset out of file: " + tensor.name);
            }
            impl->payloads_complete = false;
            tensor.data = nullptr;
            continue;
        }
        const std::uint64_t start = data_offset + tensor.offset;
        if (tensor.nbytes > impl->file_size - start) {
            if (require_payloads) {
                fail("GGUF tensor payload out of file: " + tensor.name);
            }
            impl->payloads_complete = false;
            tensor.data = nullptr;
            continue;
        }
        tensor.data = impl->map + static_cast<std::size_t>(start);
    }

    return GgufFile(std::move(impl));
}

std::uint32_t GgufFile::version() const {
    return impl_->version;
}

std::uint32_t GgufFile::alignment() const {
    return impl_->alignment;
}

std::uint64_t GgufFile::file_size() const {
    return impl_->file_size;
}

bool GgufFile::payloads_complete() const {
    return impl_->payloads_complete;
}

std::string GgufFile::architecture() const {
    if (!has_kv("general.architecture")) {
        return {};
    }
    return kv_string("general.architecture");
}

const std::vector<GgufTensor>& GgufFile::tensors() const {
    return impl_->tensors;
}

const GgufTensor* GgufFile::find(std::string_view name) const {
    const auto it = impl_->by_name.find(std::string(name));
    if (it == impl_->by_name.end()) {
        return nullptr;
    }
    return &impl_->tensors[it->second];
}

bool GgufFile::has_kv(std::string_view key) const {
    return find_kv(impl_->kv, key) != nullptr;
}

std::uint64_t GgufFile::kv_u64(std::string_view key) const {
    const KvValue& value = require_kv(impl_->kv, key);
    switch (value.type) {
        case GgufValType::UINT8:
        case GgufValType::UINT16:
        case GgufValType::UINT32:
        case GgufValType::UINT64:
            return value.u;
        case GgufValType::INT8:
        case GgufValType::INT16:
        case GgufValType::INT32:
        case GgufValType::INT64:
        case GgufValType::FLOAT32:
        case GgufValType::FLOAT64:
        case GgufValType::BOOL:
        case GgufValType::STRING:
        case GgufValType::ARRAY:
            fail("GGUF key is not an unsigned integer: " + std::string(key));
    }
    fail("unknown GGUF value type");
}

std::int64_t GgufFile::kv_i64(std::string_view key) const {
    const KvValue& value = require_kv(impl_->kv, key);
    switch (value.type) {
        case GgufValType::INT8:
        case GgufValType::INT16:
        case GgufValType::INT32:
        case GgufValType::INT64:
            return value.i;
        case GgufValType::UINT8:
        case GgufValType::UINT16:
        case GgufValType::UINT32:
        case GgufValType::UINT64:
        case GgufValType::FLOAT32:
        case GgufValType::FLOAT64:
        case GgufValType::BOOL:
        case GgufValType::STRING:
        case GgufValType::ARRAY:
            fail("GGUF key is not a signed integer: " + std::string(key));
    }
    fail("unknown GGUF value type");
}

double GgufFile::kv_f64(std::string_view key) const {
    const KvValue& value = require_kv(impl_->kv, key);
    switch (value.type) {
        case GgufValType::FLOAT32:
        case GgufValType::FLOAT64:
            return value.f;
        case GgufValType::UINT8:
        case GgufValType::INT8:
        case GgufValType::UINT16:
        case GgufValType::INT16:
        case GgufValType::UINT32:
        case GgufValType::INT32:
        case GgufValType::BOOL:
        case GgufValType::STRING:
        case GgufValType::ARRAY:
        case GgufValType::UINT64:
        case GgufValType::INT64:
            fail("GGUF key is not a float: " + std::string(key));
    }
    fail("unknown GGUF value type");
}

bool GgufFile::kv_bool(std::string_view key) const {
    const KvValue& value = require_kv(impl_->kv, key);
    switch (value.type) {
        case GgufValType::BOOL:
            return value.b;
        case GgufValType::UINT8:
        case GgufValType::INT8:
        case GgufValType::UINT16:
        case GgufValType::INT16:
        case GgufValType::UINT32:
        case GgufValType::INT32:
        case GgufValType::FLOAT32:
        case GgufValType::STRING:
        case GgufValType::ARRAY:
        case GgufValType::UINT64:
        case GgufValType::INT64:
        case GgufValType::FLOAT64:
            fail("GGUF key is not a bool: " + std::string(key));
    }
    fail("unknown GGUF value type");
}

std::string GgufFile::kv_string(std::string_view key) const {
    const KvValue& value = require_kv(impl_->kv, key);
    switch (value.type) {
        case GgufValType::STRING:
            return value.s;
        case GgufValType::UINT8:
        case GgufValType::INT8:
        case GgufValType::UINT16:
        case GgufValType::INT16:
        case GgufValType::UINT32:
        case GgufValType::INT32:
        case GgufValType::FLOAT32:
        case GgufValType::BOOL:
        case GgufValType::ARRAY:
        case GgufValType::UINT64:
        case GgufValType::INT64:
        case GgufValType::FLOAT64:
            fail("GGUF key is not a string: " + std::string(key));
    }
    fail("unknown GGUF value type");
}

std::vector<std::uint64_t> GgufFile::kv_u64_array(std::string_view key) const {
    const KvValue& value = require_kv(impl_->kv, key);
    if (value.type != GgufValType::ARRAY) {
        fail("GGUF key is not an array: " + std::string(key));
    }
    switch (value.elem) {
        case GgufValType::UINT8:
        case GgufValType::UINT16:
        case GgufValType::UINT32:
        case GgufValType::UINT64:
        case GgufValType::BOOL:
            return value.u_arr;
        case GgufValType::INT8:
        case GgufValType::INT16:
        case GgufValType::INT32:
        case GgufValType::INT64: {
            std::vector<std::uint64_t> out;
            out.reserve(value.i_arr.size());
            for (std::int64_t item : value.i_arr) {
                if (item < 0) {
                    fail("GGUF integer array has a negative value: " + std::string(key));
                }
                out.push_back(static_cast<std::uint64_t>(item));
            }
            return out;
        }
        case GgufValType::FLOAT32:
        case GgufValType::FLOAT64:
        case GgufValType::STRING:
        case GgufValType::ARRAY:
            fail("GGUF key is not an integer array: " + std::string(key));
    }
    fail("unknown GGUF array element type");
}

std::vector<std::string> GgufFile::kv_string_array(std::string_view key) const {
    const KvValue& value = require_kv(impl_->kv, key);
    if (value.type != GgufValType::ARRAY) {
        fail("GGUF key is not an array: " + std::string(key));
    }
    switch (value.elem) {
        case GgufValType::STRING:
            return value.s_arr;
        case GgufValType::UINT8:
        case GgufValType::INT8:
        case GgufValType::UINT16:
        case GgufValType::INT16:
        case GgufValType::UINT32:
        case GgufValType::INT32:
        case GgufValType::FLOAT32:
        case GgufValType::BOOL:
        case GgufValType::ARRAY:
        case GgufValType::UINT64:
        case GgufValType::INT64:
        case GgufValType::FLOAT64:
            fail("GGUF key is not a string array: " + std::string(key));
    }
    fail("unknown GGUF array element type");
}

std::size_t GgufFile::kv_count() const {
    return impl_->kv.size();
}

}  // namespace vesper
