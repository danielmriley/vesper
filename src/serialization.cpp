#include <vesper/serialization.h>
#include <vesper/core/factories.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>

namespace vesper {

constexpr const char* MAGIC_HEADER = "VSP1";

// Simple Binary Format:
// Header:
//   Magic (4 bytes) "VSP1"
//   Count (u64)
// For each Tensor:
//   Name Len (u64)
//   Name (bytes)
//   DType (u32)
//   NDim (u32)
//   Shape (NDim * i64)
//   Data Bytes (u64)
//   Data (bytes)

void save(const StateDict& state_dict, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Magic
    file.write(MAGIC_HEADER, 4);

    // Count
    uint64_t count = state_dict.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& [name, tensor] : state_dict) {
        // Name
        uint64_t name_len = name.size();
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.c_str(), name_len);

        // Metadata
        // Move to CPU for saving
        Tensor cpu_tensor = tensor.to(Device::CPU);
        if (!cpu_tensor.is_contiguous()) {
            cpu_tensor = cpu_tensor.contiguous();
        }

        uint32_t dtype = static_cast<uint32_t>(cpu_tensor.dtype());
        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

        uint32_t ndim = static_cast<uint32_t>(cpu_tensor.shape().size());
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

        for (int64_t dim : cpu_tensor.shape()) {
            file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        }

        // Data
        uint64_t data_bytes = cpu_tensor.numel() * GetDTypeSize(cpu_tensor.dtype());
        file.write(reinterpret_cast<const char*>(&data_bytes), sizeof(data_bytes));
        
        file.write(cpu_tensor.data_ptr<const char>(), data_bytes);
    }
}

StateDict load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    char magic[5] = {0};
    file.read(magic, 4);
    if (std::string(magic) != MAGIC_HEADER) {
        throw std::runtime_error("Invalid file format or magic header.");
    }

    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));

    StateDict dict;

    for (uint64_t i = 0; i < count; ++i) {
        // Name
        uint64_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        // Metadata
        uint32_t dtype_u32;
        file.read(reinterpret_cast<char*>(&dtype_u32), sizeof(dtype_u32));
        DType dtype = static_cast<DType>(dtype_u32);

        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

        std::vector<int64_t> shape(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            file.read(reinterpret_cast<char*>(&shape[d]), sizeof(int64_t));
        }

        // Data
        uint64_t data_bytes;
        file.read(reinterpret_cast<char*>(&data_bytes), sizeof(data_bytes));

        // Create tensor (CPU)
        Tensor t = empty(shape, dtype, Device::CPU);
        if (t.numel() * GetDTypeSize(dtype) != data_bytes) {
             throw std::runtime_error("Data size mismatch in file.");
        }
        
        file.read(t.data_ptr<char>(), data_bytes);
        
        dict[name] = t;
    }
    return dict;
}

void save(const nn::Module& module, const std::string& filename) {
    save(module.state_dict(), filename);
}

void load(nn::Module& module, const std::string& filename) {
    StateDict dict = load(filename);
    module.load_state_dict(dict);
}

} // namespace vesper
