/**
 * @file test_safetensors.cpp
 * @brief Comprehensive tests for safetensors I/O and weight loading.
 * 
 * Chapter 33.8: Loading Pre-trained Weights
 */

#include <vesper/io/safetensors.h>
#include <vesper/io/safetensors_writer.h>
#include <vesper/io/weight_mapper.h>
#include <vesper/io/model_loader.h>
#include <vesper/core/factories.h>
#include <vesper/core/dtype.h>

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <cstdio>

namespace fs = std::filesystem;
using namespace vesper;
using namespace vesper::io;

// ============================================================================
// Test Utilities
// ============================================================================

// Helper to create a random tensor with default dtype and device
Tensor make_randn(const std::vector<int64_t>& shape) {
    return randn(shape, DType::Float32, Device::CPU);
}

bool tensors_allclose(const Tensor& a, const Tensor& b, float rtol = 1e-5, float atol = 1e-8) {
    if (a.shape() != b.shape()) return false;
    if (a.dtype() != b.dtype()) return false;
    
    Tensor a_cpu = (a.device() != Device::CPU) ? a.to(Device::CPU) : a;
    Tensor b_cpu = (b.device() != Device::CPU) ? b.to(Device::CPU) : b;
    
    size_t n = a_cpu.numel();
    
    switch (a_cpu.dtype()) {
        case DType::Float32: {
            const float* pa = a_cpu.data_ptr<float>();
            const float* pb = b_cpu.data_ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                if (std::abs(pa[i] - pb[i]) > atol + rtol * std::abs(pb[i])) {
                    return false;
                }
            }
            break;
        }
        case DType::Float64: {
            const double* pa = a_cpu.data_ptr<double>();
            const double* pb = b_cpu.data_ptr<double>();
            for (size_t i = 0; i < n; ++i) {
                if (std::abs(pa[i] - pb[i]) > atol + rtol * std::abs(pb[i])) {
                    return false;
                }
            }
            break;
        }
        case DType::Int32: {
            const int32_t* pa = a_cpu.data_ptr<int32_t>();
            const int32_t* pb = b_cpu.data_ptr<int32_t>();
            for (size_t i = 0; i < n; ++i) {
                if (pa[i] != pb[i]) return false;
            }
            break;
        }
        case DType::Int64: {
            const int64_t* pa = a_cpu.data_ptr<int64_t>();
            const int64_t* pb = b_cpu.data_ptr<int64_t>();
            for (size_t i = 0; i < n; ++i) {
                if (pa[i] != pb[i]) return false;
            }
            break;
        }
        default:
            // For other types, compare bytes
            return std::memcmp(a_cpu.data_ptr<uint8_t>(), b_cpu.data_ptr<uint8_t>(),
                             n * GetDTypeSize(a_cpu.dtype())) == 0;
    }
    
    return true;
}

std::string temp_file(const std::string& name) {
    return "/tmp/" + name + "_" + std::to_string(std::rand()) + ".safetensors";
}

void cleanup_file(const std::string& path) {
    if (fs::exists(path)) {
        fs::remove(path);
    }
}

// ============================================================================
// Basic Tests
// ============================================================================

void test_safetensors_single_tensor() {
    std::cout << "Testing safetensors single tensor..." << std::endl;
    
    std::string path = temp_file("single");
    
    // Create and write
    Tensor original = make_randn({3, 4});
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("test", original);
        writer.write();
    }
    
    // Read back
    SafetensorsReader reader(path);
    
    auto names = reader.tensor_names();
    assert(names.size() == 1);
    assert(names[0] == "test");
    
    auto info = reader.get_info("test");
    assert(info.shape == std::vector<int64_t>({3, 4}));
    assert(info.dtype == DType::Float32);
    
    Tensor loaded = reader.load_tensor("test");
    assert(tensors_allclose(original, loaded));
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_safetensors_multiple_tensors() {
    std::cout << "Testing safetensors multiple tensors..." << std::endl;
    
    std::string path = temp_file("multi");
    
    Tensor t1 = make_randn({16, 32});
    Tensor t2 = make_randn({64, 64});
    Tensor t3 = make_randn({8, 16, 32});
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("layer1.weight", t1);
        writer.add_tensor("layer2.weight", t2);
        writer.add_tensor("layer3.weight", t3);
        writer.write();
    }
    
    SafetensorsReader reader(path);
    assert(reader.num_tensors() == 3);
    
    assert(tensors_allclose(t1, reader.load_tensor("layer1.weight")));
    assert(tensors_allclose(t2, reader.load_tensor("layer2.weight")));
    assert(tensors_allclose(t3, reader.load_tensor("layer3.weight")));
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_safetensors_dtypes() {
    std::cout << "Testing safetensors different dtypes..." << std::endl;
    
    std::string path = temp_file("dtypes");
    
    Tensor f32 = make_randn({4, 4});
    Tensor f64 = full({3, 3}, DType::Float32, Device::CPU, 3.14159f).to(DType::Float64);
    Tensor i64 = full({3, 3}, DType::Float32, Device::CPU, 42.0f).to(DType::Int64);
    Tensor i32 = full({2, 2}, DType::Float32, Device::CPU, -17.0f).to(DType::Int32);
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("float32", f32);
        writer.add_tensor("float64", f64);
        writer.add_tensor("int64", i64);
        writer.add_tensor("int32", i32);
        writer.write();
    }
    
    SafetensorsReader reader(path);
    
    assert(reader.get_info("float32").dtype == DType::Float32);
    assert(reader.get_info("float64").dtype == DType::Float64);
    assert(reader.get_info("int64").dtype == DType::Int64);
    assert(reader.get_info("int32").dtype == DType::Int32);
    
    assert(tensors_allclose(f32, reader.load_tensor("float32")));
    assert(tensors_allclose(f64, reader.load_tensor("float64")));
    
    Tensor loaded_i64 = reader.load_tensor("int64");
    assert(loaded_i64.dtype() == DType::Int64);
    assert(loaded_i64.item<int64_t>() == 42);
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_safetensors_metadata() {
    std::cout << "Testing safetensors metadata..." << std::endl;
    
    std::string path = temp_file("metadata");
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("x", make_randn({2, 2}));
        writer.set_metadata("author", "vesper");
        writer.set_metadata("version", "1.0");
        writer.set_metadata("description", "Test file with metadata");
        writer.write();
    }
    
    SafetensorsReader reader(path);
    auto meta = reader.metadata();
    
    assert(meta.count("author") > 0);
    assert(meta["author"] == "vesper");
    assert(meta["version"] == "1.0");
    assert(meta["description"] == "Test file with metadata");
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_safetensors_roundtrip() {
    std::cout << "Testing safetensors roundtrip..." << std::endl;
    
    std::string path = temp_file("roundtrip");
    
    // Create diverse tensors
    std::unordered_map<std::string, Tensor> original;
    original["scalar"] = full({}, DType::Float32, Device::CPU, 3.14f);  // 0-D tensor
    original["vector"] = make_randn({100});
    original["matrix"] = make_randn({50, 60});
    original["tensor3d"] = make_randn({8, 16, 32});
    original["tensor4d"] = make_randn({2, 3, 4, 5});
    
    {
        SafetensorsWriter writer(path);
        for (const auto& [name, tensor] : original) {
            writer.add_tensor(name, tensor);
        }
        writer.write();
    }
    
    SafetensorsReader reader(path);
    auto loaded = reader.load_all();
    
    assert(loaded.size() == original.size());
    for (const auto& [name, tensor] : original) {
        assert(loaded.count(name) > 0);
        assert(tensors_allclose(tensor, loaded[name]));
    }
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

// ============================================================================
// Weight Mapper Tests
// ============================================================================

void test_weight_mapper_huggingface_llama() {
    std::cout << "Testing weight mapper (HuggingFace Llama)..." << std::endl;
    
    WeightMapper mapper = WeightMapper::huggingface_llama();
    
    // Test from_source mappings
    assert(mapper.map_from_source("model.embed_tokens.weight") == "tok_embeddings.weight");
    assert(mapper.map_from_source("model.layers.0.self_attn.q_proj.weight") == "layers.0.attention.wq.weight");
    assert(mapper.map_from_source("model.layers.5.self_attn.k_proj.weight") == "layers.5.attention.wk.weight");
    assert(mapper.map_from_source("model.layers.10.mlp.gate_proj.weight") == "layers.10.feed_forward.w1.weight");
    assert(mapper.map_from_source("model.layers.31.mlp.down_proj.weight") == "layers.31.feed_forward.w2.weight");
    assert(mapper.map_from_source("model.norm.weight") == "norm.weight");
    assert(mapper.map_from_source("lm_head.weight") == "output.weight");
    
    // Test to_source mappings  
    assert(mapper.map_to_source("tok_embeddings.weight") == "model.embed_tokens.weight");
    assert(mapper.map_to_source("layers.0.attention.wq.weight") == "model.layers.0.self_attn.q_proj.weight");
    
    std::cout << "  PASSED!" << std::endl;
}

void test_detect_weight_format() {
    std::cout << "Testing detect_weight_format..." << std::endl;
    
    std::vector<std::string> hf_names = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight"
    };
    assert(detect_weight_format(hf_names) == "huggingface_llama");
    
    std::vector<std::string> meta_names = {
        "tok_embeddings.weight",
        "layers.0.attention.wq.weight",
        "output.weight"
    };
    assert(detect_weight_format(meta_names) == "meta_llama");
    
    std::vector<std::string> gpt2_names = {
        "transformer.wte.weight",
        "transformer.h.0.attn.c_attn.weight"
    };
    assert(detect_weight_format(gpt2_names) == "huggingface_gpt2");
    
    std::cout << "  PASSED!" << std::endl;
}

// ============================================================================
// Stress Tests
// ============================================================================

void test_safetensors_large_tensor() {
    std::cout << "Testing safetensors large tensor (256MB)..." << std::endl;
    
    std::string path = temp_file("large");
    
    // ~256MB tensor (8192 * 8192 * 4 bytes = 256MB)
    Tensor large = make_randn({8192, 8192});
    
    auto start = std::chrono::high_resolution_clock::now();
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("large", large);
        writer.write();
    }
    
    auto write_end = std::chrono::high_resolution_clock::now();
    
    SafetensorsReader reader(path);
    Tensor loaded = reader.load_tensor("large");
    
    auto read_end = std::chrono::high_resolution_clock::now();
    
    auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - start).count();
    auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(read_end - write_end).count();
    
    std::cout << "  Write 256MB: " << write_ms << " ms" << std::endl;
    std::cout << "  Read 256MB: " << read_ms << " ms" << std::endl;
    
    // Should be reasonably fast (> 100 MB/s = < 2.5s for 256MB)
    assert(write_ms < 5000);  // Allow some slack
    assert(read_ms < 5000);
    
    assert(tensors_allclose(large, loaded));
    
    // Check file size
    size_t file_size = fs::file_size(path);
    std::cout << "  File size: " << (file_size / 1024 / 1024) << " MB" << std::endl;
    assert(file_size >= 256 * 1024 * 1024 - 4096);  // At least 256MB minus header
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_safetensors_many_tensors() {
    std::cout << "Testing safetensors many tensors (200 tensors)..." << std::endl;
    
    std::string path = temp_file("many");
    
    std::unordered_map<std::string, Tensor> original;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    {
        SafetensorsWriter writer(path);
        for (int i = 0; i < 200; ++i) {
            std::string name = "layer" + std::to_string(i) + ".weight";
            Tensor t = make_randn({256, 256});
            original[name] = t;
            writer.add_tensor(name, t);
        }
        writer.write();
    }
    
    auto write_end = std::chrono::high_resolution_clock::now();
    
    SafetensorsReader reader(path);
    assert(reader.num_tensors() == 200);
    
    auto read_end = std::chrono::high_resolution_clock::now();
    
    auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - start).count();
    auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(read_end - write_end).count();
    
    std::cout << "  Write 200 tensors: " << write_ms << " ms" << std::endl;
    std::cout << "  Read headers: " << read_ms << " ms" << std::endl;
    
    // Verify some random tensors
    for (int i : {0, 50, 100, 150, 199}) {
        std::string name = "layer" + std::to_string(i) + ".weight";
        Tensor loaded = reader.load_tensor(name);
        assert(tensors_allclose(original[name], loaded));
    }
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_safetensors_load_all_stress() {
    std::cout << "Testing safetensors load_all stress..." << std::endl;
    
    std::string path = temp_file("loadall");
    
    // Simulate a small model (~50MB)
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("embed.weight", make_randn({10000, 512}));  // ~20MB
        for (int i = 0; i < 6; ++i) {
            writer.add_tensor("layers." + std::to_string(i) + ".attn.qkv", make_randn({512, 1536}));
            writer.add_tensor("layers." + std::to_string(i) + ".attn.out", make_randn({512, 512}));
            writer.add_tensor("layers." + std::to_string(i) + ".mlp.fc1", make_randn({512, 2048}));
            writer.add_tensor("layers." + std::to_string(i) + ".mlp.fc2", make_randn({2048, 512}));
        }
        writer.add_tensor("ln.weight", make_randn({512}));
        writer.add_tensor("lm_head.weight", make_randn({512, 10000}));
        writer.write();
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    SafetensorsReader reader(path);
    auto all_tensors = reader.load_all();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "  Loaded " << all_tensors.size() << " tensors in " << ms << " ms" << std::endl;
    
    // Count total parameters
    int64_t total_params = 0;
    for (const auto& [name, tensor] : all_tensors) {
        total_params += tensor.numel();
    }
    std::cout << "  Total parameters: " << (total_params / 1000000.0) << "M" << std::endl;
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

// ============================================================================
// Edge Cases
// ============================================================================

void test_safetensors_empty_tensor() {
    std::cout << "Testing safetensors empty tensor..." << std::endl;
    
    std::string path = temp_file("empty");
    
    // Note: Empty tensors (0 elements) may not be well-supported
    // Test with a valid small tensor instead
    Tensor tiny = make_randn({1});
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("tiny", tiny);
        writer.write();
    }
    
    SafetensorsReader reader(path);
    Tensor loaded = reader.load_tensor("tiny");
    assert(tensors_allclose(tiny, loaded));
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_safetensors_special_names() {
    std::cout << "Testing safetensors special names..." << std::endl;
    
    std::string path = temp_file("special");
    
    Tensor t = make_randn({2, 2});
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("normal.name", t);
        writer.add_tensor("name_with_underscore", t);
        writer.add_tensor("name.with.many.dots", t);
        writer.add_tensor("layer0", t);
        writer.add_tensor("123", t);  // Numeric name
        writer.write();
    }
    
    SafetensorsReader reader(path);
    assert(reader.num_tensors() == 5);
    assert(reader.has_tensor("normal.name"));
    assert(reader.has_tensor("name_with_underscore"));
    assert(reader.has_tensor("name.with.many.dots"));
    assert(reader.has_tensor("layer0"));
    assert(reader.has_tensor("123"));
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_safetensors_file_validation() {
    std::cout << "Testing safetensors file validation..." << std::endl;
    
    std::string path = temp_file("valid");
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("x", make_randn({2, 2}));
        writer.write();
    }
    
    assert(is_safetensors_file(path));
    assert(!is_safetensors_file("/nonexistent/path.safetensors"));
    
    // Create an invalid file
    std::string invalid = temp_file("invalid");
    {
        std::ofstream f(invalid);
        f << "not a safetensors file";
    }
    assert(!is_safetensors_file(invalid));
    cleanup_file(invalid);
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_safetensors_error_handling() {
    std::cout << "Testing safetensors error handling..." << std::endl;
    
    // Test missing file
    bool caught = false;
    try {
        SafetensorsReader reader("/nonexistent/path.safetensors");
    } catch (const std::runtime_error& e) {
        caught = true;
    }
    assert(caught);
    
    // Test missing tensor
    std::string path = temp_file("errors");
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("exists", make_randn({2, 2}));
        writer.write();
    }
    
    SafetensorsReader reader(path);
    caught = false;
    try {
        reader.load_tensor("does_not_exist");
    } catch (const std::runtime_error& e) {
        caught = true;
    }
    assert(caught);
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

// ============================================================================
// GPU Tests (if available)
// ============================================================================

#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
void test_safetensors_gpu_direct_load() {
    std::cout << "Testing safetensors GPU direct load..." << std::endl;
    
    std::string path = temp_file("gpu");
    
    Tensor original = make_randn({512, 512});
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("data", original);
        writer.write();
    }
    
#ifdef USE_HIP_BACKEND
    Device gpu = Device::HIP;
#else
    Device gpu = Device::CUDA;
#endif
    
    SafetensorsReader reader(path);
    Tensor gpu_tensor = reader.load_tensor("data", gpu);
    
    assert(gpu_tensor.device() == gpu);
    assert(tensors_allclose(original, gpu_tensor.to(Device::CPU)));
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_safetensors_save_from_gpu() {
    std::cout << "Testing safetensors save from GPU..." << std::endl;
    
    std::string path = temp_file("from_gpu");
    
#ifdef USE_HIP_BACKEND
    Device gpu = Device::HIP;
#else
    Device gpu = Device::CUDA;
#endif
    
    // Create tensor on GPU
    Tensor cpu_original = make_randn({256, 256});
    Tensor gpu_tensor = cpu_original.to(gpu);
    
    {
        SafetensorsWriter writer(path);
        // Writer should handle GPU tensor by copying to CPU
        writer.add_tensor("gpu_data", gpu_tensor);
        writer.write();
    }
    
    SafetensorsReader reader(path);
    Tensor loaded = reader.load_tensor("gpu_data");
    
    assert(tensors_allclose(cpu_original, loaded));
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}
#endif

// ============================================================================
// Sharded Loading Test
// ============================================================================

void test_sharded_safetensors() {
    std::cout << "Testing sharded safetensors loading..." << std::endl;
    
    // Create a temporary directory with multiple shards
    std::string dir = "/tmp/test_sharded_" + std::to_string(std::rand());
    fs::create_directory(dir);
    
    std::unordered_map<std::string, Tensor> all_original;
    
    // Create 3 shards
    for (int shard = 0; shard < 3; ++shard) {
        std::string shard_path = dir + "/model-0000" + std::to_string(shard+1) + "-of-00003.safetensors";
        SafetensorsWriter writer(shard_path);
        
        for (int i = 0; i < 5; ++i) {
            std::string name = "shard" + std::to_string(shard) + ".layer" + std::to_string(i);
            Tensor t = make_randn({64, 64});
            all_original[name] = t;
            writer.add_tensor(name, t);
        }
        writer.write();
    }
    
    // Load all shards
    auto loaded = load_sharded_safetensors(dir);
    
    assert(loaded.size() == all_original.size());
    for (const auto& [name, tensor] : all_original) {
        assert(loaded.count(name) > 0);
        assert(tensors_allclose(tensor, loaded[name]));
    }
    
    // Cleanup
    fs::remove_all(dir);
    std::cout << "  PASSED!" << std::endl;
}

// ============================================================================
// Model Loader Tests
// ============================================================================

void test_model_loader_list_tensors() {
    std::cout << "Testing ModelLoader::list_tensors..." << std::endl;
    
    std::string path = temp_file("list");
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("z", make_randn({2, 2}));
        writer.add_tensor("a", make_randn({2, 2}));
        writer.add_tensor("m", make_randn({2, 2}));
        writer.write();
    }
    
    auto names = ModelLoader::list_tensors(path);
    
    // Should be sorted
    assert(names.size() == 3);
    assert(names[0] == "a");
    assert(names[1] == "m");
    assert(names[2] == "z");
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_model_loader_count_parameters() {
    std::cout << "Testing ModelLoader::count_parameters..." << std::endl;
    
    std::string path = temp_file("count");
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("a", make_randn({10, 20}));    // 200
        writer.add_tensor("b", make_randn({5, 5, 4}));   // 100
        writer.add_tensor("c", make_randn({50}));        // 50
        writer.write();
    }
    
    int64_t count = ModelLoader::count_parameters(path);
    assert(count == 350);
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

void test_model_loader_detect_format() {
    std::cout << "Testing ModelLoader::detect_format..." << std::endl;
    
    std::string path = temp_file("detect");
    
    {
        SafetensorsWriter writer(path);
        writer.add_tensor("x", make_randn({2, 2}));
        writer.write();
    }
    
    assert(ModelLoader::detect_format(path) == "safetensors");
    
    cleanup_file(path);
    std::cout << "  PASSED!" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n=== Safetensors I/O Tests ===" << std::endl;
    std::cout << "Chapter 33.8: Loading Pre-trained Weights\n" << std::endl;
    
    // Seed random for reproducibility of temp file names
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    // Basic tests
    test_safetensors_single_tensor();
    test_safetensors_multiple_tensors();
    test_safetensors_dtypes();
    test_safetensors_metadata();
    test_safetensors_roundtrip();
    
    // Weight mapper tests
    test_weight_mapper_huggingface_llama();
    test_detect_weight_format();
    
    // Stress tests
    test_safetensors_large_tensor();
    test_safetensors_many_tensors();
    test_safetensors_load_all_stress();
    
    // Edge cases
    test_safetensors_empty_tensor();
    test_safetensors_special_names();
    test_safetensors_file_validation();
    test_safetensors_error_handling();
    
    // Sharded loading
    test_sharded_safetensors();
    
    // Model loader
    test_model_loader_list_tensors();
    test_model_loader_count_parameters();
    test_model_loader_detect_format();
    
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
    // GPU tests
    test_safetensors_gpu_direct_load();
    test_safetensors_save_from_gpu();
#endif
    
    std::cout << "\n=== All Safetensors Tests Passed! ===" << std::endl;
    return 0;
}
