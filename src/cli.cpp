#include "vesper/config.h"
#include "vesper/engine.h"
#include "vesper/gguf.h"
#include "vesper/hip.h"
#include "vesper/kernels.h"
#include "vesper/model_io.h"
#include "vesper/q4k.h"
#include "vesper/q8.h"
#include "vesper/report.h"
#include "vesper/target.h"
#include "vesper/tokenizer.h"
#include "vesper/weight.h"
#include "vesper/weights.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

struct Options {
    bool demo = false;
    bool demo_hybrid = false;
    bool hip_info = false;
    bool bench_q8 = false;
    bool bench_q4 = false;
    bool bench_q5 = false;
    bool bench_q6 = false;
    std::string inspect;
    std::string model;
    std::string write_tiny;
    std::string write_tiny_hybrid;
    std::string write_tiny_qwen35;
    std::string write_tiny_q4km;
    std::string prompt = "hello";
    int tokens = 32;
    int context = vesper::kDefaultContext;
    bool report_only = false;
    std::uint32_t seed = 1;
    vesper::Device device = vesper::Device::CPU;
};

void usage() {
    std::cout
        << "vesper-infer -- AMD-first local LLM inference\n"
        << "  v1 HIP target: " << vesper::kHipArch
        << " (Radeon AI Pro R9700, wave" << vesper::kWavefront << ")\n"
        << "\n"
        << "  --demo                 run the tiny random Qwen3-style model\n"
        << "  --demo-hybrid          run the tiny hybrid GDN+attention model\n"
        << "  --model PATH           generate from vesper_tiny, vesper_hybrid, qwen35, or qwen3_5\n"
        << "  --write-tiny PATH      write the demo as a Q8_0 GGUF and exit\n"
        << "  --write-tiny-hybrid PATH  write the hybrid fixture and exit\n"
        << "  --write-tiny-qwen35 PATH  write the qwen35 fixture and exit\n"
        << "  --write-tiny-q4km PATH write a mixed Q4_K/Q5_K/Q6_K fixture and exit\n"
        << "  --inspect PATH         print GGUF version, alignment, tensors\n"
        << "  --bench-q8             time official Qwen3.8-27B Q8_0 GEMV shapes\n"
        << "  --bench-q4             time official Qwen3.8-27B Q4_K FFN GEMV shapes\n"
        << "  --bench-q5             time official FFN shapes packed as Q5_K\n"
        << "  --bench-q6             time official Qwen3.8-27B Q6_K GEMV shapes\n"
        << "  --device cpu|hip       decode device (default: cpu)\n"
        << "  --hip-info             print HIP probe and exit\n"
        << "  --prompt TEXT          prompt text (default: hello)\n"
        << "  --tokens N             new tokens to generate (default: 32)\n"
        << "  --context N            cap KV length (default: 4096, 0 = file value)\n"
        << "  --report-only          print only the DecodeReport line\n"
        << "  --seed N               weight seed (default: 1)\n"
        << "  --help                 this message\n"
        << "\n"
        << "HIP is gfx1201-only. See docs/TARGET.md.\n";
}

vesper::Device parse_device(const std::string& name) {
    if (name == "cpu") {
        return vesper::Device::CPU;
    }
    if (name == "hip") {
        return vesper::Device::HIP;
    }
    vesper::fail("unknown device: " + name + " (expected cpu or hip)");
}

void print_hip_info() {
    std::cout << "hip built     " << (vesper::kHipBuilt ? "yes" : "no") << "\n";
    std::cout << "v1 target     " << vesper::kHipArch << " wave" << vesper::kWavefront
              << " cacheline=" << vesper::kCachelineBytes << "B"
              << " CUs=" << vesper::kComputeUnits << "\n";
    std::cout << "available     " << (vesper::hip_available() ? "yes" : "no") << "\n";
    if (vesper::hip_available()) {
        std::cout << "device        " << vesper::hip_device_name() << "\n";
        std::cout << "arch          " << vesper::hip_arch() << "\n";
    }
    if (!vesper::kHipBuilt) {
        std::cout << "build with    cmake -DVESPER_USE_HIP=ON\n";
    }
}

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                vesper::fail(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--demo") {
            opt.demo = true;
        } else if (arg == "--demo-hybrid") {
            opt.demo_hybrid = true;
        } else if (arg == "--model") {
            opt.model = need("--model");
        } else if (arg == "--write-tiny") {
            opt.write_tiny = need("--write-tiny");
        } else if (arg == "--write-tiny-hybrid") {
            opt.write_tiny_hybrid = need("--write-tiny-hybrid");
        } else if (arg == "--write-tiny-qwen35") {
            opt.write_tiny_qwen35 = need("--write-tiny-qwen35");
        } else if (arg == "--write-tiny-q4km") {
            opt.write_tiny_q4km = need("--write-tiny-q4km");
        } else if (arg == "--inspect") {
            opt.inspect = need("--inspect");
        } else if (arg == "--bench-q8") {
            opt.bench_q8 = true;
        } else if (arg == "--bench-q4") {
            opt.bench_q4 = true;
        } else if (arg == "--bench-q5") {
            opt.bench_q5 = true;
        } else if (arg == "--bench-q6") {
            opt.bench_q6 = true;
        } else if (arg == "--hip-info") {
            opt.hip_info = true;
        } else if (arg == "--device") {
            opt.device = parse_device(need("--device"));
        } else if (arg == "--prompt") {
            opt.prompt = need("--prompt");
        } else if (arg == "--tokens") {
            opt.tokens = std::atoi(need("--tokens"));
        } else if (arg == "--context") {
            opt.context = std::atoi(need("--context"));
        } else if (arg == "--report-only") {
            opt.report_only = true;
        } else if (arg == "--seed") {
            opt.seed = static_cast<std::uint32_t>(std::strtoul(need("--seed"), nullptr, 10));
        } else if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else {
            vesper::fail("unknown argument: " + arg);
        }
    }
    if ((opt.demo || opt.demo_hybrid) && !opt.model.empty()) {
        vesper::fail("use either a demo flag or --model, not both");
    }
    const bool ok = opt.demo || opt.demo_hybrid || opt.hip_info || opt.bench_q8 || opt.bench_q4 ||
                    opt.bench_q5 || opt.bench_q6 || !opt.inspect.empty() || !opt.model.empty() ||
                    !opt.write_tiny.empty() || !opt.write_tiny_hybrid.empty() ||
                    !opt.write_tiny_qwen35.empty() || !opt.write_tiny_q4km.empty();
    if (!ok) {
        usage();
        vesper::fail(
            "need --demo, --demo-hybrid, --model, --write-tiny, --write-tiny-hybrid, "
            "--write-tiny-qwen35, --write-tiny-q4km, --inspect, --bench-q8, --bench-q4, "
            "--bench-q5, --bench-q6, or --hip-info");
    }
    return opt;
}

const char* hybrid_prefix(const vesper::GgufFile& file) {
    const std::string arch = file.architecture();
    if (arch == "qwen35" || arch == "qwen3_5") {
        if (file.has_kv("qwen3_5.block_count")) {
            return "qwen3_5.";
        }
        if (file.has_kv("qwen35.block_count")) {
            return "qwen35.";
        }
        return arch == "qwen3_5" ? "qwen3_5." : "qwen35.";
    }
    if (arch == "vesper_hybrid") {
        return "vesper_hybrid.";
    }
    return nullptr;
}

void print_kv_if(const vesper::GgufFile& file, const std::string& key) {
    if (!file.has_kv(key)) {
        return;
    }
    std::cout << "  " << key << " " << file.kv_u64(key) << "\n";
}

void print_inspect(const std::string& path) {
    const vesper::GgufFile file = vesper::GgufFile::open_meta(path);
    std::cout << "version      " << file.version() << "\n";
    std::cout << "alignment    " << file.alignment() << "\n";
    std::cout << "architecture " << file.architecture() << "\n";
    std::cout << "file_bytes   " << file.file_size() << "\n";
    std::cout << "payloads     " << (file.payloads_complete() ? "complete" : "incomplete") << "\n";
    if (file.architecture() == "qwen35" || file.architecture() == "qwen3_5") {
        std::cout << "pin_header   qwen38-27b-q4km "
                  << (vesper::qwen38_27b_q4km_header_ok(file) ? "yes" : "no") << "\n";
    }
    std::cout << "kv           " << file.kv_count() << "\n";
    std::cout << "tensors      " << file.tensors().size() << "\n";
    if (const char* prefix = hybrid_prefix(file)) {
        std::cout << "hybrid_kv\n";
        print_kv_if(file, std::string(prefix) + "block_count");
        print_kv_if(file, std::string(prefix) + "nextn_predict_layers");
        print_kv_if(file, std::string(prefix) + "context_length");
        print_kv_if(file, std::string(prefix) + "embedding_length");
        print_kv_if(file, std::string(prefix) + "feed_forward_length");
        print_kv_if(file, std::string(prefix) + "attention.head_count");
        print_kv_if(file, std::string(prefix) + "attention.head_count_kv");
        print_kv_if(file, std::string(prefix) + "attention.key_length");
        print_kv_if(file, std::string(prefix) + "full_attention_interval");
        print_kv_if(file, std::string(prefix) + "ssm.conv_kernel");
        print_kv_if(file, std::string(prefix) + "ssm.state_size");
        print_kv_if(file, std::string(prefix) + "ssm.group_count");
        print_kv_if(file, std::string(prefix) + "ssm.time_step_rank");
        print_kv_if(file, std::string(prefix) + "ssm.inner_size");
        print_kv_if(file, std::string(prefix) + "rope.dimension_count");
        const std::string secs = std::string(prefix) + "rope.dimension_sections";
        if (file.has_kv(secs)) {
            const std::vector<std::uint64_t> arr = file.kv_u64_array(secs);
            std::cout << "  " << secs << " [";
            for (std::size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << arr[i];
            }
            std::cout << "]\n";
        }
        const std::string rec = std::string(prefix) + "attention.recurrent_layers";
        if (file.has_kv(rec)) {
            std::cout << "  " << rec << " len=" << file.kv_u64_array(rec).size() << "\n";
        } else {
            std::cout << "  " << rec << " absent\n";
        }
        if (file.has_kv("tokenizer.ggml.pre")) {
            std::cout << "  tokenizer.ggml.pre " << file.kv_string("tokenizer.ggml.pre") << "\n";
        }
    }
    std::map<std::string, int> types;
    for (const vesper::GgufTensor& tensor : file.tensors()) {
        types[vesper::ggml_type_name(tensor.type)] += 1;
    }
    std::cout << "types\n";
    for (const auto& [name, n] : types) {
        std::cout << "  " << name << " " << n << "\n";
    }
    for (const vesper::GgufTensor& tensor : file.tensors()) {
        std::cout << tensor.name << " " << vesper::ggml_type_name(tensor.type) << " ";
        for (std::size_t i = 0; i < tensor.dims.size(); ++i) {
            if (i > 0) {
                std::cout << 'x';
            }
            std::cout << tensor.dims[i];
        }
        std::cout << " " << tensor.nbytes << "\n";
    }
}

void print_generate(const std::string& label, const vesper::Engine& engine,
                    const std::string& prompt, const std::vector<int>& ids,
                    std::size_t prompt_n) {
    const vesper::DecodeReport report = engine.last_report();
    std::cout << label << "\n";
    vesper::print_report(report);
    std::cout << "prompt       " << prompt << "\n";
    std::cout << "greedy ids   ";
    for (std::size_t i = prompt_n; i < ids.size(); ++i) {
        if (i > prompt_n) {
            std::cout << ' ';
        }
        std::cout << ids[i];
    }
    std::cout << "\n";
}

vesper::WeightMatrix bench_dummy(vesper::WeightKind kind, int rows, int cols) {
    const std::size_t nbytes = vesper::packed_bytes(kind, rows, cols);
    std::vector<std::byte> zeros(nbytes);
    switch (kind) {
        case vesper::WeightKind::F32: {
            std::vector<float> host(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
                                    0.01f);
            return vesper::WeightMatrix::from_f32(host.data(), rows, cols);
        }
        case vesper::WeightKind::Q8_0:
            return vesper::WeightMatrix::q8_from_bytes(zeros.data(), rows, cols);
        case vesper::WeightKind::Q4_K:
            return vesper::WeightMatrix::q4_from_bytes(zeros.data(), rows, cols);
        case vesper::WeightKind::Q5_K:
            return vesper::WeightMatrix::q5_from_bytes(zeros.data(), rows, cols);
        case vesper::WeightKind::Q6_K:
            return vesper::WeightMatrix::q6_from_bytes(zeros.data(), rows, cols);
    }
    throw std::logic_error("unhandled WeightKind");
}

void bench_gemv(vesper::Device device, vesper::WeightKind kind, int rows, int cols,
                const char* shape) {
    std::vector<float> host_x(static_cast<std::size_t>(cols));
    for (int i = 0; i < cols; ++i) {
        host_x[static_cast<std::size_t>(i)] = 0.05f * static_cast<float>((i * 9) % 13 - 6);
    }
    vesper::WeightMatrix packed = bench_dummy(kind, rows, cols);
    const std::size_t nbytes = packed.bytes();
    const int warmup = nbytes > (32ull * 1024ull * 1024ull) ? 1 : 3;
    const int iters = nbytes > (32ull * 1024ull * 1024ull) ? 8 : 20;

    double ms = 0.0;
    switch (device) {
        case vesper::Device::CPU: {
            std::vector<float> y(static_cast<std::size_t>(rows), 0.0f);
            for (int i = 0; i < warmup; ++i) {
                vesper::gemv(y.data(), packed, host_x.data());
            }
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                vesper::gemv(y.data(), packed, host_x.data());
            }
            const auto t1 = std::chrono::steady_clock::now();
            ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            break;
        }
        case vesper::Device::HIP: {
            vesper::WeightMatrix gpu_w = packed.to(vesper::Device::HIP);
            vesper::Buffer x(static_cast<std::size_t>(cols), vesper::Device::HIP);
            vesper::Buffer y(static_cast<std::size_t>(rows), vesper::Device::HIP);
            x.copy_from(host_x.data(), host_x.size());
            for (int i = 0; i < warmup; ++i) {
                vesper::gemv(vesper::Device::HIP, y.data(), gpu_w, x.data());
            }
            vesper::hip_synchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                vesper::gemv(vesper::Device::HIP, y.data(), gpu_w, x.data());
            }
            vesper::hip_synchronize();
            const auto t1 = std::chrono::steady_clock::now();
            ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            break;
        }
    }

    const double seconds = ms / 1000.0;
    const double moved = static_cast<double>(nbytes) * static_cast<double>(iters);
    const double gbs = (seconds > 0.0) ? (moved / seconds) / 1e9 : 0.0;
    std::cout << "shape        " << shape << "\n";
    std::cout << "gemv         " << vesper::weight_kind_name(kind) << " " << rows << "x" << cols
              << "\n";
    std::cout << "device       " << vesper::device_name(device) << "\n";
    std::cout << "weight bytes " << nbytes << "\n";
    std::cout << "iters        " << iters << "\n";
    std::cout << "time         " << ms << " ms\n";
    std::cout << "achieved     " << gbs << " GB/s\n";
    std::cout << "peak         " << vesper::kPeakBandwidthGBs << " GB/s\n";
    std::cout << "frac         " << (gbs / vesper::kPeakBandwidthGBs) << " of peak\n";
}

void bench_official(vesper::Device device, vesper::WeightKind kind) {
    const vesper::ModelConfig cfg = vesper::ModelConfig::qwen38_27b();
    const int h = cfg.hidden_size;
    switch (kind) {
        case vesper::WeightKind::Q4_K:
        case vesper::WeightKind::Q5_K:
            bench_gemv(device, kind, cfg.intermediate_size, h, "ffn_up");
            bench_gemv(device, kind, h, cfg.intermediate_size, "ffn_down");
            return;
        case vesper::WeightKind::Q8_0:
            bench_gemv(device, kind, cfg.q_proj_rows(), h, "attn_q");
            bench_gemv(device, kind, cfg.gdn_qkv_dim(), h, "gdn_qkv");
            return;
        case vesper::WeightKind::Q6_K:
            bench_gemv(device, kind, h, cfg.q_dim(), "attn_o");
            if (device == vesper::Device::HIP) {
                bench_gemv(device, kind, cfg.vocab_size, h, "lm_head");
            } else {
                std::cout << "shape        lm_head\n";
                std::cout << "gemv         Q6_K " << cfg.vocab_size << "x" << h << "\n";
                std::cout << "device       cpu\n";
                std::cout << "skipped      CPU skip of 248320-row lm_head\n";
            }
            return;
        case vesper::WeightKind::F32:
            vesper::fail("no official F32 GEMV bench");
    }
    throw std::logic_error("unhandled WeightKind");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse(argc, argv);
        if (opt.hip_info) {
            print_hip_info();
            return 0;
        }
        if (!opt.inspect.empty()) {
            print_inspect(opt.inspect);
            return 0;
        }
        if (!opt.write_tiny.empty()) {
            vesper::write_tiny_q8(opt.write_tiny, opt.seed);
            std::cout << "wrote        " << opt.write_tiny << "\n";
            return 0;
        }
        if (!opt.write_tiny_hybrid.empty()) {
            vesper::write_tiny_hybrid(opt.write_tiny_hybrid, opt.seed);
            std::cout << "wrote        " << opt.write_tiny_hybrid << "\n";
            return 0;
        }
        if (!opt.write_tiny_qwen35.empty()) {
            vesper::write_tiny_qwen35(opt.write_tiny_qwen35, opt.seed);
            std::cout << "wrote        " << opt.write_tiny_qwen35 << "\n";
            return 0;
        }
        if (!opt.write_tiny_q4km.empty()) {
            vesper::write_tiny_q4km(opt.write_tiny_q4km, opt.seed);
            std::cout << "wrote        " << opt.write_tiny_q4km << "\n";
            return 0;
        }
        if (opt.bench_q8) {
            bench_official(opt.device, vesper::WeightKind::Q8_0);
            return 0;
        }
        if (opt.bench_q4) {
            bench_official(opt.device, vesper::WeightKind::Q4_K);
            return 0;
        }
        if (opt.bench_q5) {
            bench_official(opt.device, vesper::WeightKind::Q5_K);
            return 0;
        }
        if (opt.bench_q6) {
            bench_official(opt.device, vesper::WeightKind::Q6_K);
            return 0;
        }

        vesper::ModelWeights weights;
        vesper::Tokenizer tokenizer = vesper::Tokenizer::bytes();
        const char* prefix = "vesper demo  ";
        if (!opt.model.empty()) {
            weights = vesper::load_model(opt.model);
            tokenizer = vesper::Tokenizer::load(opt.model);
            prefix = "vesper model  ";
        } else if (opt.demo_hybrid) {
            weights = vesper::ModelWeights::random(vesper::ModelConfig::tiny_hybrid(), opt.seed);
            prefix = "vesper hybrid  ";
        } else {
            weights = vesper::ModelWeights::random(vesper::ModelConfig::tiny_demo(), opt.seed);
        }
        vesper::Engine engine(std::move(weights), opt.device, opt.context);
        const std::string label = std::string(prefix) + engine.config().describe();

        const std::vector<int> prompt = tokenizer.encode(opt.prompt);
        const std::vector<int> ids = engine.generate(prompt, opt.tokens);
        if (opt.report_only) {
            vesper::print_report(engine.last_report());
        } else {
            print_generate(label, engine, opt.prompt, ids, prompt.size());
            if (!tokenizer.is_bytes()) {
                std::cout << "text         " << tokenizer.decode(ids) << "\n";
            }
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "vesper-infer: " << ex.what() << "\n";
        return 1;
    }
}
