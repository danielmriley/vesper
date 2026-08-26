#include "vesper/config.h"
#include "vesper/engine.h"
#include "vesper/gguf.h"
#include "vesper/hip.h"
#include "vesper/kernels.h"
#include "vesper/model_io.h"
#include "vesper/q8.h"
#include "vesper/target.h"
#include "vesper/weight.h"
#include "vesper/weights.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    bool demo = false;
    bool hip_info = false;
    bool bench_q8 = false;
    std::string inspect;
    std::string model;
    std::string write_tiny;
    std::string prompt = "hello";
    int tokens = 32;
    std::uint32_t seed = 1;
    vesper::Device device = vesper::Device::CPU;
};

void usage() {
    std::cout
        << "vesper-infer -- AMD-first local LLM inference\n"
        << "  v1 HIP target: " << vesper::kHipArch
        << " (Radeon AI Pro R9700, wave" << vesper::kWavefront << ")\n"
        << "\n"
        << "  --demo            run the tiny random Qwen3-style model\n"
        << "  --model PATH      generate from a vesper_tiny GGUF\n"
        << "  --write-tiny PATH write the demo as a Q8_0 GGUF and exit\n"
        << "  --inspect PATH    print GGUF version, alignment, tensors\n"
        << "  --bench-q8        time fused Q8_0 GEMV and print GB/s\n"
        << "  --device cpu|hip  decode device (default: cpu)\n"
        << "  --hip-info        print HIP probe and exit\n"
        << "  --prompt TEXT     byte-tokenized prompt (default: hello)\n"
        << "  --tokens N        new tokens to generate (default: 32)\n"
        << "  --seed N          weight seed (default: 1)\n"
        << "  --help            this message\n"
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
        } else if (arg == "--model") {
            opt.model = need("--model");
        } else if (arg == "--write-tiny") {
            opt.write_tiny = need("--write-tiny");
        } else if (arg == "--inspect") {
            opt.inspect = need("--inspect");
        } else if (arg == "--bench-q8") {
            opt.bench_q8 = true;
        } else if (arg == "--hip-info") {
            opt.hip_info = true;
        } else if (arg == "--device") {
            opt.device = parse_device(need("--device"));
        } else if (arg == "--prompt") {
            opt.prompt = need("--prompt");
        } else if (arg == "--tokens") {
            opt.tokens = std::atoi(need("--tokens"));
        } else if (arg == "--seed") {
            opt.seed = static_cast<std::uint32_t>(std::strtoul(need("--seed"), nullptr, 10));
        } else if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else {
            vesper::fail("unknown argument: " + arg);
        }
    }
    if (opt.demo && !opt.model.empty()) {
        vesper::fail("use either --demo or --model, not both");
    }
    const bool ok = opt.demo || opt.hip_info || opt.bench_q8 || !opt.inspect.empty() ||
                    !opt.model.empty() || !opt.write_tiny.empty();
    if (!ok) {
        usage();
        vesper::fail("need --demo, --model, --write-tiny, --inspect, --bench-q8, or --hip-info");
    }
    return opt;
}

void print_inspect(const std::string& path) {
    const vesper::GgufFile file = vesper::GgufFile::open(path);
    std::cout << "version      " << file.version() << "\n";
    std::cout << "alignment    " << file.alignment() << "\n";
    std::cout << "architecture " << file.architecture() << "\n";
    std::cout << "kv           " << file.kv_count() << "\n";
    std::cout << "tensors      " << file.tensors().size() << "\n";
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
    const vesper::GenerateStats& stats = engine.last_stats();
    const std::size_t bytes_tok = engine.weights().linear_bytes();
    const double decode_s = stats.decode_ms / 1000.0;
    const double moved = static_cast<double>(bytes_tok) *
                         static_cast<double>(stats.generated_tokens);
    const double gbs = (decode_s > 0.0) ? (moved / decode_s) / 1e9 : 0.0;

    std::cout << label << "\n";
    std::cout << "device       " << vesper::device_name(engine.device());
    if (engine.device() == vesper::Device::HIP) {
        std::cout << " " << vesper::hip_arch() << " " << vesper::hip_device_name();
    }
    std::cout << "\n";
    std::cout << "prompt       " << prompt << " (" << stats.prompt_tokens << " bytes)\n";
    std::cout << "generated    " << stats.generated_tokens << " tokens\n";
    std::cout << "prefill      " << stats.prefill_tps() << " tok/s  (" << stats.prefill_ms
              << " ms)\n";
    std::cout << "decode       " << stats.decode_tps() << " tok/s  (" << stats.decode_ms
              << " ms)\n";
    std::cout << "weight/tok   " << bytes_tok << " B\n";
    std::cout << "achieved     " << gbs << " GB/s\n";
    std::cout << "peak         " << vesper::kPeakBandwidthGBs << " GB/s\n";
    std::cout << "greedy ids   ";
    for (std::size_t i = prompt_n; i < ids.size(); ++i) {
        if (i > prompt_n) {
            std::cout << ' ';
        }
        std::cout << ids[i];
    }
    std::cout << "\n";
}

void bench_q8(vesper::Device device) {
    const int rows = 1024;
    const int cols = 1024;
    std::vector<float> host_w(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    std::vector<float> host_x(static_cast<std::size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        host_w[static_cast<std::size_t>(i)] = 0.01f * static_cast<float>((i * 17) % 23 - 11);
    }
    for (int i = 0; i < cols; ++i) {
        host_x[static_cast<std::size_t>(i)] = 0.05f * static_cast<float>((i * 9) % 13 - 6);
    }
    vesper::WeightMatrix packed = vesper::WeightMatrix::q8_from_f32(host_w.data(), rows, cols);
    const std::size_t nbytes = packed.bytes();
    const int warmup = 3;
    const int iters = 20;

    double ms = 0.0;
    switch (device) {
        case vesper::Device::CPU: {
            std::vector<float> y(static_cast<std::size_t>(rows), 0.0f);
            for (int i = 0; i < warmup; ++i) {
                vesper::gemv_q8(y.data(), packed.packed(), host_x.data(), rows, cols);
            }
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                vesper::gemv_q8(y.data(), packed.packed(), host_x.data(), rows, cols);
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
    std::cout << "gemv_q8      " << rows << "x" << cols << "\n";
    std::cout << "device       " << vesper::device_name(device) << "\n";
    std::cout << "weight bytes " << nbytes << "\n";
    std::cout << "iters        " << iters << "\n";
    std::cout << "time         " << ms << " ms\n";
    std::cout << "achieved     " << gbs << " GB/s\n";
    std::cout << "peak         " << vesper::kPeakBandwidthGBs << " GB/s\n";
    std::cout << "frac         " << (gbs / vesper::kPeakBandwidthGBs) << " of peak\n";
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
        if (opt.bench_q8) {
            bench_q8(opt.device);
            return 0;
        }

        vesper::ModelWeights weights;
        std::string label;
        if (!opt.model.empty()) {
            weights = vesper::load_model(opt.model);
            label = "vesper model  " + weights.config.describe();
        } else {
            weights = vesper::ModelWeights::random(vesper::ModelConfig::tiny_demo(), opt.seed);
            label = "vesper demo  " + weights.config.describe();
        }
        vesper::Engine engine(std::move(weights), opt.device);

        const std::vector<int> prompt = vesper::encode_bytes(opt.prompt);
        const std::vector<int> ids = engine.generate(prompt, opt.tokens);
        print_generate(label, engine, opt.prompt, ids, prompt.size());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "vesper-infer: " << ex.what() << "\n";
        return 1;
    }
}
