#include "vesper/config.h"
#include "vesper/engine.h"
#include "vesper/hip.h"
#include "vesper/weights.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    bool demo = false;
    bool hip_info = false;
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
        << "  --device cpu|hip  decode device (default: cpu)\n"
        << "  --hip-info        print HIP probe and exit\n"
        << "  --prompt TEXT     byte-tokenized prompt (default: hello)\n"
        << "  --tokens N        new tokens to generate (default: 32)\n"
        << "  --seed N          weight seed (default: 1)\n"
        << "  --help            this message\n"
        << "\n"
        << "Weight loading is not wired yet. HIP is gfx1201-only. See docs/TARGET.md.\n";
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
    if (!opt.demo && !opt.hip_info) {
        usage();
        vesper::fail("this build supports --demo and --hip-info");
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse(argc, argv);
        if (opt.hip_info) {
            print_hip_info();
            return 0;
        }

        const vesper::ModelConfig cfg = vesper::ModelConfig::tiny_demo();
        vesper::Engine engine(vesper::ModelWeights::random(cfg, opt.seed), opt.device);

        const std::vector<int> prompt = vesper::encode_bytes(opt.prompt);
        const std::vector<int> ids = engine.generate(prompt, opt.tokens);
        const vesper::GenerateStats& stats = engine.last_stats();

        std::cout << "vesper demo  " << cfg.describe() << "\n";
        std::cout << "device       " << vesper::device_name(engine.device());
        if (engine.device() == vesper::Device::HIP) {
            std::cout << " " << vesper::hip_arch() << " " << vesper::hip_device_name();
        }
        std::cout << "\n";
        std::cout << "prompt       " << opt.prompt << " (" << stats.prompt_tokens << " bytes)\n";
        std::cout << "generated    " << stats.generated_tokens << " tokens\n";
        std::cout << "prefill      " << stats.prefill_tps() << " tok/s  (" << stats.prefill_ms
                  << " ms)\n";
        std::cout << "decode       " << stats.decode_tps() << " tok/s  (" << stats.decode_ms
                  << " ms)\n";
        std::cout << "greedy ids   ";
        for (std::size_t i = prompt.size(); i < ids.size(); ++i) {
            if (i > prompt.size()) {
                std::cout << ' ';
            }
            std::cout << ids[i];
        }
        std::cout << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "vesper-infer: " << ex.what() << "\n";
        return 1;
    }
}
