#include "vesper/config.h"
#include "vesper/engine.h"
#include "vesper/weights.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    bool demo = false;
    std::string prompt = "hello";
    int tokens = 32;
    std::uint32_t seed = 1;
};

void usage() {
    std::cout
        << "vesper-infer -- AMD-first local LLM inference\n"
        << "\n"
        << "  --demo            run the tiny random Qwen3-style model\n"
        << "  --prompt TEXT     byte-tokenized prompt (default: hello)\n"
        << "  --tokens N        new tokens to generate (default: 32)\n"
        << "  --seed N          weight seed (default: 1)\n"
        << "  --help            this message\n"
        << "\n"
        << "Weight loading and HIP are not wired yet. See docs/DESIGN.md.\n";
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
    if (!opt.demo) {
        usage();
        vesper::fail("this build only supports --demo");
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse(argc, argv);
        const vesper::ModelConfig cfg = vesper::ModelConfig::tiny_demo();
        vesper::Engine engine(vesper::ModelWeights::random(cfg, opt.seed));

        const std::vector<int> prompt = vesper::encode_bytes(opt.prompt);
        const std::vector<int> ids = engine.generate(prompt, opt.tokens);
        const vesper::GenerateStats& stats = engine.last_stats();

        std::cout << "vesper demo  " << cfg.describe() << "\n";
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
