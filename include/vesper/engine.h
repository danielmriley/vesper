#pragma once

#include "vesper/gdn.h"
#include "vesper/kv_cache.h"
#include "vesper/report.h"
#include "vesper/target.h"
#include "vesper/types.h"
#include "vesper/weights.h"

#include <string>
#include <vector>

namespace vesper {

struct GenerateStats {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    int prompt_tokens = 0;
    int generated_tokens = 0;

    double decode_tps() const;
    double prefill_tps() const;
};

class Engine {
public:
    // context > 0 caps KV length. Official qwen35 files store 262144; the
    // default is kDefaultContext (4096). context == 0 keeps the file value.
    explicit Engine(ModelWeights weights, Device device = Device::CPU,
                    int context = kDefaultContext);

    void reset();
    void step(int token);
    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens);

    const float* logits() const;
    Device device() const { return device_; }
    const ModelConfig& config() const { return weights_.config; }
    const ModelWeights& weights() const { return weights_; }
    const KVCache& cache() const { return cache_; }
    const GenerateStats& last_stats() const { return stats_; }
    DecodeReport last_report() const;

private:
    void ensure_room() const;
    void forward_token(int token);

    ModelWeights weights_;
    KVCache cache_;
    GenerateStats stats_;
    Device device_ = Device::CPU;
    mutable std::vector<float> host_logits_;

    struct Scratch {
        Buffer x;
        Buffer residual;
        Buffer q;
        Buffer k;
        Buffer v;
        Buffer attn;
        Buffer gate;
        Buffer up;
        Buffer hidden;
        Buffer logits;
        Buffer scores;
        Buffer q_full;
        Buffer attn_gate;
        GdnScratch gdn;
    } scratch_;
    std::vector<float> host_embed_;
};

std::vector<int> encode_bytes(const std::string& text);
std::string decode_bytes(const std::vector<int>& tokens);

}  // namespace vesper
