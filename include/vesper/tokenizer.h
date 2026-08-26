#pragma once

#include "vesper/gguf.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vesper {

enum class PretokKind {
    Bytes,
    Gpt2,
    Qwen2,
    Qwen35,
};

class Tokenizer {
public:
    static Tokenizer bytes();
    static Tokenizer from_gguf(const GgufFile& file);
    static Tokenizer load(const std::string& path);

    std::vector<int> encode(std::string_view text) const;
    std::string decode(const std::vector<int>& tokens) const;

    bool is_bytes() const { return bytes_; }
    PretokKind pretok() const { return pretok_; }
    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    int bos_id() const { return bos_; }
    int eos_id() const { return eos_; }
    int special_count() const { return static_cast<int>(specials_.size()); }

private:
    struct SpecialToken {
        std::string text;
        int id = -1;
    };

    void encode_plain(std::vector<int>* ids, std::string_view text) const;

    bool bytes_ = true;
    PretokKind pretok_ = PretokKind::Bytes;
    int bos_ = -1;
    int eos_ = -1;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<SpecialToken> specials_;
};

}  // namespace vesper
