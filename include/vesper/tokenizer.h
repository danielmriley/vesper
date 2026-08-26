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

private:
    bool bytes_ = true;
    PretokKind pretok_ = PretokKind::Bytes;
    int bos_ = -1;
    int eos_ = -1;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merge_rank_;
};

}  // namespace vesper
