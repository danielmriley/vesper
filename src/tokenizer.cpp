#include "vesper/tokenizer.h"

#include "vesper/types.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <utility>

namespace vesper {
namespace {

bool is_ascii_letter(unsigned char ch) {
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
}

bool is_ascii_digit(unsigned char ch) {
    return ch >= '0' && ch <= '9';
}

std::string bytes_to_unicode(unsigned char byte) {
    static const std::vector<std::string> table = [] {
        std::vector<int> bs;
        for (int i = 33; i <= 126; ++i) {
            bs.push_back(i);
        }
        for (int i = 161; i <= 172; ++i) {
            bs.push_back(i);
        }
        for (int i = 174; i <= 255; ++i) {
            bs.push_back(i);
        }
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) != bs.end()) {
                continue;
            }
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
        std::vector<std::string> out(256);
        for (std::size_t i = 0; i < bs.size(); ++i) {
            const int code = cs[i];
            std::string s;
            if (code < 128) {
                s.push_back(static_cast<char>(code));
            } else if (code < 2048) {
                s.push_back(static_cast<char>(0xc0 | (code >> 6)));
                s.push_back(static_cast<char>(0x80 | (code & 0x3f)));
            } else {
                s.push_back(static_cast<char>(0xe0 | (code >> 12)));
                s.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
                s.push_back(static_cast<char>(0x80 | (code & 0x3f)));
            }
            out[static_cast<std::size_t>(bs[i])] = s;
        }
        return out;
    }();
    return table[byte];
}

unsigned char unicode_to_byte(std::string_view ch) {
    static const std::unordered_map<std::string, unsigned char> table = [] {
        std::unordered_map<std::string, unsigned char> out;
        for (int b = 0; b < 256; ++b) {
            out.emplace(bytes_to_unicode(static_cast<unsigned char>(b)),
                        static_cast<unsigned char>(b));
        }
        return out;
    }();
    const auto it = table.find(std::string(ch));
    check(it != table.end(), "tokenizer decode hit an unknown byte symbol");
    return it->second;
}

std::vector<std::string> utf8_chars(std::string_view text) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        std::size_t n = 1;
        if ((lead & 0x80) == 0) {
            n = 1;
        } else if ((lead & 0xe0) == 0xc0) {
            n = 2;
        } else if ((lead & 0xf0) == 0xe0) {
            n = 3;
        } else if ((lead & 0xf8) == 0xf0) {
            n = 4;
        }
        check(i + n <= text.size(), "truncated UTF-8 in tokenizer");
        out.emplace_back(text.substr(i, n));
        i += n;
    }
    return out;
}

std::vector<std::string> pretokenize(std::string_view text) {
    std::vector<std::string> parts;
    std::size_t i = 0;
    while (i < text.size()) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (ch == ' ' && i + 1 < text.size()) {
            const unsigned char next = static_cast<unsigned char>(text[i + 1]);
            if (is_ascii_letter(next) || next >= 0x80) {
                std::size_t j = i + 1;
                while (j < text.size()) {
                    const unsigned char c = static_cast<unsigned char>(text[j]);
                    if (!is_ascii_letter(c) && c < 0x80) {
                        break;
                    }
                    ++j;
                }
                parts.emplace_back(text.substr(i, j - i));
                i = j;
                continue;
            }
            if (is_ascii_digit(next)) {
                std::size_t j = i + 1;
                while (j < text.size() && is_ascii_digit(static_cast<unsigned char>(text[j]))) {
                    ++j;
                }
                parts.emplace_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }
        if (is_ascii_letter(ch) || ch >= 0x80) {
            std::size_t j = i + 1;
            while (j < text.size()) {
                const unsigned char c = static_cast<unsigned char>(text[j]);
                if (!is_ascii_letter(c) && c < 0x80) {
                    break;
                }
                ++j;
            }
            parts.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        if (is_ascii_digit(ch)) {
            std::size_t j = i + 1;
            while (j < text.size() && is_ascii_digit(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            parts.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        parts.emplace_back(text.substr(i, 1));
        ++i;
    }
    return parts;
}

std::string map_bytes(std::string_view raw) {
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char ch : raw) {
        out += bytes_to_unicode(ch);
    }
    return out;
}

int pair_rank(const std::unordered_map<std::string, int>& ranks, const std::string& a,
              const std::string& b) {
    const auto it = ranks.find(a + " " + b);
    if (it == ranks.end()) {
        return std::numeric_limits<int>::max();
    }
    return it->second;
}

std::vector<std::string> bpe(const std::string& mapped,
                             const std::unordered_map<std::string, int>& ranks) {
    std::vector<std::string> parts = utf8_chars(mapped);
    if (parts.size() < 2) {
        return parts;
    }
    while (parts.size() >= 2) {
        int best = std::numeric_limits<int>::max();
        std::size_t at = 0;
        for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
            const int rank = pair_rank(ranks, parts[i], parts[i + 1]);
            if (rank < best) {
                best = rank;
                at = i;
            }
        }
        if (best == std::numeric_limits<int>::max()) {
            break;
        }
        parts[at] += parts[at + 1];
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(at + 1));
    }
    return parts;
}

}  // namespace

Tokenizer Tokenizer::bytes() {
    Tokenizer tok;
    tok.bytes_ = true;
    tok.id_to_token_.resize(256);
    for (int i = 0; i < 256; ++i) {
        tok.id_to_token_[static_cast<std::size_t>(i)] = std::string(1, static_cast<char>(i));
        tok.token_to_id_[tok.id_to_token_[static_cast<std::size_t>(i)]] = i;
    }
    return tok;
}

Tokenizer Tokenizer::from_gguf(const GgufFile& file) {
    if (!file.has_kv("tokenizer.ggml.tokens")) {
        return bytes();
    }
    Tokenizer tok;
    tok.bytes_ = false;
    tok.id_to_token_ = file.kv_string_array("tokenizer.ggml.tokens");
    check(!tok.id_to_token_.empty(), "tokenizer.ggml.tokens is empty");
    tok.token_to_id_.reserve(tok.id_to_token_.size());
    for (int i = 0; i < static_cast<int>(tok.id_to_token_.size()); ++i) {
        tok.token_to_id_.emplace(tok.id_to_token_[static_cast<std::size_t>(i)], i);
    }
    if (file.has_kv("tokenizer.ggml.merges")) {
        const std::vector<std::string> merges = file.kv_string_array("tokenizer.ggml.merges");
        tok.merge_rank_.reserve(merges.size());
        for (int i = 0; i < static_cast<int>(merges.size()); ++i) {
            tok.merge_rank_.emplace(merges[static_cast<std::size_t>(i)], i);
        }
    }
    if (file.has_kv("tokenizer.ggml.bos_token_id")) {
        tok.bos_ = static_cast<int>(file.kv_u64("tokenizer.ggml.bos_token_id"));
    }
    if (file.has_kv("tokenizer.ggml.eos_token_id")) {
        tok.eos_ = static_cast<int>(file.kv_u64("tokenizer.ggml.eos_token_id"));
    }
    return tok;
}

Tokenizer Tokenizer::load(const std::string& path) {
    return from_gguf(GgufFile::open(path));
}

std::vector<int> Tokenizer::encode(std::string_view text) const {
    if (bytes_) {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (unsigned char ch : text) {
            ids.push_back(static_cast<int>(ch));
        }
        if (ids.empty()) {
            ids.push_back(0);
        }
        return ids;
    }

    std::vector<int> ids;
    if (text.empty()) {
        ids.push_back(bos_ >= 0 ? bos_ : 0);
        return ids;
    }
    for (const std::string& part : pretokenize(text)) {
        const std::string mapped = map_bytes(part);
        const auto exact = token_to_id_.find(mapped);
        if (exact != token_to_id_.end()) {
            ids.push_back(exact->second);
            continue;
        }
        for (const std::string& piece : bpe(mapped, merge_rank_)) {
            const auto it = token_to_id_.find(piece);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
                continue;
            }
            for (unsigned char byte : piece) {
                char hex[8];
                std::snprintf(hex, sizeof(hex), "<0x%02X>", byte);
                const auto byte_it = token_to_id_.find(hex);
                check(byte_it != token_to_id_.end(),
                      "tokenizer missing piece '" + piece + "'");
                ids.push_back(byte_it->second);
            }
        }
    }
    if (ids.empty()) {
        ids.push_back(bos_ >= 0 ? bos_ : 0);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    if (bytes_) {
        std::string text;
        text.reserve(tokens.size());
        for (int id : tokens) {
            check(id >= 0 && id < 256, "byte tokenizer only accepts ids in [0, 255]");
            text.push_back(static_cast<char>(id));
        }
        return text;
    }
    std::string mapped;
    for (int id : tokens) {
        check(id >= 0 && id < static_cast<int>(id_to_token_.size()), "token id out of vocab");
        const std::string& piece = id_to_token_[static_cast<std::size_t>(id)];
        if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
            piece[5] == '>') {
            unsigned value = 0;
            std::sscanf(piece.c_str(), "<0x%02X>", &value);
            mapped += bytes_to_unicode(static_cast<unsigned char>(value));
            continue;
        }
        mapped += piece;
    }
    std::string out;
    for (const std::string& ch : utf8_chars(mapped)) {
        out.push_back(static_cast<char>(unicode_to_byte(ch)));
    }
    return out;
}

}  // namespace vesper
