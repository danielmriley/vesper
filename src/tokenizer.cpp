#include "vesper/tokenizer.h"

#include "vesper/types.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
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

bool is_mark_cp(std::uint32_t c) {
    if (c >= 0x0300 && c <= 0x036f) {
        return true;
    }
    if (c >= 0x0483 && c <= 0x0489) {
        return true;
    }
    if (c >= 0x0591 && c <= 0x05c7) {
        return true;
    }
    if (c >= 0x0610 && c <= 0x061a) {
        return true;
    }
    if (c >= 0x064b && c <= 0x065f) {
        return true;
    }
    if (c == 0x0670) {
        return true;
    }
    if (c >= 0x06d6 && c <= 0x06ed) {
        return true;
    }
    if (c >= 0x1ab0 && c <= 0x1aff) {
        return true;
    }
    if (c >= 0x1dc0 && c <= 0x1dff) {
        return true;
    }
    if (c >= 0x20d0 && c <= 0x20ff) {
        return true;
    }
    if (c >= 0xfe20 && c <= 0xfe2f) {
        return true;
    }
    return false;
}

bool is_letter_cp(std::uint32_t c) {
    if (c < 128) {
        return is_ascii_letter(static_cast<unsigned char>(c));
    }
    if (c >= 0xff10 && c <= 0xff19) {
        return false;
    }
    if (is_mark_cp(c)) {
        return false;
    }
    return true;
}

bool is_number_cp(std::uint32_t c) {
    return (c >= '0' && c <= '9') || (c >= 0xff10 && c <= 0xff19);
}

bool is_space_cp(std::uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0b || c == 0x0c;
}

struct Codepoint {
    std::uint32_t v = 0;
    int nbytes = 1;
};

Codepoint next_cp(std::string_view text, std::size_t i) {
    check(i < text.size(), "pretok past end of text");
    const unsigned char lead = static_cast<unsigned char>(text[i]);
    Codepoint cp;
    if (lead < 0x80) {
        cp.v = lead;
        cp.nbytes = 1;
        return cp;
    }
    if ((lead & 0xe0) == 0xc0 && i + 1 < text.size()) {
        cp.v = (static_cast<std::uint32_t>(lead & 0x1f) << 6) |
               (static_cast<unsigned char>(text[i + 1]) & 0x3f);
        cp.nbytes = 2;
        return cp;
    }
    if ((lead & 0xf0) == 0xe0 && i + 2 < text.size()) {
        cp.v = (static_cast<std::uint32_t>(lead & 0x0f) << 12) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3f) << 6) |
               (static_cast<unsigned char>(text[i + 2]) & 0x3f);
        cp.nbytes = 3;
        return cp;
    }
    if ((lead & 0xf8) == 0xf0 && i + 3 < text.size()) {
        cp.v = (static_cast<std::uint32_t>(lead & 0x07) << 18) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3f) << 12) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3f) << 6) |
               (static_cast<unsigned char>(text[i + 3]) & 0x3f);
        cp.nbytes = 4;
        return cp;
    }
    cp.v = lead;
    cp.nbytes = 1;
    return cp;
}

int contraction_len(std::string_view text, std::size_t i) {
    if (i >= text.size() || text[i] != '\'') {
        return 0;
    }
    auto match = [&](std::string_view lit) {
        if (i + 1 + lit.size() > text.size()) {
            return false;
        }
        for (std::size_t k = 0; k < lit.size(); ++k) {
            char ch = text[i + 1 + k];
            if (ch >= 'A' && ch <= 'Z') {
                ch = static_cast<char>(ch - 'A' + 'a');
            }
            if (ch != lit[k]) {
                return false;
            }
        }
        const std::size_t end = i + 1 + lit.size();
        if (end < text.size() && is_letter_cp(next_cp(text, end).v)) {
            return false;
        }
        return true;
    };
    if (match("re") || match("ve") || match("ll")) {
        return 3;
    }
    if (match("s") || match("t") || match("m") || match("d")) {
        return 2;
    }
    return 0;
}

bool in_letter_run(std::uint32_t c, bool attach_marks) {
    return is_letter_cp(c) || (attach_marks && is_mark_cp(c));
}

bool stops_punct(std::uint32_t c, bool attach_marks) {
    if (is_space_cp(c) || is_letter_cp(c) || is_number_cp(c)) {
        return true;
    }
    return attach_marks && is_mark_cp(c);
}

std::vector<std::string> pretok_qwen(std::string_view text, bool attach_marks) {
    std::vector<std::string> parts;
    std::size_t i = 0;
    while (i < text.size()) {
        if (const int n = contraction_len(text, i)) {
            parts.emplace_back(text.substr(i, static_cast<std::size_t>(n)));
            i += static_cast<std::size_t>(n);
            continue;
        }

        {
            std::size_t j = i;
            const Codepoint c0 = next_cp(text, j);
            if (c0.v != '\r' && c0.v != '\n' && !is_letter_cp(c0.v) && !is_number_cp(c0.v)) {
                j += static_cast<std::size_t>(c0.nbytes);
            }
            if (j < text.size() && in_letter_run(next_cp(text, j).v, attach_marks)) {
                while (j < text.size()) {
                    const Codepoint c = next_cp(text, j);
                    if (!in_letter_run(c.v, attach_marks)) {
                        break;
                    }
                    j += static_cast<std::size_t>(c.nbytes);
                }
                parts.emplace_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }

        {
            const Codepoint c = next_cp(text, i);
            if (is_number_cp(c.v)) {
                parts.emplace_back(text.substr(i, static_cast<std::size_t>(c.nbytes)));
                i += static_cast<std::size_t>(c.nbytes);
                continue;
            }
        }

        {
            std::size_t j = i;
            if (j < text.size() && text[j] == ' ') {
                ++j;
            }
            bool any = false;
            while (j < text.size()) {
                const Codepoint c = next_cp(text, j);
                if (stops_punct(c.v, attach_marks)) {
                    break;
                }
                j += static_cast<std::size_t>(c.nbytes);
                any = true;
            }
            if (any) {
                while (j < text.size() && (text[j] == '\r' || text[j] == '\n')) {
                    ++j;
                }
                parts.emplace_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }

        {
            std::size_t j = i;
            while (j < text.size()) {
                const unsigned char c = static_cast<unsigned char>(text[j]);
                if (c == ' ' || c == '\t' || c == '\v' || c == '\f') {
                    ++j;
                    continue;
                }
                break;
            }
            if (j < text.size() && (text[j] == '\r' || text[j] == '\n')) {
                while (j < text.size() && (text[j] == '\r' || text[j] == '\n')) {
                    ++j;
                }
                parts.emplace_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }

        if (is_space_cp(static_cast<unsigned char>(text[i]))) {
            std::size_t j = i;
            while (j < text.size() && is_space_cp(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            parts.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        const Codepoint c = next_cp(text, i);
        parts.emplace_back(text.substr(i, static_cast<std::size_t>(c.nbytes)));
        i += static_cast<std::size_t>(c.nbytes);
    }
    return parts;
}

PretokKind pretok_from_pre(std::string_view pre) {
    if (pre == "default" || pre == "gpt2" || pre == "llama-bpe") {
        return PretokKind::Gpt2;
    }
    if (pre == "qwen35" || pre == "qwen3_5" || pre == "qwen3.5") {
        return PretokKind::Qwen35;
    }
    return PretokKind::Qwen2;
}

enum class GgufTokenType : std::uint32_t {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
};

bool is_special_token_type(std::uint64_t type) {
    switch (static_cast<GgufTokenType>(type)) {
        case GgufTokenType::Control:
        case GgufTokenType::UserDefined:
            return true;
        case GgufTokenType::Undefined:
        case GgufTokenType::Normal:
        case GgufTokenType::Unknown:
        case GgufTokenType::Unused:
        case GgufTokenType::Byte:
            return false;
    }
    return type == 3 || type == 4;
}

}  // namespace

Tokenizer Tokenizer::bytes() {
    Tokenizer tok;
    tok.bytes_ = true;
    tok.pretok_ = PretokKind::Bytes;
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
    tok.pretok_ = PretokKind::Qwen2;
    if (file.has_kv("tokenizer.ggml.pre")) {
        tok.pretok_ = pretok_from_pre(file.kv_string("tokenizer.ggml.pre"));
    }
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
    if (file.has_kv("tokenizer.ggml.token_type")) {
        const std::vector<std::uint64_t> types = file.kv_u64_array("tokenizer.ggml.token_type");
        const std::size_t n = std::min(types.size(), tok.id_to_token_.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (!is_special_token_type(types[i])) {
                continue;
            }
            Tokenizer::SpecialToken spec;
            spec.text = tok.id_to_token_[i];
            spec.id = static_cast<int>(i);
            if (!spec.text.empty()) {
                tok.specials_.push_back(std::move(spec));
            }
        }
    }
    return tok;
}

Tokenizer Tokenizer::load(const std::string& path) {
    return from_gguf(GgufFile::open(path));
}

std::vector<std::string> pretok_parts(PretokKind kind, std::string_view text) {
    switch (kind) {
        case PretokKind::Bytes:
            fail("encode_plain is not used for the byte tokenizer");
        case PretokKind::Gpt2:
            return pretokenize(text);
        case PretokKind::Qwen2:
            return pretok_qwen(text, false);
        case PretokKind::Qwen35:
            return pretok_qwen(text, true);
    }
    throw std::logic_error("unhandled PretokKind");
}

void Tokenizer::encode_plain(std::vector<int>* ids, std::string_view text) const {
    check(ids != nullptr, "encode_plain null ids");
    if (text.empty()) {
        return;
    }
    const std::vector<std::string> parts = pretok_parts(pretok_, text);
    for (const std::string& part : parts) {
        const std::string mapped = map_bytes(part);
        const auto exact = token_to_id_.find(mapped);
        if (exact != token_to_id_.end()) {
            ids->push_back(exact->second);
            continue;
        }
        for (const std::string& piece : bpe(mapped, merge_rank_)) {
            const auto it = token_to_id_.find(piece);
            if (it != token_to_id_.end()) {
                ids->push_back(it->second);
                continue;
            }
            for (unsigned char byte : piece) {
                char hex[8];
                std::snprintf(hex, sizeof(hex), "<0x%02X>", byte);
                const auto byte_it = token_to_id_.find(hex);
                check(byte_it != token_to_id_.end(),
                      "tokenizer missing piece '" + piece + "'");
                ids->push_back(byte_it->second);
            }
        }
    }
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
    if (specials_.empty()) {
        encode_plain(&ids, text);
    } else {
        std::size_t i = 0;
        while (i < text.size()) {
            std::size_t best_len = 0;
            int best_id = -1;
            for (const SpecialToken& spec : specials_) {
                if (spec.text.size() <= best_len || i + spec.text.size() > text.size()) {
                    continue;
                }
                if (text.substr(i, spec.text.size()) == spec.text) {
                    best_len = spec.text.size();
                    best_id = spec.id;
                }
            }
            if (best_len > 0) {
                ids.push_back(best_id);
                i += best_len;
                continue;
            }
            std::size_t next = text.size();
            for (const SpecialToken& spec : specials_) {
                const std::size_t pos = text.find(spec.text, i);
                if (pos != std::string_view::npos && pos < next) {
                    next = pos;
                }
            }
            encode_plain(&ids, text.substr(i, next - i));
            i = next;
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
