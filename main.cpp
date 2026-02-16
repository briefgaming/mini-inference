#include "headers/bpe.h"
#include "headers/model.h"
#include "headers/weights.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <string>
#include <vector>

namespace {

constexpr int kBosToken = 128000;
constexpr int kEosToken = 128001;
constexpr int kDefaultMaxSeq = 128;
constexpr int kDefaultMaxNewTokens = 10;

bool debug_tokens_enabled() {
    const char* raw = std::getenv("DEBUG_TOKENS");
    return raw != nullptr && raw[0] != '\0' && std::strcmp(raw, "0") != 0;
}

std::string escape_bytes(const std::string &s) {
    std::string out;
    out.reserve(s.size() * 4);

    char buf[5] = {0};
    for (unsigned char c : s) {
        if (std::isprint(c) && c != '\\' && c != '\'') {
            out.push_back(c);
        } else {
            std::snprintf(buf, sizeof(buf), "\\x%02X", c);
            out += buf;
        }
    }
    return out;
}

std::string decode_one_token(int token, BPEDecode &decoder) {
    char outbuf[1024] = {0};
    decoder.decode(&token, 1, outbuf, sizeof(outbuf));
    return std::string(outbuf);
}

void dump_tokens(const char* title, const std::vector<int> &tokens, BPEDecode &decoder, int max_show = 64) {
    std::printf("%s (%zu)\n", title, tokens.size());
    const size_t shown = std::min(tokens.size(), static_cast<size_t>(max_show));
    for (size_t i = 0; i < shown; ++i) {
        const std::string piece = escape_bytes(decode_one_token(tokens[i], decoder));
        std::printf("  [%zu] id=%d piece='%s'\n", i, tokens[i], piece.c_str());
    }
    if (tokens.size() > shown) {
        std::printf("  ... (%zu more)\n", tokens.size() - shown);
    }
}

std::vector<int> topk_indices(const Tensor<1> &logits, int k) {
    const int n = logits.dim(0);
    if (k <= 0 || n <= 0) {
        return {};
    }
    if (k > n) {
        k = n;
    }

    std::vector<int> idx(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        idx[static_cast<size_t>(i)] = i;
    }

    std::partial_sort(
        idx.begin(),
        idx.begin() + k,
        idx.end(),
        [&logits](int a, int b) {
            return logits[a] > logits[b];
        }
    );
    idx.resize(static_cast<size_t>(k));
    return idx;
}

void dump_decode_step(int step, const Tensor<1> &logits, int chosen, BPEDecode &decoder) {
    const std::string chosen_piece = escape_bytes(decode_one_token(chosen, decoder));
    std::printf(
        "Decode step %d: chosen id=%d piece='%s' logit=%.6f\n",
        step,
        chosen,
        chosen_piece.c_str(),
        logits[chosen]
    );

    const std::vector<int> top5 = topk_indices(logits, 5);
    std::printf("  Top-5 logits:\n");
    for (int id : top5) {
        const std::string piece = escape_bytes(decode_one_token(id, decoder));
        std::printf("    id=%d logit=%.6f piece='%s'\n", id, logits[id], piece.c_str());
    }
}

// Sample best token for next iteration
int argmax_token(const Tensor<1> &logits) {
    int best = 0;
    for (int i = 1; i < logits.dim(0); ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

std::string decode_tokens(const std::vector<int> &tokens, BPEDecode &decoder) {
    if (tokens.empty()) {
        return std::string();
    }

    const int out_cap = tokens.size() * 256 + 1;
    std::vector<char> outbuf(out_cap, '\0');
    decoder.decode(tokens.data(), tokens.size(), outbuf.data(), out_cap);
    return std::string(outbuf.data());
}

std::string prompt_from_args(int argc, char** argv) {
    if (argc <= 1) {
        return "What is the capital of France?";
    }

    std::string prompt = argv[1];
    for (int i = 2; i < argc; ++i) {
        prompt += " ";
        prompt += argv[i];
    }
    return prompt;
}

int max_new_tokens_from_env_or_default() {
    const char* raw = std::getenv("MAX_NEW_TOKENS");
    if (!raw) {
        return kDefaultMaxNewTokens;
    }
    const int v = std::atoi(raw);
    return (v > 0) ? v : kDefaultMaxNewTokens;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string prompt = prompt_from_args(argc, argv);
    const int max_new_tokens = max_new_tokens_from_env_or_default();
    const bool debug_tokens = debug_tokens_enabled();

    BPEDecode decoder;
    if (!decoder.load("vocab.bin")) {
        std::fprintf(stderr, "Failed to load vocab.bin\n");
        return 1;
    }

    BPEEncode encoder;
    if (!encoder.load(decoder.vocab)) {
        std::fprintf(stderr, "Failed to build BPE trie from vocabulary\n");
        return 1;
    }

    Model m;
    if (!m.load_config("configs/llama3_config.json")) {
        std::fprintf(stderr, "Failed to load model config\n");
        return 1;
    }

    WeightMap w;
    if (!w.load("configs/model_index_f32.json")) {
        std::fprintf(stderr, "Failed to load model index\n");
        return 1;
    }

    m.load_weights(w, "llama_f32.bin");
    if (m.layers == nullptr || m.mmap_data == nullptr) {
        std::fprintf(stderr, "Failed to load model weights\n");
        return 1;
    }

    // Per-request cache/session allocation
    const int max_seq = kDefaultMaxSeq;
    std::vector<Tensor<3>> kcache;
    std::vector<Tensor<3>> vcache;
    kcache.reserve(m.n_layers);
    vcache.reserve(m.n_layers);
    for (int l = 0; l < m.n_layers; ++l) {
        kcache.emplace_back(max_seq, m.n_kv_heads, m.head_dim);
        vcache.emplace_back(max_seq, m.n_kv_heads, m.head_dim);
    }

    // Tokenize prompt as BOS + prompt tokens
    std::vector<int> prompt_buf(max_seq, -1);
    int n_prompt = 0;
    const char* rem = encoder.encode(prompt.c_str(), prompt_buf.data(), max_seq - 1, &n_prompt);
    if (rem == nullptr || *rem != '\0') {
        std::fprintf(stderr, "Prompt could not be fully tokenized. Remaining starts at: '%s'\n",
                     rem ? rem : "(null)");
        return 1;
    }

    std::vector<int> input_tokens;
    input_tokens.reserve(n_prompt + 1);
    input_tokens.push_back(kBosToken);
    for (int i = 0; i < n_prompt; ++i) {
        input_tokens.push_back(prompt_buf[i]);
    }

    if (debug_tokens) {
        std::vector<int> prompt_tokens;
        prompt_tokens.reserve(static_cast<size_t>(n_prompt));
        for (int i = 0; i < n_prompt; ++i) {
            prompt_tokens.push_back(prompt_buf[i]);
        }

        dump_tokens("Prompt tokenization (without BOS)", prompt_tokens, decoder);
        dump_tokens("Model input tokenization (with BOS)", input_tokens, decoder);

        const std::string prompt_roundtrip = decode_tokens(prompt_tokens, decoder);
        std::printf("Prompt roundtrip decoded: '%s'\n", escape_bytes(prompt_roundtrip).c_str());
        if (prompt_roundtrip != prompt) {
            std::printf("WARNING: prompt roundtrip mismatch\n");
            std::printf("  input    : '%s'\n", escape_bytes(prompt).c_str());
            std::printf("  roundtrip: '%s'\n", escape_bytes(prompt_roundtrip).c_str());
        }
    }

    if (input_tokens.size() >= max_seq) {
        std::fprintf(stderr, "Prompt is too long for max_seq=%d\n", max_seq);
        return 1;
    }

    Tensor<1> logits(m.vocab_size);

    // Prefill
    using Clock = std::chrono::steady_clock;
    const auto prefill_start = Clock::now();
    int pos = 0;
    for (; pos < input_tokens.size(); ++pos) {
        m.forward(input_tokens[pos], pos, kcache, vcache, logits);
    }
    const auto prefill_end = Clock::now();
    const auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start).count();

    // Autoregressive decode
    // start using sampled logits after exhausting the input tokens
    const auto decode_start = Clock::now();
    std::vector<int> generated;
    generated.reserve(max_new_tokens);
    for (int step = 0; step < max_new_tokens; ++step) {
        const int next = argmax_token(logits);
        if (debug_tokens) {
            dump_decode_step(step, logits, next, decoder);
        }

        // Stop conditions: EOS / context full.
        if (next == kEosToken || pos >= max_seq) {
            break;
        }

        generated.push_back(next);
        m.forward(next, pos, kcache, vcache, logits);
        ++pos;
    }
    const auto decode_end = Clock::now();
    const auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count();

    const std::string output = decode_tokens(generated, decoder);
    std::printf("Prompt: %s\n", prompt.c_str());
    std::printf("Response: %s\n", output.c_str());
    std::printf("Timing:\n");
    std::printf("  Prefill: %lld ms for %zu tokens (%.3f ms/token)\n",
                prefill_ms,
                input_tokens.size(),
                input_tokens.empty() ? 0.0 : prefill_ms / input_tokens.size());
    std::printf("  Decode:  %lld ms for %zu tokens (%.3f ms/token)\n",
                decode_ms,
                generated.size(),
                generated.empty() ? 0.0 : decode_ms / generated.size());
    return 0;
}
