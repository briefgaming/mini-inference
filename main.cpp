#include "headers/bpe.h"
#include "headers/model.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    BPEEncode encoder;
    BPEDecode decoder;

    if (!decoder.load("vocab.bin")) {
        std::fprintf(stderr, "Failed to load vocabulary\n");
        return 1;
    }

    if (!encoder.load(decoder.vocab)) {
        std::fprintf(stderr, "Failed to build vocabulary tree\n");
        return 1;
    }

    std::string sample_prompt = "What is the capital of France?";
    int token_buf[1024];
    int ntokens = 0;
    const char* remaining = encoder.encode(
        sample_prompt.c_str(), token_buf, 1024, &ntokens
    );

    if (remaining == nullptr || *remaining != '\0') {
        std::fprintf(stderr, "Failed to tokenize full input. Remaining: '%s'\n",
                     remaining ? remaining : "(null)");
        return 1;
    }

    std::printf("Tokens (%d):", ntokens);
    for (int i = 0; i < ntokens; i++) {
        std::printf(" %d", token_buf[i]);
    }
    std::printf("\n");

    char decoded[4096];
    decoder.decode(token_buf, ntokens, decoded, sizeof(decoded));
    std::printf("Decoded: %s\n", decoded);

    Model m;

    // load weights config into model
    m.load_config("configs/llama3_config.json");

    WeightMap w;

    // load weight mapping into a map for O(1) access
    w.load("configs/model_index.json");

    // pass memory location of weights into model
    m.load_weights(w, "scripts/llama.bin");


    return 0;
}
