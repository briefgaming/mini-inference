#include "headers/bpe.h"

#include <fstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstring>

BPEDecode::BPEDecode() = default;
BPEDecode::~BPEDecode() = default;

// Store word segments in vocab
bool BPEDecode::load(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file) {
        return false;
    }

    vocab.clear();

    while (true) {
        unsigned char len;
        // Read the length of the next token
        if (!file.read((char*)&len, 1)) {
            break;
        }

        // Allocate buffer memory of length len
        std::string token(len, '\0');

        // Read data from file into buffer "token"
        if (!file.read(&token[0], static_cast<std::streamsize>(len))) {
            break;
        }

        vocab.push_back(token);
    }

    return true;
}

// Turn a sequence of tokens back into a string
int BPEDecode::decode(const int* tokens, int ntokens, char* outputbuf, int outputbuf_s) {
    if (outputbuf_s <= 0 || outputbuf == nullptr || tokens == nullptr) {
        return 0;
    }
    int j = 0;
    for (int i = 0; i < ntokens; i++) {
        if (tokens[i] == -1) {
            if (j + 3 >= outputbuf_s) {
                break;
            }
            memcpy(&outputbuf[j], "(?)", 3);
            j += 3;
            continue;
        }

        if (tokens[i] < 0 || static_cast<size_t>(tokens[i]) >= vocab.size()) {
            break;
        }

        const std::string& s = vocab[tokens[i]];
        int len = static_cast<int>(s.size());

        if (j + len >= outputbuf_s) {
            break;
        }
        memcpy(&outputbuf[j], s.c_str(), len);
        j += len;
    }

    outputbuf[j] = 0;
    return j;
}

struct BPETrieNode {
    int token_length = -1;
    int token_id = -1;
    std::unordered_map<char, BPETrieNode*> children;

    ~BPETrieNode() {
        for (auto it = children.begin(); it != children.end(); it++) {
            delete it->second;
        }
    }
};

BPEEncode::BPEEncode() {
    root = new BPETrieNode();
}

BPEEncode::~BPEEncode() {
    delete root;
    root = nullptr;
}

// Build trie from a vocabulary list
bool BPEEncode::load(const std::vector<std::string>& vocab) {
    delete root;
    root = new BPETrieNode();
    for (size_t i = 0; i < vocab.size(); i++) {
        auto token = vocab[i];
        BPETrieNode* node = root;
        // Iterate over token
        for (char c : token) {
            if (node->children.count(c) == 0) {
                node->children[c] = new BPETrieNode();
            }
            node = node->children[c];
        }
        node->token_length = static_cast<int>(token.size());
        node->token_id = i;
    }
    return true;
}

/** Convert string to tokens where the return char points to the first byte of the token sequence in memory
Greedy search Prefix tree

outputbuf is max buffer size of tokens
while loop stops when input_seq is empty or outputbuf is full
*/
const char* BPEEncode::encode(const char* input_seq, int* outputbuf, int outputbuf_s, int* num_tokens) {
    if (num_tokens != nullptr) {
        *num_tokens = 0;
    }
    if (input_seq == nullptr || outputbuf == nullptr || num_tokens == nullptr || outputbuf_s <= 0) { // validate inputs
        return input_seq;
    }

    while (*input_seq && *num_tokens < outputbuf_s) {
        BPETrieNode* node = root;
        int last_token_length = -1;
        int last_token_id = -1;

        for (size_t i = 0; input_seq[i]; i++)   {
            char key = input_seq[i];
            if (node->children.count(key) == 0) {
                break;
            }
            node = node->children[key];
            if (node->token_length != -1) {
                last_token_length = node->token_length;
                last_token_id = node->token_id;
            }
        }
        if (last_token_length == -1) {
            return input_seq;
        } else {
            outputbuf[*num_tokens] = last_token_id;
            input_seq += last_token_length;
            (*num_tokens)++;
        }
    }
    return input_seq;
}
