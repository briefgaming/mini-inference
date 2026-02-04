#pragma once

#include <string>
#include <vector>


struct BPETrieNode;


class BPEDecode {
    public:
        BPEDecode();
        ~BPEDecode();

        std::vector<std::string> vocab;
        
        bool load(const std::string& vocab_path);
        int decode(const int* tokens, int ntokens, char* outbuf, int outbuf_size);
};

class BPEEncode {
    public:
        BPEEncode();
        ~BPEEncode();
        BPEEncode(const BPEEncode&) = delete;
        BPEEncode& operator=(const BPEEncode&) = delete;

        bool load(const std::vector<std::string>& vocab);

        const char* encode(const char* input, int* outbuf, int buf_s, int* ntokens);
    
    private:
        BPETrieNode* root; // root can only be modified by its class
};
