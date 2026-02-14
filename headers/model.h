#pragma once

#include "bf16.h"
#include "tensor.h"
#include "weights.h"

struct RMSNorm {
    Tensor<1, bf16> weight;

    void rmsnorm(Tensor<1> &out, Tensor<1> &in);
};

// Grouped Query Attention
struct GQAttention {
    Tensor<2, bf16> wq;
    Tensor<2, bf16> wk;
    Tensor<2, bf16> wv;
    Tensor<2, bf16> wo;

    // Apply with RoPE
    // https://arxiv.org/abs/2104.09864
    void rope();
    void gqattention(Tensor<1> &out, Tensor<1> &x, int pos, const Tensor<3> &kvcache);
};

struct SwiGLUBlock {
    Tensor<2, bf16> w1_gate; // Gate projection to hidden dim (4x)
    Tensor<2, bf16> w1_up; // Up projection to hidden dim (4x)
    Tensor<2, bf16> w1_down; // Down projection to original model dim

    void swiglu(Tensor<1> &out, Tensor<1> &in);
};

struct TransformerBlock {
    RMSNorm pre_attn_norm;
    GQAttention attn;
    RMSNorm post_ffn_norm;
    SwiGLUBlock mlp;

    void apply_transformer(Tensor<1> &x, int pos, const Tensor<3> &kvcache);
};

struct Model {
    int n_layers;
    int dim; // embedding dimension
    int kv_dim;
    int hidden_dim;
    int vocab_size;
    int n_heads;

    // points to weights address loaded from file
    char* mmap_data;
    size_t mmap_size;

    Tensor<2, bf16> token_embed;
    TransformerBlock* layers; // 16 layers for llama3
    RMSNorm final_norm; // just befor output projection
    Tensor<2, bf16> output_head;

    Model();
    ~Model();

    void load_weights(WeightMap& w, const std::string& weight_path);
    bool load_config(const std::string& config_path);

    void forward(int token_id, int input_pos, const Tensor<3> &kvcache, const Tensor<1> &emb_out);
};
