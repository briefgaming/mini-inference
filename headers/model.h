#pragma once

#include "tensor.h"
#include "weights.h"

struct RMSNorm {
    Tensor<1> weight;

    void apply(Tensor<1> &out, Tensor<1> &in);
};

// Grouped Query Attention
struct GQAttention {
    Tensor<2> wq;
    Tensor<2> wk;
    Tensor<2> wv;
    Tensor<2> wo;

    // Apply with RoPE
    // https://arxiv.org/abs/2104.09864
    void apply(Tensor<1> &out, Tensor<1> &x, int pos, const Tensor<3> &kvcache);
};

struct SwiGLUBlock {
    Tensor<2> w1_gate; // Gate projection to hidden dim (4x)
    Tensor<2> w1_up; // Up projection to hidden dim (4x)
    Tensor<2> w1_down; // Down projection to orignal model dim (2048 for llama3-1.2B)

    void apply(Tensor<1> &out, Tensor<1> &in);
};

struct TransformerBlock {
    RMSNorm pre_attn_norm;
    GQAttention attn;
    RMSNorm post_ffn_norm;
    SwiGLUBlock mlp;

    void apply(Tensor<1> &x, int pos, const Tensor<3> &kvcache);
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

    Tensor<2> token_embed;
    TransformerBlock* layers; // 12 layers for llama3
    RMSNorm final_norm; // just befor output projection
    Tensor<2> output_head;

    Model();
    ~Model();

    void load_weights(WeightMap& w, const std::string& weight_path);
    bool load_config(const std::string& config_path);
};