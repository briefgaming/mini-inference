#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "bf16.h"
#include "tensor.h"
#include "weights.h"

struct RMSNorm {
    Tensor<1, bf16> weight;

    void rmsnorm(Tensor<1> &out, const Tensor<1> &in, float eps);
};

// Grouped Query Attention
// GQA uses half the dim of q for kv matrices
struct GQAttention {
    Tensor<2, bf16> wq; // [dim, dim]
    Tensor<2, bf16> wk; // [dim, kv_dim]
    Tensor<2, bf16> wv; // [dim, kv_dim]
    Tensor<2, bf16> wo; // [dim, dim]

    // Apply with RoPE
    // https://arxiv.org/abs/2104.09864
    static void rope(float *head_vec, int head_dim, int pos, float rope_theta);
    void gqattention(
        Tensor<1> &out,
        const Tensor<1> &x,
        int pos,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        Tensor<3> &kcache,
        Tensor<3> &vcache
    );
};

struct SwiGLUBlock {
    Tensor<2, bf16> w1_gate; // Gate projection to hidden dim (4x)
    Tensor<2, bf16> w1_up; // Up projection to hidden dim (4x)
    Tensor<2, bf16> w1_down; // Down projection to original model dim

    void swiglu(Tensor<1> &out, const Tensor<1> &in);
};

struct TransformerBlock {
    RMSNorm pre_attn_norm;
    GQAttention attn;
    RMSNorm post_ffn_norm;
    SwiGLUBlock mlp;

    void apply_transformer(
        Tensor<1> &x,
        int pos,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        float rms_eps,
        Tensor<3> &kcache,
        Tensor<3> &vcache
    );
};

struct Model {
    int n_layers;
    int dim; // embedding dimension
    int kv_dim;
    int hidden_dim; // larger than dim used for ffn
    int vocab_size;
    int n_heads;
    int n_kv_heads; // Number of key/value heads (for GQA/MQA can be < n_heads)
    int head_dim; // Per head dim (dim / n_heads)
    float rope_theta;
    float rms_eps;

    // points to weights address loaded from file
    char* mmap_data;
    size_t mmap_size;

    Tensor<2, bf16> token_embed;
    TransformerBlock* layers; // 16 layers for llama3
    RMSNorm final_norm; // just before output projection
    Tensor<2, bf16> output_head;

    Model();
    ~Model();

    void load_weights(WeightMap& w, const std::string& weight_path);
    bool load_config(const std::string& config_path);

    void forward(
        int token_id,
        int input_pos,
        std::vector<Tensor<3>> &kcache,
        std::vector<Tensor<3>> &vcache,
        Tensor<1> &logits_out // vocab size
    );
};
