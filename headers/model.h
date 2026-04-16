#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "float.h"
#include "tensor.h"
#include "weights.h"

enum class MatrixDType {
    F32,
    I8,
};

struct LinearWeight {
    MatrixDType dtype = MatrixDType::F32;
    Tensor<2, float> f32;
    Tensor<2, int8_t> i8;
    Tensor<1, float> scales;

    int dim(int axis) const {
        return dtype == MatrixDType::I8 ? i8.dim(axis) : f32.dim(axis);
    }
};

struct EmbeddingWeight {
    MatrixDType dtype = MatrixDType::F32;
    Tensor<2, float> f32;
    Tensor<2, int8_t> i8;
    Tensor<1, float> scales;

    int dim(int axis) const {
        return dtype == MatrixDType::I8 ? i8.dim(axis) : f32.dim(axis);
    }

    const void* data() const {
        return dtype == MatrixDType::I8
            ? static_cast<const void*>(i8.data())
            : static_cast<const void*>(f32.data());
    }
};

struct RMSNorm {
    Tensor<1, float> weight;

    void rmsnorm(Tensor<1> &out, const Tensor<1> &in, float eps);
};

// Grouped Query Attention
// GQA uses half the dim of q for kv matrices
struct GQAttention {
    LinearWeight wq; // [dim, dim]
    LinearWeight wk; // [dim, kv_dim]
    LinearWeight wv; // [dim, kv_dim]
    LinearWeight wo; // [dim, dim]

    // Apply with RoPE
    // https://arxiv.org/abs/2104.09864
    static void rope(float *head_vec, int head_dim, int pos, float rope_theta);
    void gqattention(
        Tensor<1> &out,
        const Tensor<1> &x,
        int layer_idx,
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
    LinearWeight w1_gate; // Gate projection to hidden dim (4x)
    LinearWeight w1_up; // Up projection to hidden dim (4x)
    LinearWeight w1_down; // Down projection to original model dim

    void swiglu(Tensor<1> &out, const Tensor<1> &in, int layer_idx, int token_pos);
};

struct TransformerBlock {
    RMSNorm pre_attn_norm;
    GQAttention attn;
    RMSNorm post_ffn_norm;
    SwiGLUBlock mlp;

    void apply_transformer(
        Tensor<1> &x,
        int layer_idx,
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
    int n_layers = 0;
    int dim = 0; // embedding dimension
    int kv_dim = 0;
    int hidden_dim = 0; // larger than dim used for ffn
    int vocab_size = 0;
    int n_heads = 0;
    int n_kv_heads = 0; // Number of key/value heads (for GQA/MQA can be < n_heads)
    int head_dim = 0; // Per head dim (dim / n_heads)
    float rope_theta = 0.0f;
    float rms_eps = 0.0f;

    // points to weights address loaded from file
    char* mmap_data = nullptr;
    size_t mmap_size = 0;

    EmbeddingWeight token_embed; // embed is a 2d tensor
    TransformerBlock* layers = nullptr; // 16 layers for llama3
    RMSNorm final_norm; // just before output projection
    EmbeddingWeight output_head;

    Model();
    ~Model();

    void load_weights(WeightMap &w, const std::string &weight_path);
    bool load_config(const std::string &config_path);

    void forward(
        int token_id,
        int input_pos,
        std::vector<Tensor<3>> &kcache,
        std::vector<Tensor<3>> &vcache,
        Tensor<1> &logits_out // vocab size
    );
};
