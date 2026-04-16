#include "headers/model.h"
#include "headers/json.hpp"
#include "headers/weights.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

Model::Model() : layers(nullptr), mmap_data(nullptr), mmap_size(0) {}

Model::~Model() {
    if (layers) {
        delete[] layers;
        layers = nullptr;
    }
    if (mmap_data && mmap_size > 0) {
        munmap(mmap_data, mmap_size);
        mmap_data = nullptr;
        mmap_size = 0;
    }
}

static inline bool is_float32_meta(const TensorMeta& meta) {
    return meta.dtype == "F32";
}

static inline bool is_int8_meta(const TensorMeta& meta) {
    return meta.dtype == "I8";
}

static bool bind_linear_weight(
    LinearWeight& out,
    const WeightMap& weights,
    char* mmap_ptr,
    const std::string& key,
    int d0,
    int d1
) {
    const TensorMeta* meta = weights.get_meta(key);
    if (meta == nullptr) {
        std::fprintf(stderr, "Missing tensor metadata for %s\n", key.c_str());
        return false;
    }

    if (is_float32_meta(*meta)) {
        out.dtype = MatrixDType::F32;
        out.f32 = Tensor<2, float>(weights.get_ptr<float>(key, mmap_ptr), d0, d1);
        out.i8 = Tensor<2, int8_t>();
        out.scales = Tensor<1, float>();
        return true;
    }

    if (is_int8_meta(*meta)) {
        if (meta->scale_name.empty()) {
            std::fprintf(stderr, "Quantized tensor %s is missing scale metadata\n", key.c_str());
            return false;
        }
        const TensorMeta* scale_meta = weights.get_meta(meta->scale_name);
        if (scale_meta == nullptr || !is_float32_meta(*scale_meta)) {
            std::fprintf(stderr, "Quantized tensor %s has invalid scale tensor %s\n", key.c_str(), meta->scale_name.c_str());
            return false;
        }
        out.dtype = MatrixDType::I8;
        out.i8 = Tensor<2, int8_t>(weights.get_ptr<int8_t>(key, mmap_ptr), d0, d1);
        out.f32 = Tensor<2, float>();
        out.scales = Tensor<1, float>(weights.get_ptr<float>(meta->scale_name, mmap_ptr), d1);
        return true;
    }

    std::fprintf(stderr, "Unsupported tensor dtype %s for %s\n", meta->dtype.c_str(), key.c_str());
    return false;
}

static bool bind_embedding_weight(
    EmbeddingWeight& out,
    const WeightMap& weights,
    char* mmap_ptr,
    const std::string& key,
    int d0,
    int d1
) {
    const TensorMeta* meta = weights.get_meta(key);
    if (meta == nullptr) {
        std::fprintf(stderr, "Missing tensor metadata for %s\n", key.c_str());
        return false;
    }

    if (is_float32_meta(*meta)) {
        out.dtype = MatrixDType::F32;
        out.f32 = Tensor<2, float>(weights.get_ptr<float>(key, mmap_ptr), d0, d1);
        out.i8 = Tensor<2, int8_t>();
        out.scales = Tensor<1, float>();
        return true;
    }

    if (is_int8_meta(*meta)) {
        if (meta->scale_name.empty()) {
            std::fprintf(stderr, "Quantized embedding %s is missing scale metadata\n", key.c_str());
            return false;
        }
        const TensorMeta* scale_meta = weights.get_meta(meta->scale_name);
        if (scale_meta == nullptr || !is_float32_meta(*scale_meta)) {
            std::fprintf(stderr, "Quantized embedding %s has invalid scale tensor %s\n", key.c_str(), meta->scale_name.c_str());
            return false;
        }
        out.dtype = MatrixDType::I8;
        out.i8 = Tensor<2, int8_t>(weights.get_ptr<int8_t>(key, mmap_ptr), d0, d1);
        out.f32 = Tensor<2, float>();
        out.scales = Tensor<1, float>(weights.get_ptr<float>(meta->scale_name, mmap_ptr), d0);
        return true;
    }

    std::fprintf(stderr, "Unsupported tensor dtype %s for %s\n", meta->dtype.c_str(), key.c_str());
    return false;
}

static inline void matmul_f32(Tensor<1>& out, const Tensor<1>& x, const Tensor<2, float>& w_f32) {
    const int in_dim = x.dim(0);
    const int out_dim = out.dim(0);
    assert(w_f32.dim(0) == in_dim);
    assert(w_f32.dim(1) == out_dim);

    for (int j = 0; j < out_dim; ++j) {
        out[j] = 0.0f;
    }

    for (int i = 0; i < in_dim; ++i) {
        const float xi = x[i];
        const float* row = w_f32.data() + i * w_f32.stride(0);
        for (int j = 0; j < out_dim; ++j) {
            out[j] += xi * row[j];
        }
    }
}

static inline void matmul_i8(
    Tensor<1>& out,
    const Tensor<1>& x,
    const Tensor<2, int8_t>& w_i8,
    const Tensor<1, float>& scales
) {
    const int in_dim = x.dim(0);
    const int out_dim = out.dim(0);
    assert(w_i8.dim(0) == in_dim);
    assert(w_i8.dim(1) == out_dim);
    assert(scales.dim(0) == out_dim);

    for (int j = 0; j < out_dim; ++j) {
        out[j] = 0.0f;
    }

    for (int i = 0; i < in_dim; ++i) {
        const float xi = x[i];
        const int8_t* row = w_i8.data() + i * w_i8.stride(0);
        for (int j = 0; j < out_dim; ++j) {
            out[j] += xi * static_cast<float>(row[j]);
        }
    }

    for (int j = 0; j < out_dim; ++j) {
        out[j] *= scales[j];
    }
}

static inline void matmul(
    Tensor<1> &out,
    const Tensor<1> &x,
    const LinearWeight& weight,
    const char* op_name,
    int layer_idx,
    int token_pos
) {
    // const bool timing_enabled = matmul_timing_enabled();
    // const auto start = timing_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (weight.dtype == MatrixDType::I8) {
        matmul_i8(out, x, weight.i8, weight.scales);
    } else {
        matmul_f32(out, x, weight.f32);
    }

    // if (timing_enabled) {
    //     const auto end = std::chrono::steady_clock::now();
    //     const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //     log_kernel_timing(op_name, layer_idx, token_pos, x.dim(0), out.dim(0), elapsed_us);
    // }
}

static inline void embedding_lookup(Tensor<1>& out, int token_id, const EmbeddingWeight& embedding) {
    const int dim = out.dim(0);
    if (embedding.dtype == MatrixDType::I8) {
        assert(embedding.i8.dim(0) > token_id);
        assert(embedding.i8.dim(1) == dim);
        assert(embedding.scales.dim(0) > token_id);
        const float row_scale = embedding.scales[token_id];
        const int8_t* row = embedding.i8.data() + token_id * embedding.i8.stride(0);
        for (int i = 0; i < dim; ++i) {
            out[i] = row_scale * static_cast<float>(row[i]);
        }
        return;
    }

    assert(embedding.f32.dim(0) > token_id);
    assert(embedding.f32.dim(1) == dim);
    const float* row = embedding.f32.data() + token_id * embedding.f32.stride(0);
    for (int i = 0; i < dim; ++i) {
        out[i] = row[i];
    }
}

static inline void output_head_projection(
    Tensor<1>& logits_out,
    const Tensor<1>& norm_out,
    const EmbeddingWeight& output_head
) {
    const int vocab_size = logits_out.dim(0);
    const int dim = norm_out.dim(0);
    if (output_head.dtype == MatrixDType::I8) {
        assert(output_head.i8.dim(0) == vocab_size);
        assert(output_head.i8.dim(1) == dim);
        assert(output_head.scales.dim(0) == vocab_size);
        for (int j = 0; j < vocab_size; ++j) {
            float s = 0.0f;
            const float row_scale = output_head.scales[j];
            const int8_t* row = output_head.i8.data() + j * output_head.i8.stride(0);
            for (int i = 0; i < dim; ++i) {
                s += norm_out[i] * static_cast<float>(row[i]);
            }
            logits_out[j] = row_scale * s;
        }
        return;
    }

    assert(output_head.f32.dim(0) == vocab_size);
    assert(output_head.f32.dim(1) == dim);
    for (int j = 0; j < vocab_size; ++j) {
        float s = 0.0f;
        const float* row = output_head.f32.data() + j * output_head.f32.stride(0);
        for (int i = 0; i < dim; ++i) {
            s += norm_out[i] * row[i];
        }
        logits_out[j] = s;
    }
}

static inline float silu(float z) {
    return z / (1.0f + std::exp(-z));
}

static inline void residual_connection(Tensor<1> &out, Tensor<1> &in) {
    for (int i = 0; i < out.dim(0); ++i) {
        out[i] += in[i];
    }
}

void Model::load_weights(WeightMap& w, const std::string& weight_path) {
    if (layers) {
        delete[] layers;
        layers = nullptr;
    }
    if (mmap_data && mmap_size > 0) {
        munmap(mmap_data, mmap_size);
        mmap_data = nullptr;
        mmap_size = 0;
    }

    const char* fname = weight_path.c_str();
    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
        perror(fname);
        return;
    }

    struct stat sb {};
    if (fstat(fd, &sb) != 0) {
        perror("fstat");
        close(fd);
        return;
    }

    void* mapped = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        return;
    }

    mmap_data = static_cast<char*>(mapped);
    mmap_size = static_cast<size_t>(sb.st_size);

    std::cout << "Allocating " << n_layers << " transformer blocks..." << std::endl;
    layers = new TransformerBlock[n_layers];

    auto fail_load = [this]() {
        if (layers) {
            delete[] layers;
            layers = nullptr;
        }
        if (mmap_data && mmap_size > 0) {
            munmap(mmap_data, mmap_size);
            mmap_data = nullptr;
            mmap_size = 0;
        }
    };

    if (!bind_embedding_weight(token_embed, w, mmap_data, "model.embed_tokens.weight", vocab_size, dim)) {
        fail_load();
        return;
    }
    final_norm.weight = Tensor<1>(
        w.get_ptr<float>("model.norm.weight", mmap_data),
        dim
    );

    // Llama 3.2 1B ties output head and token embeddings.
    output_head = token_embed;

    for (int i = 0; i < n_layers; ++i) {
        const std::string prefix = "model.layers." + std::to_string(i) + ".";

        if (!bind_linear_weight(layers[i].attn.wq, w, mmap_data, prefix + "self_attn.q_proj.weight", dim, dim) ||
            !bind_linear_weight(layers[i].attn.wk, w, mmap_data, prefix + "self_attn.k_proj.weight", dim, kv_dim) ||
            !bind_linear_weight(layers[i].attn.wv, w, mmap_data, prefix + "self_attn.v_proj.weight", dim, kv_dim) ||
            !bind_linear_weight(layers[i].attn.wo, w, mmap_data, prefix + "self_attn.o_proj.weight", dim, dim) ||
            !bind_linear_weight(layers[i].mlp.w1_gate, w, mmap_data, prefix + "mlp.gate_proj.weight", dim, hidden_dim) ||
            !bind_linear_weight(layers[i].mlp.w1_up, w, mmap_data, prefix + "mlp.up_proj.weight", dim, hidden_dim) ||
            !bind_linear_weight(layers[i].mlp.w1_down, w, mmap_data, prefix + "mlp.down_proj.weight", hidden_dim, dim)) {
            fail_load();
            return;
        }

        layers[i].pre_attn_norm.weight = Tensor<1, float>(
            w.get_ptr<float>(prefix + "input_layernorm.weight", mmap_data),
            dim
        );
        layers[i].post_ffn_norm.weight = Tensor<1, float>(
            w.get_ptr<float>(prefix + "post_attention_layernorm.weight", mmap_data),
            dim
        );
    }
}

bool Model::load_config(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        return false;
    }

    nlohmann::json config;
    f >> config;

    n_layers = config["num_hidden_layers"];
    vocab_size = config["vocab_size"];
    dim = config["hidden_size"];
    hidden_dim = config["intermediate_size"];
    n_heads = config["num_attention_heads"];
    n_kv_heads = config["num_key_value_heads"];
    head_dim = config.contains("head_dim")
        ? config["head_dim"].get<int>()
        : (dim / n_heads);
    kv_dim = n_kv_heads * head_dim;
    rms_eps = config.contains("rms_norm_eps")
        ? config["rms_norm_eps"].get<float>()
        : 1e-5f;
    rope_theta = config.contains("rope_theta")
        ? config["rope_theta"].get<float>()
        : 10000.0f;

    return true;
}

void RMSNorm::rmsnorm(Tensor<1>& out, const Tensor<1>& in, float eps) {
    assert(out.dim(0) == in.dim(0));
    assert(weight.dim(0) == in.dim(0));

    const int d = in.dim(0);
    float sum_squares = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum_squares += in[i] * in[i];
    }

    const float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(d) + eps);
    for (int i = 0; i < d; ++i) {
        out[i] = in[i] * inv_rms * weight[i];
    }
}

void GQAttention::rope(float* head_vec, int head_dim, int pos, float rope_theta) {
    for (int m = 0; m < head_dim / 2; ++m) {
        const float exp_term = static_cast<float>(2 * m) / static_cast<float>(head_dim);
        const float inv_freq = std::pow(rope_theta, -exp_term);
        const float theta = static_cast<float>(pos) * inv_freq;
        const float c = std::cos(theta);
        const float s = std::sin(theta);

        const int i0 = 2 * m;
        const int i1 = i0 + 1;
        const float a = head_vec[i0];
        const float b = head_vec[i1];
        head_vec[i0] = a * c - b * s;
        head_vec[i1] = a * s + b * c;
    }
}

void GQAttention::gqattention(
    Tensor<1>& out,
    const Tensor<1>& x,
    int layer_idx,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_theta,
    Tensor<3>& kcache,
    Tensor<3>& vcache
) {
    const int dim = x.dim(0);
    const int kv_dim_local = n_kv_heads * head_dim;
    const int q_kv_head_ratio = n_heads / n_kv_heads; // 4 q_heads for every kv_head
    assert(out.dim(0) == dim);

    Tensor<1> q_flat(dim);
    Tensor<1> k_flat(kv_dim_local);
    Tensor<1> v_flat(kv_dim_local);
    matmul(q_flat, x, wq, "attn.wq", layer_idx, pos);
    matmul(k_flat, x, wk, "attn.wk", layer_idx, pos);
    matmul(v_flat, x, wv, "attn.wv", layer_idx, pos);

    // Apply RoPE on every query and key in each head.
    for (int h = 0; h < n_heads; ++h) {
        rope(q_flat.data() + h * head_dim, head_dim, pos, rope_theta);
    }
    for (int hk = 0; hk < n_kv_heads; ++hk) {
        rope(k_flat.data() + hk * head_dim, head_dim, pos, rope_theta);
    }

    // Write new K/V into cache for this token position.
    for (int hk = 0; hk < n_kv_heads; ++hk) {
        for (int d = 0; d < head_dim; ++d) {
            kcache(pos, hk, d) = k_flat[hk * head_dim + d];
            vcache(pos, hk, d) = v_flat[hk * head_dim + d];
        }
    }

    Tensor<1> ctx_flat(dim);
    std::vector<float> scores(pos + 1, 0.0f);
    const float scale = 1.0f / std::sqrt(head_dim);

    for (int h = 0; h < n_heads; ++h) {
        const int hk = h / q_kv_head_ratio;
        const float* qh = q_flat.data() + h * head_dim;

        float max_score = -std::numeric_limits<float>::infinity();
        // dot product of current q with the kv of all computed tokens in the same head
        for (int t = 0; t <= pos; ++t) {
            float s = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                s += qh[d] * kcache(t, hk, d);
            }
            s *= scale;
            scores[t] = s;
            if (s > max_score) {
                max_score = s;
            }
        }

        float denom = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            float e = std::exp(scores[t] - max_score);
            scores[t] = e;
            denom += e;
        }

        for (int d = 0; d < head_dim; ++d) {
            float c = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                c += (scores[t] / denom) * vcache(t, hk, d);
            }
            ctx_flat[h * head_dim + d] = c;
        }
    }

    matmul(out, ctx_flat, wo, "attn.wo", layer_idx, pos);
}

void SwiGLUBlock::swiglu(Tensor<1>& out, const Tensor<1>& in, int layer_idx, int token_pos) {
    Tensor<1> g(w1_gate.dim(1));
    Tensor<1> u(w1_up.dim(1));
    Tensor<1> h(w1_down.dim(0));

    matmul(g, in, w1_gate, "mlp.gate", layer_idx, token_pos);
    matmul(u, in, w1_up, "mlp.up", layer_idx, token_pos);
    for (int i = 0; i < h.dim(0); ++i) {
        h[i] = silu(g[i]) * u[i];
    }

    matmul(out, h, w1_down, "mlp.down", layer_idx, token_pos);
}

void TransformerBlock::apply_transformer(
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
) {
    Tensor<1> x_norm(x.dim(0));
    Tensor<1> attn_out(x.dim(0));
    Tensor<1> ffn_in(x.dim(0));
    Tensor<1> ffn_out(x.dim(0));

    pre_attn_norm.rmsnorm(x_norm, x, rms_eps);
    attn.gqattention(attn_out, x_norm, layer_idx, pos, n_heads, n_kv_heads, head_dim, rope_theta, kcache, vcache);
    residual_connection(x, attn_out);

    post_ffn_norm.rmsnorm(ffn_in, x, rms_eps);
    mlp.swiglu(ffn_out, ffn_in, layer_idx, pos);
    residual_connection(x, ffn_out);
}

void Model::forward(
    int token_id,
    int input_pos,
    std::vector<Tensor<3>> &kcache,
    std::vector<Tensor<3>> &vcache,
    Tensor<1> &logits_out
) {
    assert(token_id >= 0 && token_id < vocab_size);
    assert(kcache.size() == n_layers);
    assert(vcache.size() == n_layers);

    /***
    What's happening here is that we are plucking the token embedding (token projection) from token_embed.
    A row in token_embed is represented as a 1d tensor Tensor<1> x(dim).
    */
    Tensor<1> x(dim); 
    embedding_lookup(x, token_id, token_embed);

    for (int l = 0; l < n_layers; ++l) {
        layers[l].apply_transformer(
            x,
            l,
            input_pos,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
            rms_eps,
            kcache[l],
            vcache[l]
        );
    }

    Tensor<1> norm_out(dim);
    final_norm.rmsnorm(norm_out, x, rms_eps);

    // logits_out[j] = dot(norm_out, output_head[j]) with tied output embeddings.
    // const bool timing_enabled = matmul_timing_enabled();
    // const auto output_start = timing_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    output_head_projection(logits_out, norm_out, output_head);
    // if (timing_enabled) {
    //     const auto output_end = std::chrono::steady_clock::now();
    //     const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(output_end - output_start).count();
    //     log_kernel_timing("output_head", -1, input_pos, dim, vocab_size, elapsed_us, false);
    // }
}
