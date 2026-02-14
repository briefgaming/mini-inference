#include "headers/model.h"
#include "headers/json.hpp"
#include "headers/weights.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
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

static inline void matvec(Tensor<1>& out, const Tensor<1>& x, const Tensor<2>& w_f32) {
    const int in_dim = x.dim(0);
    const int out_dim = out.dim(0);
    assert(w_f32.dim(0) == in_dim);
    assert(w_f32.dim(1) == out_dim);

    for (int j = 0; j < out_dim; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            acc += x[i] * w_f32(i, j);
        }
        out[j] = acc;
    }
}

static inline float silu(float z) {
    return z / (1.0f + std::exp(-z));
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

    token_embed = Tensor<2, bf16>(
        w.get_ptr<bf16>("model.embed_tokens.weight", mmap_data),
        vocab_size,
        dim
    );
    final_norm.weight = Tensor<1, bf16>(
        w.get_ptr<bf16>("model.norm.weight", mmap_data),
        dim
    );

    // Llama 3.2 1B ties output head and token embeddings.
    output_head = token_embed;

    for (int i = 0; i < n_layers; ++i) {
        const std::string prefix = "model.layers." + std::to_string(i) + ".";

        layers[i].attn.wq = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "self_attn.q_proj.weight", mmap_data),
            dim,
            dim
        );
        layers[i].attn.wk = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "self_attn.k_proj.weight", mmap_data),
            dim,
            kv_dim
        );
        layers[i].attn.wv = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "self_attn.v_proj.weight", mmap_data),
            dim,
            kv_dim
        );
        layers[i].attn.wo = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "self_attn.o_proj.weight", mmap_data),
            dim,
            dim
        );

        layers[i].mlp.w1_gate = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "mlp.gate_proj.weight", mmap_data),
            dim,
            hidden_dim
        );
        layers[i].mlp.w1_up = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "mlp.up_proj.weight", mmap_data),
            dim,
            hidden_dim
        );
        layers[i].mlp.w1_down = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "mlp.down_proj.weight", mmap_data),
            hidden_dim,
            dim
        );

        layers[i].pre_attn_norm.weight = Tensor<1, bf16>(
            w.get_ptr<bf16>(prefix + "input_layernorm.weight", mmap_data),
            dim
        );
        layers[i].post_ffn_norm.weight = Tensor<1, bf16>(
            w.get_ptr<bf16>(prefix + "post_attention_layernorm.weight", mmap_data),
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
    Tensor<1> w_f32 = weight.dequantize_to_f32();
    assert(w_f32.dim(0) == in.dim(0));

    const int d = in.dim(0);
    float sum_squares = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum_squares += in[i] * in[i];
    }

    const float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(d) + eps);
    for (int i = 0; i < d; ++i) {
        out[i] = in[i] * inv_rms * w_f32[i];
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
    const int n_rep = n_heads / n_kv_heads;
    assert(out.dim(0) == dim);

    Tensor<2> wq_f32 = wq.dequantize_to_f32(); // [dim, dim]
    Tensor<2> wk_f32 = wk.dequantize_to_f32(); // [dim, kv_dim]
    Tensor<2> wv_f32 = wv.dequantize_to_f32(); // [dim, kv_dim]
    Tensor<2> wo_f32 = wo.dequantize_to_f32(); // [dim, dim]

    Tensor<1> q_flat(dim);
    Tensor<1> k_flat(kv_dim_local);
    Tensor<1> v_flat(kv_dim_local);
    matvec(q_flat, x, wq_f32);
    matvec(k_flat, x, wk_f32);
    matvec(v_flat, x, wv_f32);

    // Apply RoPE on every query head and KV head.
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
    std::vector<float> scores(static_cast<size_t>(pos + 1), 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int h = 0; h < n_heads; ++h) {
        const int hk = h / n_rep;
        const float* qh = q_flat.data() + h * head_dim;

        float max_score = -std::numeric_limits<float>::infinity();
        for (int t = 0; t <= pos; ++t) {
            float s = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                s += qh[d] * kcache(t, hk, d);
            }
            s *= scale;
            scores[static_cast<size_t>(t)] = s;
            if (s > max_score) {
                max_score = s;
            }
        }

        float denom = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            float e = std::exp(scores[static_cast<size_t>(t)] - max_score);
            scores[static_cast<size_t>(t)] = e;
            denom += e;
        }

        for (int d = 0; d < head_dim; ++d) {
            float c = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                c += (scores[static_cast<size_t>(t)] / denom) * vcache(t, hk, d);
            }
            ctx_flat[h * head_dim + d] = c;
        }
    }

    matvec(out, ctx_flat, wo_f32);
}

void SwiGLUBlock::swiglu(Tensor<1>& out, const Tensor<1>& in) {
    Tensor<2> gate_f32 = w1_gate.dequantize_to_f32(); // [dim, hidden_dim]
    Tensor<2> up_f32 = w1_up.dequantize_to_f32();     // [dim, hidden_dim]
    Tensor<2> down_f32 = w1_down.dequantize_to_f32(); // [hidden_dim, dim]

    Tensor<1> g(gate_f32.dim(1));
    Tensor<1> u(up_f32.dim(1));
    Tensor<1> h(down_f32.dim(0));

    matvec(g, in, gate_f32);
    matvec(u, in, up_f32);
    for (int i = 0; i < h.dim(0); ++i) {
        h[i] = silu(g[i]) * u[i];
    }

    matvec(out, h, down_f32);
}

void TransformerBlock::apply_transformer(
    Tensor<1>& x,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_theta,
    float rms_eps,
    Tensor<3>& kcache,
    Tensor<3>& vcache
) {
    Tensor<1> x_norm(x.dim(0));
    Tensor<1> attn_out(x.dim(0));
    Tensor<1> ffn_in(x.dim(0));
    Tensor<1> ffn_out(x.dim(0));

    pre_attn_norm.rmsnorm(x_norm, x, rms_eps);
    attn.gqattention(attn_out, x_norm, pos, n_heads, n_kv_heads, head_dim, rope_theta, kcache, vcache);
    for (int i = 0; i < x.dim(0); ++i) {
        x[i] += attn_out[i];
    }

    post_ffn_norm.rmsnorm(ffn_in, x, rms_eps);
    mlp.swiglu(ffn_out, ffn_in);
    for (int i = 0; i < x.dim(0); ++i) {
        x[i] += ffn_out[i];
    }
}

void Model::forward(
    int token_id,
    int input_pos,
    std::vector<Tensor<3>>& kcache,
    std::vector<Tensor<3>>& vcache,
    Tensor<1>& logits_out
) {
    assert(token_id >= 0 && token_id < vocab_size);
    assert(static_cast<int>(kcache.size()) == n_layers);
    assert(static_cast<int>(vcache.size()) == n_layers);

    Tensor<1> x(dim);
    for (int i = 0; i < dim; ++i) {
        x[i] = token_embed(token_id, i).to_float();
    }

    for (int l = 0; l < n_layers; ++l) {
        layers[l].apply_transformer(
            x,
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
    for (int j = 0; j < vocab_size; ++j) {
        float s = 0.0f;
        for (int i = 0; i < dim; ++i) {
            s += norm_out[i] * output_head(j, i).to_float();
        }
        logits_out[j] = s;
    }
}
