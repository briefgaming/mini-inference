#include "headers/model.h"
#include "headers/json.hpp"
#include "headers/weights.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>

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
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        layers[i].attn.wq = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "self_attn.q_proj.weight", mmap_data),
            dim,
            dim
        );
        layers[i].attn.wk = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "self_attn.k_proj.weight", mmap_data),
            kv_dim,
            dim
        );
        layers[i].attn.wv = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "self_attn.v_proj.weight", mmap_data),
            kv_dim,
            dim
        );
        layers[i].attn.wo = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "self_attn.o_proj.weight", mmap_data),
            dim,
            dim
        );

        layers[i].mlp.w1_gate = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "mlp.gate_proj.weight", mmap_data),
            hidden_dim,
            dim
        );
        layers[i].mlp.w1_down = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "mlp.up_proj.weight", mmap_data),
            hidden_dim,
            dim
        );
        layers[i].mlp.w1_up = Tensor<2, bf16>(
            w.get_ptr<bf16>(prefix + "mlp.down_proj.weight", mmap_data),
            dim,
            hidden_dim
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
    kv_dim = config["num_key_value_heads"];

    return true;
}

void RMSNorm::rmsnorm(Tensor<1>& out, Tensor<1>& in) {
    for (int i = 0; i < out.dim(0); ++i) {
        out[i] = in[i];
    }
}

void GQAttention::rope() {}

void GQAttention::gqattention(
    Tensor<1>& out,
    Tensor<1>& x,
    int pos,
    const Tensor<3>& kvcache
) {
    (void)pos;
    (void)kvcache;
    for (int i = 0; i < out.dim(0); ++i) {
        out[i] = x[i];
    }
}

void SwiGLUBlock::swiglu(Tensor<1>& out, Tensor<1>& in) {
    for (int i = 0; i < out.dim(0); ++i) {
        out[i] = in[i];
    }
}

void TransformerBlock::apply_transformer(Tensor<1>& x, int pos, const Tensor<3>& kvcache) {
    Tensor<1> xbuf(x.dim(0));
    pre_attn_norm.rmsnorm(xbuf, x);
    attn.rope();
    attn.gqattention(x, xbuf, pos, kvcache);
    post_ffn_norm.rmsnorm(xbuf, x);
    mlp.swiglu(x, xbuf);
}

void Model::forward(
    int token_id,
    int input_pos,
    const Tensor<3>& kvcache,
    const Tensor<1>& emb_out
) {
    (void)token_id;
    (void)emb_out;

    Tensor<1> x(dim);
    for (int i = 0; i < dim; ++i) {
        x[i] = 0.0f;
    }

    for (int l = 0; l < n_layers; ++l) {
        layers[l].apply_transformer(x, input_pos, kvcache);
    }

    Tensor<1> norm_out(dim);
    final_norm.rmsnorm(norm_out, x);
}
