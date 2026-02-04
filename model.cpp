#include "headers/model.h"
#include "headers/weights.h"
#include "headers/tensor.h"
#include "headers/json.hpp"
#include <fcntl.h>
#include <sys/stat.h> 
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <fstream>


Model::Model() : layers(nullptr), mmap_data(nullptr) {}
Model::~Model() {
    if (layers) delete[] layers;
}

void Model::load_weights(WeightMap& w, const std::string& weight_path) {
    const char *fname = weight_path.c_str();
    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
        perror(fname);
        return;
    }

    struct stat sb;
    fstat(fd, &sb);

    mmap_data = (char*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    std::cout << "Allocating " << n_layers << " transformer blocks..." << std::endl;
    layers = new TransformerBlock[n_layers];
    
    token_embed = Tensor<2>(w.get_ptr("model.embed_tokens.weight", mmap_data), vocab_size, dim);
    final_norm.weight = Tensor<1>(w.get_ptr("model.norm.weight", mmap_data), dim);
    /**
     * Llama 3.2 1B uses shared weights for the output head and the token embeddings.
     * This is a common optimization to reduce the model size.
     * Should change for support for bigger models.
    */
    output_head = token_embed;
    for (int i = 0; i < n_layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        layers[i].attn.wq = Tensor<2>(w.get_ptr(prefix + "self_attn.q_proj.weight", mmap_data), dim, dim);
        layers[i].attn.wk = Tensor<2>(w.get_ptr(prefix + "self_attn.k_proj.weight", mmap_data), kv_dim, dim);
        layers[i].attn.wv = Tensor<2>(w.get_ptr(prefix + "self_attn.v_proj.weight", mmap_data), kv_dim, dim);
        layers[i].attn.wo = Tensor<2>(w.get_ptr(prefix + "self_attn.o_proj.weight", mmap_data), dim, dim);

        layers[i].mlp.w1_gate = Tensor<2>(w.get_ptr(prefix + "mlp.gate_proj.weight", mmap_data), hidden_dim, dim);
        layers[i].mlp.w1_down = Tensor<2>(w.get_ptr(prefix + "mlp.up_proj.weight", mmap_data), hidden_dim, dim);
        layers[i].mlp.w1_up = Tensor<2>(w.get_ptr(prefix + "mlp.down_proj.weight", mmap_data), dim, hidden_dim);

        layers[i].pre_attn_norm.weight = Tensor<1>(w.get_ptr(prefix + "input_layernorm.weight", mmap_data), dim);
        layers[i].post_ffn_norm.weight = Tensor<1>(w.get_ptr(prefix + "post_attention_layernorm.weight", mmap_data), dim);
    }
}

bool Model::load_config(const std::string& config_path) {
    std::ifstream f(config_path);
    nlohmann::json config;
    f >> config;

    n_layers   = config["num_hidden_layers"];
    vocab_size = config["vocab_size"];
    dim        = config["hidden_size"];
    hidden_dim = config["intermediate_size"]; 
    n_heads    = config["num_attention_heads"];
    kv_dim     = config["num_key_value_heads"];

    return true;
}