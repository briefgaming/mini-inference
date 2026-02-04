#include "headers/weights.h"
#include "headers/json.hpp"

#include <iostream>
#include <fstream>

bool WeightMap::load(const std::string& offset_path) {
    std::ifstream f(offset_path);
    nlohmann::json j;
    f >> j;

    for (auto& element : j.items()) {
        std::string key_name = element.key();
        
        size_t offset = element.value()["offset"];
        weight_mapping[key_name].offset = offset;
    }
    return true;
}

float* WeightMap::get_ptr(const std::string& weight_name, char* weight_mmap_ptr) {
    if (weight_mapping.find(weight_name) == weight_mapping.end()) {
        std::cerr << "Warning: Weight not found: " << weight_name << std::endl;
        return nullptr;
    }

    TensorMeta& meta = weight_mapping[weight_name];
    return (float*)(weight_mmap_ptr + meta.offset);
}
