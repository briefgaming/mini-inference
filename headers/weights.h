#pragma once

#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct TensorMeta {
    size_t offset;
    std::vector<int> shape;
};

class WeightMap {
public:
    std::map<std::string, TensorMeta> weight_mapping;

    bool load(const std::string& offset_path);

    template <typename T = float>
    T* get_ptr(const std::string& weight_name, char* weight_mmap_ptr) const {
        auto it = weight_mapping.find(weight_name);
        if (it == weight_mapping.end()) {
            std::cerr << "Warning: Weight not found: " << weight_name << std::endl;
            return nullptr;
        }
        return reinterpret_cast<T*>(weight_mmap_ptr + it->second.offset);
    }
};
