#pragma once

#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct TensorMeta {
    size_t offset = 0;
    std::vector<int> shape;
    std::string dtype = "F32";
    bool transposed = false;
    std::string scale_name;
};

class WeightMap {
public:
    std::map<std::string, TensorMeta> weight_mapping;

    bool load(const std::string& offset_path);
    const TensorMeta* get_meta(const std::string& weight_name) const;

    template <typename T = float>
    T* get_ptr(const std::string& weight_name, char* weight_mmap_ptr) const {
        const TensorMeta* meta = get_meta(weight_name);
        if (meta == nullptr) {
            std::cerr << "Warning: Weight not found: " << weight_name << std::endl;
            return nullptr;
        }
        return reinterpret_cast<T*>(weight_mmap_ptr + meta->offset);
    }
};
