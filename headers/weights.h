#pragma once

#include <string>
#include <vector>
#include <map>

struct TensorMeta {
    size_t offset;
    std::vector<int> shape;
};

class WeightMap {
public:
    std::map<std::string, TensorMeta> weight_mapping;

    bool load(const std::string& offset_path);
    float* get_ptr(const std::string& weight_name, char* weight_mmap_ptr);
};