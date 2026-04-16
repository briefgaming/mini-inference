#include "headers/weights.h"
#include "headers/json.hpp"

#include <fstream>

const TensorMeta* WeightMap::get_meta(const std::string& weight_name) const {
    auto it = weight_mapping.find(weight_name);
    if (it == weight_mapping.end()) {
        return nullptr;
    }
    return &it->second;
}

bool WeightMap::load(const std::string& offset_path) {
    std::ifstream f(offset_path);
    if (!f.is_open()) {
        return false;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (...) {
        return false;
    }

    weight_mapping.clear();

    for (auto& element : j.items()) {
        std::string key_name = element.key();

        TensorMeta meta{};
        if (!element.value().contains("offset")) {
            return false;
        }
        meta.offset = element.value()["offset"];
        if (element.value().contains("shape")) {
            meta.shape = element.value()["shape"].get<std::vector<int>>();
        }
        if (element.value().contains("dtype")) {
            meta.dtype = element.value()["dtype"].get<std::string>();
        }
        if (element.value().contains("transposed")) {
            meta.transposed = element.value()["transposed"].get<bool>();
        }
        if (element.value().contains("scale_name")) {
            meta.scale_name = element.value()["scale_name"].get<std::string>();
        }
        weight_mapping[key_name] = std::move(meta);
    }
    return true;
}
