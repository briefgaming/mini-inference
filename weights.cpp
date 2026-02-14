#include "headers/weights.h"
#include "headers/json.hpp"

#include <fstream>

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
        weight_mapping[key_name] = std::move(meta);
    }
    return true;
}
