#pragma once

#include <cstdint>
#include <bit>

struct bf16 {
    uint16_t bits;

    static bf16 from_float(float x) {
        uint32_t u = std::bit_cast<uint32_t>(x);
        uint32_t lsb = (u >> 16) & 1u;
        uint32_t rounded = u + 0x7FFFu + lsb;
        return bf16{static_cast<uint16_t>(rounded >> 16)};
    }

    float to_float() const {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        return std::bit_cast<float>(u);
    }
};