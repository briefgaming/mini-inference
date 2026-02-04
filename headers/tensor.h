// Atomic representation of data (minimal tensor view)
#pragma once

#include <array>
#include <cstddef>

template <int N>
class Tensor {
public:
    Tensor() : data_(nullptr) {
        shape_.fill(0);
        stride_.fill(0);
    }

    Tensor(float* data, int d0) {
        static_assert(N == 1, "Tensor ctor (data, d0) requires N=1");
        init(data, {d0});
    }

    Tensor(float* data, int d0, int d1) {
        static_assert(N == 2, "Tensor ctor (data, d0, d1) requires N=2");
        init(data, {d0, d1});
    }

    Tensor(float* data, int d0, int d1, int d2) {
        static_assert(N == 3, "Tensor ctor (data, d0, d1, d2) requires N=3");
        init(data, {d0, d1, d2});
    }

    float* data() { return data_; }
    const float* data() const { return data_; }

    int dim(int i) const { return shape_[static_cast<size_t>(i)]; }
    int stride(int i) const { return stride_[static_cast<size_t>(i)]; }

    const std::array<int, N>& shape() const { return shape_; }
    const std::array<int, N>& strides() const { return stride_; }

    int size() const {
        int total = 1;
        for (int i = 0; i < N; i++) {
            total *= shape_[static_cast<size_t>(i)];
        }
        return total;
    }

private:
    float* data_;
    std::array<int, N> shape_;
    std::array<int, N> stride_;

    void init(float* data, const std::array<int, N>& shape) {
        data_ = data;
        shape_ = shape;
        if constexpr (N > 0) {
            stride_[N - 1] = 1;
            for (int i = N - 2; i >= 0; i--) {
                stride_[static_cast<size_t>(i)] =
                    stride_[static_cast<size_t>(i + 1)] *
                    shape_[static_cast<size_t>(i + 1)];
            }
        }
    }
};
