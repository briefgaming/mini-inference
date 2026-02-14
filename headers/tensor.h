#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "bf16.h"

template <int N, typename T = float>
class Tensor {
public:
    Tensor() : data_(nullptr) {
        shape_.fill(0);
        stride_.fill(0);
    }

    // Non-owning views over weights in memory
    Tensor(T* data, int d0) {
        static_assert(N == 1, "Tensor ctor (data, d0) requires N=1");
        init_view(data, {d0});
    }

    Tensor(T* data, int d0, int d1) {
        static_assert(N == 2, "Tensor ctor (data, d0, d1) requires N=2");
        init_view(data, {d0, d1});
    }

    Tensor(T* data, int d0, int d1, int d2) {
        static_assert(N == 3, "Tensor ctor (data, d0, d1, d2) requires N=3");
        init_view(data, {d0, d1, d2});
    }

    // Owning alloc constructors
    explicit Tensor(int d0) requires (N == 1) { allocate({d0}); }
    explicit Tensor(int d0, int d1) requires (N == 2) { allocate({d0, d1}); }
    explicit Tensor(int d0, int d1, int d2) requires (N == 3) { allocate({d0, d1, d2}); }



    T* data() { 
        return data_; 
    }
    const T* data() const { 
        return data_; 
    }

    // 1D
    T& operator[](int i) requires (N == 1) {
        assert(i >= 0 && i < shape_[0]);
        return data_[i];
    }
    const T& operator[](int i) const requires (N == 1) {
        assert(i >= 0 && i < shape_[0]);
        return data_[i];
    }

    // 2D
    T& operator()(int i, int j) requires (N == 2) {
        assert(i >= 0 && i < shape_[0]);
        assert(j >= 0 && j < shape_[1]);
        return data_[i * stride_[0] + j * stride_[1]];
    }
    const T& operator()(int i, int j) const requires (N == 2) {
        assert(i >= 0 && i < shape_[0]);
        assert(j >= 0 && j < shape_[1]);
        return data_[i * stride_[0] + j * stride_[1]];
    }

    // 3D
    T& operator()(int i, int j, int k) requires (N == 3) {
        assert(i >= 0 && i < shape_[0]);
        assert(j >= 0 && j < shape_[1]);
        assert(k >= 0 && k < shape_[2]);
        return data_[i * stride_[0] + j * stride_[1] + k * stride_[2]];
    }
    const T& operator()(int i, int j, int k) const requires (N == 3) {
        assert(i >= 0 && i < shape_[0]);
        assert(j >= 0 && j < shape_[1]);
        assert(k >= 0 && k < shape_[2]);
        return data_[i * stride_[0] + j * stride_[1] + k * stride_[2]];
    }

    Tensor<N - 1, T> slice(int i) const requires (N > 1) {
        assert(i >= 0 && i < shape_[0]);

        std::array<int, N - 1> new_shape{};
        std::array<int, N - 1> new_stride{};

        for (int d = 1; d < N; ++d) {
            new_shape[d - 1] = shape_[d];
            new_stride[d - 1] = stride_[d];
        }

        return Tensor<N - 1, T>(
            data_ + i * stride_[0],
            new_shape,
            new_stride,
            owner_
        );
    }

    Tensor<N, float> dequantize_to_f32() const requires std::is_same_v<T, bf16> {
        Tensor<N, float> out;
        if constexpr (N == 1) {
            out = Tensor<1, float>(shape_[0]);
        } else if constexpr (N == 2) {
            out = Tensor<2, float>(shape_[0], shape_[1]);
        } else if constexpr (N == 3) {
            out = Tensor<3, float>(shape_[0], shape_[1], shape_[2]);
        }

        for (int idx = 0; idx < size(); ++idx) {
            out.data()[idx] = data_[idx].to_float();
        }
        return out;
    }

    // Workspace memory to reduce memory pressure
    void dequantize_to_f32(Tensor<N, float>& out) const requires std::is_same_v<T, bf16> {
        for (int i = 0; i < N; ++i) {
            assert(out.dim(i) == shape_[i]);
        }
        for (int idx = 0; idx < size(); ++idx) {
            out.data()[idx] = data_[idx].to_float();
        }
    }

    int dim(int i) const { 
        return shape_[static_cast<size_t>(i)];
    }

    int stride(int i) const {
        return stride_[static_cast<size_t>(i)];
    }

    const std::array<int, N>& shape() const {
        return shape_;
    }

    const std::array<int, N>& strides() const {
        return stride_;
    }

    int size() const {
        int total = 1;
        for (int i = 0; i < N; i++) {
            total *= shape_[static_cast<size_t>(i)];
        }
        return total;
    }

private:
    T* data_;
    std::array<int, N> shape_;
    std::array<int, N> stride_;
    std::shared_ptr<T[]> owner_;

    void compute_stride() {
        stride_[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i) {
            stride_[i] = shape_[i + 1] * stride_[i + 1];
        }
    }

    void allocate(const std::array<int, N>& shape) {
        shape_ = shape;
        compute_stride();
        owner_.reset(new T[size()](), std::default_delete<T[]>());
        data_ = owner_.get();
    }


    void init_view(T* data, const std::array<int, N>& shape) {
        data_ = data;
        shape_ = shape;
        compute_stride();
        owner_.reset();
    }

    Tensor(T* data, const std::array<int, N>& shape, const std::array<int, N>& stride, std::shared_ptr<T[]> owner)
        : data_(data), shape_(shape), stride_(stride), owner_(std::move(owner)) {}

    template <int M, typename U>
    friend class Tensor;
};
