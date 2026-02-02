#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <memory>
#include <string>

namespace llamatelemetry {

enum class DType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    UInt8
};

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DType dtype, int device_id = 0);
    ~Tensor();

    // Copy and move constructors
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // Accessors
    void* data_ptr() const { return data_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    int device() const { return device_id_; }
    int64_t numel() const;
    size_t element_size() const;
    size_t nbytes() const;
    int ndim() const { return shape_.size(); }

    // Device operations
    Tensor to(int device_id) const;
    Tensor cpu() const;

    // Utilities
    bool is_contiguous() const { return is_contiguous_; }
    Tensor contiguous() const;

    // Static factory methods
    static Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::Float32, int device_id = 0);
    static Tensor ones(const std::vector<int64_t>& shape, DType dtype = DType::Float32, int device_id = 0);
    static Tensor from_ptr(void* data, const std::vector<int64_t>& shape, DType dtype, int device_id, bool copy = false);

private:
    void* data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DType dtype_;
    int device_id_;
    bool is_contiguous_;
    bool owns_data_;

    void allocate();
    void deallocate();
    void compute_strides();
    static size_t dtype_size(DType dtype);
};

} // namespace llamatelemetry
