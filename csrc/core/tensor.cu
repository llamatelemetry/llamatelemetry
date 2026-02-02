#include "tensor.h"
#include "device.h"
#include <algorithm>
#include <numeric>
#include <cstring>

namespace llamatelemetry {

Tensor::Tensor()
    : data_(nullptr), dtype_(DType::Float32), device_id_(0),
      is_contiguous_(true), owns_data_(false) {}

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, int device_id)
    : shape_(shape), dtype_(dtype), device_id_(device_id),
      is_contiguous_(true), owns_data_(true) {
    compute_strides();
    allocate();
}

Tensor::~Tensor() {
    deallocate();
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), strides_(other.strides_), dtype_(other.dtype_),
      device_id_(other.device_id_), is_contiguous_(other.is_contiguous_),
      owns_data_(true) {
    allocate();
    if (data_ && other.data_) {
        CUDA_CHECK(cudaMemcpy(data_, other.data_, nbytes(), cudaMemcpyDeviceToDevice));
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)), dtype_(other.dtype_),
      device_id_(other.device_id_), is_contiguous_(other.is_contiguous_),
      owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();

        shape_ = other.shape_;
        strides_ = other.strides_;
        dtype_ = other.dtype_;
        device_id_ = other.device_id_;
        is_contiguous_ = other.is_contiguous_;
        owns_data_ = true;

        allocate();
        if (data_ && other.data_) {
            CUDA_CHECK(cudaMemcpy(data_, other.data_, nbytes(), cudaMemcpyDeviceToDevice));
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();

        data_ = other.data_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = other.dtype_;
        device_id_ = other.device_id_;
        is_contiguous_ = other.is_contiguous_;
        owns_data_ = other.owns_data_;

        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

void Tensor::allocate() {
    if (numel() == 0) {
        data_ = nullptr;
        return;
    }

    int current_device = Device::get_device();
    Device::set_device(device_id_);

    CUDA_CHECK(cudaMalloc(&data_, nbytes()));

    Device::set_device(current_device);
}

void Tensor::deallocate() {
    if (data_ && owns_data_) {
        int current_device = Device::get_device();
        Device::set_device(device_id_);

        cudaFree(data_);

        Device::set_device(current_device);
        data_ = nullptr;
    }
}

void Tensor::compute_strides() {
    int ndim = shape_.size();
    strides_.resize(ndim);

    if (ndim == 0) return;

    strides_[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Float16: return sizeof(half);
        case DType::BFloat16: return sizeof(__nv_bfloat16);
        case DType::Int32: return sizeof(int32_t);
        case DType::Int64: return sizeof(int64_t);
        case DType::UInt8: return sizeof(uint8_t);
        default: throw std::runtime_error("Unknown dtype");
    }
}

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
}

size_t Tensor::element_size() const {
    return dtype_size(dtype_);
}

size_t Tensor::nbytes() const {
    return numel() * element_size();
}

Tensor Tensor::to(int target_device) const {
    if (target_device == device_id_) {
        return *this;
    }

    Tensor result(shape_, dtype_, target_device);

    if (data_ && result.data_) {
        CUDA_CHECK(cudaMemcpy(result.data_, data_, nbytes(), cudaMemcpyDeviceToDevice));
    }

    return result;
}

Tensor Tensor::cpu() const {
    // For now, return CPU data as a new tensor (device -1)
    // In full implementation, would return NumPy array via pybind11
    throw std::runtime_error("CPU conversion not yet implemented - use Python API");
}

Tensor Tensor::contiguous() const {
    if (is_contiguous_) {
        return *this;
    }

    // Create new contiguous tensor and copy data
    Tensor result(shape_, dtype_, device_id_);
    // TODO: Implement strided copy kernel
    throw std::runtime_error("Non-contiguous tensors not yet supported");
}

Tensor Tensor::zeros(const std::vector<int64_t>& shape, DType dtype, int device_id) {
    Tensor result(shape, dtype, device_id);

    if (result.data_) {
        int current_device = Device::get_device();
        Device::set_device(device_id);

        CUDA_CHECK(cudaMemset(result.data_, 0, result.nbytes()));

        Device::set_device(current_device);
    }

    return result;
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, DType dtype, int device_id) {
    Tensor result(shape, dtype, device_id);

    // TODO: Implement fill kernel for ones
    throw std::runtime_error("ones() not yet implemented");
}

Tensor Tensor::from_ptr(void* data, const std::vector<int64_t>& shape, DType dtype, int device_id, bool copy) {
    Tensor result;
    result.shape_ = shape;
    result.dtype_ = dtype;
    result.device_id_ = device_id;
    result.is_contiguous_ = true;
    result.compute_strides();

    if (copy) {
        result.owns_data_ = true;
        result.allocate();
        if (result.data_) {
            CUDA_CHECK(cudaMemcpy(result.data_, data, result.nbytes(), cudaMemcpyDeviceToDevice));
        }
    } else {
        result.data_ = data;
        result.owns_data_ = false;
    }

    return result;
}

} // namespace llamatelemetry
