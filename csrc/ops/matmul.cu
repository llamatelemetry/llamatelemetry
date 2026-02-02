#include "matmul.h"
#include "../core/device.h"
#include <cublas_v2.h>
#include <stdexcept>

namespace llamatelemetry {
namespace ops {

static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;

    if (handle == nullptr) {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
    }

    return handle;
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    // Check dimensions
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("matmul expects 2D tensors");
    }

    if (A.shape()[1] != B.shape()[0]) {
        throw std::runtime_error("Incompatible dimensions for matmul");
    }

    if (A.device() != B.device()) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    if (A.dtype() != B.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype");
    }

    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];

    // Create output tensor
    Tensor C({M, N}, A.dtype(), A.device());

    // Set device
    int current_device = Device::get_device();
    Device::set_device(A.device());

    cublasHandle_t handle = get_cublas_handle();

    if (A.dtype() == DType::Float32) {
        // C = A @ B (row-major)
        // cuBLAS assumes column-major, so we compute: C^T = B^T @ A^T
        // Which means: C = A @ B in row-major

        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_N,    // B not transposed (but treated as B^T in col-major)
            CUBLAS_OP_N,    // A not transposed (but treated as A^T in col-major)
            N,              // Rows of B^T (cols of B)
            M,              // Cols of A^T (rows of A)
            K,              // Cols of B^T = Rows of A^T
            &alpha,
            (const float*)B.data_ptr(), N,  // Leading dimension of B
            (const float*)A.data_ptr(), K,  // Leading dimension of A
            &beta,
            (float*)C.data_ptr(), N         // Leading dimension of C
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS SGEMM failed");
        }
    } else if (A.dtype() == DType::Float16) {
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(0.0f);

        cublasStatus_t status = cublasHgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            (const half*)B.data_ptr(), N,
            (const half*)A.data_ptr(), K,
            &beta,
            (half*)C.data_ptr(), N
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS HGEMM failed");
        }
    } else {
        throw std::runtime_error("Unsupported dtype for matmul");
    }

    Device::set_device(current_device);

    return C;
}

Tensor batched_matmul(const Tensor& A, const Tensor& B) {
    // Check dimensions
    if (A.ndim() != 3 || B.ndim() != 3) {
        throw std::runtime_error("batched_matmul expects 3D tensors");
    }

    if (A.shape()[0] != B.shape()[0]) {
        throw std::runtime_error("Batch sizes must match");
    }

    if (A.shape()[2] != B.shape()[1]) {
        throw std::runtime_error("Incompatible dimensions for matmul");
    }

    if (A.device() != B.device()) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    if (A.dtype() != B.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype");
    }

    int64_t batch = A.shape()[0];
    int64_t M = A.shape()[1];
    int64_t K = A.shape()[2];
    int64_t N = B.shape()[2];

    // Create output tensor
    Tensor C({batch, M, N}, A.dtype(), A.device());

    // Set device
    int current_device = Device::get_device();
    Device::set_device(A.device());

    cublasHandle_t handle = get_cublas_handle();

    if (A.dtype() == DType::Float32) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasStatus_t status = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            (const float*)B.data_ptr(), N, K * N,  // stride for B
            (const float*)A.data_ptr(), K, M * K,  // stride for A
            &beta,
            (float*)C.data_ptr(), N, M * N,        // stride for C
            batch
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS batched SGEMM failed");
        }
    } else if (A.dtype() == DType::Float16) {
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(0.0f);

        cublasStatus_t status = cublasHgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            (const half*)B.data_ptr(), N, K * N,
            (const half*)A.data_ptr(), K, M * K,
            &beta,
            (half*)C.data_ptr(), N, M * N,
            batch
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS batched HGEMM failed");
        }
    } else {
        throw std::runtime_error("Unsupported dtype for batched_matmul");
    }

    Device::set_device(current_device);

    return C;
}

} // namespace ops
} // namespace llamatelemetry
