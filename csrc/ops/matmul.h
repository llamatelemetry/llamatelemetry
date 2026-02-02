#pragma once

#include "../core/tensor.h"

namespace llamatelemetry {
namespace ops {

// Matrix multiplication: C = A @ B
// A: [M, K]
// B: [K, N]
// C: [M, N]
Tensor matmul(const Tensor& A, const Tensor& B);

// Batched matrix multiplication: C = A @ B
// A: [batch, M, K]
// B: [batch, K, N]
// C: [batch, M, N]
Tensor batched_matmul(const Tensor& A, const Tensor& B);

} // namespace ops
} // namespace llamatelemetry
