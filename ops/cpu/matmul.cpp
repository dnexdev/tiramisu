#include "tiramisu/ops/matmul.hpp"
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

namespace tiramisu::ops {

namespace {

constexpr int64_t TILE_M = 64;
constexpr int64_t TILE_N = 64;
constexpr int64_t TILE_K = 64;

// Inner kernel: accumulate C[i:i+tm, j:j+tn] += A[i:i+tm, k:k+tk] @ B[k:k+tk, j:j+tn]
// All pointers are into contiguous row-major buffers.
// M_stride, K_stride, N_stride are the full matrix column counts (for pointer arithmetic).
void matmul_tile(
    const float* A, const float* B, float* C,
    int64_t tm, int64_t tn, int64_t tk,   // actual tile sizes (may be < TILE when near edge)
    int64_t M_stride, int64_t K_stride, int64_t N_stride)
{
    for (int64_t ii = 0; ii < tm; ii++) {
        for (int64_t kk = 0; kk < tk; kk++) {
            float a_val = A[ii * M_stride + kk];
            __m256 a_bcast = _mm256_set1_ps(a_val);

            int64_t jj = 0;
            // AVX2 path: 8 floats at a time
            for (; jj + 8 <= tn; jj += 8) {
                __m256 c_vec = _mm256_loadu_ps(&C[ii * N_stride + jj]);
                __m256 b_vec = _mm256_loadu_ps(&B[kk * K_stride + jj]);
                c_vec = _mm256_fmadd_ps(a_bcast, b_vec, c_vec);
                _mm256_storeu_ps(&C[ii * N_stride + jj], c_vec);
            }
            // scalar tail: leftover columns when N % 8 != 0
            for (; jj < tn; jj++) {
              // float sum = 0.0f;
              C[ii * N_stride + jj] += a_val * B[kk * K_stride + jj];
            }
        }
    }
}

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32)
        throw std::runtime_error("matmul: only Float32 supported");
    if (a.shape().size() != 2 || b.shape().size() != 2)
        throw std::invalid_argument("matmul: only 2D tensors supported");

    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    int64_t N = b.shape()[1];
    if (K != b.shape()[0])
        throw std::invalid_argument("matmul: inner dimensions must match");

    Tensor c_a = a.contiguous();
    Tensor c_b = b.contiguous();
    Tensor out({M, N}, DType::Float32, Device::CPU);

    const float* A = c_a.data<float>();
    const float* B = c_b.data<float>();
    float* C = out.data<float>();

    // zero output (tiled accumulation assumes C starts at 0)
    std::fill_n(C, M * N, 0.0f);

    for (int64_t i = 0; i < M; i += TILE_M) {
        for (int64_t j = 0; j < N; j += TILE_N) {
            for (int64_t k = 0; k < K; k += TILE_K) {
                int64_t tm = std::min(TILE_M, M - i);
                int64_t tn = std::min(TILE_N, N - j);
                int64_t tk = std::min(TILE_K, K - k);

                matmul_tile(
                    A + i * K + k,   // A tile starts at row i, col k
                    B + k * N + j,   // B tile starts at row k, col j
                    C + i * N + j,   // C tile starts at row i, col j
                    tm, tn, tk,
                    K,               // A's row stride = K (full width of A)
                    N,               // B's row stride = N (full width of B)
                    N                // C's row stride = N (full width of C)
                );
            }
        }
    }

    return out;
}

}  // namespace tiramisu::ops