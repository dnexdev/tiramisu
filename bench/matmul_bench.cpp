// matmul_bench: tiramisu::ops::matmul vs a textbook naive i-j-k loop.
//
// Both sides are O(n^3) on the same float32 inputs. The library kernel
// (ops/cpu/matmul.cpp) gets its speed from 64x64x64 cache tiling plus an
// AVX2 FMA inner loop; it is single-threaded, so this is a fair 1-core
// comparison. Each method gets one warmup run, then the best of N timed
// runs. The library time includes allocating and zeroing the output tensor.
//
// Usage:
//   matmul_bench [n ...]   (default: 128 256 512 1000 1024)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tiramisu/core/tensor.hpp"
#include "tiramisu/ops/matmul.hpp"

using namespace tiramisu;

namespace {

// C[i,j] = sum_k A[i,k] * B[k,j], row-major, B walked down a column.
void naive_matmul(const float* A, const float* B, float* C, int64_t M,
                  int64_t K, int64_t N) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

void fill_uniform(Tensor& t, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  float* d = t.data<float>();
  for (int64_t i = 0; i < t.numel(); ++i) d[i] = dist(gen);
}

// One warmup call, then the fastest of `reps` timed calls, in ms.
template <typename F>
double best_ms(F&& f, int reps) {
  f();
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best,
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return best;
}

int reps_for(int64_t n) {
  if (n <= 256) return 10;
  if (n <= 512) return 5;
  return 3;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int64_t> sizes = {128, 256, 512, 1000, 1024};
  if (argc > 1) {
    sizes.clear();
    for (int i = 1; i < argc; ++i) {
      int64_t n = std::strtoll(argv[i], nullptr, 10);
      if (n <= 0) {
        std::fprintf(stderr, "Usage: %s [n ...]\n", argv[0]);
        return 1;
      }
      sizes.push_back(n);
    }
  }

#ifndef NDEBUG
  std::fprintf(stderr,
               "warning: assertions on (Debug build?); Debug also enables "
               "sanitizers, so timings are not representative\n");
#endif
#ifdef __VERSION__
  std::printf("compiler: %s\n", __VERSION__);
#endif
  std::printf("float32, square n x n, 1 warmup + best of N, single thread\n\n");
  std::printf("%6s %3s %12s %12s %9s %12s %10s\n", "n", "N", "naive ms",
              "tiramisu ms", "speedup", "tiramisu GF", "max|diff|");

  std::mt19937 gen(42);
  for (int64_t n : sizes) {
    Tensor a({n, n});
    Tensor b({n, n});
    Tensor c_naive({n, n});
    fill_uniform(a, gen);
    fill_uniform(b, gen);

    const int reps = reps_for(n);
    const float* A = a.data<float>();
    const float* B = b.data<float>();
    float* C = c_naive.data<float>();

    double naive = best_ms([&] { naive_matmul(A, B, C, n, n, n); }, reps);

    Tensor c_lib({n, n});
    double lib = best_ms([&] { c_lib = ops::matmul(a, b); }, reps);

    double max_diff = 0.0;
    const float* L = c_lib.data<float>();
    for (int64_t i = 0; i < n * n; ++i) {
      max_diff =
          std::max(max_diff, static_cast<double>(std::fabs(L[i] - C[i])));
    }

    double gflops = 2.0 * n * n * n / (lib * 1e6);
    std::printf("%6lld %3d %12.2f %12.2f %8.2fx %12.2f %10.2e\n",
                static_cast<long long>(n), reps, naive, lib, naive / lib,
                gflops, max_diff);
  }
  return 0;
}
