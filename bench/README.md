# bench

microbenchmarks comparing naive vs SIMD/threaded kernels.
Built only with `-DTIRAMISU_BUILD_BENCH=ON` (OFF by default).

## matmul

`matmul_bench.cpp` times the real `tiramisu::ops::matmul` against a textbook
naive i-j-k loop on the same float32 inputs, and reports the max abs diff
between the two outputs.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTIRAMISU_BUILD_BENCH=ON
cmake --build build --target matmul_bench --parallel
./build/bench/matmul_bench          # sizes default to 128 256 512 1000 1024
./build/bench/matmul_bench 2048     # or pass your own
```

Use Release: Debug turns on ASan/UBSan and the timings mean nothing.

Results, 2026-11-24:

- CPU: Intel Core i7-9750H (laptop, `powersave` governor), single thread
- Compiler: GCC 16.2.1, Arch Linux
- Flags: `-O3 -DNDEBUG -O3 -mavx2 -mfma` for both sides (the bench copies
  `tiramisu_ops`' flags, so the naive loop is built the same way)
- Method: square n x n, inputs uniform in [-1, 1) (seed 42), 1 warmup +
  best of N; tiramisu time includes allocating and zeroing the output

| n | N | naive ms | tiramisu ms | speedup | tiramisu GFLOP/s | max abs diff |
|---|---|---|---|---|---|---|
| 128 | 10 | 1.69 | 0.15 | 11.04x | 27.35 | 3.81e-06 |
| 256 | 10 | 16.83 | 1.16 | 14.55x | 29.02 | 5.72e-06 |
| 512 | 5 | 227.36 | 10.08 | 22.56x | 26.63 | 1.05e-05 |
| 1000 | 3 | 972.70 | 64.68 | 15.04x | 30.92 | 1.91e-05 |
| 1024 | 3 | 1515.34 | 91.67 | 16.53x | 23.43 | 2.10e-05 |

Over 4 runs the speedup ranged 10.8-11.6x (128), 13.5-14.6x (256),
18.5-23.5x (512), 13.7-15.3x (1000), 15.9-17.2x (1024). Most of the noise is
on the naive side.

Notes:

- Both kernels are O(n^3). The gain is a constant factor from 64x64x64 cache
  tiling plus the AVX2 `_mm256_fmadd_ps` inner loop in `ops/cpu/matmul.cpp`.
  No threading, no Strassen.
- Power-of-two n hurts naive extra (B's column walk keeps hitting the same
  cache sets), which flatters 512 and 1024. n=1000 is the fairer point.
- Small or ragged n gets less: about 4x at n=32 and 7.4x at n=100 vs 10x at
  n=96 (n not a multiple of 8 means misaligned rows and a scalar tail).
- Max abs diff is float32 rounding from a different summation order.
