// gemm_w4a8.h -- small-M int4-weight x int8-activation WMMA GEMM for decode
// and speculative verify (HALO_W4DEC=3, "W4A8-decode-all").
//
//   C[m][n] = sum_c (float)dot_c(m, n) * (fp16f(as[m][c]) * fp16f(ws[n][c]))
//
// dot_c is the exact int32 dot over the 256 codes of chunk c between the
// activation's int8 codes (k_actq8_emit: Hadamard-rotated per 256-chunk, one
// f16 scale = absmax/127) and the .i4l weight codes (two's-complement nibbles,
// f16 scale per 256). M <= 16 rows (rows past M-1 contribute nothing), N % 16
// == 0, K % 256 == 0. Every row's arithmetic is independent of the other rows,
// so a one-row serial step and an eight-row verify pass are bit-identical per
// row by construction (the reason this kernel exists: the multi-row GEMVs cost
// 1.2-1.5x a single row, this kernel costs the weight bytes once whatever M is).
//
// Layout: one block of 8 waves per 16 output columns (weight rows n0..n0+15);
// wave w takes the 256-wide K chunks c = w, w+8, ... (split-K inside the
// block), keeps the 16x16 i32 accumulator of the chunk in registers, converts
// it into its f32 accumulator with the two scales once per chunk, and the 8
// partial f32 tiles are summed in wave order through LDS at the end.
// Fragments (measured layout, results/phase3a-iu4-probe.json, gemm_i4.h): lane
// l supplies row (l & 15) of A and weight row n0 + (l & 15) of B, 16 codes of
// the current 16-wide k step as 16 int8 in 4 dwords, lanes 16-31 replicate
// lanes 0-15; C/D: lane l holds column l & 15, element e holds row 2e + (l >> 4).
// The k order inside a 16-step is the nibble-expansion order
// [0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15] (low nibbles of each source dword
// first), and k_actq8_emit writes the activation codes in that same order, so
// the weight nibbles need no byte permutation: two dwords of codes expand to
// four dwords of int8 with masks and one multiply.
// Weight tiles are double-buffered in registers (the wave's next chunk is
// loading while the current one multiplies).
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

namespace w4a8 {

typedef int   i32x4 __attribute__((ext_vector_type(4)));
typedef int   i32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float fp16f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  if (exp == 0) { float d = (float)man * (1.0f / 16777216.0f); return sign ? -d : d; }
  if (exp == 31) return man ? nanf("1") : (sign ? -INFINITY : INFINITY);
  uint32_t u = sign | ((exp - 15 + 127) << 23) | (man << 13);
  return __uint_as_float(u);
}
__device__ __forceinline__ uint16_t f32bf(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lo = u & 0xFFFFu;
  u = (u >> 16) + (lo > 0x8000u) + ((lo == 0x8000u) & ((u >> 16) & 1u));
  return (uint16_t)(u & 0xFFFFu);
}
// 8 nibbles (one dword) -> two dwords of int8: low nibbles (elements 0,2,4,6)
// then high nibbles (1,3,5,7), each sign-extended from 4 bits.
__device__ __forceinline__ void expand8(uint32_t d, uint32_t& lo, uint32_t& hi) {
  lo = d & 0x0F0F0F0Fu;
  hi = (d >> 4) & 0x0F0F0F0Fu;
  lo |= ((lo & 0x08080808u) >> 3) * 0xF0u;
  hi |= ((hi & 0x08080808u) >> 3) * 0xF0u;
}

static constexpr int NWAVES = 8, NT = NWAVES * 32;

// C: bf16 [M][N] (OUTF32 = false) or f32 [M][N]
template<bool OUTF32>
__global__ void __launch_bounds__(NT)
k_gemm_w4a8(const uint8_t* __restrict__ ac, const uint16_t* __restrict__ as,
            const uint8_t* __restrict__ wc, const uint16_t* __restrict__ ws,
            void* __restrict__ Cv, int M, int N, int K) {
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = blockIdx.x * 16;
  const int r = lane & 15;                       // A row / weight row within the tile
  const int nsc = K >> 8;
  const uint8_t* wrow = wc + (size_t)(n0 + r) * (K >> 1);
  const uint16_t* wsr = ws + (size_t)(n0 + r) * nsc;
  const uint8_t* arow = ac + (size_t)r * K;
  const uint16_t* asr = as + (size_t)r * nsc;
  const bool arow_live = r < M;
  float facc[8];
#pragma unroll
  for (int e = 0; e < 8; e++) facc[e] = 0.f;
  // chunk list of this wave: c = wave, wave + 8, ...; two register tiles so
  // the next chunk's weights load while the current chunk multiplies (static
  // buffer indices: the loop is unrolled by two)
  uint4 wbuf0[8], wbuf1[8];
  auto load_w = [&](int c, uint4* dst) {
    const uint4* p = reinterpret_cast<const uint4*>(wrow + (size_t)c * 128);
#pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = p[i];
  };
  auto compute = [&](int c, const uint4* w) {
    i32x8 iacc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int s = 0; s < 16; s++) {
      // weight nibbles of k step s: dwords 2s, 2s+1 of the 32-dword tile
      const uint4 q = w[s >> 1];
      const uint32_t d0 = (s & 1) ? q.z : q.x, d1 = (s & 1) ? q.w : q.y;
      uint32_t b0, b1, b2, b3;
      expand8(d0, b0, b1);
      expand8(d1, b2, b3);
      i32x4 bf = {(int)b0, (int)b1, (int)b2, (int)b3};
      i32x4 af = {0, 0, 0, 0};
      if (arow_live) af = *reinterpret_cast<const i32x4*>(arow + (size_t)c * 256 + s * 16);
      iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af, true, bf, iacc, false);
    }
    const float sw = fp16f(wsr[c]);
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const int m = 2 * e + (lane >> 4);
      const float sa = m < M ? fp16f(as[(size_t)m * nsc + c]) : 0.f;
      facc[e] = fmaf((float)iacc[e], sa * sw, facc[e]);
    }
  };
  if (wave < nsc) load_w(wave, wbuf0);
  for (int c = wave; c < nsc; c += 2 * NWAVES) {
    const int c1 = c + NWAVES, c2 = c + 2 * NWAVES;
    if (c1 < nsc) load_w(c1, wbuf1);
    compute(c, wbuf0);
    if (c1 < nsc) {
      if (c2 < nsc) load_w(c2, wbuf0);
      compute(c1, wbuf1);
    }
  }
  // cross-wave sum in wave order (deterministic), then the epilogue by wave 0
  __shared__ float part[NWAVES][8][32];
#pragma unroll
  for (int e = 0; e < 8; e++) part[wave][e][lane] = facc[e];
  __syncthreads();
  if (wave == 0) {
#pragma unroll
    for (int e = 0; e < 8; e++) {
      float sum = 0.f;
#pragma unroll
      for (int wv = 0; wv < NWAVES; wv++) sum += part[wv][e][lane];
      const int m = 2 * e + (lane >> 4), n = n0 + (lane & 15);
      if (m < M) {
        if constexpr (OUTF32) reinterpret_cast<float*>(Cv)[(size_t)m * N + n] = sum;
        else reinterpret_cast<uint16_t*>(Cv)[(size_t)m * N + n] = f32bf(sum);
      }
    }
  }
}

// Activation quantizer for k_gemm_w4a8: grid (16, K/256), 256 threads. Rows
// m < M: Hadamard-256 rotation of the row's chunk (the rotation k_actq_emit
// applies; H/16), f16 scale = absmax/127, codes = rint(v/scale) in [-127,127]
// written in the kernel's k order; rows m >= M are zero codes and zero scale.
__device__ __forceinline__ float bf16f_(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ uint16_t f32h_(float x) {
  uint32_t u = __float_as_uint(x);
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t e = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
  uint32_t m = u & 0x7FFFFFu;
  if (((u >> 23) & 0xFFu) == 255) return (uint16_t)(sign | 0x7C00u | (m ? 0x200u : 0));
  if (e >= 31) return (uint16_t)(sign | 0x7C00u);
  if (e <= 0) {
    if (e < -10) return (uint16_t)sign;
    m |= 0x800000u;
    int sh = 14 - e;
    uint32_t r = m >> sh, rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
    r += (rem > half || (rem == half && (r & 1))) ? 1u : 0u;
    return (uint16_t)(sign | r);
  }
  uint32_t r = ((uint32_t)e << 10) | (m >> 13);
  uint32_t rem = m & 0x1FFFu;
  r += (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ? 1u : 0u;
  return (uint16_t)(sign | r);
}
__global__ void k_actq8_emit(const uint16_t* __restrict__ X, uint8_t* __restrict__ C,
                             uint16_t* __restrict__ S, int M, int K) {
  __shared__ float sh[256], sm[256];
  const int tid = threadIdx.x, m = blockIdx.x, chunk = blockIdx.y;
  const int nsc = K >> 8;
  if (m >= M) {
    C[(size_t)m * K + (size_t)chunk * 256 + tid] = 0;
    if (tid == 0) S[(size_t)m * nsc + chunk] = 0;
    return;
  }
  sh[tid] = bf16f_(X[(size_t)m * K + (size_t)chunk * 256 + tid]);
  __syncthreads();
  for (int len = 1; len < 256; len <<= 1) {
    float a = sh[tid], b = sh[tid ^ len];
    __syncthreads();
    sh[tid] = ((tid & len) ? b - a : a + b);
    __syncthreads();
  }
  const float v = sh[tid] * (1.0f / 16.0f);
  sm[tid] = fabsf(v);
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sm[tid] = fmaxf(sm[tid], sm[tid + off]);
    __syncthreads();
  }
  const uint16_t sb = f32h_(sm[0] / 127.0f);
  const float scale = fp16f(sb);
  if (tid == 0) S[(size_t)m * nsc + chunk] = sb;
  float q = 0.f;
  if (scale > 0.f) {
    q = rintf(v / scale);
    q = fminf(fmaxf(q, -127.f), 127.f);
  }
  // k order of the GEMM fragments: within each 16-block, [0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]
  const int blk = tid >> 4, j = tid & 15;
  const int slot = (j < 8) ? (j >> 1) + ((j & 1) << 2) : 8 + ((j - 8) >> 1) + (((j - 8) & 1) << 2);
  C[(size_t)m * K + (size_t)chunk * 256 + blk * 16 + slot] = (uint8_t)((int)q & 0xFF);
}

// Host launcher. ac/as: [16][K] codes and [16][K/256] scales (from k_actq8_emit).
static inline int gemm_w4a8_launch(const uint8_t* ac, const uint16_t* as, const uint8_t* wc, const uint16_t* ws,
                                   void* C, int M, int N, int K, bool f32out, hipStream_t stream) {
  if (M < 1 || M > 16 || (N % 16) || (K % 256)) return -1;
  if (f32out) k_gemm_w4a8<true><<<dim3((unsigned)(N / 16)), NT, 0, stream>>>(ac, as, wc, ws, C, M, N, K);
  else        k_gemm_w4a8<false><<<dim3((unsigned)(N / 16)), NT, 0, stream>>>(ac, as, wc, ws, C, M, N, K);
  return (int)hipGetLastError();
}

}  // namespace w4a8
