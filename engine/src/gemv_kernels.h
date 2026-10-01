// gemv_kernels.h -- decode (M == 1) GEMV kernels. k_gemv_dq and k_gemv_bf16f are
// the phase-2 kernels moved verbatim from generator.hip. k_gemv_dq_vec and
// k_gemv_bf16f_vec keep the same thread-to-strip mapping (tid*16 + 4096*i), the
// same even/odd f32 accumulators, the same per-element weight values and the
// same 256-entry LDS reduction tree, and only widen the loads to 16 bytes, so
// every output is bit-identical to the original kernel. They require K % 16 == 0
// and 16-byte-aligned row bases; the dispatcher checks both and falls back.
#pragma once
#include "dequant_common.h"
#include <hip/hip_fp16.h>

// fused dequant GEMV (decode steps, M == 1): C[n] = sum_k A[k] * dequant(W[n][k]).
// fp8r: per-thread 16-k strips, two f32 accumulators, xor-butterfly reduce, row
// scale applied once at the end (mirrors k_gemv's structure); q4c: per-element
// codebook*groupscale. f32 accumulate throughout, one bf16 round at the output.
template<int Q4C>
__global__ void k_gemv_dq(const uint16_t* __restrict__ A,
                          const uint8_t* __restrict__ Wp,
                          const uint16_t* __restrict__ Ws,
                          const float* __restrict__ CB, long rowstride,
                          uint16_t* __restrict__ C, int K) {
  int n = blockIdx.x;
  const uint8_t* row = Wp + (size_t)n * rowstride;
  int tid = threadIdx.x;
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    int ke = k0 + 16 > K ? K : k0 + 16;
    for (int k = k0; k < ke; k += 2) {
      float a0 = bf16f(A[k]), w0, a1 = 0.f, w1 = 0.f;
      if (Q4C) {
        uint8_t b0 = row[k >> 1];
        w0 = CB[b0 & 0xF] * dequant_fp8_e4m3(row[(K >> 1) + (k >> 4)]);
        if (k + 1 < K) {
          uint8_t b1 = row[(k + 1) >> 1];
          a1 = bf16f(A[k + 1]);
          w1 = CB[(b1 >> 4) & 0xF] * dequant_fp8_e4m3(row[(K >> 1) + ((k + 1) >> 4)]);
        }
      } else {
        w0 = dequant_fp8_e4m3(row[k]);
        if (k + 1 < K) { a1 = bf16f(A[k + 1]); w1 = dequant_fp8_e4m3(row[k + 1]); }
      }
      acc0 = fmaf(a0, w0, acc0);
      if (k + 1 < K) acc1 = fmaf(a1, w1, acc1);
    }
  }
  float acc = acc0 + acc1;
  if (!Q4C) acc *= bf16f(Ws[n]);
  __shared__ float sh[256];
  sh[tid] = acc;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[n] = f32bf(sh[0]);
}

// plain bf16 GEMV with f32 output (lm_head at T == 1)
__global__ void k_gemv_bf16f(const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
                             float* __restrict__ C, int K) {
  int n = blockIdx.x;
  const uint16_t* row = W + (size_t)n * K;
  int tid = threadIdx.x;
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    int ke = k0 + 16 > K ? K : k0 + 16;
    for (int k = k0; k < ke; k += 2) {
      acc0 = fmaf(bf16f(A[k]), bf16f(row[k]), acc0);
      if (k + 1 < K) acc1 = fmaf(bf16f(A[k + 1]), bf16f(row[k + 1]), acc1);
    }
  }
  float acc = acc0 + acc1;
  __shared__ float sh[256];
  sh[tid] = acc;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[n] = sh[0];
}

// ---- vectorized, bit-identical variants (phase 3) ----
typedef unsigned int gemv_u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned int gemv_u32x4 __attribute__((ext_vector_type(4)));
// NT=1 issues the weight loads as streaming (non-temporal) loads: the pool is
// read once per token and must not displace the activations from L2. The
// values and the arithmetic are unchanged.
template<int Q4C, int NT = 0>
__global__ void k_gemv_dq_vec(const uint16_t* __restrict__ A,
                              const uint8_t* __restrict__ Wp,
                              const uint16_t* __restrict__ Ws,
                              const float* __restrict__ CB, long rowstride,
                              uint16_t* __restrict__ C, int K) {
  int n = blockIdx.x;
  const uint8_t* row = Wp + (size_t)n * rowstride;
  int tid = threadIdx.x;
  // q4c: the 16-entry codebook sits in LDS for the block (same values as CB[]).
  __shared__ float cbs[16];
  if (Q4C) {
    if (tid < 16) cbs[tid] = CB[tid];
    __syncthreads();
  }
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    const uint4 av0 = *reinterpret_cast<const uint4*>(A + k0);
    const uint4 av1 = *reinterpret_cast<const uint4*>(A + k0 + 8);
    const uint32_t av[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
    if (Q4C) {
      // 16 codes = 8 bytes at row[k0/2 ..], one fp8 group scale per 16 codes.
      uint32_t cv[2];
      if (NT) { const gemv_u32x2 cw = __builtin_nontemporal_load(reinterpret_cast<const gemv_u32x2*>(row + (k0 >> 1))); cv[0] = cw.x; cv[1] = cw.y; }
      else { const uint2 cw = *reinterpret_cast<const uint2*>(row + (k0 >> 1)); cv[0] = cw.x; cv[1] = cw.y; }
      const float gs = dequant_fp8_e4m3(row[(K >> 1) + (k0 >> 4)]);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const uint32_t byte = (cv[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
        const float w0 = cbs[byte & 0xF] * gs;           // even k: low nibble
        const float w1 = cbs[(byte >> 4) & 0xF] * gs;    // odd k: high nibble
        acc0 = fmaf(bf16f((uint16_t)(av[j] & 0xFFFFu)), w0, acc0);
        acc1 = fmaf(bf16f((uint16_t)(av[j] >> 16)), w1, acc1);
      }
    } else {
      uint32_t cv[4];
      if (NT) { const gemv_u32x4 cw = __builtin_nontemporal_load(reinterpret_cast<const gemv_u32x4*>(row + k0)); cv[0] = cw.x; cv[1] = cw.y; cv[2] = cw.z; cv[3] = cw.w; }
      else { const uint4 cw = *reinterpret_cast<const uint4*>(row + k0); cv[0] = cw.x; cv[1] = cw.y; cv[2] = cw.z; cv[3] = cw.w; }
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const uint32_t pair = cv[j >> 1] >> ((j & 1) * 16);
        acc0 = fmaf(bf16f((uint16_t)(av[j] & 0xFFFFu)), fp8_e4m3_nobranch(pair & 0xFFu), acc0);
        acc1 = fmaf(bf16f((uint16_t)(av[j] >> 16)), fp8_e4m3_nobranch((pair >> 8) & 0xFFu), acc1);
      }
    }
  }
  float acc = acc0 + acc1;
  if (!Q4C) acc *= bf16f(Ws[n]);
  __shared__ float sh[256];
  sh[tid] = acc;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[n] = f32bf(sh[0]);
}

// ---- W4 decode (HALO_W4DEC): GEMV over a resident .i4l plane ----
// A is the Hadamard-rotated bf16 activation (k_had_rows: the same H/16 rotation
// k_actq_emit applies before quantizing for the W4A4 prefill, but the rotated
// activation stays bf16 here, so this is W4A16). Weight k of row n is
// code * fp16f(Ws[n][k/256]) exactly (small integer times an f16 value is exact
// in f32). Strips, even/odd accumulators and the reduction tree are those of
// k_gemv_dq_vec, so the multi-row variant (Q4C == 2 below) is bit-identical
// per row to this kernel. Not bf16-exact against the fp8r/q4c decode path: a
// declared quality configuration like W4A4-all-400.
// Same function as fp16f in generator.hip / gi4::fp16f (exact bit conversion).
__device__ __forceinline__ float gemv_fp16f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  if (exp == 0) { float d = (float)man * (1.0f / 16777216.0f); return sign ? -d : d; }
  if (exp == 31) return man ? nanf("1") : (sign ? -INFINITY : INFINITY);
  uint32_t u = sign | ((exp - 15 + 127) << 23) | (man << 13);
  return __uint_as_float(u);
}
__device__ __forceinline__ float gemv_i4_code(uint32_t nib) {
  return (float)((int)nib - (int)((nib & 8u) << 1));     // two's complement nibble
}
template<int NT = 0>
__global__ void k_gemv_i4l_vec(const uint16_t* __restrict__ A,
                               const uint8_t* __restrict__ Wc,
                               const uint16_t* __restrict__ Ws,
                               uint16_t* __restrict__ C, int K) {
  int n = blockIdx.x;
  const uint8_t* row = Wc + (size_t)n * (K >> 1);
  const uint16_t* srow = Ws + (size_t)n * (K >> 8);
  int tid = threadIdx.x;
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    const uint4 av0 = *reinterpret_cast<const uint4*>(A + k0);
    const uint4 av1 = *reinterpret_cast<const uint4*>(A + k0 + 8);
    const uint32_t av[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
    uint32_t cv[2];
    if (NT) { const gemv_u32x2 cw = __builtin_nontemporal_load(reinterpret_cast<const gemv_u32x2*>(row + (k0 >> 1))); cv[0] = cw.x; cv[1] = cw.y; }
    else { const uint2 cw = *reinterpret_cast<const uint2*>(row + (k0 >> 1)); cv[0] = cw.x; cv[1] = cw.y; }
    const float sc = gemv_fp16f(srow[k0 >> 8]);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint32_t byte = (cv[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
      const float w0 = gemv_i4_code(byte & 0xF) * sc;          // even k: low nibble
      const float w1 = gemv_i4_code((byte >> 4) & 0xF) * sc;   // odd k: high nibble
      acc0 = fmaf(bf16f((uint16_t)(av[j] & 0xFFFFu)), w0, acc0);
      acc1 = fmaf(bf16f((uint16_t)(av[j] >> 16)), w1, acc1);
    }
  }
  float acc = acc0 + acc1;
  __shared__ float sh[256];
  sh[tid] = acc;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[n] = f32bf(sh[0]);
}

// bf16 GEMV with f32 output, 16-byte loads; bit-identical to k_gemv_bf16f.
__global__ void k_gemv_bf16f_vec(const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
                                 float* __restrict__ C, int K) {
  int n = blockIdx.x;
  const uint16_t* row = W + (size_t)n * K;
  int tid = threadIdx.x;
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    const uint4 av0 = *reinterpret_cast<const uint4*>(A + k0);
    const uint4 av1 = *reinterpret_cast<const uint4*>(A + k0 + 8);
    const uint4 wv0 = *reinterpret_cast<const uint4*>(row + k0);
    const uint4 wv1 = *reinterpret_cast<const uint4*>(row + k0 + 8);
    const uint32_t av[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
    const uint32_t wv[8] = {wv0.x, wv0.y, wv0.z, wv0.w, wv1.x, wv1.y, wv1.z, wv1.w};
#pragma unroll
    for (int j = 0; j < 8; j++) {
      acc0 = fmaf(bf16f((uint16_t)(av[j] & 0xFFFFu)), bf16f((uint16_t)(wv[j] & 0xFFFFu)), acc0);
      acc1 = fmaf(bf16f((uint16_t)(av[j] >> 16)), bf16f((uint16_t)(wv[j] >> 16)), acc1);
    }
  }
  float acc = acc0 + acc1;
  __shared__ float sh[256];
  sh[tid] = acc;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[n] = sh[0];
}

// lm_head straight from the stored fp8r plane, bit-identical to k_gemv_bf16f over
// the init-time bf16 copy: each weight is rounded to bf16 exactly as
// k_dequant_fp8r does (f32bf(fp8 * bf16(scale))) before the same fmaf chain.
__global__ void k_gemv_lmhead_fp8r_vec(const uint16_t* __restrict__ A, const uint8_t* __restrict__ Wc,
                                       const uint16_t* __restrict__ Ws, float* __restrict__ C, int K) {
  int n = blockIdx.x;
  const uint8_t* row = Wc + (size_t)n * K;
  const float scale = bf16f(Ws[n]);
  int tid = threadIdx.x;
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    const uint4 av0 = *reinterpret_cast<const uint4*>(A + k0);
    const uint4 av1 = *reinterpret_cast<const uint4*>(A + k0 + 8);
    const uint4 cw = *reinterpret_cast<const uint4*>(row + k0);
    const uint32_t av[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
    const uint32_t cv[4] = {cw.x, cw.y, cw.z, cw.w};
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint32_t pair = cv[j >> 1] >> ((j & 1) * 16);
      const float w0 = bf16f(f32bf(fp8_e4m3_nobranch(pair & 0xFFu) * scale));
      const float w1 = bf16f(f32bf(fp8_e4m3_nobranch((pair >> 8) & 0xFFu) * scale));
      acc0 = fmaf(bf16f((uint16_t)(av[j] & 0xFFFFu)), w0, acc0);
      acc1 = fmaf(bf16f((uint16_t)(av[j] >> 16)), w1, acc1);
    }
  }
  float acc = acc0 + acc1;
  __shared__ float sh[256];
  sh[tid] = acc;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[n] = sh[0];
}

// bf16 weights [N][K] x bf16 activation -> bf16 output (the 48-row GDN a/b gate
// projections at T == 1, replacing two hipBLASLt M=1 calls per layer). f32
// accumulation in the k_gemv_bf16f_vec order, one bf16 RNE round at the end.
// Not bit-identical to hipBLASLt's own accumulation order; a declared change
// guarded by the token checks.
__global__ void k_gemv_bf16_bf16out_vec(const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
                                        uint16_t* __restrict__ C, int K) {
  int n = blockIdx.x;
  const uint16_t* row = W + (size_t)n * K;
  int tid = threadIdx.x;
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    const uint4 av0 = *reinterpret_cast<const uint4*>(A + k0);
    const uint4 av1 = *reinterpret_cast<const uint4*>(A + k0 + 8);
    const uint4 wv0 = *reinterpret_cast<const uint4*>(row + k0);
    const uint4 wv1 = *reinterpret_cast<const uint4*>(row + k0 + 8);
    const uint32_t av[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
    const uint32_t wv[8] = {wv0.x, wv0.y, wv0.z, wv0.w, wv1.x, wv1.y, wv1.z, wv1.w};
#pragma unroll
    for (int j = 0; j < 8; j++) {
      acc0 = fmaf(bf16f((uint16_t)(av[j] & 0xFFFFu)), bf16f((uint16_t)(wv[j] & 0xFFFFu)), acc0);
      acc1 = fmaf(bf16f((uint16_t)(av[j] >> 16)), bf16f((uint16_t)(wv[j] >> 16)), acc1);
    }
  }
  float acc = acc0 + acc1;
  __shared__ float sh[256];
  sh[tid] = acc;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[n] = f32bf(sh[0]);
}

// P activation rows through one pass over a bf16 weight matrix; row p is
// bit-identical to k_gemv_bf16_bf16out_vec on that row (same strips, even/odd
// accumulators and tree). A rows are K apart, C rows are N apart.
template<int P>
__global__ void k_gemv_bf16_bf16out_rows(const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
                                         uint16_t* __restrict__ C, int K, int N) {
  int n = blockIdx.x;
  const uint16_t* row = W + (size_t)n * K;
  int tid = threadIdx.x;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) { acc0[p] = 0.f; acc1[p] = 0.f; }
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    const uint4 wv0 = *reinterpret_cast<const uint4*>(row + k0);
    const uint4 wv1 = *reinterpret_cast<const uint4*>(row + k0 + 8);
    const uint32_t wv[8] = {wv0.x, wv0.y, wv0.z, wv0.w, wv1.x, wv1.y, wv1.z, wv1.w};
#pragma unroll
    for (int p = 0; p < P; p++) {
      const uint16_t* Ap = A + (size_t)p * K;
      const uint4 av0 = *reinterpret_cast<const uint4*>(Ap + k0);
      const uint4 av1 = *reinterpret_cast<const uint4*>(Ap + k0 + 8);
      const uint32_t av[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
#pragma unroll
      for (int j = 0; j < 8; j++) {
        acc0[p] = fmaf(bf16f((uint16_t)(av[j] & 0xFFFFu)), bf16f((uint16_t)(wv[j] & 0xFFFFu)), acc0[p]);
        acc1[p] = fmaf(bf16f((uint16_t)(av[j] >> 16)), bf16f((uint16_t)(wv[j] >> 16)), acc1[p]);
      }
    }
  }
  __shared__ float sh[P][256];
#pragma unroll
  for (int p = 0; p < P; p++) sh[p][tid] = acc0[p] + acc1[p];
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) {
#pragma unroll
      for (int p = 0; p < P; p++) sh[p][tid] += sh[p][tid + off];
    }
    __syncthreads();
  }
  if (tid < P) C[(size_t)tid * N + n] = f32bf(sh[tid][0]);
}

// ---- multi-row (speculative verify) variants ----
// P activation rows share one pass over the weights. For every row the thread
// mapping, the per-element weight value, the even/odd accumulators and the
// reduction tree are exactly those of the P = 1 kernel above, so row r's output
// is bit-identical to a separate single-row call. Rows are P consecutive
// activation vectors of length K (stride K) and P output vectors (stride N).
template<int Q4C, int P, int NT = 0, int R = 4, bool OUTF32 = false>
__global__ void __launch_bounds__(256) k_gemv_dq_vec_mr(const uint16_t* __restrict__ A,
                                 const uint8_t* __restrict__ Wp,
                                 const uint16_t* __restrict__ Ws,
                                 const float* __restrict__ CB, long rowstride,
                                 uint16_t* __restrict__ C, int K, int N) {
  // R output rows per block: activations are loaded once per strip and reused
  // for R weight rows. Output row n still uses thread tid's strips
  // (tid*16 + 4096*i), its own even/odd accumulators and its own tree.
  // Q4C: 0 = fp8r plane (Ws = bf16 row scales), 1 = q4c plane (CB codebook,
  // fp8 group scales in the row), 2 = i4l plane (Ws = f16 scales [N][K/256],
  // rowstride = K/2; per row bit-identical to k_gemv_i4l_vec), 3 = q4c
  // variant 1 (NF4 codebook, f16 scale per 32 codes in the row, rowstride
  // K/2 + K/16; per row bit-identical to k_gemv_q4c1_vec). OUTF32 writes
  // f32 outputs (C reinterpreted as float*).
  const int n0 = blockIdx.x * R;
  int tid = threadIdx.x;
  __shared__ float cbs[16];
  if (Q4C == 1 || Q4C == 3) {
    if (tid < 16) cbs[tid] = CB[tid];
    __syncthreads();
  }
  float acc0[R][P], acc1[R][P];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int p = 0; p < P; p++) { acc0[r][p] = 0.f; acc1[r][p] = 0.f; }
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    uint32_t av[P][8];
#pragma unroll
    for (int p = 0; p < P; p++) {
      const uint16_t* Ap = A + (size_t)p * K;
      const uint4 a0 = *reinterpret_cast<const uint4*>(Ap + k0);
      const uint4 a1 = *reinterpret_cast<const uint4*>(Ap + k0 + 8);
      av[p][0] = a0.x; av[p][1] = a0.y; av[p][2] = a0.z; av[p][3] = a0.w;
      av[p][4] = a1.x; av[p][5] = a1.y; av[p][6] = a1.z; av[p][7] = a1.w;
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
      const uint8_t* row = Wp + (size_t)(n0 + r) * rowstride;
      float w0v[8], w1v[8];
      if constexpr (Q4C == 1) {
        uint32_t cv[2];
        if (NT) { const gemv_u32x2 cw = __builtin_nontemporal_load(reinterpret_cast<const gemv_u32x2*>(row + (k0 >> 1))); cv[0] = cw.x; cv[1] = cw.y; }
        else { const uint2 cw = *reinterpret_cast<const uint2*>(row + (k0 >> 1)); cv[0] = cw.x; cv[1] = cw.y; }
        const float gs = dequant_fp8_e4m3(row[(K >> 1) + (k0 >> 4)]);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const uint32_t byte = (cv[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
          w0v[j] = cbs[byte & 0xF] * gs;
          w1v[j] = cbs[(byte >> 4) & 0xF] * gs;
        }
      } else if constexpr (Q4C == 3) {
        uint32_t cv[2];
        if (NT) { const gemv_u32x2 cw = __builtin_nontemporal_load(reinterpret_cast<const gemv_u32x2*>(row + (k0 >> 1))); cv[0] = cw.x; cv[1] = cw.y; }
        else { const uint2 cw = *reinterpret_cast<const uint2*>(row + (k0 >> 1)); cv[0] = cw.x; cv[1] = cw.y; }
        const float gs = __half2float(__ushort_as_half(reinterpret_cast<const uint16_t*>(row + (K >> 1))[k0 >> 5]));
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const uint32_t byte = (cv[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
          w0v[j] = cbs[byte & 0xF] * gs;
          w1v[j] = cbs[(byte >> 4) & 0xF] * gs;
        }
      } else if constexpr (Q4C == 2) {
        uint32_t cv[2];
        if (NT) { const gemv_u32x2 cw = __builtin_nontemporal_load(reinterpret_cast<const gemv_u32x2*>(row + (k0 >> 1))); cv[0] = cw.x; cv[1] = cw.y; }
        else { const uint2 cw = *reinterpret_cast<const uint2*>(row + (k0 >> 1)); cv[0] = cw.x; cv[1] = cw.y; }
        const float sc = gemv_fp16f(Ws[(size_t)(n0 + r) * (K >> 8) + (k0 >> 8)]);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const uint32_t byte = (cv[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
          w0v[j] = gemv_i4_code(byte & 0xF) * sc;
          w1v[j] = gemv_i4_code((byte >> 4) & 0xF) * sc;
        }
      } else {
        uint32_t cv[4];
        if (NT) { const gemv_u32x4 cw = __builtin_nontemporal_load(reinterpret_cast<const gemv_u32x4*>(row + k0)); cv[0] = cw.x; cv[1] = cw.y; cv[2] = cw.z; cv[3] = cw.w; }
        else { const uint4 cw = *reinterpret_cast<const uint4*>(row + k0); cv[0] = cw.x; cv[1] = cw.y; cv[2] = cw.z; cv[3] = cw.w; }
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const uint32_t pair = cv[j >> 1] >> ((j & 1) * 16);
          w0v[j] = fp8_e4m3_nobranch(pair & 0xFFu);
          w1v[j] = fp8_e4m3_nobranch((pair >> 8) & 0xFFu);
        }
      }
#pragma unroll
      for (int p = 0; p < P; p++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
          acc0[r][p] = fmaf(bf16f((uint16_t)(av[p][j] & 0xFFFFu)), w0v[j], acc0[r][p]);
          acc1[r][p] = fmaf(bf16f((uint16_t)(av[p][j] >> 16)), w1v[j], acc1[r][p]);
        }
      }
    }
  }
  __shared__ float sh[R * P][256];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int p = 0; p < P; p++) {
      float acc = acc0[r][p] + acc1[r][p];
      if constexpr (Q4C == 0) acc *= bf16f(Ws[n0 + r]);
      sh[r * P + p][tid] = acc;
    }
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) {
#pragma unroll
      for (int q = 0; q < R * P; q++) sh[q][tid] += sh[q][tid + off];
    }
    __syncthreads();
  }
  if (tid < R * P) {
    const int r = tid / P, p = tid % P;
    if constexpr (OUTF32) reinterpret_cast<float*>(C)[(size_t)p * N + n0 + r] = sh[tid][0];
    else C[(size_t)p * N + n0 + r] = f32bf(sh[tid][0]);
  }
}

// multi-row lm_head over the fp8r plane; row p bit-identical to k_gemv_lmhead_fp8r_vec
template<int P>
__global__ void k_gemv_lmhead_fp8r_vec_mr(const uint16_t* __restrict__ A, const uint8_t* __restrict__ Wc,
                                          const uint16_t* __restrict__ Ws, float* __restrict__ C, int K, int N) {
  int n = blockIdx.x;
  const uint8_t* row = Wc + (size_t)n * K;
  const float scale = bf16f(Ws[n]);
  int tid = threadIdx.x;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) { acc0[p] = 0.f; acc1[p] = 0.f; }
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    const uint4 cw = *reinterpret_cast<const uint4*>(row + k0);
    const uint32_t cv[4] = {cw.x, cw.y, cw.z, cw.w};
    float w0v[8], w1v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint32_t pair = cv[j >> 1] >> ((j & 1) * 16);
      w0v[j] = bf16f(f32bf(fp8_e4m3_nobranch(pair & 0xFFu) * scale));
      w1v[j] = bf16f(f32bf(fp8_e4m3_nobranch((pair >> 8) & 0xFFu) * scale));
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      const uint16_t* Ap = A + (size_t)p * K;
      const uint4 av0 = *reinterpret_cast<const uint4*>(Ap + k0);
      const uint4 av1 = *reinterpret_cast<const uint4*>(Ap + k0 + 8);
      const uint32_t av[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
#pragma unroll
      for (int j = 0; j < 8; j++) {
        acc0[p] = fmaf(bf16f((uint16_t)(av[j] & 0xFFFFu)), w0v[j], acc0[p]);
        acc1[p] = fmaf(bf16f((uint16_t)(av[j] >> 16)), w1v[j], acc1[p]);
      }
    }
  }
  __shared__ float sh[P][256];
#pragma unroll
  for (int p = 0; p < P; p++) sh[p][tid] = acc0[p] + acc1[p];
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) {
#pragma unroll
      for (int p = 0; p < P; p++) sh[p][tid] += sh[p][tid + off];
    }
    __syncthreads();
  }
  if (tid < P) C[(size_t)tid * N + n] = sh[tid][0];
}

// lm_head rows selected by an index map (draft vocabulary): C[i] = row map[i]
__global__ void k_gemv_lmhead_fp8r_vec_map(const uint16_t* __restrict__ A, const uint8_t* __restrict__ Wc,
                                           const uint16_t* __restrict__ Ws, const int* __restrict__ map,
                                           float* __restrict__ C, int K) {
  const int n = map[blockIdx.x];
  const uint8_t* row = Wc + (size_t)n * K;
  const float scale = bf16f(Ws[n]);
  int tid = threadIdx.x;
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    const uint4 av0 = *reinterpret_cast<const uint4*>(A + k0);
    const uint4 av1 = *reinterpret_cast<const uint4*>(A + k0 + 8);
    const uint4 cw = *reinterpret_cast<const uint4*>(row + k0);
    const uint32_t av[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
    const uint32_t cv[4] = {cw.x, cw.y, cw.z, cw.w};
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint32_t pair = cv[j >> 1] >> ((j & 1) * 16);
      acc0 = fmaf(bf16f((uint16_t)(av[j] & 0xFFFFu)), bf16f(f32bf(fp8_e4m3_nobranch(pair & 0xFFu) * scale)), acc0);
      acc1 = fmaf(bf16f((uint16_t)(av[j] >> 16)), bf16f(f32bf(fp8_e4m3_nobranch((pair >> 8) & 0xFFu) * scale)), acc1);
    }
  }
  __shared__ float sh[256];
  sh[tid] = acc0 + acc1;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[blockIdx.x] = sh[0];
}
// argmax over V logits plus the softmax probability of the winner (drafter
// confidence). Writes the token id (through map if given) and the probability.
__global__ void k_argmax_prob(const float* __restrict__ L, int V, const int* __restrict__ map,
                              int* __restrict__ pred, float* __restrict__ prob) {
  __shared__ float smx[256]; __shared__ int smi[256]; __shared__ float ssum[256];
  int tid = threadIdx.x;
  float mx = -3.0e38f; int mi = 0;
  for (int v = tid; v < V; v += 256) { float x = L[v]; if (x > mx) { mx = x; mi = v; } }
  smx[tid] = mx; smi[tid] = mi;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off && smx[tid + off] > smx[tid]) { smx[tid] = smx[tid + off]; smi[tid] = smi[tid + off]; }
    __syncthreads();
  }
  const float M = smx[0];
  float s = 0.f;
  for (int v = tid; v < V; v += 256) s += __expf(L[v] - M);
  ssum[tid] = s;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) ssum[tid] += ssum[tid + off];
    __syncthreads();
  }
  if (tid == 0) { pred[0] = map ? map[smi[0]] : smi[0]; prob[0] = 1.0f / ssum[0]; }
}

// q4c variant 1 GEMV (NF4 codebook, one f16 scale per 32 codes, row-interleaved
// [K/2 codes | K/32 f16 scales]). Drafter weights only.
__global__ void k_gemv_q4c1_vec(const uint16_t* __restrict__ A, const uint8_t* __restrict__ Wp,
                                const float* __restrict__ CB, long rowstride,
                                uint16_t* __restrict__ C, int K) {
  int n = blockIdx.x;
  const uint8_t* row = Wp + (size_t)n * rowstride;
  const uint16_t* sc = reinterpret_cast<const uint16_t*>(row + (K >> 1));
  int tid = threadIdx.x;
  __shared__ float cbs[16];
  if (tid < 16) cbs[tid] = CB[tid];
  __syncthreads();
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    const uint4 av0 = *reinterpret_cast<const uint4*>(A + k0);
    const uint4 av1 = *reinterpret_cast<const uint4*>(A + k0 + 8);
    const uint32_t av[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
    const uint2 cw = *reinterpret_cast<const uint2*>(row + (k0 >> 1));
    const uint32_t cv[2] = {cw.x, cw.y};
    const float gs = __half2float(__ushort_as_half(sc[k0 >> 5]));
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint32_t byte = (cv[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
      acc0 = fmaf(bf16f((uint16_t)(av[j] & 0xFFFFu)), cbs[byte & 0xF] * gs, acc0);
      acc1 = fmaf(bf16f((uint16_t)(av[j] >> 16)), cbs[(byte >> 4) & 0xF] * gs, acc1);
    }
  }
  __shared__ float sh[256];
  sh[tid] = acc0 + acc1;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[n] = f32bf(sh[0]);
}
