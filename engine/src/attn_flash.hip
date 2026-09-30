// Qwen3.8-27B causal attention: 24 Q heads, 4 KV heads, head dimension 256.
// Explicit experimental alternative to the global-bf16 attn_tiled reference.
// Q/K/V are converted from bf16 to f16. Online fp32 softmax rescales each
// tile's fp32 accumulator; unnormalized probabilities round to f16 for PV.
// This changes the reference's global normalization / bf16 rounding order.
// gfx1151 WMMA fragment layout follows the instruction recipes in
// gfx1151-engine/src/gpu/vit.cpp, adapted to this model and causal KV offsets.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstring>

using flash_h16 = __attribute__((ext_vector_type(16))) short;
using flash_f8 = __attribute__((ext_vector_type(8))) float;
static constexpr int F_QH = 24, F_KVH = 4, F_HD = 256;
static constexpr int F_QT = 64, F_KT = 32, F_STR = 264, F_PSTR = 40;

__device__ __forceinline__ short flash_half(float f) {
  return (short)__half_as_ushort(__float2half_rn(f));
}
__device__ __forceinline__ float flash_bfloat(uint16_t h) {
  return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t flash_out(float f) {
  uint32_t u = __float_as_uint(f), hi = u >> 16, lo = u & 65535u;
  return (uint16_t)(hi + (lo > 0x8000u) + ((lo == 0x8000u) & (hi & 1u)));
}
__device__ __forceinline__ flash_h16 flash_load(const short* p) {
  flash_h16 r;
  memcpy(&r, p, sizeof(r));
  return r;
}

// Two waves share each 16-query group and split the 256 output columns.
// K and V reuse the same LDS array after scores are consumed. 55,808 B LDS.
__global__ void __launch_bounds__(256)
k_attention_flash_f16(const uint16_t* __restrict__ Q,
                      const uint16_t* __restrict__ K,
                      const uint16_t* __restrict__ V,
                      uint16_t* __restrict__ O, int T, int pos_base) {
  __shared__ short qs[F_QT][F_STR];
  __shared__ short kvs[F_KT][F_STR];
  __shared__ short probs[F_QT][F_PSTR];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int qgroup = wave >> 1, dgroup = wave & 1;
  const int col = lane & 15, halfrow = lane >> 4;
  const int q0 = blockIdx.x * F_QT, h = blockIdx.y, kh = h / 6;
  int arow = -1;
  if (lane < 16 && !(lane & 1)) arow = lane / 2;
  else if (lane >= 17 && (lane & 1)) arow = 8 + (lane - 17) / 2;
  for (int i = tid; i < F_QT * F_HD; i += 256) {
    int r = i / F_HD, d = i % F_HD;
    qs[r][d] = q0 + r < T
        ? flash_half(flash_bfloat(Q[((long)(q0 + r) * F_QH + h) * F_HD + d])) : 0;
  }
  __syncthreads();
  flash_f8 accum[8];
  float mxrun[8], sumrun[8];
#pragma unroll
  for (int d = 0; d < 8; d++) accum[d] = {0};
#pragma unroll
  for (int e = 0; e < 8; e++) { mxrun[e] = -INFINITY; sumrun[e] = 0; }
  const int last_query = (q0 + F_QT < T ? q0 + F_QT : T) - 1;
  const int nkeys = pos_base + last_query + 1;
  for (int k0 = 0; k0 < nkeys; k0 += F_KT) {
    __syncthreads();
    for (int i = tid; i < F_KT * F_HD; i += 256) {
      int r = i / F_HD, d = i % F_HD, key = k0 + r;
      kvs[r][d] = key < nkeys
          ? flash_half(flash_bfloat(K[((long)key * F_KVH + kh) * F_HD + d])) : 0;
    }
    __syncthreads();
    flash_f8 scores[2] = {{0}, {0}};
#pragma unroll
    for (int dk = 0; dk < 16; dk++) {
      flash_h16 a = {0};
      if (arow >= 0) a = flash_load(&qs[qgroup * 16 + arow][dk * 16]);
#pragma unroll
      for (int f = 0; f < 2; f++) {
        flash_h16 b;
#pragma unroll
        for (int k = 0; k < 16; k++) b[k] = kvs[f * 16 + col][dk * 16 + k];
        scores[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, scores[f]);
      }
    }
#pragma unroll
    for (int e = 0; e < 8; e++) {
      int query = q0 + qgroup * 16 + halfrow * 8 + e;
      float m = -INFINITY;
#pragma unroll
      for (int f = 0; f < 2; f++) {
        int key = k0 + f * 16 + col;
        float s = query < T && key <= pos_base + query
            ? scores[f][e] * 0.0625f : -INFINITY;
        scores[f][e] = s;
        m = fmaxf(m, s);
      }
#pragma unroll
      for (int off = 1; off <= 8; off <<= 1) m = fmaxf(m, __shfl_xor(m, off));
      float mnew = fmaxf(mxrun[e], m);
      float correction = isfinite(mnew) ? expf(mxrun[e] - mnew) : 0.f;
      float es = 0.f;
#pragma unroll
      for (int f = 0; f < 2; f++) {
        float p = isfinite(scores[f][e]) ? expf(scores[f][e] - mnew) : 0.f;
        scores[f][e] = p;
        es += p;
        if (dgroup == 0)
          probs[qgroup * 16 + halfrow * 8 + e][f * 16 + col] = flash_half(p);
      }
#pragma unroll
      for (int off = 1; off <= 8; off <<= 1) es += __shfl_xor(es, off);
      sumrun[e] = sumrun[e] * correction + es;
      mxrun[e] = mnew;
#pragma unroll
      for (int d = 0; d < 8; d++) accum[d][e] *= correction;
    }
    __syncthreads();
    for (int i = tid; i < F_KT * F_HD; i += 256) {
      int r = i / F_HD, d = i % F_HD, key = k0 + r;
      kvs[r][d] = key < nkeys
          ? flash_half(flash_bfloat(V[((long)key * F_KVH + kh) * F_HD + d])) : 0;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      flash_h16 a = {0};
      if (arow >= 0) a = flash_load(&probs[qgroup * 16 + arow][kk * 16]);
#pragma unroll
      for (int d = 0; d < 8; d++) {
        flash_h16 b;
#pragma unroll
        for (int k = 0; k < 16; k++)
          b[k] = kvs[kk * 16 + k][dgroup * 128 + d * 16 + col];
        accum[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, accum[d]);
      }
    }
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    int query = q0 + qgroup * 16 + halfrow * 8 + e;
    if (query < T) {
#pragma unroll
      for (int d = 0; d < 8; d++)
        O[((long)query * F_QH + h) * F_HD + dgroup * 128 + d * 16 + col] =
            flash_out(accum[d][e] / sumrun[e]);
    }
  }
}

extern "C" int chlorine_attention_flash(const uint16_t* Q, const uint16_t* K,
                                         const uint16_t* V, uint16_t* O,
                                         int T, int pos_base) {
  if (T < 1 || pos_base < 0) return -1;
  k_attention_flash_f16<<<dim3((T + F_QT - 1) / F_QT, F_QH), 256>>>(Q, K, V, O, T, pos_base);
  hipError_t rc = hipGetLastError();
  if (rc == hipSuccess) rc = hipDeviceSynchronize();
  return (int)rc;
}
