// gemm_wdec.h -- f16 WMMA decode GEMM over the stored decode planes
// (HALO_WMMA_DEC=1). One kernel for M = 1..16 activation rows, used for the
// serial step AND the speculative verify pass, so each row's arithmetic does
// not depend on how many rows share the launch: speculative output equals
// serial output by construction, and an eight-row verify streams the weight
// bytes once at the cost of one row.
//
// Operands are f16 (v_wmma_f32_16x16x16_f16), products exact in f32:
//   activations: the bf16 rows converted once per launch to f16 (exact for
//     |a| in [2^-14, 65504]; smaller values keep 11 significant bits).
//   FMT 0 fp8r projection / FMT 2 fp8r lm_head: an e4m3 byte b becomes the
//     f16 bit pattern ((b & 0x80) << 8) | ((b & 0x7F) << 7), which is exactly
//     e4m3(b) / 256 for every code including zero and subnormals; the 256 is
//     folded into the row scale applied after the sum (FMT 2 therefore uses
//     the unrounded e4m3 * scale weight, where the lm_head GEMV rounds each
//     weight to bf16 first).
//   FMT 1 q4c v0 (NVFP4): w = lv[code] (the codebook over its unit G, exact
//     f16 levels), times G * e4m3(group scale) per 16-k step.
// Only the summation order and the lm_head weight rounding differ from the
// GEMVs: a declared numerics change, guarded by the mandatory greedy tokens.
//
// 16-bit WMMA layout on gfx1151 (gfx1151-engine tools/wmma_gemm_proto.cu):
// A(r,k): r < 8 -> lane 2r, r >= 8 -> lane 17 + 2(r-8), other lanes ignored;
// B(k,c): lanes c and 16+c; C(r,c): lane c + 16*(r >= 8), element r % 8.
// The two lanes of a B column decode k 0-7 and k 8-15 respectively and swap
// halves with v_permlanex16, so no weight byte is decoded twice.
// Block = 8 waves on 16 output columns; wave w takes the 128-wide K chunks
// w, w+8, ...; each lane half loads 64 contiguous bytes of its row per
// chunk (next chunk prefetched); partial tiles are summed in wave order.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace wdec {

using s16x16 = __attribute__((ext_vector_type(16))) short;
using f32x8 = __attribute__((ext_vector_type(8))) float;
static constexpr int NWAVES = 8, NT = NWAVES * 32, KC = 128;

__device__ __forceinline__ float bf(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ uint16_t rne(float f) {
  uint32_t u = __float_as_uint(f), lo = u & 0xFFFFu;
  u = (u >> 16) + (lo > 0x8000u) + ((lo == 0x8000u) & ((u >> 16) & 1u));
  return (uint16_t)(u & 0xFFFFu);
}
__device__ __forceinline__ float e4m3(uint32_t b) {
  uint32_t s = (b & 0x80u) << 24, e = (b >> 3) & 0xFu, m = b & 0x7u;
  float normal = __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
  float sub = __uint_as_float(s | __float_as_uint((float)m * 0.001953125f));
  return e ? normal : sub;
}
// two e4m3 bytes (bytes i, i+1 of d) -> two f16 (value / 256) in one dword
__device__ __forceinline__ uint32_t fp8x2_f16(uint32_t d, uint32_t sel) {
  const uint32_t p = __builtin_amdgcn_perm(0u, d, sel);          // [0, b_i, 0, b_i+1]
  return (p & 0x80008000u) | ((p >> 1) & 0x3F803F80u);
}
__device__ __forceinline__ uint32_t xhalf(uint32_t v) {           // value of the same lane in the other 16-lane row
  return (uint32_t)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210u, 0xfedcba98u, false, false);
}

struct LvTable { uint16_t v[16]; };   // f16 codebook levels for FMT 1

__global__ void k_rows_f16(const uint16_t* __restrict__ A, uint16_t* __restrict__ H, int M, int K) {
  const int r = blockIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  H[(size_t)r * K + k] = r < M ? __half_as_ushort(__float2half_rn(bf(A[(size_t)r * K + k]))) : (uint16_t)0;
}

template<int FMT, bool OUTF32>
__global__ void __launch_bounds__(NT)
k_wdec(const uint16_t* __restrict__ Ah, const uint8_t* __restrict__ Wp, const uint16_t* __restrict__ Ws,
       long rowstride, LvTable lv, float G, void* __restrict__ Cv, int M, int N, int K) {
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = blockIdx.x * 16;
  const int c = lane & 15, half = lane >> 4;
  int arow = -1;
  if (lane < 16 && !(lane & 1)) arow = lane >> 1;
  else if (lane >= 17 && (lane & 1)) arow = 8 + ((lane - 17) >> 1);
  const bool alive = arow >= 0 && arow < M;
  const uint16_t* ap = Ah + (size_t)(alive ? arow : 0) * K;
  const uint8_t* wrow = Wp + (size_t)(n0 + c) * rowstride;
  __shared__ uint16_t lvs[16];
  if constexpr (FMT == 1) { if (tid < 16) lvs[tid] = lv.v[tid]; __syncthreads(); }
  f32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
  const int nch = K / KC;
  // v3: lane half h owns k [k0 + 64h, k0 + 64h + 64) of each 128-wide chunk:
  // fp8 = 64 contiguous bytes (4 x 16 B), q4c = 32 bytes of codes (2 x 16 B);
  // the next chunk's bytes load while this one multiplies. Step pair
  // (s, s+4): each lane decodes its own step-(4h+s) codes, the partner's
  // decode arrives by v_permlanex16, and the two steps use them in order.
  constexpr int QW = (FMT == 1) ? 2 : 4;
  uint4 cur[QW], nxt[QW];
  uint2 gsc_cur = {0, 0}, gsc_nxt = {0, 0};
  auto load = [&](int ch, uint4* dst, uint2& gq) {
    const int k0 = ch * KC;
    if constexpr (FMT == 1) {
      const uint4* p = reinterpret_cast<const uint4*>(wrow + (k0 >> 1) + half * 32);
      dst[0] = p[0]; dst[1] = p[1];
      gq = *reinterpret_cast<const uint2*>(wrow + (K >> 1) + (k0 >> 4));
    } else {
      const uint4* p = reinterpret_cast<const uint4*>(wrow + k0 + half * 64);
      dst[0] = p[0]; dst[1] = p[1]; dst[2] = p[2]; dst[3] = p[3];
    }
  };
  auto decode = [&](const uint4* q, int s, uint32_t* d) {     // 16 values of local step s (0..3) -> 8 dwords
    if constexpr (FMT == 1) {
      const uint4 w = q[s >> 1];
      const uint32_t lo = (s & 1) ? w.z : w.x, hi = (s & 1) ? w.w : w.y;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t b0 = (lo >> (8 * j)) & 0xFFu, b1 = (hi >> (8 * j)) & 0xFFu;
        d[j] = (uint32_t)lvs[b0 & 0xF] | ((uint32_t)lvs[b0 >> 4] << 16);
        d[4 + j] = (uint32_t)lvs[b1 & 0xF] | ((uint32_t)lvs[b1 >> 4] << 16);
      }
    } else {
      const uint4 w = q[s];
      const uint32_t v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; j++) { d[2 * j] = fp8x2_f16(v[j], 0x010C000Cu); d[2 * j + 1] = fp8x2_f16(v[j], 0x030C020Cu); }
    }
  };
  auto mma = [&](int k, const uint32_t* d, float gs) {
    s16x16 a = {0};
    if (alive) {
      const uint4 a0 = *reinterpret_cast<const uint4*>(ap + k);
      const uint4 a1 = *reinterpret_cast<const uint4*>(ap + k + 8);
      const uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
      for (int j = 0; j < 8; j++) { a[2 * j] = (short)(av[j] & 0xFFFFu); a[2 * j + 1] = (short)(av[j] >> 16); }
    }
    s16x16 b;
#pragma unroll
    for (int j = 0; j < 8; j++) { b[2 * j] = (short)(d[j] & 0xFFFFu); b[2 * j + 1] = (short)(d[j] >> 16); }
    if constexpr (FMT == 1) {
      f32x8 t = {0, 0, 0, 0, 0, 0, 0, 0};
      t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, t);
#pragma unroll
      for (int e = 0; e < 8; e++) acc[e] = fmaf(t[e], gs, acc[e]);
    } else {
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
  };
  auto chunk = [&](int ch, const uint4* q, uint2 gq) {
    const int k0 = ch * KC;
#pragma unroll
    for (int s = 0; s < 4; s++) {
      uint32_t mine[8], oth[8];
      decode(q, s, mine);
#pragma unroll
      for (int j = 0; j < 8; j++) oth[j] = xhalf(mine[j]);
      float g0 = 0.f, g1 = 0.f;
      if constexpr (FMT == 1) {
        g0 = G * e4m3((gq.x >> (8 * s)) & 0xFFu);            // step s   : scale byte s
        g1 = G * e4m3((gq.y >> (8 * s)) & 0xFFu);            // step s+4 : scale byte s+4
      }
      mma(k0 + 16 * s, half ? oth : mine, g0);              // step s     (owned by half 0)
      mma(k0 + 64 + 16 * s, half ? mine : oth, g1);         // step s + 4 (owned by half 1)
    }
  };
  int ch = wave;
  if (ch < nch) load(ch, cur, gsc_cur);
  for (; ch < nch; ch += NWAVES) {
    const bool more = ch + NWAVES < nch;
    if (more) load(ch + NWAVES, nxt, gsc_nxt);
    chunk(ch, cur, gsc_cur);
    if (more) {
#pragma unroll
      for (int i = 0; i < QW; i++) cur[i] = nxt[i];
      gsc_cur = gsc_nxt;
    }
  }
  __shared__ float part[NWAVES][8][32];
#pragma unroll
  for (int e = 0; e < 8; e++) part[wave][e][lane] = acc[e];
  __syncthreads();
  if (wave == 0) {
    const int n = n0 + c;
    const float post = (FMT == 1) ? 1.f : 256.f * bf(Ws[n]);
#pragma unroll
    for (int e = 0; e < 8; e++) {
      float sum = 0.f;
#pragma unroll
      for (int w = 0; w < NWAVES; w++) sum += part[w][e][lane];
      const int m = e + 8 * half;
      if (m < M) {
        if constexpr (FMT != 1) sum *= post;
        if constexpr (OUTF32) reinterpret_cast<float*>(Cv)[(size_t)m * N + n] = sum;
        else reinterpret_cast<uint16_t*>(Cv)[(size_t)m * N + n] = rne(sum);
      }
    }
  }
}

// FMT 1 needs every codebook entry to be fl(G * l) for an NVFP4 level l in
// {0, 0.5, 1, 1.5, 2, 3, 4, 6} (either sign), with G = 2 * the smallest
// positive entry (exact: a power-of-two scaling). Returns false otherwise.
static inline bool lv_from_codebook(const float* cb, LvTable* t, float* G) {
  float unit = 0.f;
  for (int i = 0; i < 16; i++) if (cb[i] > 0.f && (unit == 0.f || cb[i] < unit)) unit = cb[i];
  if (unit == 0.f) return false;
  const float g = unit * 2.0f;
  static const float L[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  static const uint16_t H[8] = {0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600};   // f16 bits
  for (int i = 0; i < 16; i++) {
    int found = -1;
    for (int j = 0; j < 8 && found < 0; j++) {
      volatile float pos = g * L[j], neg = g * -L[j];
      if (cb[i] == pos && !(L[j] == 0.f && std::signbit(cb[i]))) found = j;
      else if (cb[i] == neg) found = 8 + j;
    }
    if (found < 0) return false;
    t->v[i] = found < 8 ? H[found] : (uint16_t)(H[found - 8] | 0x8000u);
  }
  *G = g;
  return true;
}

template<bool OUTF32>
static inline int launch(int fmt, const uint16_t* A, const uint8_t* Wp, const uint16_t* Ws, long rowstride,
                         const LvTable& lv, float G, void* C, int M, int N, int K, hipStream_t st = nullptr) {
  if (M < 1 || M > 16 || (N % 16) || (K % KC)) return -1;
  static uint16_t* Ah = nullptr;
  static size_t cap = 0;
  const size_t need = (size_t)16 * K;
  if (need > cap) {
    if (Ah) (void)hipFree(Ah);
    if (hipMalloc((void**)&Ah, need * 2) != hipSuccess) return -2;
    cap = need;
  }
  k_rows_f16<<<dim3((unsigned)((K + 255) / 256), 16), 256, 0, st>>>(A, Ah, M, K);
  const dim3 g((unsigned)(N / 16));
  if (fmt == 0) k_wdec<0, OUTF32><<<g, NT, 0, st>>>(Ah, Wp, Ws, rowstride, lv, G, C, M, N, K);
  else if (fmt == 1) k_wdec<1, OUTF32><<<g, NT, 0, st>>>(Ah, Wp, Ws, rowstride, lv, G, C, M, N, K);
  else k_wdec<2, OUTF32><<<g, NT, 0, st>>>(Ah, Wp, Ws, rowstride, lv, G, C, M, N, K);
  return (int)hipGetLastError();
}

}  // namespace wdec
