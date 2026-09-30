// gemm_i4.h -- native int4 x int4 WMMA GEMM for gfx1151 (RDNA 3.5, wave32).
//
//   C[m][n] (bf16) = f32bf( sum_{c=0}^{K/256-1} (float)dot_c(m,n) * (fp16f(as[m][c]) * fp16f(ws[n][c])) )
//
// dot_c is the exact int32 dot over the 256 signed 4-bit codes of chunk c, the
// two f16 scales are converted to f32 and multiplied first (one rounding), the
// product is multiplied into the dot and added to the f32 accumulator as ONE
// fma (chunks in ascending order), one bf16 round at the end. This is what
// k_gemm_i4emul in generator.hip computes: its `acc += (float)dot * (sa * sw)`
// compiles to v_mul_f32 + v_fmac_f32 under the engine's default -ffp-contract,
// so the kernel below uses an explicit fmaf and is bit-identical to it
// (validated by variant-27b/bench/gemm_i4_check.hip).
//
// Inputs (row-major, no padding):
//   ac [M][K/2]    activation codes, two's-complement nibbles, low nibble = even k
//   as [M][K/256]  f16 activation scales
//   wc [N][K/2]    weight codes (the checkpoint's .i4l code plane)
//   ws [N][K/256]  f16 weight scales (the .i4l scale plane)
//   C  [M][N]      bf16 output
// K % 256 == 0 and N % 256 == 0 are required (host-checked). Any M works: A
// rows past M-1 are clamped while staging (duplicate compute, discarded) and
// the epilogue is predicated.
//
// Kernel design (k_gemm_i4, one 16x16x16 v_wmma_i32_16x16x16_iu4 per fragment
// pair, measured fragment layout from variant-27b/results/phase3a-iu4-probe.json):
//   - A fragment for output rows r0..r0+15: lane l supplies the 8 code bytes of
//     row r0 + (l & 15) (lanes 16-31 replicate lanes 0-15); B fragment for
//     output columns n0..n0+15: lane l supplies the 8 code bytes of weight row
//     n0 + (l & 15). Both are plain 8-byte loads from the [rows][K/2] planes:
//     nibble k sits in bits 4k..4k+3 of the two dwords, which is the packing
//     order of both planes. C/D: lane l holds column l & 15, element e holds
//     row 2e + (l >> 4).
//   - Block tile BM x BN, NT threads = MW x PW waves, warp tile (BM/MW) x
//     (BN/PW) fragments. K is stepped by 128 (half a scale chunk): the LDS
//     double buffer holds two K-steps ([rows][64 code bytes] per operand), the
//     int32 accumulators run across the two K-steps of a chunk (16 WMMAs per
//     tile), then each tile is converted: acc = fmaf((float)iacc, sa * sw, acc).
//     Per-chunk scales are staged as f32 into LDS (the fp16f conversion is done
//     once per (row, chunk) by the staging thread), row-permuted so a lane's
//     8 rows are one contiguous 8-float run.
//   - LDS rows are either padded to 72 B (SWZ = 0) or kept at 64 B with the
//     8-byte units XOR-swizzled by (row >> 1) & 7 (SWZ = 1); both make the
//     16-lane ds_load_b64 fragment reads bank-conflict-free.
//   - Unconditional global -> register -> LDS staging one K-step ahead (tail
//     index clamped, as in gfx1151-engine's k_gemm_wmma; conditional staging
//     makes the allocator spill), grouped-M block swizzle for L2 reuse, and a
//     custom barrier without the buffer_gl0_inv that __syncthreads emits.
//   - PIPE (see GI4_COMPUTE) fences the next K-step's global loads ahead of the
//     compute phase and software-pipelines the LDS fragment loads.
// Measured (variant-27b/results/phase3-gemm-i4-check.json, isolated kernel):
// the default config (variant 10: 128x256, 512 threads, PIPE 2, 211 VGPRs, 0
// scratch, 52,224 B LDS) runs 63-67 TOPS at M=4096/32768 for (N,K) =
// (17408,5120) and (5120,17408), ~1.6-2.4x hipBLASLt bf16 on the same shapes.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

namespace gi4 {

typedef int   i32x2 __attribute__((ext_vector_type(2)));
typedef int   i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

// Same function as fp16f in generator.hip (bit manipulation, exact).
__device__ __forceinline__ float fp16f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  if (exp == 0) { float d = (float)man * (1.0f / 16777216.0f); return sign ? -d : d; }
  if (exp == 31) return man ? nanf("1") : (sign ? -INFINITY : INFINITY);
  uint32_t u = sign | ((exp - 15 + 127) << 23) | (man << 13);
  return __uint_as_float(u);
}
// Same function as f32bf in dequant_common.h (round to nearest even).
__device__ __forceinline__ uint16_t f32bf(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lo = u & 0xFFFFu;
  u = (u >> 16) + (lo > 0x8000u) + ((lo == 0x8000u) & ((u >> 16) & 1u));
  return (uint16_t)(u & 0xFFFFu);
}
// Workgroup barrier for LDS-staged operands: drain LDS and store counters,
// then s_barrier. __syncthreads() would also emit buffer_gl0_inv on gfx11,
// draining the read-only global loads in flight every K-step.
__device__ __forceinline__ void lds_sync() {
  __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_waitcnt_vscnt null, 0x0\n\ts_barrier" ::: "memory");
}

template <int BM, int BN, int MW, int PW, int NT, int SWZ, int OCC = 1, int PIPE = 0>
__global__ void __launch_bounds__(NT) __attribute__((amdgpu_waves_per_eu(OCC)))
k_gemm_i4(const uint8_t* __restrict__ ac, const uint16_t* __restrict__ as,
          const uint8_t* __restrict__ wc, const uint16_t* __restrict__ ws,
          uint16_t* __restrict__ C, int M, int N, int K, int gm) {
  constexpr int NW = NT / 32;
  constexpr int WM = BM / (16 * MW), WP = BN / (16 * PW);
  static_assert(MW * PW == NW, "warp grid mismatch");
  static_assert(WM >= 1 && WP >= 1 && BM == MW * WM * 16 && BN == PW * WP * 16, "tile mismatch");
  constexpr int PITCH = SWZ ? 64 : 72;                 // LDS bytes per row per K-step
  constexpr int ASZ = BM * PITCH, BSZ = BN * PITCH, BUF = ASZ + BSZ;
  constexpr int AN = BM * 4 / NT, BNU = BN * 4 / NT;   // 16-byte staging units per thread
  static_assert(BM * 4 % NT == 0 && BN * 4 % NT == 0, "staging mismatch");
  constexpr int NS = (BM + BN + NT - 1) / NT;          // scale values per thread per chunk
  static_assert(2 * BUF + 2 * (BM + BN) * 4 <= 65536, "LDS budget");

  __shared__ __align__(16) uint8_t lds[2 * BUF];
  __shared__ __align__(16) float Sa[2][BM];
  __shared__ __align__(16) float Sw[2][BN];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int l16 = lane & 15, hh = lane >> 4;
  const int KB = K >> 1;          // code bytes per row
  const int NC = K >> 8;          // chunks per row (scales per row)

  // grouped-M block swizzle: gm consecutive M-blocks share one N-block
  const int num_m = (M + BM - 1) / BM, num_n = N / BN;
  const int bid = blockIdx.x;
  const int per_group = gm * num_n;
  const int gid = bid / per_group, rem = bid - gid * per_group;
  const int first_m = gid * gm;
  const int gs = min(gm, num_m - first_m);
  const int bm = first_m + rem % gs, bn = rem / gs;
  const int m0 = bm * BM, n0 = bn * BN;
  const int wm0 = (w / PW) * (WM * 16), wp0 = (w % PW) * (WP * 16);

  // fragment read offsets within a tile: row (l & 15), 8-byte unit kh
  uint32_t foff[8];
#pragma unroll
  for (int kh = 0; kh < 8; ++kh)
    foff[kh] = SWZ ? (uint32_t)(l16 * 64 + 8 * (kh ^ ((l16 >> 1) & 7)))
                   : (uint32_t)(l16 * PITCH + 8 * kh);

  // staging: unit idx = tid + j*NT covers row idx>>2, 16-byte quarter idx&3
  const uint8_t* ga[AN];
  const uint8_t* gb[BNU];
  uint32_t sa_off[AN], sb_off[BNU];
  int sa_swap[AN], sb_swap[BNU];
#pragma unroll
  for (int j = 0; j < AN; ++j) {
    const int idx = tid + j * NT, row = idx >> 2, q = idx & 3, s = (row >> 1) & 7;
    ga[j] = ac + (size_t)min(m0 + row, M - 1) * KB + q * 16;
    sa_off[j] = SWZ ? (uint32_t)(row * 64 + 16 * (q ^ (s >> 1))) : (uint32_t)(row * PITCH + q * 16);
    sa_swap[j] = SWZ ? (s & 1) : 0;
  }
#pragma unroll
  for (int j = 0; j < BNU; ++j) {
    const int idx = tid + j * NT, row = idx >> 2, q = idx & 3, s = (row >> 1) & 7;
    gb[j] = wc + (size_t)min(n0 + row, N - 1) * KB + q * 16;
    sb_off[j] = ASZ + (SWZ ? (uint32_t)(row * 64 + 16 * (q ^ (s >> 1))) : (uint32_t)(row * PITCH + q * 16));
    sb_swap[j] = SWZ ? (s & 1) : 0;
  }
  // scale staging: value idx = tid + j*NT is A row idx (idx < BM) or W row idx-BM
  const uint16_t* gsc[NS];
  int sc_pos[NS];
#pragma unroll
  for (int j = 0; j < NS; ++j) {
    const int idx = tid + j * NT;
    if (idx < BM) {
      gsc[j] = as + (size_t)min(m0 + idx, M - 1) * NC;
      sc_pos[j] = (idx & ~15) | ((idx & 1) << 3) | ((idx >> 1) & 7);   // lane-run permutation
    } else if (idx < BM + BN) {
      gsc[j] = ws + (size_t)(n0 + idx - BM) * NC;
      sc_pos[j] = BM + (idx - BM);
    } else {
      gsc[j] = as;   // idle thread: harmless clamped load, never stored
      sc_pos[j] = -1;
    }
  }

  i32x8 iacc[WM][WP];
  f32x8 acc[WM][WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) { iacc[i][j][e] = 0; acc[i][j][e] = 0.f; }

  uint4 ra[AN], rb[BNU];
  uint16_t rs[NS];

#define GI4_LOAD(KS)                                                     \
  do {                                                                   \
    _Pragma("unroll") for (int j = 0; j < AN; ++j)                       \
        ra[j] = *(const uint4*)(ga[j] + (size_t)(KS) * 64);              \
    _Pragma("unroll") for (int j = 0; j < BNU; ++j)                      \
        rb[j] = *(const uint4*)(gb[j] + (size_t)(KS) * 64);              \
  } while (0)
#define GI4_STORE(BUFI)                                                  \
  do {                                                                   \
    _Pragma("unroll") for (int j = 0; j < AN; ++j) {                     \
      uint4 v = ra[j];                                                   \
      if (SWZ && sa_swap[j]) v = make_uint4(v.z, v.w, v.x, v.y);         \
      *(uint4*)(lds + (BUFI) * BUF + sa_off[j]) = v;                     \
    }                                                                    \
    _Pragma("unroll") for (int j = 0; j < BNU; ++j) {                    \
      uint4 v = rb[j];                                                   \
      if (SWZ && sb_swap[j]) v = make_uint4(v.z, v.w, v.x, v.y);         \
      *(uint4*)(lds + (BUFI) * BUF + sb_off[j]) = v;                     \
    }                                                                    \
  } while (0)
#define GI4_LOAD_SCALES(CH)                                              \
  do {                                                                   \
    _Pragma("unroll") for (int j = 0; j < NS; ++j) rs[j] = gsc[j][(CH)]; \
  } while (0)
#define GI4_STORE_SCALES(SB)                                             \
  do {                                                                   \
    _Pragma("unroll") for (int j = 0; j < NS; ++j) {                     \
      const float f = fp16f(rs[j]);                                      \
      if (sc_pos[j] >= 0 && sc_pos[j] < BM) Sa[SB][sc_pos[j]] = f;       \
      else if (sc_pos[j] >= BM) Sw[SB][sc_pos[j] - BM] = f;              \
    }                                                                    \
  } while (0)
#define GI4_FRAGS(Ab, KH, AF, BF)                                              \
  do {                                                                          \
    _Pragma("unroll") for (int i = 0; i < WM; ++i)                              \
        AF[i] = *(const i32x2*)((Ab) + (wm0 + 16 * i) * PITCH + foff[KH]);      \
    _Pragma("unroll") for (int j = 0; j < WP; ++j)                              \
        BF[j] = *(const i32x2*)((Ab) + ASZ + (wp0 + 16 * j) * PITCH + foff[KH]);\
  } while (0)
#define GI4_WMMAS(AF, BF)                                                       \
  do {                                                                          \
    _Pragma("unroll") for (int i = 0; i < WM; ++i)                              \
      _Pragma("unroll") for (int j = 0; j < WP; ++j)                            \
        iacc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(                \
            true, AF[i], true, BF[j], iacc[i][j], false);                       \
  } while (0)
// PIPE = 0: the compiler schedules everything (it sinks the next K-step's
// global loads to just before their LDS store, exposing their latency).
// PIPE >= 1: a sched_barrier after the global loads at the top of each K-step
// keeps them in flight across the whole compute phase.
// PIPE = 2: additionally an explicit one-substep fragment pipeline (substep
// kh+1's LDS fragments are loaded before the WMMAs of kh, +2*(WM+WP) VGPRs),
// fenced with sched_barriers so the scheduler cannot re-serialize it.
#define GI4_FENCE() do { if constexpr (PIPE >= 1) __builtin_amdgcn_sched_barrier(0); } while (0)
#define GI4_COMPUTE(BUFI)                                                       \
  do {                                                                          \
    const uint8_t* Ab = lds + (BUFI) * BUF;                                     \
    if constexpr (PIPE >= 2) {                                                  \
      i32x2 af[2][WM], bf[2][WP];                                               \
      GI4_FRAGS(Ab, 0, af[0], bf[0]);                                           \
      _Pragma("unroll") for (int kh = 0; kh < 8; ++kh) {                        \
        if (kh < 7) GI4_FRAGS(Ab, kh + 1, af[(kh + 1) & 1], bf[(kh + 1) & 1]);  \
        __builtin_amdgcn_sched_barrier(0);                                      \
        GI4_WMMAS(af[kh & 1], bf[kh & 1]);                                      \
        __builtin_amdgcn_sched_barrier(0);                                      \
      }                                                                         \
    } else {                                                                    \
      _Pragma("unroll") for (int kh = 0; kh < 8; ++kh) {                        \
        i32x2 af[WM], bf[WP];                                                   \
        GI4_FRAGS(Ab, kh, af, bf);                                              \
        GI4_WMMAS(af, bf);                                                      \
      }                                                                         \
    }                                                                           \
  } while (0)
#define GI4_CONVERT(SB)                                                         \
  do {                                                                          \
    float swv[WP];                                                              \
    _Pragma("unroll") for (int j = 0; j < WP; ++j)                              \
        swv[j] = Sw[SB][wp0 + 16 * j + l16];                                    \
    _Pragma("unroll") for (int i = 0; i < WM; ++i) {                            \
      const float4 s0 = *(const float4*)&Sa[SB][wm0 + 16 * i + 8 * hh];         \
      const float4 s1 = *(const float4*)&Sa[SB][wm0 + 16 * i + 8 * hh + 4];     \
      const float sav[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};    \
      _Pragma("unroll") for (int j = 0; j < WP; ++j)                            \
        _Pragma("unroll") for (int e = 0; e < 8; ++e) {                         \
          const float s = sav[e] * swv[j];                                      \
          acc[i][j][e] = __builtin_fmaf((float)iacc[i][j][e], s, acc[i][j][e]); \
          iacc[i][j][e] = 0;                                                    \
        }                                                                       \
    }                                                                           \
  } while (0)

  // prologue: K-step 0 -> buffer 0, chunk 0 scales -> scale buffer 0
  GI4_LOAD(0);
  GI4_STORE(0);
  GI4_LOAD_SCALES(0);
  GI4_STORE_SCALES(0);
  lds_sync();

  const int nks = K >> 7;
#pragma unroll 1
  for (int c = 0; c < NC; ++c) {
    // first half of chunk c is in buffer 0; stage the second half into buffer 1
    GI4_LOAD(2 * c + 1);
    GI4_LOAD_SCALES(min(c + 1, NC - 1));
    GI4_FENCE();
    GI4_COMPUTE(0);
    GI4_FENCE();
    GI4_STORE(1);
    GI4_STORE_SCALES((c + 1) & 1);   // buffer (c-1)&1, last read before the previous barrier
    lds_sync();
    // second half of chunk c is in buffer 1; stage the first half of c+1 into buffer 0
    GI4_LOAD(min(2 * c + 2, nks - 1));
    GI4_FENCE();
    GI4_COMPUTE(1);
    GI4_FENCE();
    GI4_CONVERT(c & 1);
    GI4_STORE(0);
    lds_sync();
  }
#undef GI4_LOAD
#undef GI4_STORE
#undef GI4_LOAD_SCALES
#undef GI4_STORE_SCALES
#undef GI4_COMPUTE
#undef GI4_FENCE
#undef GI4_FRAGS
#undef GI4_WMMAS
#undef GI4_CONVERT

  // epilogue: lane holds column l16 of each tile, rows 2e + hh
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j) {
      const int n = n0 + wp0 + 16 * j + l16;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm0 + 16 * i + 2 * e + hh;
        if (m < M) C[(size_t)m * N + n] = f32bf(acc[i][j][e]);
      }
    }
}

template <int BM, int BN, int MW, int PW, int NT, int SWZ, int OCC = 1, int PIPE = 0>
static int launch_cfg(const uint8_t* ac, const uint16_t* as, const uint8_t* wc, const uint16_t* ws,
                      uint16_t* C, int M, int N, int K, int gm, hipStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || (K & 255) || (N % BN)) return -1;
  const int num_m = (M + BM - 1) / BM, num_n = N / BN;
  if (gm < 1) gm = 1;
  if (gm > num_m) gm = num_m;
  k_gemm_i4<BM, BN, MW, PW, NT, SWZ, OCC, PIPE><<<dim3((unsigned)(num_m * num_n)), dim3(NT), 0, stream>>>(
      ac, as, wc, ws, C, M, N, K, gm);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gi4

// Configurations the harness times; gemm_i4_launch uses 10 (4 when M <= 64).
//   0: 128x256 block, 512 threads (4x4 waves, 32x64 warp tile), XOR-swizzled 64 B rows, gm 8
//   1: 128x256 block, 512 threads, padded 72 B rows, gm 8
//   2: 128x128 block, 256 threads (2x4 waves, 64x32 warp tile), XOR-swizzled, gm 8
//   3: 128x128 block, 256 threads, padded, gm 8
//   4: 64x256 block, 256 threads (1x8 waves, 64x32 warp tile), XOR-swizzled, gm 16
//   5: 256x128 block, 512 threads (4x4 waves, 64x32 warp tile), XOR-swizzled, gm 4
//   6: as 0 but compiled for 8 waves/SIMD (<= 192 VGPRs: two blocks per WGP)
//   7: as 0 with gm 4
//   8: as 0 with gm 16
//   9: as 0 with PIPE = 1 (global loads fenced ahead of the compute phase)
//  10: as 0 with PIPE = 2 (PIPE 1 + explicit one-substep fragment pipeline)
//  11: as 2 (128x128, 256 threads) with PIPE = 2
//  12: as 2 (128x128, 256 threads) with PIPE = 1
//  13: as 4 (64x256, 256 threads) with PIPE = 2
static inline int gemm_i4_launch_variant(int v, const uint8_t* ac, const uint16_t* as, const uint8_t* wc,
                                         const uint16_t* ws, uint16_t* C, int M, int N, int K,
                                         hipStream_t stream) {
  switch (v) {
    case 0: return gi4::launch_cfg<128, 256, 4, 4, 512, 1>(ac, as, wc, ws, C, M, N, K, 8, stream);
    case 1: return gi4::launch_cfg<128, 256, 4, 4, 512, 0>(ac, as, wc, ws, C, M, N, K, 8, stream);
    case 2: return gi4::launch_cfg<128, 128, 2, 4, 256, 1>(ac, as, wc, ws, C, M, N, K, 8, stream);
    case 3: return gi4::launch_cfg<128, 128, 2, 4, 256, 0>(ac, as, wc, ws, C, M, N, K, 8, stream);
    case 4: return gi4::launch_cfg<64, 256, 1, 8, 256, 1>(ac, as, wc, ws, C, M, N, K, 16, stream);
    case 5: return gi4::launch_cfg<256, 128, 4, 4, 512, 1>(ac, as, wc, ws, C, M, N, K, 4, stream);
    case 6: return gi4::launch_cfg<128, 256, 4, 4, 512, 1, 8>(ac, as, wc, ws, C, M, N, K, 8, stream);
    case 7: return gi4::launch_cfg<128, 256, 4, 4, 512, 1>(ac, as, wc, ws, C, M, N, K, 4, stream);
    case 8: return gi4::launch_cfg<128, 256, 4, 4, 512, 1>(ac, as, wc, ws, C, M, N, K, 16, stream);
    case 9: return gi4::launch_cfg<128, 256, 4, 4, 512, 1, 1, 1>(ac, as, wc, ws, C, M, N, K, 8, stream);
    case 10: return gi4::launch_cfg<128, 256, 4, 4, 512, 1, 1, 2>(ac, as, wc, ws, C, M, N, K, 8, stream);
    case 11: return gi4::launch_cfg<128, 128, 2, 4, 256, 1, 1, 2>(ac, as, wc, ws, C, M, N, K, 8, stream);
    case 12: return gi4::launch_cfg<128, 128, 2, 4, 256, 1, 1, 1>(ac, as, wc, ws, C, M, N, K, 8, stream);
    case 13: return gi4::launch_cfg<64, 256, 1, 8, 256, 1, 1, 2>(ac, as, wc, ws, C, M, N, K, 16, stream);
    default: return -1;
  }
}
static inline int gemm_i4_num_variants() { return 14; }

// C[M][N] bf16 = int4 GEMM of ac/as [M] against wc/ws [N] over K (see header
// comment). Returns 0 on success, -1 on a shape the kernel does not support
// (K % 256, N % 256) or a launch error. stream == nullptr is the null stream.
static int gemm_i4_launch(const uint8_t* ac, const uint16_t* as, const uint8_t* wc, const uint16_t* ws,
                          uint16_t* C, int M, int N, int K, hipStream_t stream) {
  if (M <= 64) return gemm_i4_launch_variant(4, ac, as, wc, ws, C, M, N, K, stream);
  return gemm_i4_launch_variant(10, ac, as, wc, ws, C, M, N, K, stream);
}
