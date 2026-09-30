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
#include <cstdlib>

using flash_h16 = __attribute__((ext_vector_type(16))) short;
using flash_f8 = __attribute__((ext_vector_type(8))) float;
static constexpr int F_QH = 24, F_KVH = 4, F_HD = 256;
static constexpr int F_STR = 264;

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

// The XOR tree stays in exactly the same order; DPP only changes the
// lane transport. All peers are in the same sixteen-lane DPP row.
template<bool DPP>
__device__ __forceinline__ float flash_shuffle(float x, int off) {
  if constexpr (!DPP) return __shfl_xor(x, off);
  else {
    int u = __float_as_int(x), peer;
    switch (off) {
      case 1: peer = __builtin_amdgcn_update_dpp(0, u, 0x161, 0xf, 0xf, true); break;
      case 2: peer = __builtin_amdgcn_update_dpp(0, u, 0x162, 0xf, 0xf, true); break;
      case 4: peer = __builtin_amdgcn_update_dpp(0, u, 0x164, 0xf, 0xf, true); break;
      default: peer = __builtin_amdgcn_update_dpp(0, u, 0x168, 0xf, 0xf, true); break;
    }
    return __int_as_float(peer);
  }
}

// DS waves share each 16-query group and split the 256 output columns.
// K and V reuse the same LDS array after scores are consumed.
template<int F_QT, int F_KT, int DS, bool TRANSPOSE_V = false, bool FAST_EXP = false, bool GLOBAL_Q = false, int LOAD_VECTOR = 1, bool DPP = false, bool PREFETCH = false, bool HEAD_MAJOR = false, bool TRANSPOSED_CACHE = false>
__global__ void __launch_bounds__((F_QT / 16) * DS * 32)
k_attention_flash_f16(const uint16_t* __restrict__ Q,
                      const uint16_t* __restrict__ K,
                      const uint16_t* __restrict__ V,
                      uint16_t* __restrict__ O, int T, int pos_base, int vstride = 0) {
  constexpr int F_PSTR = F_KT + 8, NT = (F_QT / 16) * DS * 32;
  constexpr int NF = F_KT / 16, ND = 16 / DS;
  static_assert(F_QT % 16 == 0 && F_KT % 16 == 0 && (DS == 1 || DS == 2));
  static_assert(!TRANSPOSED_CACHE || TRANSPOSE_V);
  static_assert(LOAD_VECTOR == 1 || LOAD_VECTOR == 8 || LOAD_VECTOR == 16);
  using load_vec = short __attribute__((ext_vector_type(LOAD_VECTOR)));
  // A 16-byte row alignment lets the compiler use ds_load_b128 for WMMA
  // fragments; a 2-half pad degenerates into individual halfword gathers.
  constexpr int VSTR = F_KT + 8;
  constexpr int QROWS = GLOBAL_Q ? 1 : F_QT;
  constexpr int KVSHORTS = TRANSPOSE_V && F_HD * VSTR > F_KT * F_STR ? F_HD * VSTR : F_KT * F_STR;
  static_assert(2 * (QROWS * F_STR + KVSHORTS + F_QT * F_PSTR) <= 65536);
  __shared__ short qs[QROWS][F_STR];
  __shared__ union {
    short key[F_KT][F_STR];
    short value[TRANSPOSE_V ? F_HD : 1][TRANSPOSE_V ? VSTR : 1];
  } kv;
  __shared__ short probs[F_QT][F_PSTR];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int qgroup = wave / DS, dgroup = wave % DS;
  const int col = lane & 15, halfrow = lane >> 4;
  const int q0 = (HEAD_MAJOR ? blockIdx.y : blockIdx.x) * F_QT;
  const int h = HEAD_MAJOR ? blockIdx.x : blockIdx.y, kh = h / 6;
  int arow = -1;
  if (lane < 16 && !(lane & 1)) arow = lane / 2;
  else if (lane >= 17 && (lane & 1)) arow = 8 + (lane - 17) / 2;
  if constexpr (!GLOBAL_Q) for (int i = tid * LOAD_VECTOR; i < F_QT * F_HD; i += NT * LOAD_VECTOR) {
    int r = i / F_HD, d = i % F_HD;
    load_vec x = {0}, raw = {0};
    if (q0 + r < T) memcpy(&raw, Q + ((long)(q0 + r) * F_QH + h) * F_HD + d, sizeof(raw));
#pragma unroll
    for (int j = 0; j < LOAD_VECTOR; j++) x[j] = flash_half(flash_bfloat((uint16_t)raw[j]));
    memcpy(&qs[r][d], &x, sizeof(x));
  }
  __syncthreads();
  flash_f8 accum[ND];
  float mxrun[8], sumrun[8];
#pragma unroll
  for (int d = 0; d < ND; d++) accum[d] = {0};
#pragma unroll
  for (int e = 0; e < 8; e++) { mxrun[e] = -INFINITY; sumrun[e] = 0; }
  const int last_query = (q0 + F_QT < T ? q0 + F_QT : T) - 1;
  const int nkeys = pos_base + last_query + 1;
  for (int k0 = 0; k0 < nkeys; k0 += F_KT) {
    __syncthreads();
    for (int i = tid * LOAD_VECTOR; i < F_KT * F_HD; i += NT * LOAD_VECTOR) {
      int r = i / F_HD, d = i % F_HD, key = k0 + r;
      load_vec x = {0}, raw = {0};
      if (key < nkeys) memcpy(&raw, K + ((long)key * F_KVH + kh) * F_HD + d, sizeof(raw));
#pragma unroll
      for (int j = 0; j < LOAD_VECTOR; j++) x[j] = flash_half(flash_bfloat((uint16_t)raw[j]));
      memcpy(&kv.key[r][d], &x, sizeof(x));
    }
    __syncthreads();
    constexpr int NLOAD = (F_KT * F_HD + NT * LOAD_VECTOR - 1) / (NT * LOAD_VECTOR);
    load_vec vr[NLOAD];
    if constexpr (PREFETCH) {
#pragma unroll
      for (int u = 0; u < NLOAD; u++) {
        int i = (tid + u * NT) * LOAD_VECTOR, key = k0 + i / F_HD;
        vr[u] = {0};
        if (i < F_KT * F_HD && key < nkeys)
          memcpy(&vr[u], V + ((long)key * F_KVH + kh) * F_HD + i % F_HD, sizeof(load_vec));
      }
    }
    flash_f8 scores[NF];
#pragma unroll
    for (int f = 0; f < NF; f++) scores[f] = {0};
#pragma unroll
    for (int dk = 0; dk < 16; dk++) {
      flash_h16 a = {0};
      if (arow >= 0) {
        if constexpr (GLOBAL_Q) {
          const int query = q0 + qgroup * 16 + arow;
          flash_h16 raw = {0};
          if (query < T)
            memcpy(&raw,Q+((long)query*F_QH+h)*F_HD+dk*16,sizeof(raw));
#pragma unroll
          for (int k = 0; k < 16; k++)
            a[k] = flash_half(flash_bfloat((uint16_t)raw[k]));
        } else a = flash_load(&qs[qgroup * 16 + arow][dk * 16]);
      }
#pragma unroll
      for (int f = 0; f < NF; f++) {
        flash_h16 b;
#pragma unroll
        for (int k = 0; k < 16; k++) b[k] = kv.key[f * 16 + col][dk * 16 + k];
        scores[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, scores[f]);
      }
    }
#pragma unroll
    for (int e = 0; e < 8; e++) {
      int query = q0 + qgroup * 16 + halfrow * 8 + e;
      float m = -INFINITY;
#pragma unroll
      for (int f = 0; f < NF; f++) {
        int key = k0 + f * 16 + col;
        float s = query < T && key <= pos_base + query
            ? scores[f][e] * 0.0625f : -INFINITY;
        scores[f][e] = s;
        m = fmaxf(m, s);
      }
#pragma unroll
      for (int off = 1; off <= 8; off <<= 1) m = fmaxf(m, flash_shuffle<DPP>(m, off));
      float mnew = fmaxf(mxrun[e], m);
      float correction = 0.f;
      if (isfinite(mnew)) {
        if constexpr (FAST_EXP) correction = __builtin_amdgcn_exp2f((mxrun[e] - mnew) * 1.4426950408889634f);
        else correction = expf(mxrun[e] - mnew);
      }
      float es = 0.f;
#pragma unroll
      for (int f = 0; f < NF; f++) {
        float p = 0.f;
        if (isfinite(scores[f][e])) {
          if constexpr (FAST_EXP) p = __builtin_amdgcn_exp2f((scores[f][e] - mnew) * 1.4426950408889634f);
          else p = expf(scores[f][e] - mnew);
        }
        scores[f][e] = p;
        es += p;
        if (dgroup == 0)
          probs[qgroup * 16 + halfrow * 8 + e][f * 16 + col] = flash_half(p);
      }
#pragma unroll
      for (int off = 1; off <= 8; off <<= 1) es += flash_shuffle<DPP>(es, off);
      sumrun[e] = sumrun[e] * correction + es;
      mxrun[e] = mnew;
#pragma unroll
      for (int d = 0; d < ND; d++) accum[d][e] *= correction;
    }
    __syncthreads();
    if constexpr (!PREFETCH && !TRANSPOSED_CACHE) {
      for (int i = tid * LOAD_VECTOR; i < F_KT * F_HD; i += NT * LOAD_VECTOR) {
        int r = i / F_HD, d = i % F_HD, key = k0 + r;
        load_vec x = {0}, raw = {0};
        if (key < nkeys) memcpy(&raw, V + ((long)key * F_KVH + kh) * F_HD + d, sizeof(raw));
#pragma unroll
        for (int j = 0; j < LOAD_VECTOR; j++) x[j] = flash_half(flash_bfloat((uint16_t)raw[j]));
        if constexpr (TRANSPOSE_V) {
#pragma unroll
          for (int j = 0; j < LOAD_VECTOR; j++) kv.value[d + j][r] = x[j];
        } else memcpy(&kv.key[r][d], &x, sizeof(x));
      }
    } else {
#pragma unroll
    for (int u = 0; u < NLOAD; u++) {
      int i = (tid + u * NT) * LOAD_VECTOR;
      if (i >= F_KT * F_HD) continue;
      int r = TRANSPOSED_CACHE ? i % F_KT : i / F_HD;
      int d = TRANSPOSED_CACHE ? i / F_KT : i % F_HD;
      int key = k0 + r;
      load_vec x = {0}, raw = {0};
      if constexpr (TRANSPOSED_CACHE) {
        if (key < vstride) memcpy(&raw, V + ((long)kh * F_HD + d) * vstride + key, sizeof(raw));
#pragma unroll
        for (int j = 0; j < LOAD_VECTOR; j++)
          x[j] = key + j < nkeys ? flash_half(flash_bfloat((uint16_t)raw[j])) : 0;
        memcpy(&kv.value[d][r], &x, sizeof(x));
        continue;
      }
      if constexpr (PREFETCH) raw = vr[u];
      else if (key < nkeys) memcpy(&raw, V + ((long)key * F_KVH + kh) * F_HD + d, sizeof(raw));
#pragma unroll
      for (int j = 0; j < LOAD_VECTOR; j++) x[j] = flash_half(flash_bfloat((uint16_t)raw[j]));
      if constexpr (TRANSPOSE_V) {
#pragma unroll
        for (int j = 0; j < LOAD_VECTOR; j++) kv.value[d + j][r] = x[j];
      } else memcpy(&kv.key[r][d], &x, sizeof(x));
    }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < NF; kk++) {
      flash_h16 a = {0};
      if (arow >= 0) a = flash_load(&probs[qgroup * 16 + arow][kk * 16]);
#pragma unroll
      for (int d = 0; d < ND; d++) {
        flash_h16 b;
        if constexpr (TRANSPOSE_V) {
          b = flash_load(&kv.value[dgroup * ND * 16 + d * 16 + col][kk * 16]);
        } else {
#pragma unroll
        for (int k = 0; k < 16; k++)
          b[k] = kv.key[kk * 16 + k][dgroup * ND * 16 + d * 16 + col];
        }
        accum[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, accum[d]);
      }
    }
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    int query = q0 + qgroup * 16 + halfrow * 8 + e;
    if (query < T) {
#pragma unroll
      for (int d = 0; d < ND; d++)
        O[((long)query * F_QH + h) * F_HD + dgroup * ND * 16 + d * 16 + col] =
            flash_out(accum[d][e] / sumrun[e]);
    }
  }
}

// A bounded, per-layer transpose workspace makes V fragment loads contiguous
// without conflicting transposed stores inside every attention tile. The
// original KV cache is retained for attn_tiled and the decode reference.
__global__ void k_flash_transpose_v(const uint16_t* V, uint16_t* VT, int keys, int stride) {
  // Coalesced reads and writes, with a padded shared transpose tile. A
  // one-thread-per-output transpose reads strided cache lines and hurts decode.
  __shared__ uint16_t tile[32][33];
  const int lane = threadIdx.x & 31, row = threadIdx.x >> 5;
  const int key0 = blockIdx.x * 32, d0 = blockIdx.y * 32, h = blockIdx.z;
  for (int r = row; r < 32; r += 8)
    tile[r][lane] = key0 + r < keys ? V[((long)(key0+r)*F_KVH+h)*F_HD+d0+lane] : 0;
  __syncthreads();
  for (int r = row; r < 32; r += 8)
    if (key0 + lane < stride)
      VT[((long)h*F_HD+d0+r)*stride+key0+lane] = tile[lane][r];
}

static int attention_flash_impl(const uint16_t* Q, const uint16_t* K,
                                         const uint16_t* V, uint16_t* O,
                                         int T, int pos_base, bool sync) {
  if (T < 1 || pos_base < 0) return -1;
  const char* cfg = getenv("HALO_FLASH_OPT");
  // Keep single-token attention on the phase-2E path: transposing the whole
  // KV prefix on every decode step would add work unrelated to prefill.
  const int opt = T > 1 && cfg ? atoi(cfg) : 0;
  if (opt == 2) {
    static uint16_t* vt = nullptr;
    static size_t capacity = 0;
    const int keys = pos_base + T, stride = (keys + 15) & ~15;
    const size_t bytes = (size_t)F_KVH * F_HD * stride * sizeof(uint16_t);
    if (bytes > capacity) {
      if (vt) { hipError_t e = hipFree(vt); if (e != hipSuccess) return (int)e; vt = nullptr; capacity = 0; }
      hipError_t e = hipMalloc(&vt, bytes); if (e != hipSuccess) return (int)e;
      capacity = bytes;
    }
    k_flash_transpose_v<<<dim3((stride+31)/32,F_HD/32,F_KVH),256>>>(V, vt, keys, stride);
    hipError_t e = hipGetLastError(); if (e != hipSuccess) return (int)e;
    k_attention_flash_f16<64,32,2,true,false,false,16,true,false,true,true>
        <<<dim3(F_QH,(T+63)/64),256>>>(Q,K,vt,O,T,pos_base,stride);
  } else if (opt == 1) {
    k_attention_flash_f16<64,32,2,false,false,false,16,true,false,true>
        <<<dim3(F_QH,(T+63)/64),256>>>(Q,K,V,O,T,pos_base);
  } else {
    // The phase-2E implementation remains directly selectable.
    k_attention_flash_f16<64,32,2,false,false,false,16>
        <<<dim3((T+63)/64,F_QH),256>>>(Q,K,V,O,T,pos_base);
  }
  hipError_t rc = hipGetLastError();
  if (rc == hipSuccess && sync) rc = hipDeviceSynchronize();
  return (int)rc;
}

extern "C" int chlorine_attention_flash(const uint16_t* Q,const uint16_t* K,
                                         const uint16_t* V,uint16_t* O,int T,int pos_base) {
  return attention_flash_impl(Q,K,V,O,T,pos_base,true);
}
extern "C" int chlorine_attention_flash_async(const uint16_t* Q,const uint16_t* K,
                                               const uint16_t* V,uint16_t* O,int T,int pos_base) {
  return attention_flash_impl(Q,K,V,O,T,pos_base,false);
}
