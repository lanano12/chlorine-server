// trunk.hip — chlorine-server numeric core validation (spec: docs/halogen/*.md)
// Qwen3.8-27B hybrid trunk (48 GDN + 16 FA layers) on gfx1151.
// Math source: HF transformers Qwen3.5 modeling (Apache-2.0) — GatedDeltaNet,
// Qwen3_5Attention, Qwen3_5RMSNorm(ZeroCentered), Qwen3_5RMSNormGated.
// Weight formats: docs/halogen/KERNELS-DEQUANT.md (validated formulas reused).
//
// Modes:  tf   — teacher-forced forward over /tmp/prompt_ids.json (87 tokens),
//                compare vs /tmp/forced-text.bin (HFD2 dump from the original engine)
//         dec  — prefill + 24 greedy tokens, compare vs the original's stream
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <algorithm>
#include <sys/time.h>

#define CK(x) do { hipError_t e = (x); if (e) { printf("HIP err %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)
#define GRID1(n) (unsigned)((((long)(n)) + 255) / 256)

// ---------------- model constants ----------------
static const int   HID   = 5120;      // hidden
static const int   VOC   = 248320;    // vocab
static const int   NL    = 64;        // layers; layer%4==3 -> full_attention
static const int   INTM  = 17408;     // mlp intermediate
// GDN (linear_attn)
static const int   HK    = 16;        // k heads
static const int   HV    = 48;        // v heads
static const int   DK    = 128;       // k head dim
static const int   DV    = 128;       // v head dim
static const int   QKVW  = 10240;     // qkv width (q 2048 | k 2048 | v 6144)
// FA (self_attn)
static const int   QH    = 24;        // q heads
static const int   KVH   = 4;         // kv heads
static const int   HD    = 256;       // head dim
static const int   NPAIR = 32;        // rotary pairs (partial 0.25 -> 64 dims)
static const float SCALE = 0.0625f;   // 1/sqrt(256)
static const float THETA = 1e7f;
static const float EPS   = 1e-6f;
// Build limits are explicit CMake settings; logits hold only the last row.
#ifndef CHLORINE_CXT
#define CHLORINE_CXT 1088
#endif
#ifndef CHLORINE_TMAX
#define CHLORINE_TMAX 1024
#endif
static const int   CXT   = CHLORINE_CXT;
static const int   TMAX  = CHLORINE_TMAX;
static const int   ATTN_TILE = 32; // query rows scored at once
static_assert(TMAX <= CXT - 2, "prefill rows must fit in the KV cache with room to decode");
static_assert(ATTN_TILE >= 1, "attention tile");
static const char* CKPT  = "/home/ronin/Projects/chlorine-server/models/qwen3.8-27b-p1w4d-d2.hgn";

static int ROPE_MODE = 0;   // 0: HF half-split pairs (p, p+32); 1: interleaved (2p, 2p+1)

static double now_s() { struct timeval tv; gettimeofday(&tv, nullptr); return tv.tv_sec + tv.tv_usec * 1e-6; }

#include "dequant_common.h"
__global__ void k_dequant_fp8r(const uint8_t* __restrict__ payload, const uint16_t* __restrict__ scales,
                               uint16_t* __restrict__ out, long cols) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long r = i / cols;
  float w = dequant_fp8_e4m3(payload[i]) * bf16f(scales[r]);
  out[i] = f32bf(w);
}
__global__ void k_dequant_q4c(const float* __restrict__ cb, const uint8_t* __restrict__ rowbase,
                              uint16_t* __restrict__ out, long cols) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long r = i / cols, c = i % cols;
  const uint8_t* pay = rowbase + (long)r * (cols / 2 + cols / 16);
  uint8_t byte = pay[c / 2];
  uint32_t nib = (c & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
  float w = cb[nib] * dequant_fp8_e4m3(pay[cols / 2 + c / 16]);
  out[i] = f32bf(w);
}

// ---------------- GEMM: C[M,N] = A[M,K] x W[N,K]^T, bf16 in, f32 accum ----------------
// block 16x16; grid.x = ceil(N/64); block loops all 16-row m-tiles (M <= 96).
template<int F32OUT>
__global__ void k_gemm(const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
                       uint16_t* __restrict__ C, float* __restrict__ Cf,
                       int M, int N, int K) {
  int n0 = blockIdx.x * 64;
  int tx = threadIdx.x, ty = threadIdx.y;
  __shared__ uint16_t sw[64][64];
  __shared__ uint16_t sa[16][64];
  float acc[6][4];
  for (int i = 0; i < 6; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < K; k0 += 64) {
    for (int s = 0; s < 16; s++) {
      int l = (ty * 16 + tx) + s * 256;
      int r = l >> 6, c = l & 63;
      long n = (long)n0 + r; long k = (long)k0 + c;
      sw[r][c] = (n < N && k < K) ? W[n * K + k] : (uint16_t)0;
    }
    __syncthreads();
    for (int mt = 0; mt < 6; mt++) {
      int m0 = mt * 16;
      for (int s = 0; s < 4; s++) {
        int c = tx + s * 16;
        long k = (long)k0 + c; int m = m0 + ty;
        sa[ty][c] = (m < M && k < K) ? A[(long)m * K + k] : (uint16_t)0;
      }
      __syncthreads();
      // BLASLt MI16x16x1 emulation: wmma accumulates k as ascending pair-sums
      for (int kk = 0; kk < 64; kk += 2) {
        float a0 = bf16f(sa[ty][kk]), a1 = bf16f(sa[ty][kk + 1]);
        acc[mt][0] += (a0 * bf16f(sw[tx][kk]) + a1 * bf16f(sw[tx][kk + 1]));
        acc[mt][1] += (a0 * bf16f(sw[tx + 16][kk]) + a1 * bf16f(sw[tx + 16][kk + 1]));
        acc[mt][2] += (a0 * bf16f(sw[tx + 32][kk]) + a1 * bf16f(sw[tx + 32][kk + 1]));
        acc[mt][3] += (a0 * bf16f(sw[tx + 48][kk]) + a1 * bf16f(sw[tx + 48][kk + 1]));
      }
      __syncthreads();
    }
  }
  for (int mt = 0; mt < 6; mt++) {
    int m = mt * 16 + ty;
    if (m >= M) continue;
    for (int e = 0; e < 4; e++) {
      long n = (long)n0 + tx + e * 16;
      if (n >= N) continue;
      if (F32OUT) Cf[(long)m * N + n] = acc[mt][e];
      else C[(long)m * N + n] = f32bf(acc[mt][e]);
    }
  }
}
static void dumpx(const uint16_t* d_x) {
  static int xi = 0; static FILE* xf = 0;
  const char* dxp = getenv("HALO_DUMP_X");
  if (!dxp) return;
  if (!xf) xf = fopen(dxp, "wb");
  void* hb = malloc(512);
  CK(hipMemcpy(hb, d_x, 512, hipMemcpyDeviceToHost));
  int idx = xi++;
  fwrite(&idx, 4, 1, xf);
  fwrite(hb, 1, 512, xf);
  free(hb);
}
static void gemm_scalar(const uint16_t* A, const uint16_t* W, uint16_t* C, int M, int N, int K) {
  dim3 b(16, 16);
  for (int m0 = 0; m0 < M; m0 += 96) {          // kernel supports M <= 96; chunk by m
    int mm = M - m0 < 96 ? M - m0 : 96;
    dim3 g((N + 63) / 64);
    hipLaunchKernelGGL((k_gemm<0>), g, b, 0, 0, A + (long)m0 * K, W, C + (long)m0 * N, (float*)nullptr, mm, N, K);
  }
}
static void gemm_f32_scalar(const uint16_t* A, const uint16_t* W, float* Cf, int M, int N, int K) {
  dim3 b(16, 16);
  for (int m0 = 0; m0 < M; m0 += 96) {
    int mm = M - m0 < 96 ? M - m0 : 96;
    dim3 g((N + 63) / 64);
    hipLaunchKernelGGL((k_gemm<1>), g, b, 0, 0, A + (long)m0 * K, W, (uint16_t*)nullptr, Cf + (long)m0 * N, mm, N, K);
  }
}
// 0 ok, -1 this shape falls back, -2 hipBLASLt unusable
extern "C" int chlorine_gemm_lt(const void* A, const void* W, void* C, int M, int N, int K, int f32);
static int g_blaslt = -1;
static int use_blaslt() {
  if (g_blaslt < 0) {
    const char* e = getenv("HALO_BLASLT");
    g_blaslt = (e && atoi(e) == 0) ? 0 : 1;
  }
  return g_blaslt;
}
static void gemm(const uint16_t* A, const uint16_t* W, uint16_t* C, int M, int N, int K) {
  if (use_blaslt()) {
    int rc = chlorine_gemm_lt(A, W, C, M, N, K, 0);
    if (rc == 0) return;
    if (rc == -2) g_blaslt = 0;
  }
  gemm_scalar(A, W, C, M, N, K);
}
static void gemm_f32(const uint16_t* A, const uint16_t* W, float* Cf, int M, int N, int K) {
  if (use_blaslt()) {
    int rc = chlorine_gemm_lt(A, W, Cf, M, N, K, 1);
    if (rc == 0) return;
    if (rc == -2) g_blaslt = 0;
  }
  gemm_f32_scalar(A, W, Cf, M, N, K);
}

// ---------------- norms ----------------
__global__ void k_rmsnorm(const uint16_t* __restrict__ X, const uint16_t* __restrict__ w,
                          uint16_t* __restrict__ O, int K, int T, int zero_center) {
  int row = blockIdx.x; if (row >= T) return;
  __shared__ float sh[256];
  int tid = threadIdx.x;
  float s = 0.f;
  for (int c = tid; c < K; c += 256) { float v = bf16f(X[(long)row * K + c]); s += v * v; }
  sh[tid] = s; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) { if (tid < off) sh[tid] += sh[tid + off]; __syncthreads(); }
  float inv = rsqrtf(sh[0] / (float)K + EPS);
  for (int c = tid; c < K; c += 256) {
    float v = bf16f(X[(long)row * K + c]);
    float wv = zero_center ? (1.0f + bf16f(w[c])) : bf16f(w[c]);
    O[(long)row * K + c] = f32bf(v * inv * wv);
  }
}
// per-head RMSNorm (zero-centered (1+w)): src row stride sstride, head h at offset
// h*hstride (FA q_proj rows interleave [q_h | gate_h], so q needs hstride=2*D); dst [T,heads,D]
__global__ void k_headnorm(const uint16_t* __restrict__ X, const uint16_t* __restrict__ w,
                           uint16_t* __restrict__ O, int T, int heads, int sstride, int hstride, int D) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * heads) return;
  int t = (int)(i / heads), h = (int)(i % heads);
  const uint16_t* xp = X + (long)t * sstride + (long)h * hstride;
  float ss = 0.f;
  for (int d = 0; d < D; d++) { float v = bf16f(xp[d]); ss += v * v; }
  float inv = rsqrtf(ss / (float)D + EPS);
  uint16_t* op = O + ((long)t * heads + h) * D;
  for (int d = 0; d < D; d++) {
    float nrm = bf16f(xp[d]) * inv;
    op[d] = f32bf(nrm * (1.0f + bf16f(w[d])));
  }
}

// ---------------- embed gather / residual ----------------
__global__ void k_embed(const int* __restrict__ ids, const uint16_t* __restrict__ E,
                        uint16_t* __restrict__ X, int H) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long t = i / H;
  X[i] = E[(long)ids[t] * H + (i % H)];
}
__global__ void k_add(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
                      uint16_t* __restrict__ O, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  O[i] = f32bf(bf16f(A[i]) + bf16f(B[i]));
}

// ---------------- conv1d (causal, kernel 4, groups, silu) ----------------
// conv step reading history from sti and writing the shifted history to sto
// (sti == sto is k_conv1d_step exactly). Same arithmetic as k_conv1d_step.
__global__ void k_conv1d_step_io(const uint16_t* __restrict__ X, const uint16_t* __restrict__ Wc,
                                 uint16_t* __restrict__ O, const uint16_t* sti, uint16_t* sto) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= QKVW) return;
  const uint16_t s0 = sti[c * 3 + 0], s1 = sti[c * 3 + 1], s2 = sti[c * 3 + 2];
  float acc = fmaf(bf16f(s0), bf16f(Wc[c * 4 + 0]), 0.0f);
  acc = fmaf(bf16f(s1), bf16f(Wc[c * 4 + 1]), acc);
  acc = fmaf(bf16f(s2), bf16f(Wc[c * 4 + 2]), acc);
  acc = fmaf(bf16f(X[c]), bf16f(Wc[c * 4 + 3]), acc);
  float inv = 1.0f / (1.0f + expf(-acc));
  O[c] = f32bf(acc * inv);
  sto[c * 3 + 0] = s1;
  sto[c * 3 + 1] = s2;
  sto[c * 3 + 2] = X[c];
}
// T sequential conv steps in one launch (HALO_VERIFY_MERGE=1): row r reads the
// history slot p[r] and writes p[r+1]; channel c's history is carried in
// registers, so each row sees exactly what k_conv1d_step_io would have read.
struct ConvSlots { uint16_t* p[9]; };
__global__ void k_conv1d_step_rows(const uint16_t* __restrict__ X, const uint16_t* __restrict__ Wc,
                                   uint16_t* __restrict__ O, ConvSlots slots, int T) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= QKVW) return;
  const uint16_t* sti = slots.p[0];
  uint16_t s0 = sti[c * 3 + 0], s1 = sti[c * 3 + 1], s2 = sti[c * 3 + 2];
  const float w0 = bf16f(Wc[c * 4 + 0]), w1 = bf16f(Wc[c * 4 + 1]), w2 = bf16f(Wc[c * 4 + 2]), w3 = bf16f(Wc[c * 4 + 3]);
  for (int r = 0; r < T; r++) {
    const uint16_t x = X[(long)r * QKVW + c];
    float acc = fmaf(bf16f(s0), w0, 0.0f);
    acc = fmaf(bf16f(s1), w1, acc);
    acc = fmaf(bf16f(s2), w2, acc);
    acc = fmaf(bf16f(x), w3, acc);
    float inv = 1.0f / (1.0f + expf(-acc));
    O[(long)r * QKVW + c] = f32bf(acc * inv);
    uint16_t* sto = slots.p[r + 1];
    sto[c * 3 + 0] = s1;
    sto[c * 3 + 1] = s2;
    sto[c * 3 + 2] = x;
    s0 = s1; s1 = s2; s2 = x;
  }
}
// per-row argmax, row r identical to k_argmax1 on that row
__global__ void k_argmax_rows(const float* __restrict__ L, int* __restrict__ pred, int V) {
  const float* Lr = L + (size_t)blockIdx.x * V;
  __shared__ float smx[256]; __shared__ int smi[256];
  int tid = threadIdx.x;
  float mx = -3.0e38f; int mi = 0;
  for (int v = tid; v < V; v += 256) { float x = Lr[v]; if (x > mx) { mx = x; mi = v; } }
  smx[tid] = mx; smi[tid] = mi;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off && smx[tid + off] > smx[tid]) { smx[tid] = smx[tid + off]; smi[tid] = smi[tid + off]; }
    __syncthreads();
  }
  if (tid == 0) pred[blockIdx.x] = smi[0];
}
// concatenate two [T][H] bf16 row blocks into [T][2H] (MTP fc input)
__global__ void k_concat2(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
                          uint16_t* __restrict__ O, long T, int H) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= T * 2 * H) return;
  long t = i / (2 * H); int c = (int)(i % (2 * H));
  O[i] = c < H ? A[t * H + c] : B[t * H + c - H];
}
// q4c variant 1 (NF4 codebook, f16 scale per 32, row-interleaved) -> bf16
__global__ void k_dequant_q4c_v1(const float* __restrict__ cb, const uint8_t* __restrict__ rowbase,
                                 uint16_t* __restrict__ out, long rows, long cols) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  long r = i / cols, c = i % cols;
  const uint8_t* pay = rowbase + r * (cols / 2 + cols / 16);
  uint8_t b = pay[c >> 1];
  int code = (c & 1) ? (b >> 4) : (b & 0xF);
  const uint16_t* sc = (const uint16_t*)(pay + cols / 2);
  float s = __half2float(__ushort_as_half(sc[c >> 5]));
  out[i] = f32bf(cb[code] * s);
}
// st: the three inputs before this block of rows (oldest first), for rows whose
// window reaches before the block. At pos_base 0 the history is zero, and
// fmaf(w, 0, acc) leaves acc unchanged, so the cold path is unchanged.
__global__ void k_conv1d(const uint16_t* __restrict__ X, const uint16_t* __restrict__ Wc,
                         uint16_t* __restrict__ O, long T, const uint16_t* __restrict__ st) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= T * QKVW) return;
  int t = (int)(i / QKVW), c = (int)(i % QKVW);
  float acc = 0.f;
  for (int j = 0; j < 4; j++) {
    int tt = t + j - 3;
    uint16_t x = tt >= 0 ? X[(long)tt * QKVW + c] : st[c * 3 + (tt + 3)];
    acc = fmaf(bf16f(Wc[c * 4 + j]), bf16f(x), acc);
  }
  float inv = 1.0f / (1.0f + expf(-acc));
  O[i] = f32bf(acc * inv);
}
__global__ void k_conv1d_step(const uint16_t* __restrict__ X, const uint16_t* __restrict__ Wc,
                              uint16_t* __restrict__ O, uint16_t* __restrict__ st) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= QKVW) return;
  float acc = fmaf(bf16f(st[c * 3 + 0]), bf16f(Wc[c * 4 + 0]), 0.0f);
  acc = fmaf(bf16f(st[c * 3 + 1]), bf16f(Wc[c * 4 + 1]), acc);
  acc = fmaf(bf16f(st[c * 3 + 2]), bf16f(Wc[c * 4 + 2]), acc);
  acc = fmaf(bf16f(X[c]), bf16f(Wc[c * 4 + 3]), acc);
  float inv = 1.0f / (1.0f + expf(-acc));
  O[c] = f32bf(acc * inv);
  st[c * 3 + 0] = st[c * 3 + 1];
  st[c * 3 + 1] = st[c * 3 + 2];
  st[c * 3 + 2] = X[c];
}
__global__ void k_conv_setstate(const uint16_t* __restrict__ X, uint16_t* __restrict__ st, long T) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= QKVW) return;
  uint16_t old[3] = {st[c * 3 + 0], st[c * 3 + 1], st[c * 3 + 2]};
  for (int j = 0; j < 3; j++) {
    long tt = T - 3 + j;
    st[c * 3 + j] = (tt >= 0) ? X[tt * QKVW + c] : old[tt + 3];   // carry the previous history for short blocks
  }
}

// ---------------- RoPE (partial: 64 dims of 256, theta 1e7) ----------------
__global__ void k_rope(uint16_t* __restrict__ X, long T, int heads, int pos_base, int mode) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = (long)T * heads * NPAIR;
  if (i >= total) return;
  int p = (int)(i % NPAIR);
  int h = (int)((i / NPAIR) % heads);
  int t = (int)(i / ((long)NPAIR * heads));
  uint16_t* xp = X + ((long)t * heads + h) * HD;
  int pos = pos_base + t;
  float inv = powf(THETA, -(float)(2 * p) / 64.0f);
  float ang = (float)pos * inv;
  float c = cosf(ang), s = sinf(ang);
  uint16_t cb = f32bf(c), sb = f32bf(s);   // HF casts cos/sin to bf16 before applying
  float cf = bf16f(cb), sf = bf16f(sb);
  int d1, d2;
  if (mode) { d1 = 2 * p; d2 = 2 * p + 1; } else { d1 = p; d2 = p + NPAIR; }
  float a1 = bf16f(xp[d1]), a2 = bf16f(xp[d2]);
  // HF: (q_rot*cos) + (rotate_half(q_rot)*sin), each product and the sum rounded to bf16
  float t1 = bf16f(f32bf(a1 * cf)), t2 = bf16f(f32bf(a2 * sf));
  float t3 = bf16f(f32bf(a2 * cf)), t4 = bf16f(f32bf(a1 * sf));
  xp[d1] = f32bf(t1 - t2);
  xp[d2] = f32bf(t3 + t4);
}

// ---------------- KV cache write ----------------
__global__ void k_kvwrite(const uint16_t* __restrict__ S, uint16_t* __restrict__ KC,
                          long T, int pos_base) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * KVH * HD) return;
  int d = (int)(i % HD), h = (int)((i / HD) % KVH), t = (int)(i / ((long)KVH * HD));
  KC[((long)(pos_base + t) * KVH + h) * HD + d] = S[i];
}

// ---------------- FA attention ----------------
__global__ void k_attn_scores(const uint16_t* __restrict__ Q, const uint16_t* __restrict__ KC,
                              float* __restrict__ S, int T, int Tmax) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)QH * T * T) return;
  int j = (int)(i % T), t = (int)((i / T) % T), h = (int)(i / ((long)T * T));
  if (j > t) return;
  int kvh = h / (QH / KVH);
  const uint16_t* qp = Q + ((long)t * QH + h) * HD;
  const uint16_t* kp = KC + ((long)j * KVH + kvh) * HD;
  float s = 0.f;
  for (int d = 0; d < HD; d++) s += bf16f(qp[d]) * bf16f(kp[d]);
  S[(long)h * Tmax * Tmax + (long)t * Tmax + j] = s * SCALE;
}
__global__ void k_attn_softmax(const float* __restrict__ S, uint16_t* __restrict__ Pbf, int T, int Tmax) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)QH * T) return;
  int t = (int)(i % T), h = (int)(i / T);
  const float* row = S + (long)h * Tmax * Tmax + (long)t * Tmax;
  uint16_t* pb = Pbf + (long)h * Tmax * Tmax + (long)t * Tmax;
  float mx = -3.0e38f;
  for (int j = 0; j <= t; j++) mx = fmaxf(mx, row[j]);
  float sum = 0.f;
  for (int j = 0; j <= t; j++) sum += expf(row[j] - mx);
  float inv = 1.0f / sum;
  for (int j = 0; j <= t; j++) pb[j] = f32bf(expf(row[j] - mx) * inv);
}
__global__ void k_attn_pv(const uint16_t* __restrict__ Pbf, const uint16_t* __restrict__ VC,
                          uint16_t* __restrict__ O, int T, int Tmax) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * QH * HD) return;
  int d = (int)(i % HD), h = (int)((i / HD) % QH), t = (int)(i / ((long)QH * HD));
  int kvh = h / (QH / KVH);
  const uint16_t* pb = Pbf + (long)h * Tmax * Tmax + (long)t * Tmax;
  float acc = 0.f;
  for (int j = 0; j <= t; j++) acc += bf16f(pb[j]) * bf16f(VC[((long)j * KVH + kvh) * HD + d]);
  O[((long)t * QH + h) * HD + d] = f32bf(acc);
}
// One query tile. Key j is an absolute KV position. Query local t attends
// 0..pos_base+t. Scores and bf16 probs live in [head][tile row][key].
// The dot, the max, the sum, and the value mix each walk in address order,
// matching k_attn_scores / k_attn_softmax / k_attn_pv.
__global__ void k_attn_scores_tile(const uint16_t* __restrict__ Q, const uint16_t* __restrict__ KC,
                                   float* __restrict__ S, int T, int pos_base, int t0, int tile, int nkeys) {
  long total = (long)QH * tile * nkeys;
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= total) return;
  int j = (int)(i % nkeys);
  int tr = (int)((i / nkeys) % tile);
  int h = (int)(i / ((long)nkeys * tile));
  int t = t0 + tr;
  if (t >= T) return;
  int pos = pos_base + t;
  if (j > pos) return;
  int kvh = h / (QH / KVH);
  const uint16_t* qp = Q + ((long)t * QH + h) * HD;
  const uint16_t* kp = KC + ((long)j * KVH + kvh) * HD;
  float s = 0.f;
  for (int d = 0; d < HD; d++) s += bf16f(qp[d]) * bf16f(kp[d]);
  S[((long)h * tile + tr) * nkeys + j] = s * SCALE;
}
__global__ void k_attn_softmax_tile(const float* __restrict__ S, uint16_t* __restrict__ Pbf,
                                    int T, int pos_base, int t0, int tile, int nkeys) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)QH * tile) return;
  int tr = (int)(i % tile), h = (int)(i / tile);
  int t = t0 + tr;
  if (t >= T) return;
  int pos = pos_base + t;
  const float* row = S + ((long)h * tile + tr) * nkeys;
  uint16_t* pb = Pbf + ((long)h * tile + tr) * nkeys;
  float mx = -3.0e38f;
  for (int j = 0; j <= pos; j++) mx = fmaxf(mx, row[j]);
  float sum = 0.f;
  for (int j = 0; j <= pos; j++) sum += expf(row[j] - mx);
  float inv = 1.0f / sum;
  for (int j = 0; j <= pos; j++) pb[j] = f32bf(expf(row[j] - mx) * inv);
}
__global__ void k_attn_pv_tile(const uint16_t* __restrict__ Pbf, const uint16_t* __restrict__ VC,
                               uint16_t* __restrict__ O, int T, int pos_base, int t0, int tile, int nkeys) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)tile * QH * HD) return;
  int d = (int)(i % HD), h = (int)((i / HD) % QH), tr = (int)(i / ((long)QH * HD));
  int t = t0 + tr;
  if (t >= T) return;
  int pos = pos_base + t;
  int kvh = h / (QH / KVH);
  const uint16_t* pb = Pbf + ((long)h * tile + tr) * nkeys;
  float acc = 0.f;
  for (int j = 0; j <= pos; j++) acc += bf16f(pb[j]) * bf16f(VC[((long)j * KVH + kvh) * HD + d]);
  O[((long)t * QH + h) * HD + d] = f32bf(acc);
}
// Serial reference: one thread recomputes each score three times. Same device
// expf and f32bf as the tile path, including a nonzero pos_base.
__global__ void k_attn_ref(const uint16_t* __restrict__ Q, const uint16_t* __restrict__ KC,
                           const uint16_t* __restrict__ VC, uint16_t* __restrict__ O,
                           int T, int pos_base) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * QH * HD) return;
  int d = (int)(i % HD), h = (int)((i / HD) % QH), t = (int)(i / ((long)QH * HD));
  int pos = pos_base + t;
  int kvh = h / (QH / KVH);
  const uint16_t* qp = Q + ((long)t * QH + h) * HD;
  float mx = -3.0e38f;
  for (int j = 0; j <= pos; j++) {
    const uint16_t* kp = KC + ((long)j * KVH + kvh) * HD;
    float s = 0.f;
    for (int dd = 0; dd < HD; dd++) s += bf16f(qp[dd]) * bf16f(kp[dd]);
    mx = fmaxf(mx, s * SCALE);
  }
  float sum = 0.f;
  for (int j = 0; j <= pos; j++) {
    const uint16_t* kp = KC + ((long)j * KVH + kvh) * HD;
    float s = 0.f;
    for (int dd = 0; dd < HD; dd++) s += bf16f(qp[dd]) * bf16f(kp[dd]);
    sum += expf(s * SCALE - mx);
  }
  float inv = 1.0f / sum;
  float acc = 0.f;
  for (int j = 0; j <= pos; j++) {
    const uint16_t* kp = KC + ((long)j * KVH + kvh) * HD;
    float s = 0.f;
    for (int dd = 0; dd < HD; dd++) s += bf16f(qp[dd]) * bf16f(kp[dd]);
    float p = bf16f(f32bf(expf(s * SCALE - mx) * inv));
    acc += p * bf16f(VC[((long)j * KVH + kvh) * HD + d]);
  }
  O[((long)t * QH + h) * HD + d] = f32bf(acc);
}

// ---------------- FA output gate ----------------
__global__ void k_fa_gate(uint16_t* __restrict__ A, const uint16_t* __restrict__ QG, long T) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * QH * HD) return;
  int d = (int)(i % HD), h = (int)((i / HD) % QH), t = (int)(i / ((long)QH * HD));
  float g = bf16f(QG[(long)t * (QH * HD * 2) + (long)h * HD * 2 + HD + d]);
  float sg = 1.0f / (1.0f + expf(-g));
  A[i] = f32bf(bf16f(A[i]) * bf16f(f32bf(sg)));
}

// ---------------- GDN output gated RMSNorm ----------------
__global__ void k_gdn_gatednorm(const uint16_t* __restrict__ CO, const uint16_t* __restrict__ Z,
                                const uint16_t* __restrict__ w, uint16_t* __restrict__ O, long T) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * HV) return;
  int t = (int)(i / HV), h = (int)(i % HV);
  const uint16_t* hp = CO + (long)t * (HV * DV) + h * DV;
  const uint16_t* zp = Z + (long)t * (HV * DV) + h * DV;
  float ss = 0.f;
  for (int d = 0; d < DV; d++) { float v = bf16f(hp[d]); ss += v * v; }
  float inv = rsqrtf(ss / (float)DV + EPS);
  for (int d = 0; d < DV; d++) {
    float hn = bf16f(hp[d]) * inv;
    uint16_t hb = f32bf(hn);                 // f32 norm -> bf16 RNE, then weight (HF order)
    float m = bf16f(hb) * bf16f(w[d]);       // RMSNormGated: direct w (not zero-centered)
    uint16_t mb = f32bf(m);
    float z = bf16f(zp[d]);
    float sg = z / (1.0f + expf(-z));        // silu(z) in f32
    O[(long)t * (HV * DV) + h * DV + d] = f32bf(bf16f(mb) * sg);
  }
}

// exact device gate computation (matches engine k_dn_gates semantics)
__global__ void k_gdn_gates(const uint16_t* __restrict__ ABa, const uint16_t* __restrict__ ABb,
                            const float* __restrict__ al, const float* __restrict__ dt,
                            float* __restrict__ Gout, float* __restrict__ Bout, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  int h = (int)(i % HV);
  float a = bf16f(ABa[i]) + dt[h];
  float sp = a > 20.0f ? a : log1pf(expf(a));
  float G = -expf(al[h]) * sp;
  float b = bf16f(ABb[i]);
  float beta = 1.0f / (1.0f + expf(-b));
  Gout[i] = G;
  Bout[i] = beta;
}

// ---------------- MLP act ----------------
// engine-default W4A4: activations quantized before the q4c GEMMs (k_actq).
// Emulate: symmetric int4 grid, per-row bf16 scale, RNE round, then dequant to bf16.
__global__ void k_actq(uint16_t* __restrict__ X, int K, int mode) {
  // one block per row
  __shared__ float sh[256];
  int tid = threadIdx.x;
  const uint16_t* row = X + (long)blockIdx.x * K;
  float am = 0.f;
  for (int c = tid; c < K; c += 256) { float v = bf16f(row[c]); am = fmaxf(am, fabsf(v)); }
  sh[tid] = am; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) { if (tid < off) sh[tid] = fmaxf(sh[tid], sh[tid + off]); __syncthreads(); }
  float scale = bf16f(f32bf(sh[0] / 7.0f));
  if (scale <= 0.f) scale = 1.0f;
  uint16_t* orow = X + (long)blockIdx.x * K;
  for (int c = tid; c < K; c += 256) {
    float v = bf16f(orow[c]);
    float q = rintf(v / scale);
    q = fminf(fmaxf(q, -8.f), 7.f);
    orow[c] = f32bf(q * scale);
  }
}
__global__ void k_silu_mul(const uint16_t* __restrict__ G, const uint16_t* __restrict__ U,
                           uint16_t* __restrict__ Hh, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = bf16f(G[i]);
  float sg = g / (1.0f + expf(-g));
  uint16_t gb = f32bf(sg);
  Hh[i] = f32bf(bf16f(gb) * bf16f(U[i]));
}

// ---------------- i4l (Hadamard-rotated int4) ----------------
// Weights stored as: N rows x K/2 bytes of 4-bit two's-complement codes
// (lo nib = even col), then N x (K/256) fp16 scales. Dequant gives W'
// (the rotated weights); W[n, 256c:(c+1)256] = W'[same] @ Hadamard256/16.
// Acts are rotated by the same H before quantization (k_actq_had), so
// fp16 GEMM on (a', W') equals the engine's int4 W4A4 products exactly.
__device__ inline float fp16f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  if (exp == 0) { float d = (float)man * (1.0f / 16777216.0f); return sign ? -d : d; }
  if (exp == 31) return man ? nanf("1") : (sign ? -INFINITY : INFINITY);
  uint32_t u = sign | ((exp - 15 + 127) << 23) | (man << 13);
  return __uint_as_float(u);
}

__global__ void k_dequant_i4l(const uint8_t* __restrict__ pay,
                              const uint16_t* __restrict__ sc,
                              uint16_t* __restrict__ out, int cols, long nr) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long n = nr * (long)cols;
  if (i >= n) return;
  int col = (int)(i % cols);
  long row = i / cols;
  uint8_t b = pay[row * (cols >> 1) + (col >> 1)];
  int nib = (col & 1) ? (int)(b >> 4) : (int)(b & 0xF);
  int code = (nib > 7) ? nib - 16 : nib;
  float s = fp16f(sc[row * (cols >> 8) + (col >> 8)]);
  out[i] = f32bf((float)code * s);
}

// in-place Hadamard-256 along cols, per (row, 256-chunk): X = X @ H/16
__global__ void k_had_cols(uint16_t* __restrict__ X) {
  __shared__ float sh[256];
  int tid = threadIdx.x;
  long base = (long)blockIdx.x * 256;
  sh[tid] = bf16f(X[base + tid]);
  __syncthreads();
  for (int len = 1; len < 256; len <<= 1) {
    float a = sh[tid], b = sh[tid ^ len];
    __syncthreads();
    sh[tid] = ((tid & len) ? b - a : a + b);
    __syncthreads();
  }
  X[base + tid] = f32bf(sh[tid] * (1.0f / 16.0f));
}

// W4A4 exact act-quant emission (engine k_actq<true> semantics, verified
// bit-exact against the original): H-rotate the row's 256-chunk, scale =
// fp16(absmax/7), q = rint(v/scale) clamped [-7,7], emit packed codes (lo nib
// = even col) + one fp16 scale per (row, 256-chunk). Input NOT modified.
__device__ inline uint16_t f32h(float x) {
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

__global__ void k_actq_emit(const uint16_t* __restrict__ X, uint8_t* __restrict__ C,
                            uint16_t* __restrict__ S, long cols) {
  __shared__ float sh[256], sm[256];
  __shared__ uint8_t sc[256];
  int tid = threadIdx.x;
  long tok = blockIdx.x;
  int chunk = blockIdx.y;
  long base = tok * cols + (long)chunk * 256;
  sh[tid] = bf16f(X[base + tid]);
  __syncthreads();
  for (int len = 1; len < 256; len <<= 1) {
    float a = sh[tid], b = sh[tid ^ len];
    __syncthreads();
    sh[tid] = ((tid & len) ? b - a : a + b);
    __syncthreads();
  }
  float v = sh[tid] * (1.0f / 16.0f);
  sm[tid] = fabsf(v);
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sm[tid] = fmaxf(sm[tid], sm[tid + off]);
    __syncthreads();
  }
  uint16_t sb = f32h(sm[0] / 7.0f);
  float scale = fp16f(sb);
  if (tid == 0) S[tok * (cols >> 8) + chunk] = sb;
  float q = 0.f;
  if (scale > 0.f) {                    // engine: exec-masked when scale == 0
    q = rintf(v / scale);
    q = fminf(fmaxf(q, -7.f), 7.f);
  }
  sc[tid] = (uint8_t)((int)q & 0xF);
  __syncthreads();
  if (tid < 128) C[tok * (cols >> 1) + chunk * 128 + tid] =
      (uint8_t)(sc[2 * tid] | (sc[2 * tid + 1] << 4));
}

// Engine k_gemm_i4 semantics (verified vs dumped outputs): per 256-chunk exact
// int dot of the 4-bit codes, scaled by f32(act_scale * weight_scale), f32
// accumulated over chunks in ascending order, one bf16 round at the end.
__global__ void k_gemm_i4emul(const uint8_t* __restrict__ ac, const uint16_t* __restrict__ as,
                              const uint8_t* __restrict__ wc, const uint16_t* __restrict__ ws,
                              uint16_t* __restrict__ C, int N, long K) {
  int m = blockIdx.x;
  int n = blockIdx.y * blockDim.x + threadIdx.x;
  if (n >= N) return;
  long per = K >> 1;
  int nsc = (int)(K >> 8);
  const uint8_t* arow = ac + (size_t)m * per;
  const uint8_t* wrow = wc + (size_t)n * per;
  const uint16_t* asr = as + (size_t)m * nsc;
  const uint16_t* wsr = ws + (size_t)n * nsc;
  __shared__ uint8_t sab[128];
  float acc = 0.f;
  for (int c = 0; c < nsc; c++) {
    if (threadIdx.x < 128) sab[threadIdx.x] = arow[c * 128 + threadIdx.x];
    __syncthreads();
    const uint8_t* wp = wrow + c * 128;
    int dot = 0;
    #pragma unroll 8
    for (int j = 0; j < 128; j++) {
      uint8_t ab = sab[j], wb = wp[j];
      dot += ((ab & 15) - ((ab & 8) << 1)) * ((wb & 15) - ((wb & 8) << 1))
           + ((ab >> 4) - ((ab & 0x80) >> 3)) * ((wb >> 4) - ((wb & 0x80) >> 3));
    }
    acc += (float)dot * (fp16f(asr[c]) * fp16f(wsr[c]));
    __syncthreads();
  }
  C[(size_t)m * N + n] = f32bf(acc);
}

// W4A4 exact act-quant emulation: H-rotate the row's 256-chunk, quantize
// with chunk scale absmax/7, rotate back. X ends as H(quant(H x)).
__global__ void k_actq_had(uint16_t* __restrict__ X) {
  __shared__ float sh[256], sm[256];
  int tid = threadIdx.x;
  long base = (long)blockIdx.x * 256;
  sh[tid] = bf16f(X[base + tid]);
  __syncthreads();
  for (int len = 1; len < 256; len <<= 1) {
    float a = sh[tid], b = sh[tid ^ len];
    __syncthreads();
    sh[tid] = ((tid & len) ? b - a : a + b);
    __syncthreads();
  }
  float v = sh[tid] * (1.0f / 16.0f);
  sm[tid] = fabsf(v);
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sm[tid] = fmaxf(sm[tid], sm[tid + off]);
    __syncthreads();
  }
  float scale = bf16f(f32bf(sm[0] / 7.0f));
  if (scale <= 0.f) scale = 1.0f;
  float q = rintf(v / scale);
  q = fminf(fmaxf(q, -8.f), 7.f);
  sh[tid] = q * scale;
  __syncthreads();
  for (int len = 1; len < 256; len <<= 1) {
    float a = sh[tid], b = sh[tid ^ len];
    __syncthreads();
    sh[tid] = ((tid & len) ? b - a : a + b);
    __syncthreads();
  }
  X[base + tid] = f32bf(sh[tid] * (1.0f / 16.0f));
}

// ---------------- logits reduction ----------------
__global__ void k_bf16logits(float* Lg, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  Lg[i] = bf16f(f32bf(Lg[i]));
}
__global__ void k_topnll(const float* __restrict__ Lg, const int* __restrict__ rowmap,
                         const int* __restrict__ tgt, int* __restrict__ pred, float* __restrict__ nll,
                         int V, long T) {
  long row = blockIdx.x; if (row >= T) return;
  __shared__ float smx[256]; __shared__ int smi[256]; __shared__ float ss[256];
  int tid = threadIdx.x;
  const float* L = Lg + (long)rowmap[row] * V;
  float mx = -3.0e38f; int mi = 0;
  for (int v = tid; v < V; v += 256) { float x = L[v]; if (x > mx) { mx = x; mi = v; } }
  smx[tid] = mx; smi[tid] = mi;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off && smx[tid + off] > smx[tid]) { smx[tid] = smx[tid + off]; smi[tid] = smi[tid + off]; }
    __syncthreads();
  }
  float gmx = smx[0];
  int gmi = smi[0];
  float s = 0.f;
  for (int v = tid; v < V; v += 256) s += expf(L[v] - gmx);
  ss[tid] = s; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) { if (tid < off) ss[tid] += ss[tid + off]; __syncthreads(); }
  if (tid == 0) {
    pred[row] = gmi;
    nll[row] = (gmx + logf(ss[0])) - L[tgt[row]];
  }
}
__global__ void k_argmax1(const float* __restrict__ L, int* __restrict__ pred, int V) {
  __shared__ float smx[256]; __shared__ int smi[256];
  int tid = threadIdx.x;
  float mx = -3.0e38f; int mi = 0;
  for (int v = tid; v < V; v += 256) { float x = L[v]; if (x > mx) { mx = x; mi = v; } }
  smx[tid] = mx; smi[tid] = mi;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off && smx[tid + off] > smx[tid]) { smx[tid] = smx[tid + off]; smi[tid] = smi[tid + off]; }
    __syncthreads();
  }
  if (tid == 0) pred[0] = smi[0];
}

// ================= host side =================
struct Ent { char name[96]; long off, size; int dt; };
static std::vector<Ent> g_tbl;
static FILE* g_f = nullptr;
static char nm[160];

static void load_table() {
  fseek(g_f, 0x08, SEEK_SET);
  uint64_t n; if (fread(&n, 8, 1, g_f) != 1) { printf("hdr read fail\n"); exit(1); }
  fseek(g_f, 0x68, SEEK_SET);
  g_tbl.resize(n);
  std::vector<uint8_t> raw((size_t)n * 0xA0);
  if (fread(raw.data(), 1, raw.size(), g_f) != raw.size()) { printf("table read fail\n"); exit(1); }
  for (uint64_t i = 0; i < n; i++) {
    const uint8_t* e = raw.data() + i * 0xA0;
    memcpy(g_tbl[i].name, e, 96); g_tbl[i].name[95] = 0;
    memcpy(&g_tbl[i].off, e + 0x88, 8);
    memcpy(&g_tbl[i].size, e + 0x90, 8);
    uint64_t f1; memcpy(&f1, e + 0x60, 8);
    g_tbl[i].dt = (int)(f1 & 0xFFFFFFFF);
  }
}
static const Ent* find(const char* name) {
  for (auto& e : g_tbl) if (strcmp(e.name, name) == 0) return &e;
  printf("tensor not found: %s\n", name); exit(1);
}

static uint8_t*  h_stage;    // 64 MB pinned
static uint8_t*  h_stage2;   // 1 MB pinned
static uint8_t*  d_raw;      // 64 MB
static uint16_t* d_raws;     // 512 KB
static float*    d_cb;

static void rd(long off, void* dst, size_t n) {
  fseek(g_f, off, SEEK_SET);
  if (fread(dst, 1, n, g_f) != n) { printf("short read @%ld\n", off); exit(1); }
}
static void up(void* d, const void* h, size_t n) { CK(hipMemcpy(d, h, n, hipMemcpyHostToDevice)); }

// fp8r -> device bf16, row-chunked (scales at tensor tail)
static void load_fp8r(const char* name, long rows, long cols, uint16_t* d_out) {
  const Ent* e = find(name);
  long chunk = (long)(32e6 / cols); if (chunk < 1) chunk = 1;
  for (long r0 = 0; r0 < rows; r0 += chunk) {
    long nr = chunk < rows - r0 ? chunk : rows - r0;
    rd(e->off + r0 * cols, h_stage, (size_t)nr * cols);
    up(d_raw, h_stage, (size_t)nr * cols);
    rd(e->off + rows * cols + r0 * 2, h_stage2, (size_t)nr * 2);
    up(d_raws, h_stage2, (size_t)nr * 2);
    long n = nr * cols;
    k_dequant_fp8r<<<GRID1(n), 256>>>(d_raw, d_raws, d_out + r0 * cols, cols);
    CK(hipDeviceSynchronize());   // d_raw is reused next chunk
  }
}
// q4c -> device bf16, row-chunked (64B codebook + per-row [payload|scales])
static void load_q4c(const char* name, long rows, long cols, uint16_t* d_out) {
  const Ent* e = find(name);
  rd(e->off, h_stage, 64); up(d_cb, h_stage, 64);
  long per = cols / 2 + cols / 16;
  long chunk = (long)(32e6 / per); if (chunk < 1) chunk = 1;
  for (long r0 = 0; r0 < rows; r0 += chunk) {
    long nr = chunk < rows - r0 ? chunk : rows - r0;
    rd(e->off + 64 + r0 * per, h_stage, (size_t)nr * per);
    up(d_raw, h_stage, (size_t)nr * per);
    long n = nr * cols;
    k_dequant_q4c<<<GRID1(n), 256>>>(d_cb, d_raw, d_out + r0 * cols, cols);
    CK(hipDeviceSynchronize());
  }
}
// i4l -> device bf16 (unrotated W'), row-chunked:
//   [N x K/2 code bytes][N x (K/256) fp16 scales]
static void load_i4l(const char* name, long rows, long cols, uint16_t* d_out) {
  const Ent* e = find(name);
  long per = cols >> 1;
  long nsc = cols >> 8;
  long chunk = (long)(24e6 / per); if (chunk < 1) chunk = 1;
  for (long r0 = 0; r0 < rows; r0 += chunk) {
    long nr = chunk < rows - r0 ? chunk : rows - r0;
    rd(e->off + r0 * per, h_stage, (size_t)nr * per);
    up(d_raw, h_stage, (size_t)nr * per);
    rd(e->off + rows * per + r0 * nsc * 2, h_stage2, (size_t)nr * nsc * 2);
    up(d_raws, h_stage2, (size_t)nr * nsc * 2);
    long n = nr * cols;
    k_dequant_i4l<<<GRID1(n), 256>>>(d_raw, d_raws, d_out + r0 * cols, (int)cols, nr);
    k_had_cols<<<(unsigned)(nr * (cols >> 8)), 256>>>(d_out + r0 * cols);
    CK(hipDeviceSynchronize());
  }
}
static const Ent* try_find(const char* name) {
  for (auto& e : g_tbl) if (strcmp(e.name, name) == 0) return &e;
  return nullptr;
}
static void load_bf16(const char* name, long bytes, void* d_out) {
  const Ent* e = find(name);
  long chunk = 32 << 20;
  for (long o = 0; o < bytes; o += chunk) {
    long nb = chunk < bytes - o ? chunk : bytes - o;
    rd(e->off + o, h_stage, nb);
    up((uint8_t*)d_out + o, h_stage, nb);
  }
}

static void load_auto(const char* name, long rows, long cols, uint16_t* d_out) {
  if (find(name)->dt == 5) load_q4c(name, rows, cols, d_out);
  else load_fp8r(name, rows, cols, d_out);
}

// ---------------- device weight pool ----------------
// All quantized payloads resident on device; forward() dequants per use. Falls
// back to streaming (per-use file reads) when the allocation doesn't fit.
static uint8_t* d_wpool = nullptr;
static size_t g_wpool_cap = 0, g_wpool_used = 0;
struct WEnt { size_t off; long rows, cols; int dt; };
static std::map<std::string, WEnt> g_wmap;
// Optional exact bf16 prefill cache on the 128 GB unified heap. Decode keeps
// the compressed pool; cached GEMMs read the same dequantized bits as getw.
static uint16_t* d_bfpool = nullptr;
static std::map<std::string, uint16_t*> g_bfmap;

static bool pool_add(const char* name, long rows, long cols) {
  const Ent* e = find(name);
  if (g_wpool_used + (size_t)e->size > g_wpool_cap) return false;
  long chunk = 32 << 20;
  for (long o = 0; o < e->size; o += chunk) {
    long nb = chunk < e->size - o ? chunk : e->size - o;
    rd(e->off + o, h_stage, nb);
    up(d_wpool + g_wpool_used + o, h_stage, nb);
  }
  g_wmap[name] = {g_wpool_used, rows, cols, e->dt};
  g_wpool_used += (size_t)e->size;
  return true;
}

static int g_prefill = 0;  // 1 = multi-token forward: use Hadamard-rotated i4l weights

static int gate_dt5(int l) {
  char nm[176];
  snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l);
  const Ent* e = find(nm);
  return e && e->dt == 5;
}

#include "gemm_i4.h"
#include "gemm_w4a8.h"
#include "gemm_wdec.h"

// ---- W4A4 prefill (HALO_W4A4): resident .i4l planes + native iu4 WMMA GEMM ----
// 1 = all 400 projection planes (Halogen's shipped "W4A4-all-400"); 2 = the
// FFN planes of the q4c layers only. Prefills of 64 rows or more quantize the
// activation per 256-chunk (k_actq_emit: Hadamard, f16 absmax/7 scale, codes
// in [-7,7]) and call gemm_i4_launch, which is bit-identical to k_gemm_i4emul.
// Decode, speculative verify and prefills under 64 rows keep the bf16 path.
// This is a declared quality configuration, not bf16-exact.
static int g_w4a4 = 0;
// HALO_W4DEC (W4 decode, a declared quality configuration): decode and verify
// GEMVs read the resident .i4l plane instead of the fp8r/q4c plane. 1 = the
// fp8r-stored planes only (attention q/k/v/o, GDN qkv/z/out: 9.4 GB of fp8 ->
// 4.7 GB of int4 per step), 2 = every plane with an .i4l twin. The activation
// is Hadamard-rotated per 256-chunk (k_had_rows) and stays bf16 (W4A16).
static int g_w4dec = 0;
static int g_verify_merge = 0;   // HALO_VERIFY_MERGE=1: per-layer multi-row kernels in the verify pass
struct I4Plane { uint8_t* c = nullptr; uint16_t* s = nullptr; long rows = 0, cols = 0; };
static std::map<std::string, I4Plane> g_i4map;
static size_t g_i4_bytes = 0;
static bool i4_add(const char* base, long rows, long cols) {
  char full[176];
  snprintf(full, sizeof(full), "%s.i4l", base);
  const Ent* e = find(full);
  if (!e || e->dt != 8) { fprintf(stderr, "w4a4: %s missing\n", full); return false; }
  const long per = cols >> 1, nsc = cols >> 8;
  if ((size_t)e->size != (size_t)rows * per + (size_t)rows * nsc * 2) { fprintf(stderr, "w4a4: %s size mismatch\n", full); return false; }
  I4Plane p; p.rows = rows; p.cols = cols;
  CK(hipMalloc((void**)&p.c, (size_t)rows * per));
  CK(hipMalloc((void**)&p.s, (size_t)rows * nsc * 2));
  long chunk = (long)((32L << 20) / per); if (chunk < 1) chunk = 1;
  for (long r0 = 0; r0 < rows; r0 += chunk) {
    long nr = chunk < rows - r0 ? chunk : rows - r0;
    rd(e->off + r0 * per, h_stage, (size_t)nr * per);
    up(p.c + r0 * per, h_stage, (size_t)nr * per);
    rd(e->off + rows * per + r0 * nsc * 2, h_stage2, (size_t)nr * nsc * 2);
    up((uint8_t*)(p.s + r0 * nsc), h_stage2, (size_t)nr * nsc * 2);
  }
  g_i4_bytes += (size_t)e->size;
  g_i4map[base] = p;
  return true;
}
static inline const I4Plane* i4_plane(const char* base) {
  auto it = g_i4map.find(base);
  return it == g_i4map.end() ? nullptr : &it->second;
}
static void init_i4_planes();

static void getw(const char* name, long rows, long cols, uint16_t*& d_out) {
  if (!g_bfmap.empty()) {
    auto cached = g_bfmap.find(name);
    auto source = g_wmap.find(name);
    if (cached == g_bfmap.end() || source == g_wmap.end() ||
        rows != source->second.rows || cols != source->second.cols) {
      fprintf(stderr, "weight cache: missing tensor or shape mismatch %s\n", name);
      exit(1);
    }
    d_out = cached->second;
    return;
  }
  auto it = g_wmap.find(name);
  if (it != g_wmap.end()) {
    const WEnt& w = it->second;
    long n = rows * cols;
    if (w.dt == 5)
      k_dequant_q4c<<<GRID1(n), 256>>>((const float*)(d_wpool + w.off),
                                       d_wpool + w.off + 64, d_out, cols);
    else
      k_dequant_fp8r<<<GRID1(n), 256>>>(d_wpool + w.off,
                                        (const uint16_t*)(d_wpool + w.off + (size_t)rows * cols),
                                        d_out, cols);
    return;
  }
  if (g_prefill && find(name)->dt == 5) {
    // Only q4c-based tensors take the rotated-i4l W4A4 route (engine rule:
    // base dt 5 -> i4l W4A4; base dt 6 (fp8r) -> dequant + bf16 GEMM).
    char nm2[176];
    snprintf(nm2, sizeof(nm2), "%s.i4l", name);
    if (try_find(nm2)) { load_i4l(nm2, rows, cols, d_out); return; }
  }
  load_auto(name, rows, cols, d_out);
}
// skip the bf16 weight preparation for a plane the int4 path will use
static inline void getw_i4(const char* name, long rows, long cols, uint16_t*& d_out, bool i4) {
  if (i4 && g_i4map.count(name)) return;
  getw(name, rows, cols, d_out);
}

static void init_bf16_weight_cache() {
  const char* option = getenv("HALO_WEIGHT_CACHE");
  if (!option || !strcmp(option, "0")) return;
  if (strcmp(option, "1")) {
    fprintf(stderr, "HALO_WEIGHT_CACHE must be 0 or 1\n"); exit(1);
  }
  if (!d_wpool || g_wmap.empty()) {
    fprintf(stderr, "weight cache: unavailable without the resident compressed pool\n"); return;
  }
  size_t bytes = 0;
  for (const auto& entry : g_wmap) bytes += (size_t)entry.second.rows * entry.second.cols * 2;
  size_t free_bytes = 0, total_bytes = 0;
  CK(hipMemGetInfo(&free_bytes, &total_bytes));
  unsigned long long available_kb = 0;
  if (FILE* mem = fopen("/proc/meminfo", "r")) {
    char line[256];
    while (fgets(line, sizeof(line), mem))
      if (sscanf(line, "MemAvailable: %llu kB", &available_kb) == 1) break;
    fclose(mem);
  }
  constexpr size_t headroom = 8ull << 30;
  if (free_bytes < bytes + headroom || available_kb * 1024 < bytes + headroom) {
    fprintf(stderr, "weight cache: disabled need=%zu headroom=%zu hip_free=%zu mem_available=%llu\n",
            bytes, headroom, free_bytes, available_kb * 1024); return;
  }
  hipError_t rc = hipMalloc(&d_bfpool, bytes);
  if (rc != hipSuccess) {
    d_bfpool = nullptr;
    fprintf(stderr, "weight cache: allocation disabled (%s)\n", hipGetErrorString(rc)); return;
  }
  std::map<std::string, uint16_t*> ready;
  uint16_t* cursor = d_bfpool;
  double start = now_s();
  for (const auto& entry : g_wmap) {
    uint16_t* destination = cursor;
    getw(entry.first.c_str(), entry.second.rows, entry.second.cols, destination);
    ready.emplace(entry.first, cursor);
    cursor += (size_t)entry.second.rows * entry.second.cols;
  }
  CK(hipDeviceSynchronize());
  g_bfmap.swap(ready);
  fprintf(stderr, "weight cache: bf16 resident tensors=%zu bytes=%zu prepare_s=%.3f decode=compressed\n",
          g_bfmap.size(), bytes, now_s() - start);
}

// stream one i4l tensor's raw codes + fp16 scales to device staging buffers
// (used by the exact W4A4 int-dot GEMM emulation; no dequant, no unrotation)
static uint8_t* d_wc[3]; static uint16_t* d_ws[3];
static void load_i4l_raw(const char* name, long rows, long cols, int slot) {
  const Ent* e = find(name);
  long per = cols >> 1;
  long nsc = cols >> 8;
  uint8_t* dc = d_wc[slot];
  uint16_t* ds = d_ws[slot];
  long chunk = (long)(24e6 / per); if (chunk < 1) chunk = 1;
  for (long r0 = 0; r0 < rows; r0 += chunk) {
    long nr = chunk < rows - r0 ? chunk : rows - r0;
    rd(e->off + r0 * per, h_stage, (size_t)nr * per);
    up(dc + r0 * per, h_stage, (size_t)nr * per);
    rd(e->off + rows * per + r0 * nsc * 2, h_stage2, (size_t)nr * nsc * 2);
    up((uint8_t*)(ds + r0 * nsc), h_stage2, (size_t)nr * nsc * 2);
  }
  CK(hipDeviceSynchronize());
}

#include "gemv_kernels.h"
// tiled fp8r GEMM matching BLASLt accumulation (single per-element k-chain)
template<int PAIR>
__global__ void k_gemm_fp8r(const uint16_t* __restrict__ A, const uint8_t* __restrict__ Wc,
                            const uint16_t* __restrict__ Ws, uint16_t* __restrict__ C,
                            int M, int N, int K) {
  __shared__ uint16_t sw[64][64];
  __shared__ uint16_t sa[16][64];
  int n0 = blockIdx.x * 64;
  int tx = threadIdx.x, ty = threadIdx.y;
  float wscale = (n0 + (blockIdx.y ? 0 : 0)) >= 0 ? bf16f(Ws[n0 + tx + 0]) : 0.f; // placeholder
  (void)wscale;
  float acc[4];
  for (int e = 0; e < 4; e++) acc[e] = 0.f;
  int m = blockIdx.y * 16 + ty;
  for (int k0 = 0; k0 < K; k0 += 64) {
    for (int s = 0; s < 16; s++) {
      int l = (ty * 16 + tx) + s * 256;
      int r = l & 63, c = l >> 6;
      long n = (long)n0 + r, k = (long)k0 + c;
      if (n < N && k < K) {
        float w = dequant_fp8_e4m3(Wc[(size_t)n * K + k]) * bf16f(Ws[n]);
        sw[r][c] = f32bf(w);
      } else sw[r][c] = 0;
    }
    __syncthreads();
    for (int s = 0; s < 4; s++) {
      int c = tx + s * 16;
      long k = (long)k0 + c;
      sa[ty][c] = (m < M && k < K) ? A[(long)m * K + k] : (uint16_t)0;
    }
    __syncthreads();
    if (PAIR) {
      for (int kk = 0; kk < 64; kk += 2) {
        float a0 = bf16f(sa[ty][kk]), a1 = bf16f(sa[ty][kk + 1]);
        acc[0] += (a0 * bf16f(sw[tx][kk]) + a1 * bf16f(sw[tx][kk + 1]));
        acc[1] += (a0 * bf16f(sw[tx + 16][kk]) + a1 * bf16f(sw[tx + 16][kk + 1]));
        acc[2] += (a0 * bf16f(sw[tx + 32][kk]) + a1 * bf16f(sw[tx + 32][kk + 1]));
        acc[3] += (a0 * bf16f(sw[tx + 48][kk]) + a1 * bf16f(sw[tx + 48][kk + 1]));
      }
    } else {
      for (int kk = 0; kk < 64; kk++) {
        float av = bf16f(sa[ty][kk]);
        acc[0] += av * bf16f(sw[tx][kk]);
        acc[1] += av * bf16f(sw[tx + 16][kk]);
        acc[2] += av * bf16f(sw[tx + 32][kk]);
        acc[3] += av * bf16f(sw[tx + 48][kk]);
      }
    }
    __syncthreads();
  }
  for (int e = 0; e < 4; e++) {
    long n = (long)n0 + tx + e * 16;
    if (n < N) C[(long)m * N + n] = f32bf(acc[e]);
  }
}

// HALO_GEMV_VEC=1 selects the 16-byte-load variants in gemv_kernels.h. They are
// bit-identical to k_gemv_dq / k_gemv_bf16f (same strips, accumulators, weight
// values and reduction tree) and are used only when K % 16 == 0 and the row
// base is 16-byte aligned; otherwise the phase-2 kernel runs.
static int g_gemv_vec = 0;
static int g_ab_gemv = 0;   // HALO_AB_GEMV=1: GDN a/b gate projections by GEMV at T == 1
// out-of-place Hadamard-256 rotation of T bf16 rows (same arithmetic as
// k_had_cols / the rotation in k_actq_emit); grid (T, cols/256)
__global__ void k_had_rows(const uint16_t* __restrict__ X, uint16_t* __restrict__ Y, long cols) {
  __shared__ float sh[256];
  int tid = threadIdx.x;
  long base = (long)blockIdx.x * cols + (long)blockIdx.y * 256;
  sh[tid] = bf16f(X[base + tid]);
  __syncthreads();
  for (int len = 1; len < 256; len <<= 1) {
    float a = sh[tid], b = sh[tid ^ len];
    __syncthreads();
    sh[tid] = ((tid & len) ? b - a : a + b);
    __syncthreads();
  }
  Y[base + tid] = f32bf(sh[tid] * (1.0f / 16.0f));
}
static uint16_t* d_arot = nullptr;          // [8][INTM] rotated activation rows
static const uint16_t* g_arot_src = nullptr; static int g_arot_T = 0, g_arot_K = 0;
static const uint16_t* g_a8_src = nullptr; static int g_a8_T = 0, g_a8_K = 0;   // W4A8 quantization cache (a8q below)
static inline void arot_reset() { g_arot_src = nullptr; g_a8_src = nullptr; }
// rotated copy of T rows of A (length K); cached per (A, T, K) within a layer
static const uint16_t* arot(const uint16_t* A, int T, int K) {
  if (A != g_arot_src || T != g_arot_T || K != g_arot_K) {
    k_had_rows<<<dim3((unsigned)T, (unsigned)(K / 256)), 256>>>(A, d_arot, K);
    g_arot_src = A; g_arot_T = T; g_arot_K = K;
  }
  return d_arot;
}
static inline const I4Plane* w4dec_plane(const char* name, const WEnt& w) {
  if (!g_w4dec) return nullptr;
  // mode 1: exactly the attention and GDN projections (fp8r-stored); modes 2 and 3: any resident twin
  if (g_w4dec == 1 && !(strstr(name, ".self_attn.") || strstr(name, ".linear_attn."))) return nullptr;
  (void)w;
  return i4_plane(name);
}
// HALO_W4DEC=3 (W4A8-decode-all): int8 activation codes for k_gemm_w4a8, one
// quantization per distinct (A, T, K) within a layer, like arot()
static uint8_t* d_a8c = nullptr; static uint16_t* d_a8s = nullptr;   // [16][INTM] codes, [16][INTM/256] scales
static void a8q(const uint16_t* A, int T, int K) {
  if (A != g_a8_src || T != g_a8_T || K != g_a8_K) {
    w4a8::k_actq8_emit<<<dim3(16, (unsigned)(K / 256)), 256>>>(A, d_a8c, d_a8s, T, K);
    g_a8_src = A; g_a8_T = T; g_a8_K = K;
  }
}
// HALO_WMMA_DEC=1: bf16 WMMA decode GEMM for serial steps, verify rows and
// lm_head (gemm_wdec.h); per-row arithmetic independent of the row count.
static int g_wmma_dec = 0;
struct WdecInfo { bool ok = false; wdec::LvTable lv{}; float G = 0.f; };
static std::map<std::string, WdecInfo> g_wdec_cb;
static bool wdec_gemm(const char* name, const WEnt& w, const uint16_t* A, uint16_t* C, int T) {
  if (!g_wmma_dec || w.cols % wdec::KC || w.rows % 16) return false;
  int rc;
  if (w.dt == 5) {
    auto it = g_wdec_cb.find(name);
    if (it == g_wdec_cb.end()) {
      float cb[16];
      CK(hipMemcpy(cb, d_wpool + w.off, 64, hipMemcpyDeviceToHost));
      WdecInfo inf; inf.ok = wdec::lv_from_codebook(cb, &inf.lv, &inf.G);
      if (!inf.ok) fprintf(stderr, "wmma-dec: %s codebook is not G * bf16 levels; GEMV kept\n", name);
      it = g_wdec_cb.emplace(name, inf).first;
    }
    if (!it->second.ok) return false;
    rc = wdec::launch<false>(1, A, d_wpool + w.off + 64, nullptr, w.cols / 2 + w.cols / 16, it->second.lv, it->second.G, C, T, (int)w.rows, (int)w.cols);
  } else {
    rc = wdec::launch<false>(0, A, d_wpool + w.off, (const uint16_t*)(d_wpool + w.off + (size_t)w.rows * w.cols), w.cols,
                             wdec::LvTable{}, 0.f, C, T, (int)w.rows, (int)w.cols);
  }
  if (rc) { fprintf(stderr, "wmma-dec: launch failed for %s (%d)\n", name, rc); exit(1); }
  return true;
}
static uint8_t* d_lmhead_q_fwd = nullptr; static uint16_t* d_lmhead_s_fwd = nullptr;   // set at lm_head load
static void wdec_lmhead(const uint16_t* A, float* C, int T) {
  int rc = wdec::launch<true>(2, A, d_lmhead_q_fwd, d_lmhead_s_fwd, HID, wdec::LvTable{}, 0.f, C, T, VOC, HID);
  if (rc) { fprintf(stderr, "wmma-dec: lm_head launch failed (%d)\n", rc); exit(1); }
}
static void w4a8_gemm(const uint16_t* A, const I4Plane* p, uint16_t* C, int T, int N, int K) {
  a8q(A, T, K);
  int rc = w4a8::gemm_w4a8_launch(d_a8c, d_a8s, p->c, p->s, C, T, N, K, false, nullptr);
  if (rc != 0) { fprintf(stderr, "w4a8: launch failed (%d) N %d K %d T %d\n", rc, N, K, T); exit(1); }
}
static inline bool gemv_vec_ok(const void* base, long rowstride, int K) {
  return g_gemv_vec && (K % 16) == 0 && (rowstride % 16) == 0 && (((uintptr_t)base) & 15u) == 0;
}
static bool gemv_dq(const uint16_t* A, const char* name, uint16_t* C) {  if (!d_wpool) return false;
  auto it = g_wmap.find(name);
  if (it == g_wmap.end()) return false;
  const WEnt& w = it->second;
  if (!w4dec_plane(name, w) && wdec_gemm(name, w, A, C, 1)) return true;
  if (const I4Plane* p = w4dec_plane(name, w)) {
    if (g_w4dec == 3) { w4a8_gemm(A, p, C, 1, (int)w.rows, (int)w.cols); return true; }
    const uint16_t* R = arot(A, 1, (int)w.cols);
    if (g_gemv_vec >= 2) k_gemv_i4l_vec<1><<<dim3((unsigned)w.rows), 256>>>(R, p->c, p->s, C, (int)w.cols);
    else                 k_gemv_i4l_vec<0><<<dim3((unsigned)w.rows), 256>>>(R, p->c, p->s, C, (int)w.cols);
    return true;
  }
  if (w.dt == 5) {
    const long rs = w.cols / 2 + w.cols / 16;
    if (gemv_vec_ok(d_wpool + w.off + 64, rs, (int)w.cols) && (((uintptr_t)A) & 15u) == 0) {
      if (g_gemv_vec >= 2)
        k_gemv_dq_vec<1, 1><<<dim3((unsigned)w.rows), 256>>>(A, d_wpool + w.off + 64, nullptr,
                                                             (const float*)(d_wpool + w.off), rs, C, (int)w.cols);
      else
        k_gemv_dq_vec<1><<<dim3((unsigned)w.rows), 256>>>(A, d_wpool + w.off + 64, nullptr,
                                                          (const float*)(d_wpool + w.off), rs, C, (int)w.cols);
    } else
      k_gemv_dq<1><<<dim3((unsigned)w.rows), 256>>>(A, d_wpool + w.off + 64, nullptr,
                                                    (const float*)(d_wpool + w.off), rs, C, (int)w.cols);
  } else {
    const uint16_t* ws = (const uint16_t*)(d_wpool + w.off + (size_t)w.rows * w.cols);
    if (gemv_vec_ok(d_wpool + w.off, w.cols, (int)w.cols) && (((uintptr_t)A) & 15u) == 0) {
      if (g_gemv_vec >= 2)
        k_gemv_dq_vec<0, 1><<<dim3((unsigned)w.rows), 256>>>(A, d_wpool + w.off, ws, nullptr, w.cols, C, (int)w.cols);
      else
        k_gemv_dq_vec<0><<<dim3((unsigned)w.rows), 256>>>(A, d_wpool + w.off, ws, nullptr, w.cols, C, (int)w.cols);
    } else
      k_gemv_dq<0><<<dim3((unsigned)w.rows), 256>>>(A, d_wpool + w.off, ws, nullptr, w.cols, C, (int)w.cols);
  }
  return true;
}

// Speculative verify: forward() with g_verify = 1 processes T rows as T
// independent decode steps at positions pos_base..pos_base+T-1. Weights are
// streamed once for all rows (multi-row GEMVs); convolution, GDN recurrence and
// attention run row by row with the single-token kernels, writing state into
// the rings. Every row is bit-identical to a serial single-token step.
static uint8_t* d_lmhead_q = nullptr;   // HALO_LMHEAD_FP8=1: raw fp8r codes [VOC][HID]
static uint16_t* d_lmhead_s = nullptr;  // and the bf16 row scales [VOC]
static int g_verify = 0;
// DFlash2 (dflash2.h, included later): forward() captures the residual stream
// after these target layers for every row when g_df_capture is set.
static int g_df_capture = 0;
static uint16_t* d_df_aux = nullptr;                 // [5][TMAX][HID]
static int g_df_tl[5] = {5, 19, 33, 47, 61};
static inline int df_layer_index(int l) {
  for (int k = 0; k < 5; k++) if (g_df_tl[k] == l) return k;
  return -1;
}
static float* d_logits_mr = nullptr;   // [8][VOC] verify logits
static int* d_pred_mr = nullptr;       // [8]
template<int P>
static void gemv_dq_mr(const uint16_t* A, const char* name, const WEnt& w, uint16_t* C) {
  const int N = (int)w.rows, K = (int)w.cols;
  if (N % 4) { fprintf(stderr, "verify: %d rows not a multiple of 4\n", N); exit(1); }
  const bool nt = g_gemv_vec >= 2;
  if (!w4dec_plane(name, w) && wdec_gemm(name, w, A, C, P)) return;
  if (const I4Plane* p = w4dec_plane(name, w)) {
    if (g_w4dec == 3) { w4a8_gemm(A, p, C, P, N, K); return; }
    const uint16_t* R = arot(A, P, K);
    if (nt) k_gemv_dq_vec_mr<2, P, 1><<<dim3((unsigned)N / 4), 256>>>(R, p->c, p->s, nullptr, (long)K / 2, C, K, N);
    else    k_gemv_dq_vec_mr<2, P, 0><<<dim3((unsigned)N / 4), 256>>>(R, p->c, p->s, nullptr, (long)K / 2, C, K, N);
    return;
  }
  if (w.dt == 5) {
    const long rs = w.cols / 2 + w.cols / 16;
    if (nt) k_gemv_dq_vec_mr<1, P, 1><<<dim3((unsigned)N / 4), 256>>>(A, d_wpool + w.off + 64, nullptr, (const float*)(d_wpool + w.off), rs, C, K, N);
    else    k_gemv_dq_vec_mr<1, P, 0><<<dim3((unsigned)N / 4), 256>>>(A, d_wpool + w.off + 64, nullptr, (const float*)(d_wpool + w.off), rs, C, K, N);
  } else {
    const uint16_t* ws = (const uint16_t*)(d_wpool + w.off + (size_t)w.rows * w.cols);
    if (nt) k_gemv_dq_vec_mr<0, P, 1><<<dim3((unsigned)N / 4), 256>>>(A, d_wpool + w.off, ws, nullptr, w.cols, C, K, N);
    else    k_gemv_dq_vec_mr<0, P, 0><<<dim3((unsigned)N / 4), 256>>>(A, d_wpool + w.off, ws, nullptr, w.cols, C, K, N);
  }
}
static bool gemv_dq_rows(const uint16_t* A, int T, const char* name, uint16_t* C) {
  if (T == 1) return gemv_dq(A, name, C);
  auto it = g_wmap.find(name);
  if (it == g_wmap.end()) return false;
  const WEnt& w = it->second;
  switch (T) {
    case 2: gemv_dq_mr<2>(A, name, w, C); break;
    case 3: gemv_dq_mr<3>(A, name, w, C); break;
    case 4: gemv_dq_mr<4>(A, name, w, C); break;
    case 5: gemv_dq_mr<5>(A, name, w, C); break;
    case 6: gemv_dq_mr<6>(A, name, w, C); break;
    case 7: gemv_dq_mr<7>(A, name, w, C); break;
    case 8: gemv_dq_mr<8>(A, name, w, C); break;
    default: fprintf(stderr, "verify: unsupported row count %d\n", T); exit(1);
  }
  return true;
}
static void ab_gemv_rows(const uint16_t* A, const uint16_t* W, uint16_t* C, int T) {
  switch (T) {
    case 2: k_gemv_bf16_bf16out_rows<2><<<HV, 256>>>(A, W, C, HID, HV); break;
    case 3: k_gemv_bf16_bf16out_rows<3><<<HV, 256>>>(A, W, C, HID, HV); break;
    case 4: k_gemv_bf16_bf16out_rows<4><<<HV, 256>>>(A, W, C, HID, HV); break;
    case 5: k_gemv_bf16_bf16out_rows<5><<<HV, 256>>>(A, W, C, HID, HV); break;
    case 6: k_gemv_bf16_bf16out_rows<6><<<HV, 256>>>(A, W, C, HID, HV); break;
    case 7: k_gemv_bf16_bf16out_rows<7><<<HV, 256>>>(A, W, C, HID, HV); break;
    case 8: k_gemv_bf16_bf16out_rows<8><<<HV, 256>>>(A, W, C, HID, HV); break;
    default:
      for (int r = 0; r < T; r++) k_gemv_bf16_bf16out_vec<<<HV, 256>>>(A + (long)r * HID, W, C + (long)r * HV, HID);
  }
}
static void lmhead_mr(const uint16_t* A, int T) {
  if (g_wmma_dec) { d_lmhead_q_fwd = d_lmhead_q; d_lmhead_s_fwd = d_lmhead_s; wdec_lmhead(A, d_logits_mr, T); return; }
  switch (T) {
    case 2: k_gemv_lmhead_fp8r_vec_mr<2><<<VOC, 256>>>(A, d_lmhead_q, d_lmhead_s, d_logits_mr, HID, VOC); break;
    case 3: k_gemv_lmhead_fp8r_vec_mr<3><<<VOC, 256>>>(A, d_lmhead_q, d_lmhead_s, d_logits_mr, HID, VOC); break;
    case 4: k_gemv_lmhead_fp8r_vec_mr<4><<<VOC, 256>>>(A, d_lmhead_q, d_lmhead_s, d_logits_mr, HID, VOC); break;
    case 5: k_gemv_lmhead_fp8r_vec_mr<5><<<VOC, 256>>>(A, d_lmhead_q, d_lmhead_s, d_logits_mr, HID, VOC); break;
    case 6: k_gemv_lmhead_fp8r_vec_mr<6><<<VOC, 256>>>(A, d_lmhead_q, d_lmhead_s, d_logits_mr, HID, VOC); break;
    case 7: k_gemv_lmhead_fp8r_vec_mr<7><<<VOC, 256>>>(A, d_lmhead_q, d_lmhead_s, d_logits_mr, HID, VOC); break;
    case 8: k_gemv_lmhead_fp8r_vec_mr<8><<<VOC, 256>>>(A, d_lmhead_q, d_lmhead_s, d_logits_mr, HID, VOC); break;
    default: k_gemv_lmhead_fp8r_vec<<<VOC, 256>>>(A, d_lmhead_q, d_lmhead_s, d_logits_mr, HID); break;
  }
}

// ---------------- device memory ----------------
static uint16_t *d_embed, *d_lmhead;
static uint16_t *d_ln_in_all, *d_ln_post_all, *d_final_norm;
static uint16_t *d_gdn_conv, *d_gdn_a, *d_gdn_b, *d_gdn_norm;
static float *d_alog, *d_dtbias, *d_gateG, *d_gateB;
static int g_gemm_mode = 0;
static int g_deq_trunc = 0;   // 0 = gemv_dq (split-k), 1 = tiled fp8r plain, 2 = tiled fp8r pairs
static float *h_gateG_dev, *h_gateB_dev;   // host mirrors of device gates
static uint16_t *d_qnorm, *d_knorm;
static uint16_t *d_qkv, *d_z, *d_outp;
static uint16_t *d_q, *d_k, *d_v, *d_o;
static uint16_t *d_gate, *d_up, *d_down;
static uint8_t* d_qc; static uint16_t* d_qs;
static uint16_t *d_convstate;                 // [48][10240][3]
static uint16_t *d_kcache, *d_vcache;         // [16][CXT][4][256]
static uint16_t *d_x, *d_ln1, *d_ln2, *d_tmp1;
static uint16_t *d_qkvraw, *d_qkvc, *d_ab_a, *d_ab_b, *d_zout, *d_co, *d_gn;
static uint16_t *d_qg, *d_kk, *d_vv, *d_att, *d_qn, *d_kn;
static uint16_t *d_g17, *d_u17, *d_act;
static float *d_scores, *d_logits;
static uint16_t *d_pbf;
static int *d_tok, *d_tgt, *d_pred;
static int* d_drafts = nullptr;   // [8] device-side draft chain
static float *d_nll;

// host
static float* h_S;            // [48][48][128][128] GDN state f32 (GDN-ordinal layer index)
static float* d_S = nullptr; // same layout, resident across a request when HALO_GDN is on
// Speculative decoding keeps the GDN state and the conv history in rings of
// NS slots. d_S / d_convstate always point at the live slot; a verify pass of
// P rows writes the state after row r into slot live+r+1, and accepting a
// prefix of length a moves the live slot forward by a+1. Rejected slots are
// simply never read again. NS == 1 without speculation.
static float* d_S_ring = nullptr;
static uint16_t* d_conv_ring = nullptr;
static int g_ring_ns = 1, g_ring_cur = 0;
static const size_t kSlotS = (size_t)48 * 48 * 128 * 128;       // floats per state slot
static const size_t kSlotConv = (size_t)48 * 10240 * 3;           // bf16 per conv slot
static int spec_maxg() {
  static int v = -1;
  if (v < 0) {
    const char* m = getenv("HALO_MTP");
    const char* g = getenv("HALO_SPEC_MAXG");
    v = (m && atoi(m) == 1) ? (g ? atoi(g) : 4) : 0;
    const char* df = getenv("HALO_DFLASH");
    if (df && atoi(df) == 1 && v < 7) v = 7;   // DFlash2 verifies a block of 8 rows
    if (v < 0) v = 0;
    if (v > 7) v = 7;
  }
  return v;
}
static inline int ring_ns() { return spec_maxg() > 0 ? spec_maxg() + 2 : 1; }
// MTP drafts per round (HALO_SPEC_MAXG, default 4), independent of the ring size
static int mtp_gamma() {
  static int v = -1;
  if (v < 0) { const char* g = getenv("HALO_SPEC_MAXG"); v = g ? atoi(g) : 4; if (v < 1) v = 1; if (v > 7) v = 7; }
  return v;
}
static inline float* ring_S(int k) { return d_S_ring + (size_t)((g_ring_cur + k) % g_ring_ns) * kSlotS; }
static inline uint16_t* ring_conv(int k) { return d_conv_ring + (size_t)((g_ring_cur + k) % g_ring_ns) * kSlotConv; }
static inline void ring_set(int cur) {
  g_ring_cur = cur % g_ring_ns;
  d_S = ring_S(0);
  d_convstate = ring_conv(0);
}
static int g_gdn_gpu = 0;    // 1 after the init compare accepts the device scan
extern "C" int chlorine_gdn_gpu(const uint16_t* qkv, const float* gateG, const float* gateB,
                                float* S, uint16_t* out, int T, int dnprep, float qscale);
extern "C" int chlorine_gdn_gpu_async(const uint16_t* qkv, const float* gateG, const float* gateB,
                                      float* S, uint16_t* out, int T, int dnprep, float qscale);
extern "C" int chlorine_gdn_reserve(int T);
extern "C" int chlorine_gdn_step_io(const uint16_t* qkv, const float* G, const float* B,
                                    const float* Sin, float* Sout, uint16_t* out,
                                    int dnprep, float qscale, int sync);
extern "C" int chlorine_attention_decode_splitk(const uint16_t* Q, const uint16_t* K, const uint16_t* V,
                                                uint16_t* O, int pos_base);
// verify pass, HALO_VERIFY_MERGE=1: T query rows at pos_base..pos_base+T-1 in
// one launch pair, row r bit-identical to the single-row call at pos_base+r
extern "C" int chlorine_attention_decode_splitk_rows(const uint16_t* Q, const uint16_t* K, const uint16_t* V,
                                                     uint16_t* O, int pos_base, int T);
struct GdnSlots { float* p[9]; };
// T sequential GDN steps in one launch: row r reads slot p[r], writes p[r+1]
extern "C" int chlorine_gdn_step_rows(const uint16_t* qkv, const float* G, const float* B,
                                      GdnSlots slots, int T, uint16_t* out, int dnprep, float qscale, int sync);
static int g_attn_splitk = 0;          // HALO_ATTN_SPLITK=1: flash-decoding kernel for T == 1
static int g_splitk_real_checked = 0;
static bool g_prefill_async = false;
static float* h_alog;         // [48][48]
static float* h_dtbias;       // [48][48]
static uint16_t* h_qkvbf;     // [TMAX][10240]
static uint16_t* h_abbf;      // [TMAX][48]
static float* h_af;           // [TMAX][48]
static float* h_bf;           // [TMAX][48]
static uint16_t* h_co;        // [TMAX][6144]
static float* h_gateG;        // [TMAX][48] device-computed exact gates
static float* h_gateB;
static int* h_prompt; static int h_nprompt;
static float h_qscale;

static inline float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
static inline int gdn_ord(int l) { return l - (l + 1) / 4; }   // layer idx -> GDN ordinal (0..47)
static inline int fa_ord(int l)  { return l / 4; }             // layer idx -> FA ordinal (0..15)

static int g_dbg = 0;
static int g_steps = 0;
static int g_bf16logits = 0;
static int g_actq = 0;    // engine-default serve prefill is CLEAN; HALO_ACTQ=3 emulates
                          // the forced-bench W4A4 prefill (mean-NLL 5.9406 vs 5.918531)
static int g_dnprep = 1;  // 1: round l2normed q/k to bf16 (engine inter-kernel dtype)
static int g_beta32 = 0;  // 0: beta stays f32 (k_dn_gates emits f32) — matches engine clean path
static int g_flash = 0;   // HALO_FLASH_WMMA=1, explicit opt-in; default reference
static int g_flash_real_checked = 0;
extern "C" int chlorine_attention_flash(const uint16_t*, const uint16_t*,
                                         const uint16_t*, uint16_t*, int, int);
static void dbg_all(const char* tag, const uint16_t* d_buf, long rows, long width) {
  // Bounded staging even when the activation is a full 32K prefill.
  std::vector<uint16_t> tmp(std::min<long>(rows * width, 1 << 20));
  double mx = 0, sm = 0; long nan = 0;
  for (long base = 0; base < rows * width; base += (long)tmp.size()) {
    long count = std::min<long>((long)tmp.size(), rows * width - base);
    CK(hipMemcpy(tmp.data(), d_buf + base, (size_t)count * 2, hipMemcpyDeviceToHost));
    for (long i = 0; i < count; i++) {
      float v = h_bf16f(tmp[i]);
      if (v != v) nan++;
      if (fabs(v) > mx) mx = fabs(v);
      sm += v;
    }
  }
  printf("  [%s] allrows mean=%.4g maxabs=%.4g nan=%ld\n", tag, sm / (rows * width), mx, nan);
}
static void dbg_row(const char* tag, const uint16_t* d_buf, long row, long width) {
  static uint16_t tmp[5120];
  CK(hipMemcpy(tmp, d_buf + row * width, width * 2, hipMemcpyDeviceToHost));
  double mx = 0, sm = 0; int nan = 0;
  for (long i = 0; i < width; i++) {
    float v = h_bf16f(tmp[i]);
    if (v != v) nan++;
    if (fabs(v) > mx) mx = fabs(v);
    sm += v;
  }
  printf("  [%s] mean=%.4g maxabs=%.4g nan=%d v0..3= %.4g %.4g %.4g %.4g\n",
         tag, sm / width, mx, nan, h_bf16f(tmp[0]), h_bf16f(tmp[1]), h_bf16f(tmp[2]), h_bf16f(tmp[3]));
}
static void dbg_f32(const char* tag, const float* d_buf, long row, long width) {
  static float tmp[248320];
  CK(hipMemcpy(tmp, d_buf + row * width, width * 4, hipMemcpyDeviceToHost));
  double mx = 0, sm = 0; int nan = 0;
  for (long i = 0; i < width; i++) {
    float v = tmp[i];
    if (v != v) nan++;
    if (fabs(v) > mx) mx = fabs(v);
    sm += v;
  }
  printf("  [%s] mean=%.4g maxabs=%.4g nan=%d v0..3= %.4g %.4g %.4g %.4g\n",
         tag, sm / width, mx, nan, tmp[0], tmp[1], tmp[2], tmp[3]);
}

// GDN: fla-style CHUNKED gated delta rule (f32, host), chunk 64.
// Spec: torch_chunk_gated_delta_rule in qwen35_ref.py (the engine runs k_dnc_ut/scan
// kernels — the chunked algorithm; the sequential form is numerically different).
static void gdn_scan(int T, const uint16_t* qkvbf, const float* af, const float* bfv,
                     int ord, uint16_t* out) {
  const float* al = h_alog + (long)ord * HV;
  const float* dt = h_dtbias + (long)ord * HV;
  const int CS = 64;
  (void)al; (void)dt; (void)af; (void)bfv;
  const int nch = (T + CS - 1) / CS;
  const int NP = nch * CS;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int hh = 0; hh < HV; hh++) {
    float* S = h_S + ((long)ord * HV + hh) * DK * DV;
    std::vector<float> Q((long)NP * DK, 0.f), K((long)NP * DK, 0.f), V((long)NP * DV, 0.f);
    std::vector<float> G(NP, 0.f), B(NP, 0.f);
    for (int t = 0; t < T; t++) {
      const uint16_t* qp = qkvbf + (long)t * QKVW + (hh / 3) * DK;
      const uint16_t* kp = qkvbf + (long)t * QKVW + HK * DK + (hh / 3) * DK;
      const uint16_t* vp = qkvbf + (long)t * QKVW + HK * DK * 2 + hh * DV;
      float qs = 0.f, ks = 0.f;
      for (int i = 0; i < DK; i++) {
        float a = h_bf16f(qp[i]); Q[(long)t * DK + i] = a; qs += a * a;
        float b = h_bf16f(kp[i]); K[(long)t * DK + i] = b; ks += b * b;
      }
      float qi = 1.0f / sqrtf(qs + EPS) * h_qscale, ki = 1.0f / sqrtf(ks + EPS);
      for (int i = 0; i < DK; i++) {
        // engine passes l2normed q/k between kernels in bf16 (k_dnc_prep -> k_dnc_scan)
        float qv = Q[(long)t * DK + i] * qi, kv_ = K[(long)t * DK + i] * ki;
        if (g_dnprep) { qv = h_bf16f(h_f32bf(qv)); kv_ = h_bf16f(h_f32bf(kv_)); }
        Q[(long)t * DK + i] = qv; K[(long)t * DK + i] = kv_;
      }
      for (int i = 0; i < DV; i++) V[(long)t * DV + i] = h_bf16f(vp[i]);
      G[t] = h_gateG_dev[(long)t * HV + hh];
      B[t] = h_gateB_dev[(long)t * HV + hh];
    }
    std::vector<float> vb((long)NP * DV), kb((long)NP * DK);
    for (int t = 0; t < NP; t++) {
      for (int i = 0; i < DV; i++) vb[(long)t * DV + i] = V[(long)t * DV + i] * B[t];
      for (int i = 0; i < DK; i++) kb[(long)t * DK + i] = K[(long)t * DK + i] * B[t];
    }
    std::vector<float> pd((long)CS * CS), ut((long)CS * CS), intra((long)CS * CS);
    std::vector<float> nv((long)CS * DV), kcd((long)CS * DK), qd((long)CS * DK), kd((long)CS * DK);
    std::vector<float> vn((long)CS * DV), inter((long)CS * DV), cum(CS);
    for (int c = 0; c < nch; c++) {
      int t0 = c * CS;
      { float acc = 0.f;
        for (int a = 0; a < CS; a++) { acc += G[t0 + a]; cum[a] = acc; } }
      for (int a = 0; a < CS; a++) {
        for (int b = 0; b < CS; b++) {
          float p = b <= a ? expf(cum[a] - cum[b]) : 0.f;
          pd[(long)a * CS + b] = p;
          float kbd = 0.f, qkd = 0.f;
          if (p != 0.f) {
            const float* kba = &kb[(long)(t0 + a) * DK];
            const float* kb2 = &K[(long)(t0 + b) * DK];
            const float* qa = &Q[(long)(t0 + a) * DK];
            for (int i = 0; i < DK; i++) { kbd += kba[i] * kb2[i]; qkd += qa[i] * kb2[i]; }
          }
          ut[(long)a * CS + b] = kbd * p;
          intra[(long)a * CS + b] = qkd * p;
        }
      }
      // unitriangular forward substitution: x[a] = b[a] - sum_{s<a} ut[a][s] x[s]
      for (int a = 0; a < CS; a++) {
        for (int j = 0; j < DV; j++) {
          float s = vb[(long)(t0 + a) * DV + j];
          for (int sc = 0; sc < a; sc++) s -= ut[(long)a * CS + sc] * nv[(long)sc * DV + j];
          nv[(long)a * DV + j] = s;
        }
        const float* kba = &kb[(long)(t0 + a) * DK];
        float ec = expf(cum[a]);
        for (int i = 0; i < DK; i++) { kcd[(long)a * DK + i] = kba[i] * ec; qd[(long)a * DK + i] = Q[(long)(t0 + a) * DK + i] * ec; }
        float ecd = expf(cum[CS - 1] - cum[a]);
        for (int i = 0; i < DK; i++) kd[(long)a * DK + i] = K[(long)(t0 + a) * DK + i] * ecd;
      }
      for (int a = 1; a < CS; a++) {
        for (int sc = 0; sc < a; sc++) {
          float u = ut[(long)a * CS + sc];
          const float* kcs = &kcd[(long)sc * DK];
          float* kca = &kcd[(long)a * DK];
          for (int i = 0; i < DK; i++) kca[i] -= u * kcs[i];
        }
      }
      const float cdec = expf(cum[CS - 1]);
      for (int a = 0; a < CS; a++) {
        const float* kca = &kcd[(long)a * DK];
        const float* qda = &qd[(long)a * DK];
        for (int j = 0; j < DV; j++) {
          float s1 = 0.f, s2 = 0.f;
          for (int i = 0; i < DK; i++) { s1 += kca[i] * S[(long)i * DV + j]; s2 += qda[i] * S[(long)i * DV + j]; }
          vn[(long)a * DV + j] = nv[(long)a * DV + j] - s1;
          inter[(long)a * DV + j] = s2;
        }
        for (int b = 0; b <= a; b++) {
          float w = intra[(long)a * CS + b];
          if (w == 0.f) continue;
          const float* vb2 = &vn[(long)b * DV];
          for (int j = 0; j < DV; j++) inter[(long)a * DV + j] += w * vb2[j];
        }
      }
      for (int a = 0; a < CS; a++)
        for (int j = 0; j < DV; j++) {
          int t = t0 + a;
          if (t < T) out[(long)t * (HV * DV) + hh * DV + j] = h_f32bf(inter[(long)a * DV + j]);
        }
      for (int i = 0; i < DK; i++) {
        float* Sp = S + (long)i * DV;
        for (int j = 0; j < DV; j++) Sp[j] *= cdec;
      }
      for (int a = 0; a < CS; a++) {
        const float* kda = &kd[(long)a * DK];
        const float* va = &vn[(long)a * DV];
        for (int i = 0; i < DK; i++) {
          float kk_ = kda[i];
          float* Sp = S + (long)i * DV;
          for (int j = 0; j < DV; j++) Sp[j] += kk_ * va[j];
        }
      }
    }
  }
}

// Score workspace is one query tile: [QH][tile][nkeys], nkeys <= CXT.
static void attn_tiled(const uint16_t* Q, const uint16_t* KC, const uint16_t* VC,
                       uint16_t* O, int T, int pos_base) {
  if (T < 1) return;
  for (int t0 = 0; t0 < T; t0 += ATTN_TILE) {
    int tile = T - t0;
    if (tile > ATTN_TILE) tile = ATTN_TILE;
    int nkeys = pos_base + t0 + tile;
    if (nkeys > CXT) {
      fprintf(stderr, "attn: nkeys %d exceeds CXT %d\n", nkeys, CXT);
      exit(1);
    }
    k_attn_scores_tile<<<GRID1((long)QH * tile * nkeys), 256>>>(
        Q, KC, d_scores, T, pos_base, t0, tile, nkeys);
    k_attn_softmax_tile<<<GRID1((long)QH * tile), 256>>>(
        d_scores, d_pbf, T, pos_base, t0, tile, nkeys);
    k_attn_pv_tile<<<GRID1((long)tile * QH * HD), 256>>>(
        d_pbf, VC, O, T, pos_base, t0, tile, nkeys);
  }
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
}

// Numeric-change guard, not an exact-match assertion. Both maximum error and
// bf16 differences are printed, along with the changed softmax/rounding order.
static void flash_compare(const uint16_t* reference, const uint16_t* candidate,
                          long n, const char* tag, int T, int pos_base) {
  std::vector<uint16_t> a(n), b(n);
  CK(hipMemcpy(a.data(), reference, (size_t)n * 2, hipMemcpyDeviceToHost));
  CK(hipMemcpy(b.data(), candidate, (size_t)n * 2, hipMemcpyDeviceToHost));
  double mx = 0, refmax = 0;
  long diffs = 0, nonfinite = 0;
  for (long i = 0; i < n; i++) {
    double x = h_bf16f(a[i]), y = h_bf16f(b[i]);
    if (!std::isfinite(x) || !std::isfinite(y)) nonfinite++;
    mx = std::max(mx, fabs(x - y));
    refmax = std::max(refmax, fabs(x));
    diffs += a[i] != b[i];
  }
  double limit = 0.01 + 0.01 * refmax;
  fprintf(stderr, "flash compare %s T=%d pos_base=%d maxabs=%.9g diffs=%ld n=%ld refmax=%.9g limit=%.9g nonfinite=%ld semantics=online_f32_softmax_f16_tile_probs_vs_global_bf16\n",
          tag, T, pos_base, mx, diffs, n, refmax, limit, nonfinite);
  if (nonfinite || mx > limit) {
    fprintf(stderr, "flash: numeric guard failed; no token served\n");
    exit(1);
  }
}
extern "C" int chlorine_attention_flash_async(const uint16_t*, const uint16_t*,
                                               const uint16_t*, uint16_t*, int, int);
static void flash_call(const uint16_t* Q, const uint16_t* K, const uint16_t* V,
                       uint16_t* O, int T, int pos_base) {
  CK((hipError_t)chlorine_attention_flash(Q, K, V, O, T, pos_base));
}
static void attention(const uint16_t* Q, const uint16_t* K, const uint16_t* V,
                      uint16_t* O, int T, int pos_base, int layer) {
  if (!g_flash_real_checked && layer == 3) {
    if (T > 1024) {
      fprintf(stderr, "flash: validate one real layer with a <=1024 prompt before a long request\n");
      exit(1);
    }
    uint16_t* other = nullptr;
    CK(hipMalloc(&other, (size_t)T * QH * HD * 2));
    attn_tiled(Q, K, V, O, T, pos_base);
    flash_call(Q, K, V, other, T, pos_base);
    flash_compare(O, other, (long)T * QH * HD, "real_layer3", T, pos_base);
    if (g_flash) CK(hipMemcpy(O, other, (size_t)T * QH * HD * 2, hipMemcpyDeviceToDevice));
    CK(hipFree(other));
    g_flash_real_checked = 1;
  } else if (g_flash && T == 1 && g_attn_splitk && !g_splitk_real_checked && layer == 3) {
    // first real single-token step: guard the split-K decode kernel against the
    // bf16 reference exactly as the prefill flash kernel is guarded
    uint16_t* other = nullptr;
    CK(hipMalloc(&other, (size_t)QH * HD * 2));
    attn_tiled(Q, K, V, other, 1, pos_base);
    CK((hipError_t)chlorine_attention_decode_splitk(Q, K, V, O, pos_base));
    flash_compare(other, O, (long)QH * HD, "real_layer3_decode_splitk", 1, pos_base);
    CK(hipFree(other));
    g_splitk_real_checked = 1;
  } else if (g_flash) {
    if (g_prefill_async)
      CK((hipError_t)chlorine_attention_flash_async(Q,K,V,O,T,pos_base));
    else flash_call(Q, K, V, O, T, pos_base);
  } else {
    attn_tiled(Q, K, V, O, T, pos_base);
  }
}

// ---------------- forward ----------------
// T tokens at positions pos_base..; last-token logits [VOC] in d_logits.
static void forward(const int* d_tokens, int T, int pos_base, int want_logits) {
  g_prefill = (T > 1 && !g_verify);
  const char* async = getenv("HALO_PREFILL_ASYNC");
  const char* dasync = getenv("HALO_DECODE_ASYNC");
  // All kernels and hipBLASLt use the same default stream. Debug/profile and
  // host GDN retain their fences; every forward still completes at logits.
  // HALO_DECODE_ASYNC=1 applies the same deferral to the single-token step:
  // nothing on that path reads device memory from the host before the final
  // logits fence, so the per-layer fences only serialize launches.
  const bool async_ok = async && atoi(async) == 1 && g_gdn_gpu && !g_dbg && !getenv("HALO_TIME");
  g_prefill_async = async_ok && (T > 1 || (dasync && atoi(dasync) == 1));
  k_embed<<<GRID1((long)T * HID), 256>>>(d_tokens, d_embed, d_x, HID);
  if (!g_prefill_async) CK(hipDeviceSynchronize());
  if (g_dbg) { printf("after embed:\n"); dbg_row("embed", d_x, 0, HID); }
  for (int l = 0; l < NL; l++) {
    double tw = now_s();
    int fa = (l % 4 == 3);
    int gi = fa ? fa_ord(l) : gdn_ord(l);
    bool fused = ((T == 1 || g_verify) && d_wpool);   // decode step: fused dequant-GEMV, no dequanted W
    arot_reset();
    int w4 = g_actq && T >= 64 && d_wc[0] && gate_dt5(l);
    const bool i4 = g_w4a4 && !fused && T >= 64 && !w4;
    const uint16_t* i4_lastA = nullptr; int i4_lastK = 0;
    // activation quantization once per distinct input tensor of this layer
    auto proj = [&](const uint16_t* A, const char* name, const uint16_t* W, uint16_t* C, int N, int K) {
      const I4Plane* p = i4 ? i4_plane(name) : nullptr;
      if (!p) { gemm(A, W, C, T, N, K); return; }
      if (A != i4_lastA || K != i4_lastK) {
        k_actq_emit<<<dim3(T, K / 256), 256>>>(A, d_qc, d_qs, K);
        i4_lastA = A; i4_lastK = K;
      }
      int rc = gemm_i4_launch(d_qc, d_qs, p->c, p->s, C, T, N, K, nullptr);
      if (rc != 0) { fprintf(stderr, "w4a4: gemm_i4_launch failed for %s (%d)\n", name, rc); exit(1); }
    };
    // ---- load weights for this layer (or fused from the pool) ----
    if (fa) {
      if (!fused) {
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_proj.weight", l);  getw_i4(nm, QH * HD * 2, HID, d_q, i4);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_proj.weight", l);  getw_i4(nm, KVH * HD, HID, d_k, i4);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.v_proj.weight", l);  getw_i4(nm, KVH * HD, HID, d_v, i4);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.o_proj.weight", l);  getw_i4(nm, HID, QH * HD, d_o, i4);
      }
    } else {
      if (!fused) {
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l); getw_i4(nm, QKVW, HID, d_qkv, i4);
        { const char* dp = getenv("HALO_DUMP_WQ");
          if (dp && l == 0) { CK(hipDeviceSynchronize()); void* hb = malloc((long)QKVW * HID * 2);
            CK(hipMemcpy(hb, d_qkv, (long)QKVW * HID * 2, hipMemcpyDeviceToHost));
            FILE* f = fopen(dp, "wb"); fwrite(hb, 1, (long)QKVW * HID * 2, f); fclose(f); free(hb); }
          if (getenv("HALO_DEQ_DBG") && l == 0) {
            uint16_t v; CK(hipMemcpy(&v, d_qkv + 5L * HID + 16, 2, hipMemcpyDeviceToHost));
            printf("[ours qkv(5,16)] = %u (%f)\n", (unsigned)v, h_bf16f(v)); } }
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l);   getw_i4(nm, HV * DV, HID, d_z, i4);
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.out_proj.weight", l);    getw_i4(nm, HID, HV * DV, d_outp, i4);
      }
    }
    if (!fused) {
      if (w4) {
        snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight.i4l", l);
        load_i4l_raw(nm, INTM, HID, 0);
        snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight.i4l", l);
        load_i4l_raw(nm, INTM, HID, 1);
        snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight.i4l", l);
        load_i4l_raw(nm, HID, INTM, 2);
      } else {
        snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l); getw_i4(nm, INTM, HID, d_gate, i4);
        snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight", l);   getw_i4(nm, INTM, HID, d_up, i4);
        snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight", l); getw_i4(nm, HID, INTM, d_down, i4);
      }
      if (!g_prefill_async) CK(hipDeviceSynchronize());
    }
    double tload = now_s() - tw;

    // ---- input layernorm ----
    dumpx(d_x);
    k_rmsnorm<<<T, 256>>>(d_x, d_ln_in_all + (long)l * HID, d_ln1, HID, T, 1);
    if (!g_prefill_async) CK(hipDeviceSynchronize());

    double tc = now_s();
    if (!fa) {
      // ---- GDN branch ----
      { const char* qp = getenv("HALO_GEMM_MODE"); if (qp) g_gemm_mode = atoi(qp); }
      { const char* qp = getenv("HALO_DEQ_TRUNC"); if (qp) g_deq_trunc = atoi(qp); }
      { const char* qp = getenv("HALO_QKV_IN");
        if (qp && l == 0 && T > 1) {
          FILE* fp = fopen(qp, "rb");
          if (fp) { void* hb = malloc((long)T * QKVW * 2);
            size_t rn = fread(hb, 1, (long)T * QKVW * 2, fp); fclose(fp);
            CK(hipMemcpy(d_qkvraw, hb, rn, hipMemcpyHostToDevice)); free(hb);
            fprintf(stderr, "[qkv-in] replaced d_qkvraw with %zu bytes\n", rn); }
        } else if (fused) {
        if (g_gemm_mode) {
          const WEnt* wq = &g_wmap.at((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l), nm));
          const WEnt* wz = &g_wmap.at((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l), nm));
          int Nq = (int)wq->rows, Nz = (int)wz->rows;
          const uint16_t* wqs = (const uint16_t*)(d_wpool + wq->off + (size_t)wq->rows * wq->cols);
          const uint16_t* wzs = (const uint16_t*)(d_wpool + wz->off + (size_t)wz->rows * wz->cols);
          dim3 gq((unsigned)((Nq + 63) / 64), (unsigned)T), gz((unsigned)((Nz + 63) / 64), (unsigned)T);
          if (g_gemm_mode == 1) {
            k_gemm_fp8r<0><<<gq, dim3(16, 16)>>>(d_ln1, d_wpool + wq->off, wqs, d_qkvraw, T, Nq, (int)wq->cols);
            k_gemm_fp8r<0><<<gz, dim3(16, 16)>>>(d_ln1, d_wpool + wz->off, wzs, d_zout, T, Nz, (int)wz->cols);
          } else {
            k_gemm_fp8r<1><<<gq, dim3(16, 16)>>>(d_ln1, d_wpool + wq->off, wqs, d_qkvraw, T, Nq, (int)wq->cols);
            k_gemm_fp8r<1><<<gz, dim3(16, 16)>>>(d_ln1, d_wpool + wz->off, wzs, d_zout, T, Nz, (int)wz->cols);
          }
        } else {
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l); gemv_dq_rows(d_ln1, T, nm, d_qkvraw);
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l);   gemv_dq_rows(d_ln1, T, nm, d_zout);
        }
      } else {
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l); proj(d_ln1, nm, d_qkv, d_qkvraw, QKVW, HID);
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l);   proj(d_ln1, nm, d_z, d_zout, HV * DV, HID);
      } }
      if (g_verify && g_verify_merge) {
        ConvSlots cs;
        for (int r = 0; r <= T; r++) cs.p[r] = ring_conv(r) + (long)gi * QKVW * 3;
        k_conv1d_step_rows<<<QKVW / 256, 256>>>(d_qkvraw, d_gdn_conv + (long)gi * QKVW * 4, d_qkvc, cs, T);
      } else if (g_verify) {
        for (int r = 0; r < T; r++)
          k_conv1d_step_io<<<QKVW / 256, 256>>>(d_qkvraw + (long)r * QKVW, d_gdn_conv + (long)gi * QKVW * 4,
                                                d_qkvc + (long)r * QKVW,
                                                ring_conv(r) + (long)gi * QKVW * 3, ring_conv(r + 1) + (long)gi * QKVW * 3);
      } else if (T > 1) {
        k_conv1d<<<GRID1((long)T * QKVW), 256>>>(d_qkvraw, d_gdn_conv + (long)gi * QKVW * 4, d_qkvc, T,
                                                 d_convstate + (long)gi * QKVW * 3);
        k_conv_setstate<<<QKVW / 256, 256>>>(d_qkvraw, d_convstate + (long)gi * QKVW * 3, T);
      } else {
        k_conv1d_step<<<QKVW / 256, 256>>>(d_qkvraw, d_gdn_conv + (long)gi * QKVW * 4, d_qkvc,
                                           d_convstate + (long)gi * QKVW * 3);
      }
      if (fused && g_ab_gemv && g_verify && g_verify_merge && T > 1) {
        // one pass over the a/b weights for all verify rows (per row bit-identical)
        ab_gemv_rows(d_ln1, d_gdn_a + (long)gi * HV * HID, d_ab_a, T);
        ab_gemv_rows(d_ln1, d_gdn_b + (long)gi * HV * HID, d_ab_b, T);
      } else if (fused && g_ab_gemv) {
        // HALO_AB_GEMV=1: two 48-row GEMVs instead of two hipBLASLt M=1 calls
        for (int r = 0; r < T; r++) {
          k_gemv_bf16_bf16out_vec<<<HV, 256>>>(d_ln1 + (long)r * HID, d_gdn_a + (long)gi * HV * HID, d_ab_a + (long)r * HV, HID);
          k_gemv_bf16_bf16out_vec<<<HV, 256>>>(d_ln1 + (long)r * HID, d_gdn_b + (long)gi * HV * HID, d_ab_b + (long)r * HV, HID);
        }
      } else if (g_verify) {
        // same M = 1 hipBLASLt call per row as the serial step
        for (int r = 0; r < T; r++) {
          gemm(d_ln1 + (long)r * HID, d_gdn_a + (long)gi * HV * HID, d_ab_a + (long)r * HV, 1, HV, HID);
          gemm(d_ln1 + (long)r * HID, d_gdn_b + (long)gi * HV * HID, d_ab_b + (long)r * HV, 1, HV, HID);
        }
      } else {
        gemm(d_ln1, d_gdn_a + (long)gi * HV * HID, d_ab_a, T, HV, HID);
        gemm(d_ln1, d_gdn_b + (long)gi * HV * HID, d_ab_b, T, HV, HID);
      }
      if (!g_prefill_async) CK(hipDeviceSynchronize());
      { const char* dp = getenv("HALO_DUMP_QKVRAW");
        if (dp && l == 0) { void* hb = malloc((long)T * QKVW * 2);
          CK(hipDeviceSynchronize()); CK(hipMemcpy(hb, d_qkvraw, (long)T * QKVW * 2, hipMemcpyDeviceToHost));
          FILE* f = fopen(dp, "wb"); fwrite(hb, 1, (long)T * QKVW * 2, f); fclose(f); free(hb); } }
      { const char* dp = getenv("HALO_DUMP_QKVC");
        if (dp && l == 0) { void* hb = malloc(4096);
          CK(hipDeviceSynchronize()); CK(hipMemcpy(hb, d_qkvc, 4096, hipMemcpyDeviceToHost));
          FILE* f = fopen(dp, "wb"); fwrite(hb, 1, 4096, f); fclose(f); free(hb); } }
      double tscan = 0;
      int gpu_ok = 0;
      if (g_gdn_gpu) {
        double ts = now_s();
        k_gdn_gates<<<GRID1((long)T * HV), 256>>>(d_ab_a, d_ab_b, d_alog + (long)gi * HV,
                                                  d_dtbias + (long)gi * HV, d_gateG, d_gateB, (long)T * HV);
        int gr = 0;
        if (g_verify && g_verify_merge) {
          GdnSlots gs;
          for (int r = 0; r <= T; r++) gs.p[r] = ring_S(r) + ((long)gi * HV) * DK * DV;
          gr = chlorine_gdn_step_rows(d_qkvc, d_gateG, d_gateB, gs, T, d_co, g_dnprep, h_qscale, g_prefill_async ? 0 : 1);
        } else if (g_verify) {
          for (int r = 0; r < T && gr == 0; r++)
            gr = chlorine_gdn_step_io(d_qkvc + (long)r * QKVW, d_gateG + (long)r * HV, d_gateB + (long)r * HV,
                                      ring_S(r) + ((long)gi * HV) * DK * DV, ring_S(r + 1) + ((long)gi * HV) * DK * DV,
                                      d_co + (long)r * HV * DV, g_dnprep, h_qscale, g_prefill_async ? 0 : 1);
        } else {
          gr = (g_prefill_async ? chlorine_gdn_gpu_async : chlorine_gdn_gpu)(d_qkvc, d_gateG, d_gateB,
                                  d_S + ((long)gi * HV) * DK * DV,
                                  d_co, T, g_dnprep, h_qscale);
        }
        tscan = now_s() - ts;
        if (gr == 0) gpu_ok = 1;
        else {
          fprintf(stderr, "gdn gpu layer %d rc %d\n", l, gr);
          exit(1);
        }
      }
      if (!gpu_ok) {
      CK(hipMemcpy(h_qkvbf, d_qkvc, (long)T * QKVW * 2, hipMemcpyDeviceToHost));
      CK(hipMemcpy(h_abbf, d_ab_a, (long)T * HV * 2, hipMemcpyDeviceToHost));
      for (long i = 0; i < (long)T * HV; i++) h_af[i] = h_bf16f(h_abbf[i]);
      CK(hipMemcpy(h_abbf, d_ab_b, (long)T * HV * 2, hipMemcpyDeviceToHost));
      for (long i = 0; i < (long)T * HV; i++) h_bf[i] = h_bf16f(h_abbf[i]);
      double ts = now_s();
      k_gdn_gates<<<GRID1((long)T * HV), 256>>>(d_ab_a, d_ab_b, d_alog + (long)gi * HV,
                                                d_dtbias + (long)gi * HV, d_gateG, d_gateB, (long)T * HV);
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(h_gateG_dev, d_gateG, (long)T * HV * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(h_gateB_dev, d_gateB, (long)T * HV * 4, hipMemcpyDeviceToHost));
      { const char* dp = getenv("HALO_DUMP_GB");
        if (dp && l == 0) { void* hb = malloc(4096 * 2);
          CK(hipMemcpy(hb, d_gateG, 4096, hipMemcpyDeviceToHost));
          CK(hipMemcpy((char*)hb + 4096, d_gateB, 4096, hipMemcpyDeviceToHost));
          FILE* f = fopen(dp, "wb"); fwrite(hb, 1, 8192, f); fclose(f); free(hb); } }
      { const char* dp = getenv("HALO_DUMP_AB");
        if (dp && l == 0) { FILE* f = fopen(dp, "wb");
          fwrite(h_af, 4, 1024, f); fwrite(h_bf, 4, 1024, f); fclose(f); } }
      gdn_scan(T, h_qkvbf, h_af, h_bf, gi, h_co);
      tscan = now_s() - ts;
      if (g_dbg) {
        double qm = 0, cm = 0, am = 0, bm = 0;
        for (long i = 0; i < (long)T * QKVW; i++) {
          float v = h_bf16f(h_qkvbf[i]);
          if (fabs(v) > qm) qm = fabs(v);
        }
        for (long i = 0; i < (long)T * HV * DV; i++) {
          float v = h_bf16f(h_co[i]);
          if (fabs(v) > cm) cm = fabs(v);
        }
        for (long i = 0; i < (long)T * HV; i++) {
          am = fmax(am, fabs(h_af[i]));
          bm = fmax(bm, fabs(h_bf[i]));
        }
        printf("  L%d gdn: qkvconv_max=%.3g co_max=%.3g a_max=%.3g b_max=%.3g g0=%.4g beta0=%.4g\n",
               l, qm, cm, am, bm,
               -expf(h_alog[(long)gdn_ord(l) * HV]) * softplusf(h_af[0] + h_dtbias[(long)gdn_ord(l) * HV]),
               1.f / (1.f + expf(-h_bf[0])));
      }
      up(d_co, h_co, (long)T * HV * DV * 2);
      }
      { const char* dp = getenv("HALO_DUMP_CO");
        if (dp && l == 0) { void* hb = malloc(2048);
          CK(hipMemcpy(hb, d_co, 2048, hipMemcpyDeviceToHost));
          FILE* f = fopen(dp, "wb"); fwrite(hb, 1, 2048, f); fclose(f); free(hb); } }
      k_gdn_gatednorm<<<GRID1((long)T * HV), 256>>>(d_co, d_zout, d_gdn_norm + (long)gi * DV, d_gn, T);
      { const char* gp = getenv("HALO_DUMP_GN");
        if (gp && l == 0) { CK(hipDeviceSynchronize()); void* hb2 = malloc(2048);
          CK(hipMemcpy(hb2, d_gn, 2048, hipMemcpyDeviceToHost));
          FILE* f = fopen(gp, "wb"); fwrite(hb2, 1, 2048, f); fclose(f); free(hb2); } }
      if (fused) {
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.out_proj.weight", l); gemv_dq_rows(d_gn, T, nm, d_tmp1);
      } else {
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.out_proj.weight", l); proj(d_gn, nm, d_outp, d_tmp1, HID, HV * DV);
      }
      if (g_dbg) { dbg_all("zout", d_zout, T, HV * DV); dbg_all("gn", d_gn, T, HV * DV); dbg_all("gdn_attnout", d_tmp1, T, HID); }
      if (getenv("HALO_TIME")) printf("  L%d gdn load %.2fs scan %.3fs\n", l, tload, tscan);
    } else {
      // ---- FA branch ----
      if (fused) {
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_proj.weight", l);  gemv_dq_rows(d_ln1, T, nm, d_qg);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_proj.weight", l);  gemv_dq_rows(d_ln1, T, nm, d_kk);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.v_proj.weight", l);  gemv_dq_rows(d_ln1, T, nm, d_vv);
      } else {
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_proj.weight", l); proj(d_ln1, nm, d_q, d_qg, QH * HD * 2, HID);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_proj.weight", l); proj(d_ln1, nm, d_k, d_kk, KVH * HD, HID);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.v_proj.weight", l); proj(d_ln1, nm, d_v, d_vv, KVH * HD, HID);
      }
      k_headnorm<<<GRID1((long)T * QH), 256>>>(d_qg, d_qnorm + (long)gi * HD, d_qn, T, QH, QH * HD * 2, HD * 2, HD);
      k_headnorm<<<GRID1((long)T * KVH), 256>>>(d_kk, d_knorm + (long)gi * HD, d_kn, T, KVH, KVH * HD, HD, HD);
      k_rope<<<GRID1((long)T * QH * NPAIR), 256>>>(d_qn, T, QH, pos_base, ROPE_MODE);
      k_rope<<<GRID1((long)T * KVH * NPAIR), 256>>>(d_kn, T, KVH, pos_base, ROPE_MODE);
      uint16_t* kc = d_kcache + (long)gi * CXT * KVH * HD;
      uint16_t* vc = d_vcache + (long)gi * CXT * KVH * HD;
      k_kvwrite<<<GRID1((long)T * KVH * HD), 256>>>(d_kn, kc, T, pos_base);
      k_kvwrite<<<GRID1((long)T * KVH * HD), 256>>>(d_vv, vc, T, pos_base);
      // KV is written at pos_base+t before this. Query t attends keys 0..pos_base+t.
      if (g_verify && g_verify_merge && g_flash && g_attn_splitk && g_splitk_real_checked && g_flash_real_checked) {
        CK((hipError_t)chlorine_attention_decode_splitk_rows(d_qn, kc, vc, d_att, pos_base, T));
      } else if (g_verify) {
        for (int r = 0; r < T; r++)
          attention(d_qn + (long)r * QH * HD, kc, vc, d_att + (long)r * QH * HD, 1, pos_base + r, l);
      } else {
        attention(d_qn, kc, vc, d_att, T, pos_base, l);
      }
      k_fa_gate<<<GRID1((long)T * QH * HD), 256>>>(d_att, d_qg, T);
      if (fused) {
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.o_proj.weight", l); gemv_dq_rows(d_att, T, nm, d_tmp1);
      } else {
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.o_proj.weight", l); proj(d_att, nm, d_o, d_tmp1, HID, QH * HD);
      }
      if (g_dbg) dbg_all("q_rope", d_qn, T, QH * HD);
    }
    k_add<<<GRID1((long)T * HID), 256>>>(d_x, d_tmp1, d_x, (long)T * HID);

    // ---- MLP ----
    dumpx(d_x);
    k_rmsnorm<<<T, 256>>>(d_x, d_ln_post_all + (long)l * HID, d_ln2, HID, T, 1);
    if (g_dbg) dbg_all("ln2", d_ln2, T, HID);
    { const char* dp = getenv("HALO_DUMP_LN2");
      if (dp && l == 0) { void* hb = malloc(TMAX * HID * 2);
        CK(hipMemcpy(hb, d_ln2, 87L * HID * 2, hipMemcpyDeviceToHost));
        FILE* f = fopen(dp, "wb"); fwrite(hb, 1, 87L * HID * 2, f); fclose(f); free(hb); } }
    if (w4) {
      k_actq_emit<<<dim3(T, HID / 256), 256>>>(d_ln2, d_qc, d_qs, HID);
      k_gemm_i4emul<<<dim3(T, INTM / 256), 256>>>(d_qc, d_qs, d_wc[0], d_ws[0], d_g17, INTM, HID);
      k_gemm_i4emul<<<dim3(T, INTM / 256), 256>>>(d_qc, d_qs, d_wc[1], d_ws[1], d_u17, INTM, HID);
      { const char* dp = getenv("HALO_DUMP_W4");
        if (dp && l == 0) { CK(hipDeviceSynchronize());
          void* hb = malloc(34816 * 2); void* hb2 = malloc(4096);
          CK(hipMemcpy(hb, d_g17, 34816 * 2, hipMemcpyDeviceToHost));
          CK(hipMemcpy(hb2, d_qc, 4096, hipMemcpyDeviceToHost));
          FILE* f = fopen(dp, "wb"); fwrite(hb, 1, 34816 * 2, f);
          fwrite(hb2, 1, 4096, f); fclose(f); free(hb); free(hb2);
          CK(hipMemcpy(hb2, d_qs, 4096, hipMemcpyDeviceToHost));
          f = fopen("/tmp/ours_qs.bin", "wb"); fwrite(hb2, 1, 4096, f); fclose(f); free(hb2); } }
    } else if (fused) {
      snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l); gemv_dq_rows(d_ln2, T, nm, d_g17);
      snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight", l);   gemv_dq_rows(d_ln2, T, nm, d_u17);
    } else {
      snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l); proj(d_ln2, nm, d_gate, d_g17, INTM, HID);
      snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight", l);   proj(d_ln2, nm, d_up, d_u17, INTM, HID);
    }
    if (g_dbg) { dbg_all("g17", d_g17, T, INTM); dbg_all("u17", d_u17, T, INTM); }
    k_silu_mul<<<GRID1((long)T * INTM), 256>>>(d_g17, d_u17, d_act, (long)T * INTM);
    if (w4) {
      k_actq_emit<<<dim3(T, INTM / 256), 256>>>(d_act, d_qc, d_qs, INTM);
      k_gemm_i4emul<<<dim3(T, HID / 256), 256>>>(d_qc, d_qs, d_wc[2], d_ws[2], d_tmp1, HID, INTM);
    } else if (fused) {
      snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight", l); gemv_dq_rows(d_act, T, nm, d_tmp1);
    } else {
      snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight", l); proj(d_act, nm, d_down, d_tmp1, HID, INTM);
    }
    if (g_dbg) { dbg_all("mlp_act", d_act, T, INTM); dbg_all("mlp_out", d_tmp1, T, HID); }
    k_add<<<GRID1((long)T * HID), 256>>>(d_x, d_tmp1, d_x, (long)T * HID);
    if (g_df_capture) {           // DFlash2: residual stream after this layer, all T rows
      const int k = df_layer_index(l);
      if (k >= 0) CK(hipMemcpyAsync(d_df_aux + (size_t)k * TMAX * HID, d_x, (size_t)T * HID * 2, hipMemcpyDeviceToDevice, 0));
    }
    if (!g_prefill_async) CK(hipDeviceSynchronize());
    if (g_dbg) { printf("after L%d (%s):\n", l, fa ? "fa" : "gdn"); dbg_all("x", d_x, T, HID); }
    if (getenv("HALO_TIME")) printf("  L%d load %.2fs compute %.2fs\n", l, tload, now_s() - tc);
  }
  if (want_logits) {
    dumpx(d_x);
    k_rmsnorm<<<T, 256>>>(d_x, d_final_norm, d_ln1, HID, T, 1);
    // The next token is the argmax of the last row. A full T x vocab GEMM
    // writes rows nothing reads.
    if (g_verify) {
      lmhead_mr(d_ln1, T);
    } else if (d_lmhead_q && g_wmma_dec) {
      d_lmhead_q_fwd = d_lmhead_q; d_lmhead_s_fwd = d_lmhead_s;
      wdec_lmhead(d_ln1 + (long)(T - 1) * HID, d_logits, 1);
    } else if (d_lmhead_q) {
      // HALO_LMHEAD_FP8=1: the stored fp8r plane (1.27 GB) instead of the bf16
      // copy (2.54 GB); each weight is rounded to bf16 exactly as k_dequant_fp8r
      // did, so the logits are bit-identical to the bf16-copy path.
      k_gemv_lmhead_fp8r_vec<<<VOC, 256>>>(d_ln1 + (long)(T - 1) * HID, d_lmhead_q, d_lmhead_s,
                                           d_logits, HID);
    } else if (g_gemv_vec) {
      k_gemv_bf16f_vec<<<VOC, 256>>>(d_ln1 + (long)(T - 1) * HID, d_lmhead, d_logits, HID);
    } else {
      k_gemv_bf16f<<<VOC, 256>>>(d_ln1 + (long)(T - 1) * HID, d_lmhead,
                                 d_logits, HID);
    }
    if (g_bf16logits) k_bf16logits<<<GRID1(VOC), 256>>>(d_logits, VOC);
    CK(hipDeviceSynchronize());
    if (g_dbg) { printf("final:\n"); dbg_row("finalnorm", d_ln1, 0, HID); dbg_f32("logits", d_logits, 0, VOC); }
  } else {
    CK(hipDeviceSynchronize());
  }
}


// ================= MTP drafter (drafter 1) =================
// Qwen3-Next style multi-token-prediction head: fc([norm_e(embed(t)) ;
// norm_h(h)]) -> one gated full-attention layer + SwiGLU -> final norm ->
// shared lm_head. h is the trunk's final-normed hidden at the previous
// position; the MTP layer has its own KV cache indexed by position. Weights
// (q4c variant 1, NF4) are dequantized to bf16 once at init. The drafter only
// proposes tokens; the trunk verifies every one, so its numerics never change
// the output.
static int g_mtp_ok = 0;
static uint16_t *d_mw_fc, *d_mw_q, *d_mw_k, *d_mw_v, *d_mw_o, *d_mw_g, *d_mw_u, *d_mw_d;
// HALO_MTP_Q4=1: the drafter's single-row steps read the stored NF4 blobs
struct MtpQ4 { uint8_t* blob = nullptr; long rows = 0, cols = 0; };
static MtpQ4 g_mq_q, g_mq_k, g_mq_v, g_mq_o, g_mq_g, g_mq_u, g_mq_d;
static int g_mtp_q4 = 0;
static uint16_t *d_mn_e, *d_mn_h, *d_mn_in, *d_mn_post, *d_mn_q, *d_mn_k, *d_mn_final;
static uint16_t *d_mkc, *d_mvc;          // MTP KV cache [CXT][KVH][HD]
static uint16_t *d_mtp_hin;              // [TMAX][HID] trunk hidden rows fed to the MTP layer
static uint16_t *d_mtp_hout;             // [HID] MTP output hidden of the last drafted row
static uint16_t *d_hlast;                // [HID] trunk hidden of the last processed position
static int *d_mtp_ids;                   // [TMAX]
static int *d_draft_map = nullptr;       // draft vocabulary (sorted trunk ids) or null
static int g_draft_nv = 0;
static float *d_draft_prob = nullptr;    // drafter confidence of the last draft

static uint16_t* mtp_load_bf16(const char* name, long n) {
  const Ent* e = find(name);
  if (!e || e->dt != 0 || e->size != n * 2) { fprintf(stderr, "mtp: %s missing or not bf16\n", name); return nullptr; }
  uint16_t* d = nullptr;
  CK(hipMalloc((void**)&d, n * 2));
  long chunk = 32 << 20;
  for (long o = 0; o < n * 2; o += chunk) {
    long nb = chunk < n * 2 - o ? chunk : n * 2 - o;
    rd(e->off + o, h_stage, nb); up((uint8_t*)d + o, h_stage, nb);
  }
  return d;
}
static void mtp_keep_q4(const char* name, long rows, long cols, MtpQ4& q) {
  const Ent* te = find(name);
  CK(hipMalloc((void**)&q.blob, te->size));
  long chunk = 32 << 20;
  for (long o = 0; o < te->size; o += chunk) {
    long nb = chunk < te->size - o ? chunk : te->size - o;
    rd(te->off + o, h_stage, nb); up(q.blob + o, h_stage, nb);
  }
  q.rows = rows; q.cols = cols;
}
static uint16_t* mtp_load_q4c1(const char* name, long rows, long cols) {
  const Ent* e = find(name);
  const long per = cols / 2 + cols / 16;
  if (!e || e->dt != 5 || e->size != 64 + rows * per) { fprintf(stderr, "mtp: %s missing or not q4c\n", name); return nullptr; }
  uint16_t* d = nullptr;
  CK(hipMalloc((void**)&d, (size_t)rows * cols * 2));
  float* cb = nullptr; CK(hipMalloc((void**)&cb, 64));
  rd(e->off, h_stage, 64); up(cb, h_stage, 64);
  uint8_t* raw = nullptr; const long chunkrows = (long)((48L << 20) / per);
  CK(hipMalloc((void**)&raw, (size_t)chunkrows * per));
  for (long r0 = 0; r0 < rows; r0 += chunkrows) {
    long nr = chunkrows < rows - r0 ? chunkrows : rows - r0;
    rd(e->off + 64 + r0 * per, h_stage, (size_t)nr * per); up(raw, h_stage, (size_t)nr * per);
    k_dequant_q4c_v1<<<GRID1(nr * cols), 256>>>(cb, raw, d + r0 * cols, nr, cols);
    CK(hipDeviceSynchronize());
  }
  CK(hipFree(raw)); CK(hipFree(cb));
  return d;
}
static void mtp_load() {
  if (!find("mtp.fc.weight")) { fprintf(stderr, "mtp: checkpoint has no MTP head\n"); return; }
  if (!d_lmhead_q || !g_gemv_vec || !g_gdn_gpu) {
    fprintf(stderr, "mtp: speculation needs HALO_LMHEAD_FP8=1, HALO_GEMV_VEC>=1 and the device GDN scan; drafter disabled\n");
    return;
  }
  double t0 = now_s();
  d_mw_fc = mtp_load_bf16("mtp.fc.weight", (long)HID * 2 * HID);
  d_mn_e = mtp_load_bf16("mtp.pre_fc_norm_embedding.weight", HID);
  d_mn_h = mtp_load_bf16("mtp.pre_fc_norm_hidden.weight", HID);
  d_mn_final = mtp_load_bf16("mtp.norm.weight", HID);
  d_mn_in = mtp_load_bf16("mtp.layers.0.input_layernorm.weight", HID);
  d_mn_post = mtp_load_bf16("mtp.layers.0.post_attention_layernorm.weight", HID);
  d_mn_q = mtp_load_bf16("mtp.layers.0.self_attn.q_norm.weight", HD);
  d_mn_k = mtp_load_bf16("mtp.layers.0.self_attn.k_norm.weight", HD);
  d_mw_q = mtp_load_q4c1("mtp.layers.0.self_attn.q_proj.weight", QH * HD * 2, HID);
  d_mw_k = mtp_load_q4c1("mtp.layers.0.self_attn.k_proj.weight", KVH * HD, HID);
  d_mw_v = mtp_load_q4c1("mtp.layers.0.self_attn.v_proj.weight", KVH * HD, HID);
  d_mw_o = mtp_load_q4c1("mtp.layers.0.self_attn.o_proj.weight", HID, QH * HD);
  d_mw_g = mtp_load_q4c1("mtp.layers.0.mlp.gate_proj.weight", INTM, HID);
  d_mw_u = mtp_load_q4c1("mtp.layers.0.mlp.up_proj.weight", INTM, HID);
  d_mw_d = mtp_load_q4c1("mtp.layers.0.mlp.down_proj.weight", HID, INTM);
  if (!d_mw_fc || !d_mn_e || !d_mn_h || !d_mn_final || !d_mn_in || !d_mn_post || !d_mn_q || !d_mn_k ||
      !d_mw_q || !d_mw_k || !d_mw_v || !d_mw_o || !d_mw_g || !d_mw_u || !d_mw_d) return;
  { const char* q4 = getenv("HALO_MTP_Q4");
    if (q4 && atoi(q4) == 1) {
      mtp_keep_q4("mtp.layers.0.self_attn.q_proj.weight", QH * HD * 2, HID, g_mq_q);
      mtp_keep_q4("mtp.layers.0.self_attn.k_proj.weight", KVH * HD, HID, g_mq_k);
      mtp_keep_q4("mtp.layers.0.self_attn.v_proj.weight", KVH * HD, HID, g_mq_v);
      mtp_keep_q4("mtp.layers.0.self_attn.o_proj.weight", HID, QH * HD, g_mq_o);
      mtp_keep_q4("mtp.layers.0.mlp.gate_proj.weight", INTM, HID, g_mq_g);
      mtp_keep_q4("mtp.layers.0.mlp.up_proj.weight", INTM, HID, g_mq_u);
      mtp_keep_q4("mtp.layers.0.mlp.down_proj.weight", HID, INTM, g_mq_d);
      g_mtp_q4 = 1;
    } }
  CK(hipMalloc((void**)&d_mkc, (size_t)CXT * KVH * HD * 2));
  CK(hipMalloc((void**)&d_mvc, (size_t)CXT * KVH * HD * 2));
  CK(hipMalloc((void**)&d_mtp_hin, (size_t)TMAX * HID * 2));
  CK(hipMalloc((void**)&d_mtp_hout, (size_t)HID * 2));
  CK(hipMalloc((void**)&d_hlast, (size_t)HID * 2));
  CK(hipMalloc((void**)&d_mtp_ids, (size_t)TMAX * 4));
  CK(hipMalloc((void**)&d_draft_prob, 4));
  { const char* ev = getenv("HALO_MTP_VOCAB");
    const Ent* dm = find("draft_vocab_map");
    if (ev && atoi(ev) == 1 && dm && dm->dt == 3) {
      g_draft_nv = (int)(dm->size / 4);
      CK(hipMalloc((void**)&d_draft_map, dm->size));
      rd(dm->off, h_stage, dm->size); up(d_draft_map, h_stage, dm->size);
      fprintf(stderr, "mtp: draft argmax over the %d-token draft vocabulary\n", g_draft_nv);
    } }
  g_mtp_ok = 1;
  fprintf(stderr, "mtp: drafter loaded (bf16 layer %.2f GB, %.1fs), max draft %d, state ring %d slots\n",
          (double)(2L * HID * 2 * HID + 2L * (QH * HD * 2 + 2 * KVH * HD) * HID + 2L * HID * QH * HD + 6L * INTM * HID) / 1e9,
          now_s() - t0, spec_maxg(), g_ring_ns);
}
// y[T][N] = x[T][K] . W[N][K]^T with bf16 weights: GEMV for one row, hipBLASLt otherwise
static void mproj(const uint16_t* A, const uint16_t* W, uint16_t* C, int T, int N, int K,
                  const MtpQ4* q = nullptr) {
  if (T == 1 && q && q->blob)
    k_gemv_q4c1_vec<<<N, 256>>>(A, q->blob + 64, (const float*)q->blob, (long)(K / 2 + K / 16), C, K);
  else if (T == 1) k_gemv_bf16_bf16out_vec<<<N, 256>>>(A, W, C, K);
  else gemm(A, W, C, T, N, K);
}
// MTP forward over T rows: row t combines trunk hidden h[t] with token ids[t]
// at position pos0 + t and writes the MTP KV entry for that position. With
// want_logits the last row's logits go to d_logits and its argmax to d_pred.
// Leaves the final-normed MTP hidden of the last row in d_mtp_hout.
static void mtp_rows(const uint16_t* h, const int* ids, int T, int pos0, bool want_logits, int* pred_out = nullptr) {
  if (!pred_out) pred_out = d_pred;
  k_embed<<<GRID1((long)T * HID), 256>>>(ids, d_embed, d_x, HID);
  k_rmsnorm<<<T, 256>>>(d_x, d_mn_e, d_ln2, HID, T, 1);
  k_rmsnorm<<<T, 256>>>(h, d_mn_h, d_tmp1, HID, T, 1);
  k_concat2<<<GRID1((long)T * 2 * HID), 256>>>(d_ln2, d_tmp1, d_qkvraw, T, HID);
  mproj(d_qkvraw, d_mw_fc, d_x, T, HID, 2 * HID);
  k_rmsnorm<<<T, 256>>>(d_x, d_mn_in, d_ln1, HID, T, 1);
  mproj(d_ln1, d_mw_q, d_qg, T, QH * HD * 2, HID, &g_mq_q);
  mproj(d_ln1, d_mw_k, d_kk, T, KVH * HD, HID, &g_mq_k);
  mproj(d_ln1, d_mw_v, d_vv, T, KVH * HD, HID, &g_mq_v);
  k_headnorm<<<GRID1((long)T * QH), 256>>>(d_qg, d_mn_q, d_qn, T, QH, QH * HD * 2, HD * 2, HD);
  k_headnorm<<<GRID1((long)T * KVH), 256>>>(d_kk, d_mn_k, d_kn, T, KVH, KVH * HD, HD, HD);
  k_rope<<<GRID1((long)T * QH * NPAIR), 256>>>(d_qn, T, QH, pos0, ROPE_MODE);
  k_rope<<<GRID1((long)T * KVH * NPAIR), 256>>>(d_kn, T, KVH, pos0, ROPE_MODE);
  k_kvwrite<<<GRID1((long)T * KVH * HD), 256>>>(d_kn, d_mkc, T, pos0);
  k_kvwrite<<<GRID1((long)T * KVH * HD), 256>>>(d_vv, d_mvc, T, pos0);
  if (T == 1) CK((hipError_t)chlorine_attention_decode_splitk(d_qn, d_mkc, d_mvc, d_att, pos0));
  else CK((hipError_t)chlorine_attention_flash(d_qn, d_mkc, d_mvc, d_att, T, pos0));
  k_fa_gate<<<GRID1((long)T * QH * HD), 256>>>(d_att, d_qg, T);
  mproj(d_att, d_mw_o, d_tmp1, T, HID, QH * HD, &g_mq_o);
  k_add<<<GRID1((long)T * HID), 256>>>(d_x, d_tmp1, d_x, (long)T * HID);
  k_rmsnorm<<<T, 256>>>(d_x, d_mn_post, d_ln2, HID, T, 1);
  mproj(d_ln2, d_mw_g, d_g17, T, INTM, HID, &g_mq_g);
  mproj(d_ln2, d_mw_u, d_u17, T, INTM, HID, &g_mq_u);
  k_silu_mul<<<GRID1((long)T * INTM), 256>>>(d_g17, d_u17, d_act, (long)T * INTM);
  mproj(d_act, d_mw_d, d_tmp1, T, HID, INTM, &g_mq_d);
  k_add<<<GRID1((long)T * HID), 256>>>(d_x, d_tmp1, d_x, (long)T * HID);
  k_rmsnorm<<<1, 256>>>(d_x + (long)(T - 1) * HID, d_mn_final, d_mtp_hout, HID, 1, 1);
  if (want_logits) {
    if (d_draft_map) {
      k_gemv_lmhead_fp8r_vec_map<<<g_draft_nv, 256>>>(d_mtp_hout, d_lmhead_q, d_lmhead_s, d_draft_map, d_logits, HID);
      k_argmax_prob<<<1, 256>>>(d_logits, g_draft_nv, d_draft_map, pred_out, d_draft_prob);
    } else {
      k_gemv_lmhead_fp8r_vec<<<VOC, 256>>>(d_mtp_hout, d_lmhead_q, d_lmhead_s, d_logits, HID);
      k_argmax_prob<<<1, 256>>>(d_logits, VOC, nullptr, pred_out, d_draft_prob);
    }
  }
}

static void init_i4_planes() {
  double t0 = now_s();
  bool ok = true;
  for (int l = 0; l < NL && ok; l++) {
    const bool fa = (l % 4 == 3);
    // mode 3 (W4A4-ffn-out): the FFN and the output projections only; the
    // projections that feed the attention and the GDN recurrence stay bf16
    if (g_w4a4 == 1 || g_w4dec >= 1) {
      if (fa) {
        ok &= i4_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_proj.weight", l), nm), QH * HD * 2, HID);
        ok &= i4_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_proj.weight", l), nm), KVH * HD, HID);
        ok &= i4_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.v_proj.weight", l), nm), KVH * HD, HID);
      } else {
        ok &= i4_add((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l), nm), QKVW, HID);
        ok &= i4_add((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l), nm), HV * DV, HID);
      }
    }
    if (g_w4a4 == 1 || g_w4a4 == 3 || g_w4dec >= 1) {
      if (fa) ok &= i4_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.o_proj.weight", l), nm), HID, QH * HD);
      else    ok &= i4_add((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.out_proj.weight", l), nm), HID, HV * DV);
    }
    if (g_w4a4 == 1 || g_w4a4 == 3 || g_w4dec >= 2 || (g_w4a4 == 2 && gate_dt5(l))) {
      ok &= i4_add((snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l), nm), INTM, HID);
      ok &= i4_add((snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight", l), nm), INTM, HID);
      ok &= i4_add((snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight", l), nm), HID, INTM);
    }
  }
  if (!ok) { fprintf(stderr, "w4a4: plane load failed\n"); exit(1); }
  fprintf(stderr, "w4a4: %zu .i4l planes resident (%.2f GB, %.1fs), prefill mode %d (%s), decode mode %d (%s)\n",
          g_i4map.size(), g_i4_bytes / 1e9, now_s() - t0, g_w4a4,
          g_w4a4 == 1 ? "W4A4-all-400, gemm_i4_launch for T >= 64" : g_w4a4 == 2 ? "W4A4-ffn-q4c" : g_w4a4 == 3 ? "W4A4-ffn-out" : "bf16 prefill",
          g_w4dec, g_w4dec == 1 ? "W4-decode-attn-gdn (attention/GDN projections int4, MLP q4c)" : g_w4dec == 2 ? "W4-decode-all" : g_w4dec == 3 ? "W4A8-decode-all (int8 activations, iu8 WMMA)" : "bf16-exact decode");
}

// ================= engine API (used by the server and the harness) =================
// Numerics validated against the original engine (see docs/halogen/TRUNK-NOTES.md):
// clean serve path (HALO_ACTQ=0): greedy stream 271 51 1618 579 1558 (5 exact), 15/24
// tokens match, residual = MFMA accumulation-order noise; forced-bench W4A4 path
// emulated by HALO_ACTQ=3 (mean-NLL 5.9406 vs 5.918531).

static int g_ready = 0;

static void trunk_reset_states() {
  if (d_S_ring && d_conv_ring) ring_set(0);
  CK(hipMemset(d_convstate, 0, (long)48 * QKVW * 3 * 2));
  CK(hipMemset(d_kcache, 0, (long)16 * CXT * KVH * HD * 2));
  CK(hipMemset(d_vcache, 0, (long)16 * CXT * KVH * HD * 2));
  memset(h_S, 0, (size_t)48 * HV * DK * DV * 4);
  if (d_S) CK(hipMemset(d_S, 0, (size_t)48 * HV * DK * DV * 4));
}

// ================= prompt-prefix cache (phase 5) =================
// A cold prefill runs in fixed chunks of g_pf_chunk rows (HALO_PREFILL_CHUNK,
// default 2048 when the cache is on). After the chunk that ends at the largest
// multiple B of the chunk size with B <= n_prompt - 1, the state after B tokens
// is copied into a cache entry: KV rows [0, B) of the 16 attention layers, the
// live GDN state slot, the conv history, the MTP KV rows and the last hidden.
// A later prompt whose first B ids equal the entry's ids restores that copy
// and prefills only the suffix, in the same chunk shapes as a cold run, so the
// suffix rows see the same GEMM shapes and the same recurrence boundaries.
// Snapshots never alias the live state; speculative rollback mutates only the
// live rings.
#include "dflash2.h"

struct CacheEntry {
  std::vector<int> ids;
  int B = 0;
  uint16_t* kc = nullptr; uint16_t* vc = nullptr;   // [16][B][KVH][HD]
  float* S = nullptr;                               // one slot
  uint16_t* conv = nullptr;                         // [48][QKVW][3]
  uint16_t* mkc = nullptr; uint16_t* mvc = nullptr; // MTP KV [B][KVH][HD], when MTP is loaded
  uint16_t* dkc = nullptr; uint16_t* dvc = nullptr; // DFlash2 context K/V [L][B][NKV][HD], when loaded
  uint16_t* hlast = nullptr;                        // trunk final-normed hidden of position B-1
  uint64_t last_use = 0;
};
static std::vector<CacheEntry> g_cache;
static int g_cache_max = 0, g_pf_chunk = 0;
static uint64_t g_cache_tick = 0;
static long g_cs_hits = 0, g_cs_misses = 0, g_cs_stores = 0, g_cs_evicted = 0, g_cs_tokens_saved = 0, g_cs_refused = 0;
static double g_cs_store_ms = 0, g_cs_restore_ms = 0, g_cs_last_store_ms = 0, g_cs_last_restore_ms = 0;
static size_t cache_entry_bytes(const CacheEntry& ce) {
  size_t b = (size_t)16 * ce.B * KVH * HD * 2 * 2 + kSlotS * 4 + kSlotConv * 2 + (size_t)HID * 2;
  if (ce.mkc) b += (size_t)ce.B * KVH * HD * 2 * 2;
  if (ce.dkc) b += (size_t)DF_L * ce.B * DF_KW * 2 * 2;
  return b;
}
static size_t cache_bytes() { size_t b = 0; for (auto& e : g_cache) b += cache_entry_bytes(e); return b; }
static void cache_free(CacheEntry& ce) {
  if (ce.kc) (void)hipFree(ce.kc); if (ce.vc) (void)hipFree(ce.vc); if (ce.S) (void)hipFree(ce.S); if (ce.conv) (void)hipFree(ce.conv);
  if (ce.mkc) (void)hipFree(ce.mkc); if (ce.mvc) (void)hipFree(ce.mvc); if (ce.hlast) (void)hipFree(ce.hlast);
  if (ce.dkc) (void)hipFree(ce.dkc); if (ce.dvc) (void)hipFree(ce.dvc);
  ce = CacheEntry{};
}
static void cache_init() {
  const char* c = getenv("HALO_CACHE_ENTRIES");
  g_cache_max = c ? atoi(c) : 0;
  const char* p = getenv("HALO_PREFILL_CHUNK");
  g_pf_chunk = p ? atoi(p) : (g_cache_max > 0 ? 2048 : 0);
  if (g_pf_chunk % 64) { fprintf(stderr, "HALO_PREFILL_CHUNK must be a multiple of 64\n"); exit(1); }
  if (g_cache_max > 0)
    fprintf(stderr, "prompt cache: up to %d entries, prefix boundary aligned to %d tokens\n", g_cache_max, g_pf_chunk);
}
// longest cached prefix of prompt[0..n) at an aligned boundary <= n-1
static CacheEntry* cache_lookup(const int* prompt, int n) {
  CacheEntry* best = nullptr;
  for (auto& e : g_cache)
    if (e.B <= n - 1 && (!best || e.B > best->B) && memcmp(e.ids.data(), prompt, (size_t)e.B * 4) == 0) best = &e;
  return best;
}
// copy the live state after B tokens into a (new or evicted) entry
static void cache_store(const int* prompt, int B) {
  double t0 = now_s();
  for (auto& x : g_cache) if (x.B == B && memcmp(x.ids.data(), prompt, (size_t)B * 4) == 0) return;   // already cached
  CacheEntry ce;
  ce.B = B; ce.ids.assign(prompt, prompt + B);
  size_t kvb = (size_t)16 * B * KVH * HD * 2;
  if (hipMalloc((void**)&ce.kc, kvb) != hipSuccess || hipMalloc((void**)&ce.vc, kvb) != hipSuccess ||
      hipMalloc((void**)&ce.S, kSlotS * 4) != hipSuccess || hipMalloc((void**)&ce.conv, kSlotConv * 2) != hipSuccess ||
      hipMalloc((void**)&ce.hlast, (size_t)HID * 2) != hipSuccess ||
      (g_mtp_ok && (hipMalloc((void**)&ce.mkc, (size_t)B * KVH * HD * 2) != hipSuccess ||
                    hipMalloc((void**)&ce.mvc, (size_t)B * KVH * HD * 2) != hipSuccess)) ||
      (g_df_ok && (hipMalloc((void**)&ce.dkc, (size_t)DF_L * B * DF_KW * 2) != hipSuccess ||
                   hipMalloc((void**)&ce.dvc, (size_t)DF_L * B * DF_KW * 2) != hipSuccess))) {
    cache_free(ce); g_cs_refused++; return;
  }
  for (int l = 0; l < 16; l++) {
    CK(hipMemcpy(ce.kc + (size_t)l * B * KVH * HD, d_kcache + (size_t)l * CXT * KVH * HD, (size_t)B * KVH * HD * 2, hipMemcpyDeviceToDevice));
    CK(hipMemcpy(ce.vc + (size_t)l * B * KVH * HD, d_vcache + (size_t)l * CXT * KVH * HD, (size_t)B * KVH * HD * 2, hipMemcpyDeviceToDevice));
  }
  CK(hipMemcpy(ce.S, d_S, kSlotS * 4, hipMemcpyDeviceToDevice));
  CK(hipMemcpy(ce.conv, d_convstate, kSlotConv * 2, hipMemcpyDeviceToDevice));
  CK(hipMemcpy(ce.hlast, d_ln1 + (size_t)(B - 1 - (B - g_pf_chunk)) * HID, (size_t)HID * 2, hipMemcpyDeviceToDevice));
  if (ce.dkc) for (int l = 0; l < DF_L; l++) {
    CK(hipMemcpy(ce.dkc + (size_t)l * B * DF_KW, d_df_kc + (size_t)l * CXT * DF_KW, (size_t)B * DF_KW * 2, hipMemcpyDeviceToDevice));
    CK(hipMemcpy(ce.dvc + (size_t)l * B * DF_KW, d_df_vc + (size_t)l * CXT * DF_KW, (size_t)B * DF_KW * 2, hipMemcpyDeviceToDevice));
  }
  if (ce.mkc) {
    CK(hipMemcpy(ce.mkc, d_mkc, (size_t)B * KVH * HD * 2, hipMemcpyDeviceToDevice));
    CK(hipMemcpy(ce.mvc, d_mvc, (size_t)B * KVH * HD * 2, hipMemcpyDeviceToDevice));
  }
  CK(hipDeviceSynchronize());
  if ((int)g_cache.size() >= g_cache_max) {
    size_t victim = 0;
    for (size_t i = 1; i < g_cache.size(); i++) if (g_cache[i].last_use < g_cache[victim].last_use) victim = i;
    cache_free(g_cache[victim]); g_cache.erase(g_cache.begin() + (long)victim); g_cs_evicted++;
  }
  ce.last_use = ++g_cache_tick;
  g_cache.push_back(ce);
  g_cs_stores++;
  g_cs_last_store_ms = (now_s() - t0) * 1000.0; g_cs_store_ms += g_cs_last_store_ms;
}
static void cache_restore(CacheEntry& ce) {
  double t0 = now_s();
  const int B = ce.B;
  for (int l = 0; l < 16; l++) {
    CK(hipMemcpy(d_kcache + (size_t)l * CXT * KVH * HD, ce.kc + (size_t)l * B * KVH * HD, (size_t)B * KVH * HD * 2, hipMemcpyDeviceToDevice));
    CK(hipMemcpy(d_vcache + (size_t)l * CXT * KVH * HD, ce.vc + (size_t)l * B * KVH * HD, (size_t)B * KVH * HD * 2, hipMemcpyDeviceToDevice));
  }
  CK(hipMemcpy(d_S, ce.S, kSlotS * 4, hipMemcpyDeviceToDevice));
  CK(hipMemcpy(d_convstate, ce.conv, kSlotConv * 2, hipMemcpyDeviceToDevice));
  if (ce.dkc && d_df_kc) for (int l = 0; l < DF_L; l++) {
    CK(hipMemcpy(d_df_kc + (size_t)l * CXT * DF_KW, ce.dkc + (size_t)l * B * DF_KW, (size_t)B * DF_KW * 2, hipMemcpyDeviceToDevice));
    CK(hipMemcpy(d_df_vc + (size_t)l * CXT * DF_KW, ce.dvc + (size_t)l * B * DF_KW, (size_t)B * DF_KW * 2, hipMemcpyDeviceToDevice));
  }
  if (ce.mkc && d_mkc) {
    CK(hipMemcpy(d_mkc, ce.mkc, (size_t)B * KVH * HD * 2, hipMemcpyDeviceToDevice));
    CK(hipMemcpy(d_mvc, ce.mvc, (size_t)B * KVH * HD * 2, hipMemcpyDeviceToDevice));
  }
  CK(hipDeviceSynchronize());
  ce.last_use = ++g_cache_tick;
  g_cs_hits++; g_cs_tokens_saved += B;
  g_cs_last_restore_ms = (now_s() - t0) * 1000.0; g_cs_restore_ms += g_cs_last_restore_ms;
}
extern "C" int chlorine_trunk_cstat(char* buf, int cap) {
  // entries, bytes, reserved, cap, hits, misses, stores, evicted, tokens_saved,
  // store_ms, restore_ms, last_store_ms, last_restore_ms, refused
  return snprintf(buf, (size_t)cap, "C %d %lld %lld %lld %ld %ld %ld %ld %ld %.1f %.1f %.1f %.1f %ld",
                  (int)g_cache.size(), (long long)cache_bytes(), (long long)0, (long long)g_cache_max,
                  g_cs_hits, g_cs_misses, g_cs_stores, g_cs_evicted, g_cs_tokens_saved,
                  g_cs_store_ms, g_cs_restore_ms, g_cs_last_store_ms, g_cs_last_restore_ms, g_cs_refused);
}
// Prefill prompt[0..n) into the reset state, using the cache when enabled.
// Returns the number of tokens served from the cache. Afterwards d_ln1 holds the
// final-normed rows of the LAST chunk and d_logits the last row's logits.
static int prefill_prompt(const int* prompt, int n, int* prefix_hidden_rows) {
  int start = 0;
  CacheEntry* hit = g_cache_max > 0 ? cache_lookup(prompt, n) : nullptr;
  if (hit) { cache_restore(*hit); start = hit->B; }
  else if (g_cache_max > 0) g_cs_misses++;
  const int C = g_pf_chunk > 0 ? g_pf_chunk : n;
  // boundary to snapshot: largest multiple of C that is <= n-1 and beyond the restored prefix
  int snapB = g_cache_max > 0 && C > 0 ? ((n - 1) / C) * C : 0;
  if (snapB <= start) snapB = 0;
  for (int off = start; off < n; off += C) {
    int T = std::min(C, n - off);
    up(d_tok, prompt + off, (size_t)T * 4);
    forward(d_tok, T, off, 1);
    if (g_df_capture) df_ingest(off, T);
    if (snapB && off + T == snapB) cache_store(prompt, snapB);
  }
  *prefix_hidden_rows = n - ((n - start - 1) / C) * C - start;   // rows of the last chunk present in d_ln1
  return start;
}


static float gdn_rnd(uint32_t& rng) {
  rng = rng * 1664525u + 1013904223u;
  return (rng >> 8) * (1.0f / 16777216.0f);
}

// Compare the device scan with gdn_scan on synthetic tokens. Disable the
// device path unless it satisfies the strict host recurrence contract.
static void gdn_selfcheck() {
  const char* env = getenv("HALO_GDN");
  int force_off = env && atoi(env) == 0;
  g_ring_ns = ring_ns();
  hipError_t ae = hipMalloc(&d_S_ring, (size_t)g_ring_ns * 48 * HV * DK * DV * 4);
  d_S = d_S_ring;
  if (ae != hipSuccess) {
    fprintf(stderr, "gdn: d_S alloc failed (%s); host scan\n", hipGetErrorString(ae));
    d_S = nullptr; d_S_ring = nullptr;
    g_gdn_gpu = 0;
    return;
  }
  CK(hipMemset(d_S, 0, (size_t)48 * HV * DK * DV * 4));
  uint32_t rng = 1;
  double worst_o = 0, worst_s = 0;
  auto one = [&](const char* tag, int T, int dirty) {
    for (long i = 0; i < (long)T * QKVW; i++)
      h_qkvbf[i] = h_f32bf((gdn_rnd(rng) - 0.5f) * 0.25f);
    for (long i = 0; i < (long)T * HV; i++) {
      h_gateG_dev[i] = -0.03f * (0.3f + gdn_rnd(rng));
      h_gateB_dev[i] = 0.05f + 0.8f * gdn_rnd(rng);
    }
    if (!dirty) {
      memset(h_S, 0, (size_t)HV * DK * DV * 4);
      CK(hipMemset(d_S, 0, (size_t)HV * DK * DV * 4));
    }
    CK(hipMemcpy(d_qkvc, h_qkvbf, (long)T * QKVW * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_gateG, h_gateG_dev, (long)T * HV * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_gateB, h_gateB_dev, (long)T * HV * 4, hipMemcpyHostToDevice));
    gdn_scan(T, h_qkvbf, h_af, h_bf, 0, h_co);
    int rc = chlorine_gdn_gpu(d_qkvc, d_gateG, d_gateB, d_S, d_co, T, g_dnprep, h_qscale);
    std::vector<uint16_t> gout((size_t)T * HV * DV);
    std::vector<float> gs((size_t)HV * DK * DV);
    CK(hipMemcpy(gout.data(), d_co, gout.size() * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(gs.data(), d_S, gs.size() * 4, hipMemcpyDeviceToHost));
    double omax = 0, smax = 0;
    bool finite = true;
    for (size_t i = 0; i < gout.size(); i++) {
      finite = finite && std::isfinite(h_bf16f(h_co[i])) && std::isfinite(h_bf16f(gout[i]));
      double d = fabs((double)h_bf16f(h_co[i]) - (double)h_bf16f(gout[i]));
      if (d > omax) omax = d;
    }
    for (size_t i = 0; i < gs.size(); i++) {
      finite = finite && std::isfinite(h_S[i]) && std::isfinite(gs[i]);
      double d = fabs((double)h_S[i] - (double)gs[i]);
      if (d > smax) smax = d;
    }
    fprintf(stderr, "gdn check %s T=%d dirty=%d out_maxabs %.4g state_maxabs %.4g rc %d\n",
            tag, T, dirty, omax, smax, rc);
    if (rc != 0 || !finite) omax = INFINITY;
    if (omax > worst_o) worst_o = omax;
    if (smax > worst_s) worst_s = smax;
  };
  one("zero", 1, 0);
  one("zero", 65, 0);
  one("dirty", 65, 1);
  memset(h_S, 0, (size_t)48 * HV * DK * DV * 4);
  CK(hipMemset(d_S, 0, (size_t)48 * HV * DK * DV * 4));
  int pass = worst_o == 0.0 && worst_s < 1e-5;
  if (force_off || !pass) {
    g_gdn_gpu = 0;
    fprintf(stderr, "gdn: host scan (HALO_GDN=%s out %.4g state %.4g)\n",
            force_off ? "0" : "check", worst_o, worst_s);
    return;
  }
  int rr = chlorine_gdn_reserve(TMAX);
  if (rr != 0) {
    g_gdn_gpu = 0;
    fprintf(stderr, "gdn: reserve %d failed (%d); host scan\n", TMAX, rr);
    return;
  }
  g_gdn_gpu = 1;
  fprintf(stderr, "gdn: device scan on (check out %.4g state %.4g)\n", worst_o, worst_s);
}

// Bit-compare the tile path with the full-matrix kernels at pos_base 0, and
// with the serial kernel when the prefix is already in the cache.
static void attn_selfcheck() {
  const int Tm = 33;
  const int Tf = 129; // crosses both the flash query and key tile boundaries
  const int Ts = 4;
  const int ps = 5;
  const int pd = 17;
  const int npos = Tf;
  std::vector<uint16_t> hq((size_t)Tf * QH * HD);
  std::vector<uint16_t> hk((size_t)npos * KVH * HD);
  std::vector<uint16_t> hv(hk.size());
  uint32_t rng = 7;
  for (size_t i = 0; i < hq.size(); i++) hq[i] = h_f32bf(gdn_rnd(rng) * 2.f - 1.f);
  for (size_t i = 0; i < hk.size(); i++) hk[i] = h_f32bf(gdn_rnd(rng) * 2.f - 1.f);
  for (size_t i = 0; i < hv.size(); i++) hv[i] = h_f32bf(gdn_rnd(rng) * 2.f - 1.f);
  CK(hipMemcpy(d_qn, hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_kcache, hk.data(), hk.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_vcache, hv.data(), hv.size() * 2, hipMemcpyHostToDevice));

  auto cmp = [](const uint16_t* a, const uint16_t* b, long n, const char* tag) {
    std::vector<uint16_t> ha(n), hb(n);
    CK(hipMemcpy(ha.data(), a, (size_t)n * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hb.data(), b, (size_t)n * 2, hipMemcpyDeviceToHost));
    long diffs = 0;
    double mx = 0;
    for (long i = 0; i < n; i++) {
      if (ha[i] == hb[i]) continue;
      diffs++;
      double d = fabs((double)h_bf16f(ha[i]) - (double)h_bf16f(hb[i]));
      if (d > mx) mx = d;
    }
    fprintf(stderr, "attn check %s diffs %ld maxabs %.4g\n", tag, diffs, mx);
    if (diffs) exit(1);
  };

  float* sref = nullptr;
  uint16_t* pref = nullptr;
  CK(hipMalloc((void**)&sref, (long)QH * Tm * Tm * 4));
  CK(hipMalloc((void**)&pref, (long)QH * Tm * Tm * 2));
  k_attn_scores<<<GRID1((long)QH * Tm * Tm), 256>>>(d_qn, d_kcache, sref, Tm, Tm);
  k_attn_softmax<<<GRID1((long)QH * Tm), 256>>>(sref, pref, Tm, Tm);
  k_attn_pv<<<GRID1((long)Tm * QH * HD), 256>>>(pref, d_vcache, d_qg, Tm, Tm);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  attn_tiled(d_qn, d_kcache, d_vcache, d_att, Tm, 0);
  cmp(d_qg, d_att, (long)Tm * QH * HD, "matrix T=33 pos0");
  flash_call(d_qn, d_kcache, d_vcache, d_qg, Tm, 0);
  flash_compare(d_att, d_qg, (long)Tm * QH * HD, "synthetic", Tm, 0);

  k_attn_ref<<<GRID1((long)Ts * QH * HD), 256>>>(d_qn, d_kcache, d_vcache, d_qg, Ts, ps);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  attn_tiled(d_qn, d_kcache, d_vcache, d_att, Ts, ps);
  cmp(d_qg, d_att, (long)Ts * QH * HD, "serial T=4 pos5");
  flash_call(d_qn, d_kcache, d_vcache, d_qg, Ts, ps);
  flash_compare(d_att, d_qg, (long)Ts * QH * HD, "synthetic", Ts, ps);

  k_attn_ref<<<GRID1((long)QH * HD), 256>>>(d_qn, d_kcache, d_vcache, d_qg, 1, pd);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  attn_tiled(d_qn, d_kcache, d_vcache, d_att, 1, pd);
  cmp(d_qg, d_att, (long)QH * HD, "serial T=1 pos17");
  flash_call(d_qn, d_kcache, d_vcache, d_qg, 1, pd);
  flash_compare(d_att, d_qg, (long)QH * HD, "synthetic", 1, pd);
  if (g_attn_splitk && g_flash) {
    CK((hipError_t)chlorine_attention_decode_splitk(d_qn, d_kcache, d_vcache, d_qg, pd));
    flash_compare(d_att, d_qg, (long)QH * HD, "synthetic-splitk", 1, pd);
  }

  attn_tiled(d_qn, d_kcache, d_vcache, d_att, Tf, 0);
  flash_call(d_qn, d_kcache, d_vcache, d_qg, Tf, 0);
  flash_compare(d_att, d_qg, (long)Tf * QH * HD, "synthetic", Tf, 0);
  attn_tiled(d_qn, d_kcache, d_vcache, d_att, 65, 17);
  flash_call(d_qn, d_kcache, d_vcache, d_qg, 65, 17);
  flash_compare(d_att, d_qg, (long)65 * QH * HD, "synthetic", 65, 17);

  CK(hipFree(sref));
  CK(hipFree(pref));
  CK(hipMemset(d_kcache, 0, (long)16 * CXT * KVH * HD * 2));
  CK(hipMemset(d_vcache, 0, (long)16 * CXT * KVH * HD * 2));
}

extern "C" int chlorine_trunk_init(const char* ckpt_path) {
  if (g_ready) return 0;
  const char* flash_env = getenv("HALO_FLASH_WMMA");
  if (flash_env && strcmp(flash_env, "0") && strcmp(flash_env, "1")) {
    fprintf(stderr, "HALO_FLASH_WMMA must be 0 or 1\n");
    return -1;
  }
  g_flash = flash_env && !strcmp(flash_env, "1");
  fprintf(stderr, "attention: %s (HALO_FLASH_WMMA=%d)\n",
          g_flash ? "f16 WMMA online softmax" : "global bf16 reference", g_flash);
  if (getenv("HALO_ROPE")) ROPE_MODE = atoi(getenv("HALO_ROPE"));
  if (getenv("HALO_DEBUG")) g_dbg = 1;
  if (getenv("HALO_BF16LOG")) g_bf16logits = atoi(getenv("HALO_BF16LOG"));
  if (getenv("HALO_ACTQ")) g_actq = atoi(getenv("HALO_ACTQ"));
  if (getenv("HALO_GEMV_VEC")) g_gemv_vec = atoi(getenv("HALO_GEMV_VEC"));
  if (getenv("HALO_ATTN_SPLITK")) g_attn_splitk = atoi(getenv("HALO_ATTN_SPLITK"));
  if (getenv("HALO_AB_GEMV")) g_ab_gemv = atoi(getenv("HALO_AB_GEMV"));
  if (getenv("HALO_W4A4")) g_w4a4 = atoi(getenv("HALO_W4A4"));
  if (getenv("HALO_W4DEC")) g_w4dec = atoi(getenv("HALO_W4DEC"));
  if (getenv("HALO_VERIFY_MERGE")) g_verify_merge = atoi(getenv("HALO_VERIFY_MERGE"));
  if (getenv("HALO_WMMA_DEC")) g_wmma_dec = atoi(getenv("HALO_WMMA_DEC"));
  if (g_wmma_dec) fprintf(stderr, "wmma-dec: bf16 WMMA decode GEMM for serial, verify and lm_head (declared numerics change)\n");
  if (getenv("HALO_DNPREP")) g_dnprep = atoi(getenv("HALO_DNPREP"));
  if (getenv("HALO_BETA32")) g_beta32 = atoi(getenv("HALO_BETA32"));

  int ndev; CK(hipGetDeviceCount(&ndev));
  if (ndev < 1) { fprintf(stderr, "trunk: no HIP device\n"); return -1; }
  CK(hipSetDevice(0));

  double t0 = now_s();
  g_f = fopen(ckpt_path, "rb");
  if (!g_f) { fprintf(stderr, "trunk: cannot open %s\n", ckpt_path); return -1; }
  load_table();
  fprintf(stderr, "trunk: table %zu tensors\n", g_tbl.size());

  CK(hipHostMalloc(&h_stage, 64 << 20));
  CK(hipHostMalloc(&h_stage2, 1 << 20));
  CK(hipMalloc(&d_raw, 64 << 20));
  CK(hipMalloc((void**)&d_raws, 1 << 19));
  CK(hipMalloc((void**)&d_cb, 64));

  CK(hipMalloc((void**)&d_embed, (long)VOC * HID * 2));
  fprintf(stderr, "trunk: loading embed...\n");
  load_bf16("embed_tokens.weight", (long)VOC * HID * 2, d_embed);
  { const char* lq = getenv("HALO_LMHEAD_FP8");
    if (lq && atoi(lq) == 1 && find("lm_head.weight") && find("lm_head.weight")->dt == 6) {
      // keep the stored fp8r plane; no bf16 copy of lm_head is made
      const Ent* e = find("lm_head.weight");
      CK(hipMalloc((void**)&d_lmhead_q, (size_t)VOC * HID));
      CK(hipMalloc((void**)&d_lmhead_s, (size_t)VOC * 2));
      long chunk = 32 << 20;
      for (long o = 0; o < (long)VOC * HID; o += chunk) {
        long nb = chunk < (long)VOC * HID - o ? chunk : (long)VOC * HID - o;
        rd(e->off + o, h_stage, nb); up(d_lmhead_q + o, h_stage, nb);
      }
      rd(e->off + (long)VOC * HID, h_stage, (size_t)VOC * 2); up(d_lmhead_s, h_stage, (size_t)VOC * 2);
      d_lmhead = nullptr;
      printf("lm_head: fp8r plane resident (%.2f GB), no bf16 copy\n", (double)VOC * HID / 1e9);
    } else {
      CK(hipMalloc((void**)&d_lmhead, (long)VOC * HID * 2));
      fprintf(stderr, "trunk: loading lm_head (fp8r dequant)...\n");
      load_fp8r("lm_head.weight", VOC, HID, d_lmhead);
    }
  }
  CK(hipDeviceSynchronize());
  fprintf(stderr, "trunk: big tensors loaded (%.1fs)\n", now_s() - t0);

  CK(hipMalloc((void**)&d_ln_in_all, (long)NL * HID * 2));
  CK(hipMalloc((void**)&d_ln_post_all, (long)NL * HID * 2));
  for (int l = 0; l < NL; l++) {
    snprintf(nm, sizeof(nm), "layers.%d.input_layernorm.weight", l);
    load_bf16(nm, HID * 2, d_ln_in_all + (long)l * HID);
    snprintf(nm, sizeof(nm), "layers.%d.post_attention_layernorm.weight", l);
    load_bf16(nm, HID * 2, d_ln_post_all + (long)l * HID);
  }
  CK(hipMalloc((void**)&d_final_norm, HID * 2)); load_bf16("norm.weight", HID * 2, d_final_norm);
  CK(hipMalloc((void**)&d_gdn_conv, (long)48 * QKVW * 4 * 2));
  CK(hipMalloc((void**)&d_gdn_a, (long)48 * HV * HID * 2));
  CK(hipMalloc((void**)&d_gdn_b, (long)48 * HV * HID * 2));
  CK(hipMalloc((void**)&d_gdn_norm, (long)48 * DV * 2));
  h_alog = (float*)malloc((long)48 * HV * 4);
  h_dtbias = (float*)malloc((long)48 * HV * 4);
  h_gateG_dev = (float*)malloc((long)TMAX * HV * 4);
  h_gateB_dev = (float*)malloc((long)TMAX * HV * 4);
  CK(hipMalloc((void**)&d_alog, (long)48 * HV * 4));
  CK(hipMalloc((void**)&d_dtbias, (long)48 * HV * 4));
  CK(hipMalloc((void**)&d_gateG, (long)TMAX * HV * 4));
  CK(hipMalloc((void**)&d_gateB, (long)TMAX * HV * 4));
  { int gi = 0;
    for (int l = 0; l < NL; l++) {
      if (l % 4 == 3) continue;
      snprintf(nm, sizeof(nm), "layers.%d.linear_attn.conv1d.weight", l);
      load_bf16(nm, (long)QKVW * 4 * 2, d_gdn_conv + (long)gi * QKVW * 4);
      snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_a.weight", l);
      load_bf16(nm, (long)HV * HID * 2, d_gdn_a + (long)gi * HV * HID);
      snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_b.weight", l);
      load_bf16(nm, (long)HV * HID * 2, d_gdn_b + (long)gi * HV * HID);
      snprintf(nm, sizeof(nm), "layers.%d.linear_attn.norm.weight", l);
      load_bf16(nm, DV * 2, d_gdn_norm + (long)gi * DV);
      const Ent* e = find((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.A_log", l), nm));
      rd(e->off, h_stage2, 96);
      { uint16_t* u = (uint16_t*)h_stage2;
        for (int i = 0; i < HV; i++) h_alog[(long)gi * HV + i] = h_bf16f(u[i]); }
      e = find((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.dt_bias", l), nm));
      rd(e->off, h_stage2, 96);
      { uint16_t* u = (uint16_t*)h_stage2;
        for (int i = 0; i < HV; i++) h_dtbias[(long)gi * HV + i] = h_bf16f(u[i]); }
      CK(hipMemcpy(d_alog + (long)gi * HV, h_alog + (long)gi * HV, HV * 4, hipMemcpyHostToDevice));
      CK(hipMemcpy(d_dtbias + (long)gi * HV, h_dtbias + (long)gi * HV, HV * 4, hipMemcpyHostToDevice));
      gi++;
    } }
  CK(hipMalloc((void**)&d_qnorm, (long)16 * HD * 2));
  CK(hipMalloc((void**)&d_knorm, (long)16 * HD * 2));
  { int fi = 0;
    for (int l = 0; l < NL; l++) {
      if (l % 4 != 3) continue;
      snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_norm.weight", l); load_bf16(nm, HD * 2, d_qnorm + (long)fi * HD);
      snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_norm.weight", l); load_bf16(nm, HD * 2, d_knorm + (long)fi * HD);
      fi++;
    } }
  fprintf(stderr, "trunk: small tensors loaded\n");

  CK(hipMalloc((void**)&d_qkv, (long)QKVW * HID * 2));
  CK(hipMalloc((void**)&d_z, (long)HV * DV * HID * 2));
  CK(hipMalloc((void**)&d_outp, (long)HID * HV * DV * 2));
  CK(hipMalloc((void**)&d_q, (long)QH * HD * 2 * HID * 2));
  CK(hipMalloc((void**)&d_k, (long)KVH * HD * HID * 2));
  CK(hipMalloc((void**)&d_v, (long)KVH * HD * HID * 2));
  CK(hipMalloc((void**)&d_o, (long)HID * QH * HD * 2));
  CK(hipMalloc((void**)&d_gate, (long)INTM * HID * 2));
  CK(hipMalloc((void**)&d_up, (long)INTM * HID * 2));
  CK(hipMalloc((void**)&d_down, (long)HID * INTM * 2));
  for (int i = 0; i < 3; i++) {
    CK(hipMalloc((void**)&d_wc[i], (long)INTM * HID / 2 + 1024));
    CK(hipMalloc((void**)&d_ws[i], (long)INTM * HID / 256 * 2 + 1024));
  }
  CK(hipMalloc((void**)&d_qc, (long)TMAX * INTM / 2 + 1024));
  CK(hipMalloc((void**)&d_qs, (long)TMAX * (INTM / 256) * 2 + 1024));

  CK(hipMalloc((void**)&d_conv_ring, (size_t)ring_ns() * kSlotConv * 2));
  d_convstate = d_conv_ring;
  CK(hipMemset(d_convstate, 0, (long)48 * QKVW * 3 * 2));
  CK(hipMalloc((void**)&d_kcache, (long)16 * CXT * KVH * HD * 2));
  CK(hipMalloc((void**)&d_vcache, (long)16 * CXT * KVH * HD * 2));
  CK(hipMemset(d_kcache, 0, (long)16 * CXT * KVH * HD * 2));
  CK(hipMemset(d_vcache, 0, (long)16 * CXT * KVH * HD * 2));
  h_S = (float*)calloc((size_t)48 * HV * DK * DV, 4);

  CK(hipMalloc((void**)&d_x, (long)TMAX * HID * 2));
  CK(hipMalloc((void**)&d_ln1, (long)TMAX * HID * 2));
  CK(hipMalloc((void**)&d_ln2, (long)TMAX * HID * 2));
  CK(hipMalloc((void**)&d_tmp1, (long)TMAX * HID * 2));
  CK(hipMalloc((void**)&d_qkvraw, (long)TMAX * QKVW * 2));
  CK(hipMalloc((void**)&d_qkvc, (long)TMAX * QKVW * 2));
  CK(hipMalloc((void**)&d_ab_a, (long)TMAX * HV * 2));
  CK(hipMalloc((void**)&d_ab_b, (long)TMAX * HV * 2));
  CK(hipMalloc((void**)&d_zout, (long)TMAX * HV * DV * 2));
  CK(hipMalloc((void**)&d_co, (long)TMAX * HV * DV * 2));
  CK(hipMalloc((void**)&d_gn, (long)TMAX * HV * DV * 2));
  CK(hipMalloc((void**)&d_qg, (long)TMAX * QH * HD * 2 * 2));
  CK(hipMalloc((void**)&d_kk, (long)TMAX * KVH * HD * 2));
  CK(hipMalloc((void**)&d_vv, (long)TMAX * KVH * HD * 2));
  CK(hipMalloc((void**)&d_att, (long)TMAX * QH * HD * 2));
  CK(hipMalloc((void**)&d_qn, (long)TMAX * QH * HD * 2));
  CK(hipMalloc((void**)&d_kn, (long)TMAX * KVH * HD * 2));
  CK(hipMalloc((void**)&d_g17, (long)TMAX * INTM * 2));
  CK(hipMalloc((void**)&d_u17, (long)TMAX * INTM * 2));
  CK(hipMalloc((void**)&d_act, (long)TMAX * INTM * 2));
  CK(hipMalloc((void**)&d_scores, (long)ATTN_TILE * QH * CXT * 4));
  CK(hipMalloc((void**)&d_pbf, (long)ATTN_TILE * QH * CXT * 2));
  CK(hipMalloc((void**)&d_logits, (long)VOC * 4));
  CK(hipMalloc((void**)&d_logits_mr, (long)8 * VOC * 4));
  CK(hipMalloc((void**)&d_pred_mr, 64));
  CK(hipMalloc((void**)&d_tok, TMAX * 4));
  CK(hipMalloc((void**)&d_tgt, TMAX * 4));
  CK(hipMalloc((void**)&d_pred, TMAX * 4));
  CK(hipMalloc((void**)&d_nll, TMAX * 4));
  CK(hipMalloc((void**)&d_drafts, 64));
  CK(hipMalloc((void**)&d_arot, (size_t)8 * INTM * 2));
  CK(hipMalloc((void**)&d_a8c, (size_t)16 * INTM));
  CK(hipMalloc((void**)&d_a8s, (size_t)16 * (INTM / 256) * 2));

  h_qkvbf = (uint16_t*)malloc((long)TMAX * QKVW * 2);
  h_abbf = (uint16_t*)malloc((long)TMAX * HV * 2);
  h_af = (float*)malloc((long)TMAX * HV * 4);
  h_bf = (float*)malloc((long)TMAX * HV * 4);
  h_co = (uint16_t*)malloc((long)TMAX * HV * DV * 2);
  h_qscale = 1.0f / sqrtf((float)DK);

  // device weight pool: all quantized projection payloads resident; forward()
  // dequants from device memory per use (fallback: per-use file streaming)
  {
    size_t need = 0;
    for (int l = 0; l < NL; l++) {
      if (l % 4 == 3) {
        need += find((snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_proj.weight", l), nm))->size;
        need += find((snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_proj.weight", l), nm))->size;
        need += find((snprintf(nm, sizeof(nm), "layers.%d.self_attn.v_proj.weight", l), nm))->size;
        need += find((snprintf(nm, sizeof(nm), "layers.%d.self_attn.o_proj.weight", l), nm))->size;
      } else {
        need += find((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l), nm))->size;
        need += find((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l), nm))->size;
        need += find((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.out_proj.weight", l), nm))->size;
      }
      need += find((snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l), nm))->size;
      need += find((snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight", l), nm))->size;
      need += find((snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight", l), nm))->size;
    }
    hipError_t pe = hipMalloc(&d_wpool, need);
    if (pe != hipSuccess) {
      // APUs expose system RAM to the GPU; pinned host memory is the bigger pool
      pe = hipHostMalloc(&d_wpool, need);
      if (pe != hipSuccess) { d_wpool = nullptr; g_wpool_cap = 0; }
    }
    if (d_wpool) {
      g_wpool_cap = need;
      bool ok = true;
      for (int l = 0; l < NL && ok; l++) {
        if (l % 4 == 3) {
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_proj.weight", l), nm), QH * HD * 2, HID);
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_proj.weight", l), nm), KVH * HD, HID);
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.v_proj.weight", l), nm), KVH * HD, HID);
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.o_proj.weight", l), nm), HID, QH * HD);
        } else {
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l), nm), QKVW, HID);
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l), nm), HV * DV, HID);
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.out_proj.weight", l), nm), HID, HV * DV);
        }
        ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l), nm), INTM, HID);
        ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight", l), nm), INTM, HID);
        ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight", l), nm), HID, INTM);
      }
      if (ok) fprintf(stderr, "trunk: weight pool resident (%.1f GB)\n", g_wpool_used / 1e9);
      else { hipFree(d_wpool); d_wpool = nullptr; g_wmap.clear(); g_wpool_used = 0;
             fprintf(stderr, "trunk: weight pool incomplete, streaming mode\n"); }
    } else {
      d_wpool = nullptr;
      fprintf(stderr, "trunk: weight pool alloc failed (%.1f GB), streaming mode\n", need / 1e9);
    }
  }
  if (d_wpool && g_wmap.count("layers.0.mlp.gate_proj.weight") && use_blaslt()) {
    const int M = 64;
    std::vector<uint16_t> hin((size_t)M * HID);
    for (int i = 0; i < (int)hin.size(); i++) hin[i] = h_f32bf(((i % 97) - 48) / 48.f);
    CK(hipMemcpy(d_ln2, hin.data(), hin.size() * 2, hipMemcpyHostToDevice));
    getw("layers.0.mlp.gate_proj.weight", INTM, HID, d_gate);
    CK(hipDeviceSynchronize());
    gemm_scalar(d_ln2, d_gate, d_g17, M, INTM, HID);
    CK(hipDeviceSynchronize());
    int rc = chlorine_gemm_lt(d_ln2, d_gate, d_u17, M, INTM, HID, 0);
    CK(hipDeviceSynchronize());
    if (rc == 0) {
      std::vector<uint16_t> ga((size_t)M * INTM), gb((size_t)M * INTM);
      CK(hipMemcpy(ga.data(), d_g17, ga.size() * 2, hipMemcpyDeviceToHost));
      CK(hipMemcpy(gb.data(), d_u17, gb.size() * 2, hipMemcpyDeviceToHost));
      double mx = 0, ref = 0;
      for (size_t i = 0; i < ga.size(); i++) {
        float a = h_bf16f(ga[i]), b = h_bf16f(gb[i]);
        mx = std::max(mx, (double)fabsf(a - b));
        ref = std::max(ref, (double)fabsf(a));
      }
      fprintf(stderr, "gemm: layer0 gate scalar-vs-lt maxabs %.4g refmax %.4g\n", mx, ref);
    } else {
      fprintf(stderr, "gemm: layer0 gate compare skipped (%d)\n", rc);
    }
  }
  attn_selfcheck();
  gdn_selfcheck();
  if (g_w4a4 || g_w4dec) init_i4_planes();
  if (g_w4a4 != 1) init_bf16_weight_cache();   // all-400 int4 prefill has no use for the bf16 copies
  if (spec_maxg() > 0) mtp_load();
  df_load();
  cache_init();
  g_ready = 1;
  fprintf(stderr, "trunk: ready (%.1fs)\n", now_s() - t0);
  return 0;
}

// Greedy generate: clean prefill, then per-step argmax with per-layer KV cache +
// GDN state advance. Emits each token via `emit`. Returns tokens emitted, or
// -1 (not ready) / -2 (prompt or max_tokens out of trunk capacity).
extern "C" int chlorine_trunk_generate(const int* prompt, int n_prompt, long max_tokens,
                                       const int* eos, int n_eos,
                                       void (*emit)(void*, int), void* ctx,
                                       double* prefill_ms, double* decode_ms, int* stop_hit) {
  if (!g_ready) return -1;
  if (n_prompt < 1 || n_prompt > TMAX || n_prompt > CXT - 2 || max_tokens < 1 ||
      max_tokens > (long)(CXT - n_prompt)) return -2;
  trunk_reset_states();
  double tp = now_s();
  up(d_tok, prompt, (size_t)n_prompt * 4);
  forward(d_tok, n_prompt, 0, 1);
  *prefill_ms = (now_s() - tp) * 1000.0;

  double td = now_s();
  int n_gen = 0, tok = 0, stop = 0;
  k_argmax1<<<1, 256>>>(d_logits, d_pred, VOC);
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(&tok, d_pred, 4, hipMemcpyDeviceToHost));
  while (!stop && n_gen < max_tokens) {
    emit(ctx, tok);
    n_gen++;
    for (int i = 0; i < n_eos; i++)
      if (eos[i] == tok) { stop = 1; break; }
    if (stop || n_gen >= max_tokens) break;
    up(d_tok, &tok, 4);
    forward(d_tok, 1, n_prompt - 1 + n_gen, 1);
    k_argmax1<<<1, 256>>>(d_logits, d_pred, VOC);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(&tok, d_pred, 4, hipMemcpyDeviceToHost));
  }
  *decode_ms = (now_s() - td) * 1000.0;
  *stop_hit = stop;
  return n_gen;
}

// ---------------- sampler (HOST-LOGIC.md section 5) ----------------
// k_sample semantics: penalties -> bf16 logits -> temp softmax -> top-k ->
// top-p -> min-p -> renormalize -> draw. Counter RNG decoded from the k_sample
// ISA (obj1.so): c = posctr*0xd1b54a32d192ed03 ^ seed ^ (aux*0x9e3779b97f4a7c15);
// c += 0x9e3779b97f4a7c15; splitmix64 finalizer; u = (c >> 40) * 2^-24.
typedef struct {
  int has_sample;
  float temp;
  int top_k;
  float top_p, min_p;
  unsigned long long seed;
  float presence, frequency;
  int n_bias;
  const int* bias_ids;
  const float* bias_vals;
  int logprobs;
} chlorine_sample_opts;

static float* g_pinrow = nullptr;   // pinned VOC f32 staging for sampler copies

static inline unsigned long long sm_fin(unsigned long long x) {
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  return x;
}

static float sample_uniform(unsigned long long seed, unsigned int posctr, unsigned int aux) {
  unsigned long long c = (unsigned long long)posctr * 0xd1b54a32d192ed03ULL;
  c ^= (unsigned long long)aux * 0x9e3779b97f4a7c15ULL;
  c ^= seed;
  c += 0x9e3779b97f4a7c15ULL;
  c = sm_fin(c);
  return (float)(unsigned)(c >> 40) * (1.0f / 16777216.0f);
}

// One sampling step. d_row: device f32 logits [VOC]; counts: per-request
// generated-token counts (host, VOC); posctr: n_prompt + generated-so-far.
extern "C" int chlorine_sample_host(const float* d_row, int* counts, unsigned int posctr,
                                    const chlorine_sample_opts* o,
                                    int* out_tok, float* out_logp) {
  if (!g_pinrow) CK(hipHostMalloc((void**)&g_pinrow, (size_t)VOC * 4));
  CK(hipMemcpy(g_pinrow, d_row, (size_t)VOC * 4, hipMemcpyDeviceToHost));
  float* L = g_pinrow;
  // engine logits are bf16 (k_nll/k_argmax read u16); penalties scatter into a
  // bf16 copy, so round first and re-round after every f32 delta add
  for (int v = 0; v < VOC; v++) L[v] = h_bf16f(h_f32bf(L[v]));
  if (o->presence != 0.f || o->frequency != 0.f) {
    for (int t = 0; t < VOC; t++) {
      if (counts[t] > 0) {
        float d = -(o->presence + o->frequency * (float)counts[t]);
        L[t] = h_bf16f(h_f32bf(L[t] + d));
      }
    }
  }
  for (int i = 0; i < o->n_bias; i++) {
    int t = o->bias_ids[i];
    if (t >= 0 && t < VOC) L[t] = h_bf16f(h_f32bf(L[t] + o->bias_vals[i]));
  }
  // temp softmax (f32 throughout, like k_sample)
  float mx = L[0];
  for (int v = 1; v < VOC; v++) if (L[v] > mx) mx = L[v];
  float inv_t = 1.0f / o->temp;
  static std::vector<float> p;
  static std::vector<unsigned char> keep;
  static std::vector<int> idx;
  p.resize(VOC); keep.resize(VOC); idx.resize(VOC);
  double tot = 0.0;
  for (int v = 0; v < VOC; v++) { float e = expf((L[v] - mx) * inv_t); p[v] = e; tot += e; }
  float ftot = (float)tot;
  int nkeep = 0;
  for (int v = 0; v < VOC; v++) idx[v] = v;
  if (o->top_k > 0 && o->top_k < VOC) {
    // top-k: k largest probs (ties -> lower id), via nth_element
    int k = o->top_k;
    std::nth_element(idx.data(), idx.data() + k, idx.data() + VOC,
                     [&](int a, int b) {
                       if (p[a] != p[b]) return p[a] > p[b];
                       return a < b;
                     });
    for (int i = 0; i < k; i++) keep[idx[i]] = 1;
    nkeep = k;
  } else {
    for (int v = 0; v < VOC; v++) keep[v] = 1;
    nkeep = VOC;
  }
  if (o->top_p > 0.f && o->top_p < 1.f) {
    // nucleus within the kept set: sort desc, keep until cum >= top_p (inclusive)
    std::vector<int> nk;
    nk.reserve(nkeep);
    for (int v = 0; v < VOC; v++) if (keep[v]) nk.push_back(v);
    std::sort(nk.begin(), nk.end(), [&](int a, int b) {
      if (p[a] != p[b]) return p[a] > p[b];
      return a < b;
    });
    float cum = 0.f;
    int cut = (int)nk.size();
    for (int i = 0; i < (int)nk.size(); i++) {
      cum += p[nk[i]];
      if (cum >= o->top_p * ftot) { cut = i + 1; break; }
    }
    for (int i = 0; i < cut; i++) keep[nk[i]] = 1;
    for (int i = cut; i < (int)nk.size(); i++) keep[nk[i]] = 0;
  }
  if (o->min_p > 0.f) {
    // min-p: normalized prob >= min_p * p_max (p_max = 1/tot after temp scaling)
    float thr = o->min_p / ftot;
    for (int v = 0; v < VOC; v++)
      if (keep[v] && p[v] < thr) keep[v] = 0;
  }
  double tkept = 0.0;
  for (int v = 0; v < VOC; v++) if (keep[v]) tkept += p[v];
  if (tkept <= 0.0) {  // degenerate: fall back to argmax
    int am = 0;
    for (int v = 1; v < VOC; v++) if (L[v] > L[am]) am = v;
    *out_tok = am;
    *out_logp = 0.f;
    return 0;
  }
  float u = sample_uniform(o->seed, posctr, 0);
  float target = u * (float)tkept;
  // the engine scans the support in PROB-DESC order (verified: seeded coin A/B
  // vs the original server — 92/100 bit agreement with the sorted-order model)
  static std::vector<int> sorder;
  sorder.clear();
  for (int v = 0; v < VOC; v++)
    if (keep[v]) sorder.push_back(v);
  std::sort(sorder.begin(), sorder.end(), [&](int a, int b) {
    if (p[a] != p[b]) return p[a] > p[b];
    return a < b;
  });
  float cum = 0.f;
  int tok = -1;
  for (int v : sorder) {
    cum += p[v];
    if (cum >= target) { tok = v; break; }
  }
  if (tok < 0 && !sorder.empty()) tok = sorder.back();
  *out_tok = tok;
  *out_logp = logf(p[tok] / (float)tkept);
  return 0;
}

// Sampling generate: same trunk loop as chlorine_trunk_generate with the host
// sampler (temp/top-k/top-p/min-p/seed, presence/frequency penalties, bias,
// logprobs). posctr for the counter RNG = n_prompt + generated-so-far (the
// engine passes base + loop index; first token = n_prompt).
// HALO_LOGIT_HASH=1 prints an FNV-1a hash of the full f32 logits row that
// decided each committed token ("LH <position> <hash>"). A speculative run and
// a serial run of the same prompt must print identical lines: that is the
// bit-identity test of the verify path.
// Teacher-forced scoring for the W4A4 quality gate: prefill ids[0..n) once,
// then for every position r in [n_prompt-1, n-1) compute the logits row from
// the final-normed hidden states and report NLL of ids[r+1] and the argmax.
// Uses the prefill path (int4 when HALO_W4A4 is set and n >= 64, bf16
// otherwise), so two servers can be compared on the same continuation.
static int* d_rowmap8 = nullptr;
extern "C" int chlorine_trunk_score(const int* ids, int n, int n_prompt, float* nll_out, int* pred_out) {
  if (!g_ready) return -1;
  if (n < 2 || n > TMAX || n_prompt < 1 || n_prompt >= n) return -2;
  if (!d_lmhead_q) return -3;   // needs HALO_LMHEAD_FP8=1 (multi-row lm_head reads the fp8r plane)
  if (!d_rowmap8) { int rm[8] = {0, 1, 2, 3, 4, 5, 6, 7}; CK(hipMalloc((void**)&d_rowmap8, 32)); up(d_rowmap8, rm, 32); }
  trunk_reset_states();
  {
    const int C = g_pf_chunk > 0 ? g_pf_chunk : n;
    for (int off = 0; off < n; off += C) {
      int T = std::min(C, n - off);
      up(d_tok, ids + off, (size_t)T * 4);
      forward(d_tok, T, off, 1);
    }
  }
  {
    const int C = g_pf_chunk > 0 ? g_pf_chunk : n;
    const int first = ((n - 1) / C) * C;              // first row of the last chunk in d_ln1
    if (n_prompt - 1 < first) return -4;              // continuation must lie inside the last chunk
    ids += 0;
  }
  int scored = 0;
  const int rowbase = g_pf_chunk > 0 ? ((n - 1) / g_pf_chunk) * g_pf_chunk : 0;
  for (int r0 = n_prompt - 1; r0 < n - 1; r0 += 8) {
    const int P = std::min(8, n - 1 - r0);
    lmhead_mr(d_ln1 + (size_t)(r0 - rowbase) * HID, P);
    up(d_tgt, ids + r0 + 1, (size_t)P * 4);
    k_topnll<<<P, 256>>>(d_logits_mr, d_rowmap8, d_tgt, d_pred, d_nll, VOC, P);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(nll_out + scored, d_nll, (size_t)P * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(pred_out + scored, d_pred, (size_t)P * 4, hipMemcpyDeviceToHost));
    scored += P;
  }
  return scored;
}

// SCORE through the decode path (HALO_SCORE_DECODE=1): the prompt rows go
// through the prefill path as in serving, then every continuation position is
// processed by the decode kernels, in verify chunks of up to 8 rows when the
// state rings exist (HALO_MTP=1) or one serial step at a time otherwise. The
// logits are therefore those of the decode GEMVs, which is what a decode
// precision configuration (HALO_W4DEC) changes; the prefill-path SCORE above
// is the gate for prefill configurations (HALO_W4A4).
extern "C" int chlorine_trunk_score_decode(const int* ids, int n, int n_prompt, float* nll_out, int* pred_out) {
  if (!g_ready) return -1;
  if (n < 2 || n > CXT - 2 || n_prompt < 1 || n_prompt >= n) return -2;
  if (!d_lmhead_q) return -3;
  if (!d_rowmap8) { int rm[8] = {0, 1, 2, 3, 4, 5, 6, 7}; CK(hipMalloc((void**)&d_rowmap8, 32)); up(d_rowmap8, rm, 32); }
  trunk_reset_states();
  if (d_S_ring && d_conv_ring) ring_set(0);
  {
    const int C = g_pf_chunk > 0 ? g_pf_chunk : n_prompt;
    for (int off = 0; off < n_prompt; off += C) {
      int T = std::min(C, n_prompt - off);
      up(d_tok, ids + off, (size_t)T * 4);
      forward(d_tok, T, off, 1);
    }
  }
  int scored = 0;
  // position n_prompt-1: the prefill's last-row logits (as the first generated token in serving)
  up(d_tgt, ids + n_prompt, 4);
  k_topnll<<<1, 256>>>(d_logits, d_rowmap8, d_tgt, d_pred, d_nll, VOC, 1);
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(nll_out + scored, d_nll, 4, hipMemcpyDeviceToHost));
  CK(hipMemcpy(pred_out + scored, d_pred, 4, hipMemcpyDeviceToHost));
  scored++;
  const int maxrows = (d_S_ring && d_conv_ring && g_ring_ns > 1) ? std::min(8, g_ring_ns - 1) : 1;
  for (int r0 = n_prompt; r0 < n - 1; ) {
    const int P = std::min(maxrows, n - 1 - r0);
    up(d_tok, ids + r0, (size_t)P * 4);
    up(d_tgt, ids + r0 + 1, (size_t)P * 4);
    if (P == 1 && maxrows == 1) {
      forward(d_tok, 1, r0, 1);
      k_topnll<<<1, 256>>>(d_logits, d_rowmap8, d_tgt, d_pred, d_nll, VOC, 1);
    } else {
      g_verify = 1;
      forward(d_tok, P, r0, 1);
      g_verify = 0;
      k_topnll<<<P, 256>>>(d_logits_mr, d_rowmap8, d_tgt, d_pred, d_nll, VOC, P);
      ring_set(g_ring_cur + P);      // accept every row
    }
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(nll_out + scored, d_nll, (size_t)P * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(pred_out + scored, d_pred, (size_t)P * 4, hipMemcpyDeviceToHost));
    scored += P;
    r0 += P;
  }
  return scored;
}

static int g_logit_hash = -1;
static void logit_hash(const float* d_row, int pos) {
  if (g_logit_hash < 0) { const char* e = getenv("HALO_LOGIT_HASH"); g_logit_hash = e && atoi(e) == 1; }
  if (!g_logit_hash) return;
  static std::vector<uint32_t> h;
  h.resize(VOC);
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(h.data(), d_row, (size_t)VOC * 4, hipMemcpyDeviceToHost));
  uint64_t x = 1469598103934665603ull;
  for (int i = 0; i < VOC; i++) { x ^= h[i]; x *= 1099511628211ull; }
  fprintf(stderr, "LH %d %016llx\n", pos, (unsigned long long)x);
}

struct chlorine_spec_stats {
  int drafter, rounds, proposed, accepted, committed;
  double draft_ms, verify_ms, ingest_ms, mtp_prefill_ms;
  int n_cached;
};

extern "C" int chlorine_trunk_generate3(const int* prompt, int n_prompt, long max_tokens,
                                        const int* eos, int n_eos,
                                        void (*emit)(void*, int, float), void* ctx,
                                        double* prefill_ms, double* decode_ms, int* stop_hit,
                                        const chlorine_sample_opts* opts, int drafter,
                                        chlorine_spec_stats* st) {
  if (!g_ready) return -1;
  if (n_prompt < 1 || n_prompt > TMAX || n_prompt > CXT - 2 || max_tokens < 1 ||
      max_tokens > (long)(CXT - n_prompt)) return -2;
  bool sample = opts && opts->has_sample;
  const bool dfl = drafter == 2 && g_df_ok && !sample;
  const bool spec = (drafter == 1 && g_mtp_ok && !sample) || dfl;
  g_df_capture = dfl ? 1 : 0;
  chlorine_spec_stats local{};
  if (!st) st = &local;
  *st = chlorine_spec_stats{};
  st->drafter = dfl ? 2 : (spec ? 1 : 0);
  static std::vector<int> counts;
  counts.assign(VOC, 0);
  trunk_reset_states();
  double tp = now_s();
  int last_rows = n_prompt;
  const int n_cached = prefill_prompt(prompt, n_prompt, &last_rows);
  st->n_cached = n_cached;
  int tok = 0;
  float logp = 0.f;
  if (sample) {
    chlorine_sample_host(d_logits, counts.data(),
                         (unsigned)n_prompt, opts, &tok, &logp);
    counts[tok]++;
  } else {
    k_argmax1<<<1, 256>>>(d_logits, d_pred, VOC);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(&tok, d_pred, 4, hipMemcpyDeviceToHost));
    logit_hash(d_logits, n_prompt - 1);
  }
  if (spec && !dfl) {
    // MTP prompt ingest: position q (1..n-1) combines trunk hidden h[q-1]
    // with token q. The trunk's final norm left all prompt rows in d_ln1.
    double tm = now_s();
    // d_ln1 holds the last prefill chunk (last_rows rows ending at n_prompt-1).
    // MTP position q needs h[q-1] and token q: ingest positions max(1, first)..
    // n_prompt-1 where first is the first row of that chunk plus one; with a
    // cache hit the earlier MTP KV rows were restored with the entry.
    const int first = n_prompt - last_rows;            // position of d_ln1 row 0
    const int q0 = std::max(1, first + 1);
    CK(hipMemcpy(d_hlast, d_ln1 + (size_t)(last_rows - 1) * HID, (size_t)HID * 2, hipMemcpyDeviceToDevice));
    if (q0 <= n_prompt - 1) {
      const int cnt = n_prompt - q0;
      CK(hipMemcpy(d_mtp_hin, d_ln1 + (size_t)(q0 - 1 - first) * HID, (size_t)cnt * HID * 2, hipMemcpyDeviceToDevice));
      up(d_mtp_ids, prompt + q0, (size_t)cnt * 4);
      mtp_rows(d_mtp_hin, d_mtp_ids, cnt, q0, false);
    }
    CK(hipDeviceSynchronize());
    st->mtp_prefill_ms = (now_s() - tm) * 1000.0;
  }
  *prefill_ms = (now_s() - tp) * 1000.0;

  double td = now_s();
  int n_gen = 0, stop = 0;
  auto commit = [&](int t) {
    emit(ctx, t, sample ? logp : 0.f);
    n_gen++;
    for (int i = 0; i < n_eos; i++)
      if (eos[i] == t) { stop = 1; break; }
    return !stop && n_gen < max_tokens;
  };
  int n_proc = n_prompt;     // positions the trunk has processed
  bool more = commit(tok);
  const int maxg = dfl ? DF_B - 1 : mtp_gamma();
  // HALO_SPEC_ADAPT=1: verify only as many drafts as recent rounds accept
  // (EMA of the accepted count + 2). Drafter-side only; the emitted tokens
  // are the serial ones whatever the length.
  static int adapt = -1;
  if (adapt < 0) { const char* e = getenv("HALO_SPEC_ADAPT"); adapt = (e && atoi(e) == 1) ? 1 : 0; }
  float acc_ema = dfl ? 3.0f : 2.0f;
  while (more) {
    const int remaining = (int)(max_tokens - n_gen);
    int gd = spec ? std::min(maxg, remaining - 1) : 0;
    const int gcap = adapt ? std::max(1, std::min(maxg, (int)ceilf(acc_ema) + 2)) : maxg;
    if (!dfl) gd = std::min(gd, gcap);
    gd = std::min(gd, CXT - 1 - n_proc);
    if (gd < 1) {
      // serial step
      up(d_tok, &tok, 4);
      forward(d_tok, 1, n_proc, 1);
      if (sample) {
        chlorine_sample_host(d_logits, counts.data(), (unsigned)(n_proc + 1), opts, &tok, &logp);
        counts[tok]++;
      } else {
        k_argmax1<<<1, 256>>>(d_logits, d_pred, VOC);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(&tok, d_pred, 4, hipMemcpyDeviceToHost));
        logit_hash(d_logits, n_proc);
      }
      n_proc++;
      more = commit(tok);
      continue;
    }
    // ---- draft gd tokens with the MTP head ----
    double t0 = now_s();
    int drafts[8];
    const uint16_t* hp = d_hlast;
    int tk = tok;
    static float pmin = -1.f;
    if (pmin < 0.f) { const char* e = getenv("HALO_SPEC_PMIN"); pmin = e ? (float)atof(e) : 0.f; }
    int nd = 0;
    if (dfl) {
      // DFlash2: one block forward drafts seven tokens from the anchor at n_proc
      up(d_tok, &tk, 4);
      df_draft(d_tok, n_proc);
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(drafts, d_df_out, (size_t)(DF_B - 1) * 4, hipMemcpyDeviceToHost));
      nd = gd;
    } else if (pmin <= 0.f) {
      // Device-side chain: draft j's argmax feeds draft j+1's embedding through
      // d_drafts without a host round trip; one fence and one copy per round.
      // Same kernels and values as the per-draft loop below.
      up(d_mtp_ids, &tk, 4);
      for (int j = 0; j < gd; j++) {
        mtp_rows(hp, j == 0 ? d_mtp_ids : d_drafts + (j - 1), 1, n_proc + j, true, d_drafts + j);
        hp = d_mtp_hout;
      }
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(drafts, d_drafts, (size_t)gd * 4, hipMemcpyDeviceToHost));
      nd = gd;
    } else
    for (int j = 0; j < gd; j++) {
      up(d_mtp_ids, &tk, 4);
      mtp_rows(hp, d_mtp_ids, 1, n_proc + j, true);
      CK(hipDeviceSynchronize());
      float pr = 1.f;
      CK(hipMemcpy(&drafts[j], d_pred, 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(&pr, d_draft_prob, 4, hipMemcpyDeviceToHost));
      nd = j + 1;
      hp = d_mtp_hout;
      tk = drafts[j];
      // stop early when the drafter is unsure: a low-probability draft is
      // unlikely to be accepted, and every draft costs a verify row
      if (pr < pmin) break;
    }
    gd = nd;
    if (dfl) gd = std::min(gd, gcap);
    double t1 = now_s();
    // ---- verify [tok, drafts] as gd+1 decode rows at n_proc.. ----
    int vt[9], v[9];
    vt[0] = tok;
    for (int j = 0; j < gd; j++) vt[j + 1] = drafts[j];
    up(d_tok, vt, (size_t)(gd + 1) * 4);
    g_verify = 1;
    forward(d_tok, gd + 1, n_proc, 1);
    g_verify = 0;
    k_argmax_rows<<<gd + 1, 256>>>(d_logits_mr, d_pred_mr, VOC);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(v, d_pred_mr, (size_t)(gd + 1) * 4, hipMemcpyDeviceToHost));
    int a = 0;
    while (a < gd && drafts[a] == v[a]) a++;
    for (int i = 0; i <= a; i++) logit_hash(d_logits_mr + (size_t)i * VOC, n_proc + i);
    double t2 = now_s();
    // ---- accept a drafts + the trunk's token v[a]; roll state to row a ----
    if (!dfl) {
      CK(hipMemcpy(d_mtp_hin, d_ln1, (size_t)(a + 1) * HID * 2, hipMemcpyDeviceToDevice));
      CK(hipMemcpy(d_hlast, d_ln1 + (size_t)a * HID, (size_t)HID * 2, hipMemcpyDeviceToDevice));
    }
    ring_set(g_ring_cur + a + 1);
    const int base = n_proc;
    n_proc += a + 1;
    // DFlash2: the anchor and the accepted drafts (verify rows 0..a) join the drafter context
    if (dfl) df_ingest(base, a + 1);
    // MTP KV repair: positions base+1..base+a get the trunk's hidden, not the draft's
    if (!dfl && a >= 1) {
      up(d_mtp_ids, drafts, (size_t)a * 4);
      mtp_rows(d_mtp_hin, d_mtp_ids, a, base + 1, false);
    }
    CK(hipDeviceSynchronize());
    double t3 = now_s();
    st->rounds++;
    st->proposed += gd;
    st->accepted += a;
    acc_ema = 0.75f * acc_ema + 0.25f * (float)a;
    st->draft_ms += (t1 - t0) * 1000.0;
    st->verify_ms += (t2 - t1) * 1000.0;
    st->ingest_ms += (t3 - t2) * 1000.0;
    for (int i = 0; i <= a && more; i++) {
      more = commit(v[i]);
      st->committed++;
    }
    tok = v[a];
  }
  *decode_ms = (now_s() - td) * 1000.0;
  *stop_hit = stop;
  g_df_capture = 0;
  if (spec)
    fprintf(stderr, "spec: rounds %d proposed %d accepted %d (%.3f) committed %d tokens/round %.2f draft %.1f ms verify %.1f ms ingest %.1f ms mtp-prefill %.1f ms\n",
            st->rounds, st->proposed, st->accepted, st->proposed ? (double)st->accepted / st->proposed : 0.0,
            st->committed, st->rounds ? (double)st->committed / st->rounds : 0.0,
            st->draft_ms, st->verify_ms, st->ingest_ms, st->mtp_prefill_ms);
  return n_gen;
}

extern "C" int chlorine_trunk_generate2(const int* prompt, int n_prompt, long max_tokens,
                                        const int* eos, int n_eos,
                                        void (*emit)(void*, int, float), void* ctx,
                                        double* prefill_ms, double* decode_ms, int* stop_hit,
                                        const chlorine_sample_opts* opts) {
  return chlorine_trunk_generate3(prompt, n_prompt, max_tokens, eos, n_eos, emit, ctx,
                                  prefill_ms, decode_ms, stop_hit, opts, 0, nullptr);
}

extern "C" int chlorine_trunk_ctx_cap(void) { return CXT; }
extern "C" int chlorine_trunk_tmax(void) { return TMAX; }

extern "C" void chlorine_trunk_shutdown(void) {
  // weights and buffers live for the process lifetime; nothing to do
}
