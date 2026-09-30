// GPU chunked GatedDeltaNet scan. Same chunk-64 recurrence as gdn_scan.
// q/k bf16 rounding uses the host helper's 0x7FFF threshold.
// Sums and dots run in address order so they match the host loop.
// LDS stride 129 keeps a column read off a single bank. Resident LDS stays
// under the gfx1151 64 KiB workgroup limit.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include "gdn_expf.h"

// Match the existing host gdn_scan, including its separate mul/add rounding.
#pragma clang fp contract(off)

static const int GDN_HK = 16;
static const int GDN_HV = 48;
static const int GDN_DK = 128;
static const int GDN_DV = 128;
static const int GDN_QKVW = 10240;
static const int GDN_CS = 64;
static const int GDN_STR = 129;
static const float GDN_EPS = 1e-6f;

__device__ inline float gdn_bf16f(uint16_t h) {
  return __uint_as_float((uint32_t)h << 16);
}
__device__ inline uint16_t gdn_f32bf_host(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lo = u & 0xFFFFu;
  uint32_t hi = u >> 16;
  uint32_t up = (lo > 0x7FFFu) | ((lo == 0x7FFFu) & (hi & 1u));
  return (uint16_t)((hi + up) & 0xFFFFu);
}

// Eight warps per block, one warp per (token, head). Squares land in shared
// memory and lane 0 adds them from i = 0 upward, matching the host sum.
__global__ void k_gdn_prep(const uint16_t* __restrict__ qkv, float* __restrict__ Q,
                           float* __restrict__ K, float* __restrict__ V,
                           int T, int Tpad, int dnprep, float qscale) {
  const int lane = (int)threadIdx.x & 31;
  const int sub = (int)threadIdx.x >> 5;
  const int group = ((int)blockIdx.x << 3) + sub;
  const int valid = group < T * GDN_HV;
  const int t = valid ? group / GDN_HV : 0;
  const int hh = valid ? group - t * GDN_HV : 0;
  __shared__ float sq[8][128];
  __shared__ float sk[8][128];
  const uint16_t* qp = qkv + (long)t * GDN_QKVW + (hh / 3) * GDN_DK;
  const uint16_t* kp = qkv + (long)t * GDN_QKVW + GDN_HK * GDN_DK + (hh / 3) * GDN_DK;
  const uint16_t* vp = qkv + (long)t * GDN_QKVW + GDN_HK * GDN_DK * 2 + hh * GDN_DV;
  for (int i = lane; i < GDN_DK; i += 32) {
    float a = valid ? gdn_bf16f(qp[i]) : 0.f;
    float b = valid ? gdn_bf16f(kp[i]) : 0.f;
    sq[sub][i] = a * a;
    sk[sub][i] = b * b;
  }
  __syncthreads();
  if (lane == 0 && valid) {
    float qs = 0.f, ks = 0.f;
    for (int i = 0; i < GDN_DK; i++) {
      qs += sq[sub][i];
      ks += sk[sub][i];
    }
    sq[sub][0] = 1.0f / sqrtf(qs + GDN_EPS) * qscale;
    sk[sub][0] = 1.0f / sqrtf(ks + GDN_EPS);
  }
  __syncthreads();
  if (!valid) return;
  const float qi = sq[sub][0];
  const float ki = sk[sub][0];
  float* Qr = Q + ((long)hh * Tpad + t) * GDN_DK;
  float* Kr = K + ((long)hh * Tpad + t) * GDN_DK;
  float* Vr = V + ((long)hh * Tpad + t) * GDN_DV;
  for (int i = lane; i < GDN_DK; i += 32) {
    float qv = gdn_bf16f(qp[i]) * qi;
    float kv = gdn_bf16f(kp[i]) * ki;
    if (dnprep) {
      qv = gdn_bf16f(gdn_f32bf_host(qv));
      kv = gdn_bf16f(gdn_f32bf_host(kv));
    }
    Qr[i] = qv;
    Kr[i] = kv;
    Vr[i] = gdn_bf16f(vp[i]);
  }
}

// Lower-triangular grams. K for the chunk and a 32-row Q tile sit in LDS.
// Each pair is one thread, and the 128-dot adds in i order.
// sK is 64*129 floats, sQ is 32*129, plus two 64-float rows: 50048 bytes.
__global__ void k_gdn_grams(const float* __restrict__ Q, const float* __restrict__ K,
                            const float* __restrict__ gateG, const float* __restrict__ gateB,
                            float* __restrict__ ut, float* __restrict__ intra,
                            float* __restrict__ exps, int T, int Tpad) {
  const int chunk = (int)blockIdx.x;
  const int hh = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int CS = GDN_CS;
  const int STR = GDN_STR;
  const int t0 = chunk * CS;
  __shared__ float sK[64 * GDN_STR];
  __shared__ float sQ[32 * GDN_STR];
  __shared__ float cum[GDN_CS];
  __shared__ float Brow[GDN_CS];

  const float* Qh = Q + ((long)hh * Tpad) * GDN_DK;
  const float* Kh = K + ((long)hh * Tpad) * GDN_DK;
  for (int e = tid; e < CS * GDN_DK; e += (int)blockDim.x) {
    int a = e >> 7;
    int i = e & 127;
    int t = t0 + a;
    sK[a * STR + i] = (t < T) ? Kh[(long)t * GDN_DK + i] : 0.f;
  }
  if (tid < CS) {
    int t = t0 + tid;
    Brow[tid] = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
  }
  if (tid == 0) {
    float acc = 0.f;
    for (int a = 0; a < CS; a++) {
      int t = t0 + a;
      acc += (t < T) ? gateG[(long)t * GDN_HV + hh] : 0.f;
      cum[a] = acc;
    }
  }
  __syncthreads();

  // Gate exponentials are identical across all 128 solve columns. Compute
  // once per chunk/head instead of repeating the double-precision expf.
  float* eh = exps + ((long)chunk * GDN_HV + hh) * (2 * CS + 1);
  if (tid < CS) {
    eh[tid] = gdn_expf(cum[tid]);
    eh[CS + tid] = gdn_expf(cum[CS - 1] - cum[tid]);
  }
  if (tid == 0) eh[2 * CS] = gdn_expf(cum[CS - 1]);

  float* uth = ut + (((long)chunk * GDN_HV + hh) * CS) * CS;
  float* inh = intra + (((long)chunk * GDN_HV + hh) * CS) * CS;
  for (int a0 = 0; a0 < CS; a0 += 32) {
    for (int e = tid; e < 32 * GDN_DK; e += (int)blockDim.x) {
      int ar = e >> 7;
      int i = e & 127;
      int a = a0 + ar;
      int t = t0 + a;
      sQ[ar * STR + i] = (t < T) ? Qh[(long)t * GDN_DK + i] : 0.f;
    }
    __syncthreads();
    const int nlocal = 32 * a0 + 528;
    for (int p = tid; p < nlocal; p += (int)blockDim.x) {
      int remain = p;
      int a = a0;
      while (remain >= a + 1) {
        remain -= a + 1;
        a++;
      }
      int b = remain;
      float decay = gdn_expf(cum[a] - cum[b]);
      float kbd = 0.f, qkd = 0.f;
      if (decay != 0.f) {
        const float* Ka = sK + a * STR;
        const float* Kb = sK + b * STR;
        const float* Qa = sQ + (a - a0) * STR;
        float Ba = Brow[a];
        for (int i = 0; i < GDN_DK; i++) {
          float kv = Kb[i];
          kbd += (Ka[i] * Ba) * kv;
          qkd += Qa[i] * kv;
        }
      }
      uth[(long)a * CS + b] = kbd * decay;
      inh[(long)a * CS + b] = qkd * decay;
    }
    __syncthreads();
  }
}

// One column of the 64-wide substitution, kept in named VGPRs.
// An indexed float col[64] spilled to scratch in the earlier fused kernel.
// Grid z covers columns 32 at a time.
template<bool LDS_SOLVE = false>
__global__ void k_gdn_solve(const float* __restrict__ Q, const float* __restrict__ K,
                            const float* __restrict__ V, const float* __restrict__ gateG,
                            const float* __restrict__ gateB, const float* __restrict__ ut,
                            float* __restrict__ nv, float* __restrict__ kcd,
                            float* __restrict__ qd, float* __restrict__ kd,
                            const float* __restrict__ exps, int T, int Tpad) {
  const int chunk = (int)blockIdx.x;
  const int hh = (int)blockIdx.y;
  const int lane = (int)threadIdx.x;
  const int col = (int)blockIdx.z * (LDS_SOLVE ? 128 : 32) + lane;
  const int CS = GDN_CS;
  const int t0 = chunk * CS;
  const float* eh = exps + ((long)chunk * GDN_HV + hh) * (2 * CS + 1);

  long slot = ((long)chunk * GDN_HV + hh) * CS;
  __shared__ float shared_ut[LDS_SOLVE ? GDN_CS * GDN_CS : 1];
  const float* uth = ut + slot * CS;
  if constexpr (LDS_SOLVE) {
    for (int e = lane; e < GDN_CS * GDN_CS; e += 128)
      shared_ut[e] = e % GDN_CS <= e / GDN_CS ? uth[e] : 0.f;
    __syncthreads();
    uth = shared_ut;
  }
  const float* Qh = Q + ((long)hh * Tpad) * GDN_DK;
  const float* Kh = K + ((long)hh * Tpad) * GDN_DK;
  const float* Vh = V + ((long)hh * Tpad) * GDN_DV;
  float* nvh = nv + slot * GDN_DV;
  float* kcdh = kcd + slot * GDN_DK;
  float* qdh = qd + slot * GDN_DK;
  float* kdh = kd + slot * GDN_DK;

  float c0; float c1; float c2; float c3; float c4; float c5; float c6; float c7; float c8; float c9; float c10; float c11; float c12; float c13; float c14; float c15; float c16; float c17; float c18; float c19; float c20; float c21; float c22; float c23; float c24; float c25; float c26; float c27; float c28; float c29; float c30; float c31; float c32; float c33; float c34; float c35; float c36; float c37; float c38; float c39; float c40; float c41; float c42; float c43; float c44; float c45; float c46; float c47; float c48; float c49; float c50; float c51; float c52; float c53; float c54; float c55; float c56; float c57; float c58; float c59; float c60; float c61; float c62; float c63;
  {
    int t = t0 + 0;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c0 = Vv * B;
  }
  {
    int t = t0 + 1;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c1 = Vv * B;
  }
  {
    int t = t0 + 2;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c2 = Vv * B;
  }
  {
    int t = t0 + 3;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c3 = Vv * B;
  }
  {
    int t = t0 + 4;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c4 = Vv * B;
  }
  {
    int t = t0 + 5;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c5 = Vv * B;
  }
  {
    int t = t0 + 6;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c6 = Vv * B;
  }
  {
    int t = t0 + 7;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c7 = Vv * B;
  }
  {
    int t = t0 + 8;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c8 = Vv * B;
  }
  {
    int t = t0 + 9;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c9 = Vv * B;
  }
  {
    int t = t0 + 10;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c10 = Vv * B;
  }
  {
    int t = t0 + 11;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c11 = Vv * B;
  }
  {
    int t = t0 + 12;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c12 = Vv * B;
  }
  {
    int t = t0 + 13;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c13 = Vv * B;
  }
  {
    int t = t0 + 14;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c14 = Vv * B;
  }
  {
    int t = t0 + 15;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c15 = Vv * B;
  }
  {
    int t = t0 + 16;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c16 = Vv * B;
  }
  {
    int t = t0 + 17;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c17 = Vv * B;
  }
  {
    int t = t0 + 18;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c18 = Vv * B;
  }
  {
    int t = t0 + 19;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c19 = Vv * B;
  }
  {
    int t = t0 + 20;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c20 = Vv * B;
  }
  {
    int t = t0 + 21;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c21 = Vv * B;
  }
  {
    int t = t0 + 22;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c22 = Vv * B;
  }
  {
    int t = t0 + 23;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c23 = Vv * B;
  }
  {
    int t = t0 + 24;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c24 = Vv * B;
  }
  {
    int t = t0 + 25;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c25 = Vv * B;
  }
  {
    int t = t0 + 26;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c26 = Vv * B;
  }
  {
    int t = t0 + 27;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c27 = Vv * B;
  }
  {
    int t = t0 + 28;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c28 = Vv * B;
  }
  {
    int t = t0 + 29;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c29 = Vv * B;
  }
  {
    int t = t0 + 30;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c30 = Vv * B;
  }
  {
    int t = t0 + 31;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c31 = Vv * B;
  }
  {
    int t = t0 + 32;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c32 = Vv * B;
  }
  {
    int t = t0 + 33;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c33 = Vv * B;
  }
  {
    int t = t0 + 34;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c34 = Vv * B;
  }
  {
    int t = t0 + 35;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c35 = Vv * B;
  }
  {
    int t = t0 + 36;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c36 = Vv * B;
  }
  {
    int t = t0 + 37;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c37 = Vv * B;
  }
  {
    int t = t0 + 38;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c38 = Vv * B;
  }
  {
    int t = t0 + 39;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c39 = Vv * B;
  }
  {
    int t = t0 + 40;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c40 = Vv * B;
  }
  {
    int t = t0 + 41;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c41 = Vv * B;
  }
  {
    int t = t0 + 42;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c42 = Vv * B;
  }
  {
    int t = t0 + 43;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c43 = Vv * B;
  }
  {
    int t = t0 + 44;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c44 = Vv * B;
  }
  {
    int t = t0 + 45;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c45 = Vv * B;
  }
  {
    int t = t0 + 46;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c46 = Vv * B;
  }
  {
    int t = t0 + 47;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c47 = Vv * B;
  }
  {
    int t = t0 + 48;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c48 = Vv * B;
  }
  {
    int t = t0 + 49;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c49 = Vv * B;
  }
  {
    int t = t0 + 50;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c50 = Vv * B;
  }
  {
    int t = t0 + 51;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c51 = Vv * B;
  }
  {
    int t = t0 + 52;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c52 = Vv * B;
  }
  {
    int t = t0 + 53;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c53 = Vv * B;
  }
  {
    int t = t0 + 54;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c54 = Vv * B;
  }
  {
    int t = t0 + 55;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c55 = Vv * B;
  }
  {
    int t = t0 + 56;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c56 = Vv * B;
  }
  {
    int t = t0 + 57;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c57 = Vv * B;
  }
  {
    int t = t0 + 58;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c58 = Vv * B;
  }
  {
    int t = t0 + 59;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c59 = Vv * B;
  }
  {
    int t = t0 + 60;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c60 = Vv * B;
  }
  {
    int t = t0 + 61;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c61 = Vv * B;
  }
  {
    int t = t0 + 62;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c62 = Vv * B;
  }
  {
    int t = t0 + 63;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Vv = (t < T) ? Vh[(long)t * GDN_DV + col] : 0.f;
    c63 = Vv * B;
  }
  { float s = c0;
    c0 = s; nvh[(long)0 * GDN_DV + col] = s; }
  { float s = c1;
    s -= uth[64] * c0;
    c1 = s; nvh[(long)1 * GDN_DV + col] = s; }
  { float s = c2;
    s -= uth[128] * c0;
    s -= uth[129] * c1;
    c2 = s; nvh[(long)2 * GDN_DV + col] = s; }
  { float s = c3;
    s -= uth[192] * c0;
    s -= uth[193] * c1;
    s -= uth[194] * c2;
    c3 = s; nvh[(long)3 * GDN_DV + col] = s; }
  { float s = c4;
    s -= uth[256] * c0;
    s -= uth[257] * c1;
    s -= uth[258] * c2;
    s -= uth[259] * c3;
    c4 = s; nvh[(long)4 * GDN_DV + col] = s; }
  { float s = c5;
    s -= uth[320] * c0;
    s -= uth[321] * c1;
    s -= uth[322] * c2;
    s -= uth[323] * c3;
    s -= uth[324] * c4;
    c5 = s; nvh[(long)5 * GDN_DV + col] = s; }
  { float s = c6;
    s -= uth[384] * c0;
    s -= uth[385] * c1;
    s -= uth[386] * c2;
    s -= uth[387] * c3;
    s -= uth[388] * c4;
    s -= uth[389] * c5;
    c6 = s; nvh[(long)6 * GDN_DV + col] = s; }
  { float s = c7;
    s -= uth[448] * c0;
    s -= uth[449] * c1;
    s -= uth[450] * c2;
    s -= uth[451] * c3;
    s -= uth[452] * c4;
    s -= uth[453] * c5;
    s -= uth[454] * c6;
    c7 = s; nvh[(long)7 * GDN_DV + col] = s; }
  { float s = c8;
    s -= uth[512] * c0;
    s -= uth[513] * c1;
    s -= uth[514] * c2;
    s -= uth[515] * c3;
    s -= uth[516] * c4;
    s -= uth[517] * c5;
    s -= uth[518] * c6;
    s -= uth[519] * c7;
    c8 = s; nvh[(long)8 * GDN_DV + col] = s; }
  { float s = c9;
    s -= uth[576] * c0;
    s -= uth[577] * c1;
    s -= uth[578] * c2;
    s -= uth[579] * c3;
    s -= uth[580] * c4;
    s -= uth[581] * c5;
    s -= uth[582] * c6;
    s -= uth[583] * c7;
    s -= uth[584] * c8;
    c9 = s; nvh[(long)9 * GDN_DV + col] = s; }
  { float s = c10;
    s -= uth[640] * c0;
    s -= uth[641] * c1;
    s -= uth[642] * c2;
    s -= uth[643] * c3;
    s -= uth[644] * c4;
    s -= uth[645] * c5;
    s -= uth[646] * c6;
    s -= uth[647] * c7;
    s -= uth[648] * c8;
    s -= uth[649] * c9;
    c10 = s; nvh[(long)10 * GDN_DV + col] = s; }
  { float s = c11;
    s -= uth[704] * c0;
    s -= uth[705] * c1;
    s -= uth[706] * c2;
    s -= uth[707] * c3;
    s -= uth[708] * c4;
    s -= uth[709] * c5;
    s -= uth[710] * c6;
    s -= uth[711] * c7;
    s -= uth[712] * c8;
    s -= uth[713] * c9;
    s -= uth[714] * c10;
    c11 = s; nvh[(long)11 * GDN_DV + col] = s; }
  { float s = c12;
    s -= uth[768] * c0;
    s -= uth[769] * c1;
    s -= uth[770] * c2;
    s -= uth[771] * c3;
    s -= uth[772] * c4;
    s -= uth[773] * c5;
    s -= uth[774] * c6;
    s -= uth[775] * c7;
    s -= uth[776] * c8;
    s -= uth[777] * c9;
    s -= uth[778] * c10;
    s -= uth[779] * c11;
    c12 = s; nvh[(long)12 * GDN_DV + col] = s; }
  { float s = c13;
    s -= uth[832] * c0;
    s -= uth[833] * c1;
    s -= uth[834] * c2;
    s -= uth[835] * c3;
    s -= uth[836] * c4;
    s -= uth[837] * c5;
    s -= uth[838] * c6;
    s -= uth[839] * c7;
    s -= uth[840] * c8;
    s -= uth[841] * c9;
    s -= uth[842] * c10;
    s -= uth[843] * c11;
    s -= uth[844] * c12;
    c13 = s; nvh[(long)13 * GDN_DV + col] = s; }
  { float s = c14;
    s -= uth[896] * c0;
    s -= uth[897] * c1;
    s -= uth[898] * c2;
    s -= uth[899] * c3;
    s -= uth[900] * c4;
    s -= uth[901] * c5;
    s -= uth[902] * c6;
    s -= uth[903] * c7;
    s -= uth[904] * c8;
    s -= uth[905] * c9;
    s -= uth[906] * c10;
    s -= uth[907] * c11;
    s -= uth[908] * c12;
    s -= uth[909] * c13;
    c14 = s; nvh[(long)14 * GDN_DV + col] = s; }
  { float s = c15;
    s -= uth[960] * c0;
    s -= uth[961] * c1;
    s -= uth[962] * c2;
    s -= uth[963] * c3;
    s -= uth[964] * c4;
    s -= uth[965] * c5;
    s -= uth[966] * c6;
    s -= uth[967] * c7;
    s -= uth[968] * c8;
    s -= uth[969] * c9;
    s -= uth[970] * c10;
    s -= uth[971] * c11;
    s -= uth[972] * c12;
    s -= uth[973] * c13;
    s -= uth[974] * c14;
    c15 = s; nvh[(long)15 * GDN_DV + col] = s; }
  { float s = c16;
    s -= uth[1024] * c0;
    s -= uth[1025] * c1;
    s -= uth[1026] * c2;
    s -= uth[1027] * c3;
    s -= uth[1028] * c4;
    s -= uth[1029] * c5;
    s -= uth[1030] * c6;
    s -= uth[1031] * c7;
    s -= uth[1032] * c8;
    s -= uth[1033] * c9;
    s -= uth[1034] * c10;
    s -= uth[1035] * c11;
    s -= uth[1036] * c12;
    s -= uth[1037] * c13;
    s -= uth[1038] * c14;
    s -= uth[1039] * c15;
    c16 = s; nvh[(long)16 * GDN_DV + col] = s; }
  { float s = c17;
    s -= uth[1088] * c0;
    s -= uth[1089] * c1;
    s -= uth[1090] * c2;
    s -= uth[1091] * c3;
    s -= uth[1092] * c4;
    s -= uth[1093] * c5;
    s -= uth[1094] * c6;
    s -= uth[1095] * c7;
    s -= uth[1096] * c8;
    s -= uth[1097] * c9;
    s -= uth[1098] * c10;
    s -= uth[1099] * c11;
    s -= uth[1100] * c12;
    s -= uth[1101] * c13;
    s -= uth[1102] * c14;
    s -= uth[1103] * c15;
    s -= uth[1104] * c16;
    c17 = s; nvh[(long)17 * GDN_DV + col] = s; }
  { float s = c18;
    s -= uth[1152] * c0;
    s -= uth[1153] * c1;
    s -= uth[1154] * c2;
    s -= uth[1155] * c3;
    s -= uth[1156] * c4;
    s -= uth[1157] * c5;
    s -= uth[1158] * c6;
    s -= uth[1159] * c7;
    s -= uth[1160] * c8;
    s -= uth[1161] * c9;
    s -= uth[1162] * c10;
    s -= uth[1163] * c11;
    s -= uth[1164] * c12;
    s -= uth[1165] * c13;
    s -= uth[1166] * c14;
    s -= uth[1167] * c15;
    s -= uth[1168] * c16;
    s -= uth[1169] * c17;
    c18 = s; nvh[(long)18 * GDN_DV + col] = s; }
  { float s = c19;
    s -= uth[1216] * c0;
    s -= uth[1217] * c1;
    s -= uth[1218] * c2;
    s -= uth[1219] * c3;
    s -= uth[1220] * c4;
    s -= uth[1221] * c5;
    s -= uth[1222] * c6;
    s -= uth[1223] * c7;
    s -= uth[1224] * c8;
    s -= uth[1225] * c9;
    s -= uth[1226] * c10;
    s -= uth[1227] * c11;
    s -= uth[1228] * c12;
    s -= uth[1229] * c13;
    s -= uth[1230] * c14;
    s -= uth[1231] * c15;
    s -= uth[1232] * c16;
    s -= uth[1233] * c17;
    s -= uth[1234] * c18;
    c19 = s; nvh[(long)19 * GDN_DV + col] = s; }
  { float s = c20;
    s -= uth[1280] * c0;
    s -= uth[1281] * c1;
    s -= uth[1282] * c2;
    s -= uth[1283] * c3;
    s -= uth[1284] * c4;
    s -= uth[1285] * c5;
    s -= uth[1286] * c6;
    s -= uth[1287] * c7;
    s -= uth[1288] * c8;
    s -= uth[1289] * c9;
    s -= uth[1290] * c10;
    s -= uth[1291] * c11;
    s -= uth[1292] * c12;
    s -= uth[1293] * c13;
    s -= uth[1294] * c14;
    s -= uth[1295] * c15;
    s -= uth[1296] * c16;
    s -= uth[1297] * c17;
    s -= uth[1298] * c18;
    s -= uth[1299] * c19;
    c20 = s; nvh[(long)20 * GDN_DV + col] = s; }
  { float s = c21;
    s -= uth[1344] * c0;
    s -= uth[1345] * c1;
    s -= uth[1346] * c2;
    s -= uth[1347] * c3;
    s -= uth[1348] * c4;
    s -= uth[1349] * c5;
    s -= uth[1350] * c6;
    s -= uth[1351] * c7;
    s -= uth[1352] * c8;
    s -= uth[1353] * c9;
    s -= uth[1354] * c10;
    s -= uth[1355] * c11;
    s -= uth[1356] * c12;
    s -= uth[1357] * c13;
    s -= uth[1358] * c14;
    s -= uth[1359] * c15;
    s -= uth[1360] * c16;
    s -= uth[1361] * c17;
    s -= uth[1362] * c18;
    s -= uth[1363] * c19;
    s -= uth[1364] * c20;
    c21 = s; nvh[(long)21 * GDN_DV + col] = s; }
  { float s = c22;
    s -= uth[1408] * c0;
    s -= uth[1409] * c1;
    s -= uth[1410] * c2;
    s -= uth[1411] * c3;
    s -= uth[1412] * c4;
    s -= uth[1413] * c5;
    s -= uth[1414] * c6;
    s -= uth[1415] * c7;
    s -= uth[1416] * c8;
    s -= uth[1417] * c9;
    s -= uth[1418] * c10;
    s -= uth[1419] * c11;
    s -= uth[1420] * c12;
    s -= uth[1421] * c13;
    s -= uth[1422] * c14;
    s -= uth[1423] * c15;
    s -= uth[1424] * c16;
    s -= uth[1425] * c17;
    s -= uth[1426] * c18;
    s -= uth[1427] * c19;
    s -= uth[1428] * c20;
    s -= uth[1429] * c21;
    c22 = s; nvh[(long)22 * GDN_DV + col] = s; }
  { float s = c23;
    s -= uth[1472] * c0;
    s -= uth[1473] * c1;
    s -= uth[1474] * c2;
    s -= uth[1475] * c3;
    s -= uth[1476] * c4;
    s -= uth[1477] * c5;
    s -= uth[1478] * c6;
    s -= uth[1479] * c7;
    s -= uth[1480] * c8;
    s -= uth[1481] * c9;
    s -= uth[1482] * c10;
    s -= uth[1483] * c11;
    s -= uth[1484] * c12;
    s -= uth[1485] * c13;
    s -= uth[1486] * c14;
    s -= uth[1487] * c15;
    s -= uth[1488] * c16;
    s -= uth[1489] * c17;
    s -= uth[1490] * c18;
    s -= uth[1491] * c19;
    s -= uth[1492] * c20;
    s -= uth[1493] * c21;
    s -= uth[1494] * c22;
    c23 = s; nvh[(long)23 * GDN_DV + col] = s; }
  { float s = c24;
    s -= uth[1536] * c0;
    s -= uth[1537] * c1;
    s -= uth[1538] * c2;
    s -= uth[1539] * c3;
    s -= uth[1540] * c4;
    s -= uth[1541] * c5;
    s -= uth[1542] * c6;
    s -= uth[1543] * c7;
    s -= uth[1544] * c8;
    s -= uth[1545] * c9;
    s -= uth[1546] * c10;
    s -= uth[1547] * c11;
    s -= uth[1548] * c12;
    s -= uth[1549] * c13;
    s -= uth[1550] * c14;
    s -= uth[1551] * c15;
    s -= uth[1552] * c16;
    s -= uth[1553] * c17;
    s -= uth[1554] * c18;
    s -= uth[1555] * c19;
    s -= uth[1556] * c20;
    s -= uth[1557] * c21;
    s -= uth[1558] * c22;
    s -= uth[1559] * c23;
    c24 = s; nvh[(long)24 * GDN_DV + col] = s; }
  { float s = c25;
    s -= uth[1600] * c0;
    s -= uth[1601] * c1;
    s -= uth[1602] * c2;
    s -= uth[1603] * c3;
    s -= uth[1604] * c4;
    s -= uth[1605] * c5;
    s -= uth[1606] * c6;
    s -= uth[1607] * c7;
    s -= uth[1608] * c8;
    s -= uth[1609] * c9;
    s -= uth[1610] * c10;
    s -= uth[1611] * c11;
    s -= uth[1612] * c12;
    s -= uth[1613] * c13;
    s -= uth[1614] * c14;
    s -= uth[1615] * c15;
    s -= uth[1616] * c16;
    s -= uth[1617] * c17;
    s -= uth[1618] * c18;
    s -= uth[1619] * c19;
    s -= uth[1620] * c20;
    s -= uth[1621] * c21;
    s -= uth[1622] * c22;
    s -= uth[1623] * c23;
    s -= uth[1624] * c24;
    c25 = s; nvh[(long)25 * GDN_DV + col] = s; }
  { float s = c26;
    s -= uth[1664] * c0;
    s -= uth[1665] * c1;
    s -= uth[1666] * c2;
    s -= uth[1667] * c3;
    s -= uth[1668] * c4;
    s -= uth[1669] * c5;
    s -= uth[1670] * c6;
    s -= uth[1671] * c7;
    s -= uth[1672] * c8;
    s -= uth[1673] * c9;
    s -= uth[1674] * c10;
    s -= uth[1675] * c11;
    s -= uth[1676] * c12;
    s -= uth[1677] * c13;
    s -= uth[1678] * c14;
    s -= uth[1679] * c15;
    s -= uth[1680] * c16;
    s -= uth[1681] * c17;
    s -= uth[1682] * c18;
    s -= uth[1683] * c19;
    s -= uth[1684] * c20;
    s -= uth[1685] * c21;
    s -= uth[1686] * c22;
    s -= uth[1687] * c23;
    s -= uth[1688] * c24;
    s -= uth[1689] * c25;
    c26 = s; nvh[(long)26 * GDN_DV + col] = s; }
  { float s = c27;
    s -= uth[1728] * c0;
    s -= uth[1729] * c1;
    s -= uth[1730] * c2;
    s -= uth[1731] * c3;
    s -= uth[1732] * c4;
    s -= uth[1733] * c5;
    s -= uth[1734] * c6;
    s -= uth[1735] * c7;
    s -= uth[1736] * c8;
    s -= uth[1737] * c9;
    s -= uth[1738] * c10;
    s -= uth[1739] * c11;
    s -= uth[1740] * c12;
    s -= uth[1741] * c13;
    s -= uth[1742] * c14;
    s -= uth[1743] * c15;
    s -= uth[1744] * c16;
    s -= uth[1745] * c17;
    s -= uth[1746] * c18;
    s -= uth[1747] * c19;
    s -= uth[1748] * c20;
    s -= uth[1749] * c21;
    s -= uth[1750] * c22;
    s -= uth[1751] * c23;
    s -= uth[1752] * c24;
    s -= uth[1753] * c25;
    s -= uth[1754] * c26;
    c27 = s; nvh[(long)27 * GDN_DV + col] = s; }
  { float s = c28;
    s -= uth[1792] * c0;
    s -= uth[1793] * c1;
    s -= uth[1794] * c2;
    s -= uth[1795] * c3;
    s -= uth[1796] * c4;
    s -= uth[1797] * c5;
    s -= uth[1798] * c6;
    s -= uth[1799] * c7;
    s -= uth[1800] * c8;
    s -= uth[1801] * c9;
    s -= uth[1802] * c10;
    s -= uth[1803] * c11;
    s -= uth[1804] * c12;
    s -= uth[1805] * c13;
    s -= uth[1806] * c14;
    s -= uth[1807] * c15;
    s -= uth[1808] * c16;
    s -= uth[1809] * c17;
    s -= uth[1810] * c18;
    s -= uth[1811] * c19;
    s -= uth[1812] * c20;
    s -= uth[1813] * c21;
    s -= uth[1814] * c22;
    s -= uth[1815] * c23;
    s -= uth[1816] * c24;
    s -= uth[1817] * c25;
    s -= uth[1818] * c26;
    s -= uth[1819] * c27;
    c28 = s; nvh[(long)28 * GDN_DV + col] = s; }
  { float s = c29;
    s -= uth[1856] * c0;
    s -= uth[1857] * c1;
    s -= uth[1858] * c2;
    s -= uth[1859] * c3;
    s -= uth[1860] * c4;
    s -= uth[1861] * c5;
    s -= uth[1862] * c6;
    s -= uth[1863] * c7;
    s -= uth[1864] * c8;
    s -= uth[1865] * c9;
    s -= uth[1866] * c10;
    s -= uth[1867] * c11;
    s -= uth[1868] * c12;
    s -= uth[1869] * c13;
    s -= uth[1870] * c14;
    s -= uth[1871] * c15;
    s -= uth[1872] * c16;
    s -= uth[1873] * c17;
    s -= uth[1874] * c18;
    s -= uth[1875] * c19;
    s -= uth[1876] * c20;
    s -= uth[1877] * c21;
    s -= uth[1878] * c22;
    s -= uth[1879] * c23;
    s -= uth[1880] * c24;
    s -= uth[1881] * c25;
    s -= uth[1882] * c26;
    s -= uth[1883] * c27;
    s -= uth[1884] * c28;
    c29 = s; nvh[(long)29 * GDN_DV + col] = s; }
  { float s = c30;
    s -= uth[1920] * c0;
    s -= uth[1921] * c1;
    s -= uth[1922] * c2;
    s -= uth[1923] * c3;
    s -= uth[1924] * c4;
    s -= uth[1925] * c5;
    s -= uth[1926] * c6;
    s -= uth[1927] * c7;
    s -= uth[1928] * c8;
    s -= uth[1929] * c9;
    s -= uth[1930] * c10;
    s -= uth[1931] * c11;
    s -= uth[1932] * c12;
    s -= uth[1933] * c13;
    s -= uth[1934] * c14;
    s -= uth[1935] * c15;
    s -= uth[1936] * c16;
    s -= uth[1937] * c17;
    s -= uth[1938] * c18;
    s -= uth[1939] * c19;
    s -= uth[1940] * c20;
    s -= uth[1941] * c21;
    s -= uth[1942] * c22;
    s -= uth[1943] * c23;
    s -= uth[1944] * c24;
    s -= uth[1945] * c25;
    s -= uth[1946] * c26;
    s -= uth[1947] * c27;
    s -= uth[1948] * c28;
    s -= uth[1949] * c29;
    c30 = s; nvh[(long)30 * GDN_DV + col] = s; }
  { float s = c31;
    s -= uth[1984] * c0;
    s -= uth[1985] * c1;
    s -= uth[1986] * c2;
    s -= uth[1987] * c3;
    s -= uth[1988] * c4;
    s -= uth[1989] * c5;
    s -= uth[1990] * c6;
    s -= uth[1991] * c7;
    s -= uth[1992] * c8;
    s -= uth[1993] * c9;
    s -= uth[1994] * c10;
    s -= uth[1995] * c11;
    s -= uth[1996] * c12;
    s -= uth[1997] * c13;
    s -= uth[1998] * c14;
    s -= uth[1999] * c15;
    s -= uth[2000] * c16;
    s -= uth[2001] * c17;
    s -= uth[2002] * c18;
    s -= uth[2003] * c19;
    s -= uth[2004] * c20;
    s -= uth[2005] * c21;
    s -= uth[2006] * c22;
    s -= uth[2007] * c23;
    s -= uth[2008] * c24;
    s -= uth[2009] * c25;
    s -= uth[2010] * c26;
    s -= uth[2011] * c27;
    s -= uth[2012] * c28;
    s -= uth[2013] * c29;
    s -= uth[2014] * c30;
    c31 = s; nvh[(long)31 * GDN_DV + col] = s; }
  { float s = c32;
    s -= uth[2048] * c0;
    s -= uth[2049] * c1;
    s -= uth[2050] * c2;
    s -= uth[2051] * c3;
    s -= uth[2052] * c4;
    s -= uth[2053] * c5;
    s -= uth[2054] * c6;
    s -= uth[2055] * c7;
    s -= uth[2056] * c8;
    s -= uth[2057] * c9;
    s -= uth[2058] * c10;
    s -= uth[2059] * c11;
    s -= uth[2060] * c12;
    s -= uth[2061] * c13;
    s -= uth[2062] * c14;
    s -= uth[2063] * c15;
    s -= uth[2064] * c16;
    s -= uth[2065] * c17;
    s -= uth[2066] * c18;
    s -= uth[2067] * c19;
    s -= uth[2068] * c20;
    s -= uth[2069] * c21;
    s -= uth[2070] * c22;
    s -= uth[2071] * c23;
    s -= uth[2072] * c24;
    s -= uth[2073] * c25;
    s -= uth[2074] * c26;
    s -= uth[2075] * c27;
    s -= uth[2076] * c28;
    s -= uth[2077] * c29;
    s -= uth[2078] * c30;
    s -= uth[2079] * c31;
    c32 = s; nvh[(long)32 * GDN_DV + col] = s; }
  { float s = c33;
    s -= uth[2112] * c0;
    s -= uth[2113] * c1;
    s -= uth[2114] * c2;
    s -= uth[2115] * c3;
    s -= uth[2116] * c4;
    s -= uth[2117] * c5;
    s -= uth[2118] * c6;
    s -= uth[2119] * c7;
    s -= uth[2120] * c8;
    s -= uth[2121] * c9;
    s -= uth[2122] * c10;
    s -= uth[2123] * c11;
    s -= uth[2124] * c12;
    s -= uth[2125] * c13;
    s -= uth[2126] * c14;
    s -= uth[2127] * c15;
    s -= uth[2128] * c16;
    s -= uth[2129] * c17;
    s -= uth[2130] * c18;
    s -= uth[2131] * c19;
    s -= uth[2132] * c20;
    s -= uth[2133] * c21;
    s -= uth[2134] * c22;
    s -= uth[2135] * c23;
    s -= uth[2136] * c24;
    s -= uth[2137] * c25;
    s -= uth[2138] * c26;
    s -= uth[2139] * c27;
    s -= uth[2140] * c28;
    s -= uth[2141] * c29;
    s -= uth[2142] * c30;
    s -= uth[2143] * c31;
    s -= uth[2144] * c32;
    c33 = s; nvh[(long)33 * GDN_DV + col] = s; }
  { float s = c34;
    s -= uth[2176] * c0;
    s -= uth[2177] * c1;
    s -= uth[2178] * c2;
    s -= uth[2179] * c3;
    s -= uth[2180] * c4;
    s -= uth[2181] * c5;
    s -= uth[2182] * c6;
    s -= uth[2183] * c7;
    s -= uth[2184] * c8;
    s -= uth[2185] * c9;
    s -= uth[2186] * c10;
    s -= uth[2187] * c11;
    s -= uth[2188] * c12;
    s -= uth[2189] * c13;
    s -= uth[2190] * c14;
    s -= uth[2191] * c15;
    s -= uth[2192] * c16;
    s -= uth[2193] * c17;
    s -= uth[2194] * c18;
    s -= uth[2195] * c19;
    s -= uth[2196] * c20;
    s -= uth[2197] * c21;
    s -= uth[2198] * c22;
    s -= uth[2199] * c23;
    s -= uth[2200] * c24;
    s -= uth[2201] * c25;
    s -= uth[2202] * c26;
    s -= uth[2203] * c27;
    s -= uth[2204] * c28;
    s -= uth[2205] * c29;
    s -= uth[2206] * c30;
    s -= uth[2207] * c31;
    s -= uth[2208] * c32;
    s -= uth[2209] * c33;
    c34 = s; nvh[(long)34 * GDN_DV + col] = s; }
  { float s = c35;
    s -= uth[2240] * c0;
    s -= uth[2241] * c1;
    s -= uth[2242] * c2;
    s -= uth[2243] * c3;
    s -= uth[2244] * c4;
    s -= uth[2245] * c5;
    s -= uth[2246] * c6;
    s -= uth[2247] * c7;
    s -= uth[2248] * c8;
    s -= uth[2249] * c9;
    s -= uth[2250] * c10;
    s -= uth[2251] * c11;
    s -= uth[2252] * c12;
    s -= uth[2253] * c13;
    s -= uth[2254] * c14;
    s -= uth[2255] * c15;
    s -= uth[2256] * c16;
    s -= uth[2257] * c17;
    s -= uth[2258] * c18;
    s -= uth[2259] * c19;
    s -= uth[2260] * c20;
    s -= uth[2261] * c21;
    s -= uth[2262] * c22;
    s -= uth[2263] * c23;
    s -= uth[2264] * c24;
    s -= uth[2265] * c25;
    s -= uth[2266] * c26;
    s -= uth[2267] * c27;
    s -= uth[2268] * c28;
    s -= uth[2269] * c29;
    s -= uth[2270] * c30;
    s -= uth[2271] * c31;
    s -= uth[2272] * c32;
    s -= uth[2273] * c33;
    s -= uth[2274] * c34;
    c35 = s; nvh[(long)35 * GDN_DV + col] = s; }
  { float s = c36;
    s -= uth[2304] * c0;
    s -= uth[2305] * c1;
    s -= uth[2306] * c2;
    s -= uth[2307] * c3;
    s -= uth[2308] * c4;
    s -= uth[2309] * c5;
    s -= uth[2310] * c6;
    s -= uth[2311] * c7;
    s -= uth[2312] * c8;
    s -= uth[2313] * c9;
    s -= uth[2314] * c10;
    s -= uth[2315] * c11;
    s -= uth[2316] * c12;
    s -= uth[2317] * c13;
    s -= uth[2318] * c14;
    s -= uth[2319] * c15;
    s -= uth[2320] * c16;
    s -= uth[2321] * c17;
    s -= uth[2322] * c18;
    s -= uth[2323] * c19;
    s -= uth[2324] * c20;
    s -= uth[2325] * c21;
    s -= uth[2326] * c22;
    s -= uth[2327] * c23;
    s -= uth[2328] * c24;
    s -= uth[2329] * c25;
    s -= uth[2330] * c26;
    s -= uth[2331] * c27;
    s -= uth[2332] * c28;
    s -= uth[2333] * c29;
    s -= uth[2334] * c30;
    s -= uth[2335] * c31;
    s -= uth[2336] * c32;
    s -= uth[2337] * c33;
    s -= uth[2338] * c34;
    s -= uth[2339] * c35;
    c36 = s; nvh[(long)36 * GDN_DV + col] = s; }
  { float s = c37;
    s -= uth[2368] * c0;
    s -= uth[2369] * c1;
    s -= uth[2370] * c2;
    s -= uth[2371] * c3;
    s -= uth[2372] * c4;
    s -= uth[2373] * c5;
    s -= uth[2374] * c6;
    s -= uth[2375] * c7;
    s -= uth[2376] * c8;
    s -= uth[2377] * c9;
    s -= uth[2378] * c10;
    s -= uth[2379] * c11;
    s -= uth[2380] * c12;
    s -= uth[2381] * c13;
    s -= uth[2382] * c14;
    s -= uth[2383] * c15;
    s -= uth[2384] * c16;
    s -= uth[2385] * c17;
    s -= uth[2386] * c18;
    s -= uth[2387] * c19;
    s -= uth[2388] * c20;
    s -= uth[2389] * c21;
    s -= uth[2390] * c22;
    s -= uth[2391] * c23;
    s -= uth[2392] * c24;
    s -= uth[2393] * c25;
    s -= uth[2394] * c26;
    s -= uth[2395] * c27;
    s -= uth[2396] * c28;
    s -= uth[2397] * c29;
    s -= uth[2398] * c30;
    s -= uth[2399] * c31;
    s -= uth[2400] * c32;
    s -= uth[2401] * c33;
    s -= uth[2402] * c34;
    s -= uth[2403] * c35;
    s -= uth[2404] * c36;
    c37 = s; nvh[(long)37 * GDN_DV + col] = s; }
  { float s = c38;
    s -= uth[2432] * c0;
    s -= uth[2433] * c1;
    s -= uth[2434] * c2;
    s -= uth[2435] * c3;
    s -= uth[2436] * c4;
    s -= uth[2437] * c5;
    s -= uth[2438] * c6;
    s -= uth[2439] * c7;
    s -= uth[2440] * c8;
    s -= uth[2441] * c9;
    s -= uth[2442] * c10;
    s -= uth[2443] * c11;
    s -= uth[2444] * c12;
    s -= uth[2445] * c13;
    s -= uth[2446] * c14;
    s -= uth[2447] * c15;
    s -= uth[2448] * c16;
    s -= uth[2449] * c17;
    s -= uth[2450] * c18;
    s -= uth[2451] * c19;
    s -= uth[2452] * c20;
    s -= uth[2453] * c21;
    s -= uth[2454] * c22;
    s -= uth[2455] * c23;
    s -= uth[2456] * c24;
    s -= uth[2457] * c25;
    s -= uth[2458] * c26;
    s -= uth[2459] * c27;
    s -= uth[2460] * c28;
    s -= uth[2461] * c29;
    s -= uth[2462] * c30;
    s -= uth[2463] * c31;
    s -= uth[2464] * c32;
    s -= uth[2465] * c33;
    s -= uth[2466] * c34;
    s -= uth[2467] * c35;
    s -= uth[2468] * c36;
    s -= uth[2469] * c37;
    c38 = s; nvh[(long)38 * GDN_DV + col] = s; }
  { float s = c39;
    s -= uth[2496] * c0;
    s -= uth[2497] * c1;
    s -= uth[2498] * c2;
    s -= uth[2499] * c3;
    s -= uth[2500] * c4;
    s -= uth[2501] * c5;
    s -= uth[2502] * c6;
    s -= uth[2503] * c7;
    s -= uth[2504] * c8;
    s -= uth[2505] * c9;
    s -= uth[2506] * c10;
    s -= uth[2507] * c11;
    s -= uth[2508] * c12;
    s -= uth[2509] * c13;
    s -= uth[2510] * c14;
    s -= uth[2511] * c15;
    s -= uth[2512] * c16;
    s -= uth[2513] * c17;
    s -= uth[2514] * c18;
    s -= uth[2515] * c19;
    s -= uth[2516] * c20;
    s -= uth[2517] * c21;
    s -= uth[2518] * c22;
    s -= uth[2519] * c23;
    s -= uth[2520] * c24;
    s -= uth[2521] * c25;
    s -= uth[2522] * c26;
    s -= uth[2523] * c27;
    s -= uth[2524] * c28;
    s -= uth[2525] * c29;
    s -= uth[2526] * c30;
    s -= uth[2527] * c31;
    s -= uth[2528] * c32;
    s -= uth[2529] * c33;
    s -= uth[2530] * c34;
    s -= uth[2531] * c35;
    s -= uth[2532] * c36;
    s -= uth[2533] * c37;
    s -= uth[2534] * c38;
    c39 = s; nvh[(long)39 * GDN_DV + col] = s; }
  { float s = c40;
    s -= uth[2560] * c0;
    s -= uth[2561] * c1;
    s -= uth[2562] * c2;
    s -= uth[2563] * c3;
    s -= uth[2564] * c4;
    s -= uth[2565] * c5;
    s -= uth[2566] * c6;
    s -= uth[2567] * c7;
    s -= uth[2568] * c8;
    s -= uth[2569] * c9;
    s -= uth[2570] * c10;
    s -= uth[2571] * c11;
    s -= uth[2572] * c12;
    s -= uth[2573] * c13;
    s -= uth[2574] * c14;
    s -= uth[2575] * c15;
    s -= uth[2576] * c16;
    s -= uth[2577] * c17;
    s -= uth[2578] * c18;
    s -= uth[2579] * c19;
    s -= uth[2580] * c20;
    s -= uth[2581] * c21;
    s -= uth[2582] * c22;
    s -= uth[2583] * c23;
    s -= uth[2584] * c24;
    s -= uth[2585] * c25;
    s -= uth[2586] * c26;
    s -= uth[2587] * c27;
    s -= uth[2588] * c28;
    s -= uth[2589] * c29;
    s -= uth[2590] * c30;
    s -= uth[2591] * c31;
    s -= uth[2592] * c32;
    s -= uth[2593] * c33;
    s -= uth[2594] * c34;
    s -= uth[2595] * c35;
    s -= uth[2596] * c36;
    s -= uth[2597] * c37;
    s -= uth[2598] * c38;
    s -= uth[2599] * c39;
    c40 = s; nvh[(long)40 * GDN_DV + col] = s; }
  { float s = c41;
    s -= uth[2624] * c0;
    s -= uth[2625] * c1;
    s -= uth[2626] * c2;
    s -= uth[2627] * c3;
    s -= uth[2628] * c4;
    s -= uth[2629] * c5;
    s -= uth[2630] * c6;
    s -= uth[2631] * c7;
    s -= uth[2632] * c8;
    s -= uth[2633] * c9;
    s -= uth[2634] * c10;
    s -= uth[2635] * c11;
    s -= uth[2636] * c12;
    s -= uth[2637] * c13;
    s -= uth[2638] * c14;
    s -= uth[2639] * c15;
    s -= uth[2640] * c16;
    s -= uth[2641] * c17;
    s -= uth[2642] * c18;
    s -= uth[2643] * c19;
    s -= uth[2644] * c20;
    s -= uth[2645] * c21;
    s -= uth[2646] * c22;
    s -= uth[2647] * c23;
    s -= uth[2648] * c24;
    s -= uth[2649] * c25;
    s -= uth[2650] * c26;
    s -= uth[2651] * c27;
    s -= uth[2652] * c28;
    s -= uth[2653] * c29;
    s -= uth[2654] * c30;
    s -= uth[2655] * c31;
    s -= uth[2656] * c32;
    s -= uth[2657] * c33;
    s -= uth[2658] * c34;
    s -= uth[2659] * c35;
    s -= uth[2660] * c36;
    s -= uth[2661] * c37;
    s -= uth[2662] * c38;
    s -= uth[2663] * c39;
    s -= uth[2664] * c40;
    c41 = s; nvh[(long)41 * GDN_DV + col] = s; }
  { float s = c42;
    s -= uth[2688] * c0;
    s -= uth[2689] * c1;
    s -= uth[2690] * c2;
    s -= uth[2691] * c3;
    s -= uth[2692] * c4;
    s -= uth[2693] * c5;
    s -= uth[2694] * c6;
    s -= uth[2695] * c7;
    s -= uth[2696] * c8;
    s -= uth[2697] * c9;
    s -= uth[2698] * c10;
    s -= uth[2699] * c11;
    s -= uth[2700] * c12;
    s -= uth[2701] * c13;
    s -= uth[2702] * c14;
    s -= uth[2703] * c15;
    s -= uth[2704] * c16;
    s -= uth[2705] * c17;
    s -= uth[2706] * c18;
    s -= uth[2707] * c19;
    s -= uth[2708] * c20;
    s -= uth[2709] * c21;
    s -= uth[2710] * c22;
    s -= uth[2711] * c23;
    s -= uth[2712] * c24;
    s -= uth[2713] * c25;
    s -= uth[2714] * c26;
    s -= uth[2715] * c27;
    s -= uth[2716] * c28;
    s -= uth[2717] * c29;
    s -= uth[2718] * c30;
    s -= uth[2719] * c31;
    s -= uth[2720] * c32;
    s -= uth[2721] * c33;
    s -= uth[2722] * c34;
    s -= uth[2723] * c35;
    s -= uth[2724] * c36;
    s -= uth[2725] * c37;
    s -= uth[2726] * c38;
    s -= uth[2727] * c39;
    s -= uth[2728] * c40;
    s -= uth[2729] * c41;
    c42 = s; nvh[(long)42 * GDN_DV + col] = s; }
  { float s = c43;
    s -= uth[2752] * c0;
    s -= uth[2753] * c1;
    s -= uth[2754] * c2;
    s -= uth[2755] * c3;
    s -= uth[2756] * c4;
    s -= uth[2757] * c5;
    s -= uth[2758] * c6;
    s -= uth[2759] * c7;
    s -= uth[2760] * c8;
    s -= uth[2761] * c9;
    s -= uth[2762] * c10;
    s -= uth[2763] * c11;
    s -= uth[2764] * c12;
    s -= uth[2765] * c13;
    s -= uth[2766] * c14;
    s -= uth[2767] * c15;
    s -= uth[2768] * c16;
    s -= uth[2769] * c17;
    s -= uth[2770] * c18;
    s -= uth[2771] * c19;
    s -= uth[2772] * c20;
    s -= uth[2773] * c21;
    s -= uth[2774] * c22;
    s -= uth[2775] * c23;
    s -= uth[2776] * c24;
    s -= uth[2777] * c25;
    s -= uth[2778] * c26;
    s -= uth[2779] * c27;
    s -= uth[2780] * c28;
    s -= uth[2781] * c29;
    s -= uth[2782] * c30;
    s -= uth[2783] * c31;
    s -= uth[2784] * c32;
    s -= uth[2785] * c33;
    s -= uth[2786] * c34;
    s -= uth[2787] * c35;
    s -= uth[2788] * c36;
    s -= uth[2789] * c37;
    s -= uth[2790] * c38;
    s -= uth[2791] * c39;
    s -= uth[2792] * c40;
    s -= uth[2793] * c41;
    s -= uth[2794] * c42;
    c43 = s; nvh[(long)43 * GDN_DV + col] = s; }
  { float s = c44;
    s -= uth[2816] * c0;
    s -= uth[2817] * c1;
    s -= uth[2818] * c2;
    s -= uth[2819] * c3;
    s -= uth[2820] * c4;
    s -= uth[2821] * c5;
    s -= uth[2822] * c6;
    s -= uth[2823] * c7;
    s -= uth[2824] * c8;
    s -= uth[2825] * c9;
    s -= uth[2826] * c10;
    s -= uth[2827] * c11;
    s -= uth[2828] * c12;
    s -= uth[2829] * c13;
    s -= uth[2830] * c14;
    s -= uth[2831] * c15;
    s -= uth[2832] * c16;
    s -= uth[2833] * c17;
    s -= uth[2834] * c18;
    s -= uth[2835] * c19;
    s -= uth[2836] * c20;
    s -= uth[2837] * c21;
    s -= uth[2838] * c22;
    s -= uth[2839] * c23;
    s -= uth[2840] * c24;
    s -= uth[2841] * c25;
    s -= uth[2842] * c26;
    s -= uth[2843] * c27;
    s -= uth[2844] * c28;
    s -= uth[2845] * c29;
    s -= uth[2846] * c30;
    s -= uth[2847] * c31;
    s -= uth[2848] * c32;
    s -= uth[2849] * c33;
    s -= uth[2850] * c34;
    s -= uth[2851] * c35;
    s -= uth[2852] * c36;
    s -= uth[2853] * c37;
    s -= uth[2854] * c38;
    s -= uth[2855] * c39;
    s -= uth[2856] * c40;
    s -= uth[2857] * c41;
    s -= uth[2858] * c42;
    s -= uth[2859] * c43;
    c44 = s; nvh[(long)44 * GDN_DV + col] = s; }
  { float s = c45;
    s -= uth[2880] * c0;
    s -= uth[2881] * c1;
    s -= uth[2882] * c2;
    s -= uth[2883] * c3;
    s -= uth[2884] * c4;
    s -= uth[2885] * c5;
    s -= uth[2886] * c6;
    s -= uth[2887] * c7;
    s -= uth[2888] * c8;
    s -= uth[2889] * c9;
    s -= uth[2890] * c10;
    s -= uth[2891] * c11;
    s -= uth[2892] * c12;
    s -= uth[2893] * c13;
    s -= uth[2894] * c14;
    s -= uth[2895] * c15;
    s -= uth[2896] * c16;
    s -= uth[2897] * c17;
    s -= uth[2898] * c18;
    s -= uth[2899] * c19;
    s -= uth[2900] * c20;
    s -= uth[2901] * c21;
    s -= uth[2902] * c22;
    s -= uth[2903] * c23;
    s -= uth[2904] * c24;
    s -= uth[2905] * c25;
    s -= uth[2906] * c26;
    s -= uth[2907] * c27;
    s -= uth[2908] * c28;
    s -= uth[2909] * c29;
    s -= uth[2910] * c30;
    s -= uth[2911] * c31;
    s -= uth[2912] * c32;
    s -= uth[2913] * c33;
    s -= uth[2914] * c34;
    s -= uth[2915] * c35;
    s -= uth[2916] * c36;
    s -= uth[2917] * c37;
    s -= uth[2918] * c38;
    s -= uth[2919] * c39;
    s -= uth[2920] * c40;
    s -= uth[2921] * c41;
    s -= uth[2922] * c42;
    s -= uth[2923] * c43;
    s -= uth[2924] * c44;
    c45 = s; nvh[(long)45 * GDN_DV + col] = s; }
  { float s = c46;
    s -= uth[2944] * c0;
    s -= uth[2945] * c1;
    s -= uth[2946] * c2;
    s -= uth[2947] * c3;
    s -= uth[2948] * c4;
    s -= uth[2949] * c5;
    s -= uth[2950] * c6;
    s -= uth[2951] * c7;
    s -= uth[2952] * c8;
    s -= uth[2953] * c9;
    s -= uth[2954] * c10;
    s -= uth[2955] * c11;
    s -= uth[2956] * c12;
    s -= uth[2957] * c13;
    s -= uth[2958] * c14;
    s -= uth[2959] * c15;
    s -= uth[2960] * c16;
    s -= uth[2961] * c17;
    s -= uth[2962] * c18;
    s -= uth[2963] * c19;
    s -= uth[2964] * c20;
    s -= uth[2965] * c21;
    s -= uth[2966] * c22;
    s -= uth[2967] * c23;
    s -= uth[2968] * c24;
    s -= uth[2969] * c25;
    s -= uth[2970] * c26;
    s -= uth[2971] * c27;
    s -= uth[2972] * c28;
    s -= uth[2973] * c29;
    s -= uth[2974] * c30;
    s -= uth[2975] * c31;
    s -= uth[2976] * c32;
    s -= uth[2977] * c33;
    s -= uth[2978] * c34;
    s -= uth[2979] * c35;
    s -= uth[2980] * c36;
    s -= uth[2981] * c37;
    s -= uth[2982] * c38;
    s -= uth[2983] * c39;
    s -= uth[2984] * c40;
    s -= uth[2985] * c41;
    s -= uth[2986] * c42;
    s -= uth[2987] * c43;
    s -= uth[2988] * c44;
    s -= uth[2989] * c45;
    c46 = s; nvh[(long)46 * GDN_DV + col] = s; }
  { float s = c47;
    s -= uth[3008] * c0;
    s -= uth[3009] * c1;
    s -= uth[3010] * c2;
    s -= uth[3011] * c3;
    s -= uth[3012] * c4;
    s -= uth[3013] * c5;
    s -= uth[3014] * c6;
    s -= uth[3015] * c7;
    s -= uth[3016] * c8;
    s -= uth[3017] * c9;
    s -= uth[3018] * c10;
    s -= uth[3019] * c11;
    s -= uth[3020] * c12;
    s -= uth[3021] * c13;
    s -= uth[3022] * c14;
    s -= uth[3023] * c15;
    s -= uth[3024] * c16;
    s -= uth[3025] * c17;
    s -= uth[3026] * c18;
    s -= uth[3027] * c19;
    s -= uth[3028] * c20;
    s -= uth[3029] * c21;
    s -= uth[3030] * c22;
    s -= uth[3031] * c23;
    s -= uth[3032] * c24;
    s -= uth[3033] * c25;
    s -= uth[3034] * c26;
    s -= uth[3035] * c27;
    s -= uth[3036] * c28;
    s -= uth[3037] * c29;
    s -= uth[3038] * c30;
    s -= uth[3039] * c31;
    s -= uth[3040] * c32;
    s -= uth[3041] * c33;
    s -= uth[3042] * c34;
    s -= uth[3043] * c35;
    s -= uth[3044] * c36;
    s -= uth[3045] * c37;
    s -= uth[3046] * c38;
    s -= uth[3047] * c39;
    s -= uth[3048] * c40;
    s -= uth[3049] * c41;
    s -= uth[3050] * c42;
    s -= uth[3051] * c43;
    s -= uth[3052] * c44;
    s -= uth[3053] * c45;
    s -= uth[3054] * c46;
    c47 = s; nvh[(long)47 * GDN_DV + col] = s; }
  { float s = c48;
    s -= uth[3072] * c0;
    s -= uth[3073] * c1;
    s -= uth[3074] * c2;
    s -= uth[3075] * c3;
    s -= uth[3076] * c4;
    s -= uth[3077] * c5;
    s -= uth[3078] * c6;
    s -= uth[3079] * c7;
    s -= uth[3080] * c8;
    s -= uth[3081] * c9;
    s -= uth[3082] * c10;
    s -= uth[3083] * c11;
    s -= uth[3084] * c12;
    s -= uth[3085] * c13;
    s -= uth[3086] * c14;
    s -= uth[3087] * c15;
    s -= uth[3088] * c16;
    s -= uth[3089] * c17;
    s -= uth[3090] * c18;
    s -= uth[3091] * c19;
    s -= uth[3092] * c20;
    s -= uth[3093] * c21;
    s -= uth[3094] * c22;
    s -= uth[3095] * c23;
    s -= uth[3096] * c24;
    s -= uth[3097] * c25;
    s -= uth[3098] * c26;
    s -= uth[3099] * c27;
    s -= uth[3100] * c28;
    s -= uth[3101] * c29;
    s -= uth[3102] * c30;
    s -= uth[3103] * c31;
    s -= uth[3104] * c32;
    s -= uth[3105] * c33;
    s -= uth[3106] * c34;
    s -= uth[3107] * c35;
    s -= uth[3108] * c36;
    s -= uth[3109] * c37;
    s -= uth[3110] * c38;
    s -= uth[3111] * c39;
    s -= uth[3112] * c40;
    s -= uth[3113] * c41;
    s -= uth[3114] * c42;
    s -= uth[3115] * c43;
    s -= uth[3116] * c44;
    s -= uth[3117] * c45;
    s -= uth[3118] * c46;
    s -= uth[3119] * c47;
    c48 = s; nvh[(long)48 * GDN_DV + col] = s; }
  { float s = c49;
    s -= uth[3136] * c0;
    s -= uth[3137] * c1;
    s -= uth[3138] * c2;
    s -= uth[3139] * c3;
    s -= uth[3140] * c4;
    s -= uth[3141] * c5;
    s -= uth[3142] * c6;
    s -= uth[3143] * c7;
    s -= uth[3144] * c8;
    s -= uth[3145] * c9;
    s -= uth[3146] * c10;
    s -= uth[3147] * c11;
    s -= uth[3148] * c12;
    s -= uth[3149] * c13;
    s -= uth[3150] * c14;
    s -= uth[3151] * c15;
    s -= uth[3152] * c16;
    s -= uth[3153] * c17;
    s -= uth[3154] * c18;
    s -= uth[3155] * c19;
    s -= uth[3156] * c20;
    s -= uth[3157] * c21;
    s -= uth[3158] * c22;
    s -= uth[3159] * c23;
    s -= uth[3160] * c24;
    s -= uth[3161] * c25;
    s -= uth[3162] * c26;
    s -= uth[3163] * c27;
    s -= uth[3164] * c28;
    s -= uth[3165] * c29;
    s -= uth[3166] * c30;
    s -= uth[3167] * c31;
    s -= uth[3168] * c32;
    s -= uth[3169] * c33;
    s -= uth[3170] * c34;
    s -= uth[3171] * c35;
    s -= uth[3172] * c36;
    s -= uth[3173] * c37;
    s -= uth[3174] * c38;
    s -= uth[3175] * c39;
    s -= uth[3176] * c40;
    s -= uth[3177] * c41;
    s -= uth[3178] * c42;
    s -= uth[3179] * c43;
    s -= uth[3180] * c44;
    s -= uth[3181] * c45;
    s -= uth[3182] * c46;
    s -= uth[3183] * c47;
    s -= uth[3184] * c48;
    c49 = s; nvh[(long)49 * GDN_DV + col] = s; }
  { float s = c50;
    s -= uth[3200] * c0;
    s -= uth[3201] * c1;
    s -= uth[3202] * c2;
    s -= uth[3203] * c3;
    s -= uth[3204] * c4;
    s -= uth[3205] * c5;
    s -= uth[3206] * c6;
    s -= uth[3207] * c7;
    s -= uth[3208] * c8;
    s -= uth[3209] * c9;
    s -= uth[3210] * c10;
    s -= uth[3211] * c11;
    s -= uth[3212] * c12;
    s -= uth[3213] * c13;
    s -= uth[3214] * c14;
    s -= uth[3215] * c15;
    s -= uth[3216] * c16;
    s -= uth[3217] * c17;
    s -= uth[3218] * c18;
    s -= uth[3219] * c19;
    s -= uth[3220] * c20;
    s -= uth[3221] * c21;
    s -= uth[3222] * c22;
    s -= uth[3223] * c23;
    s -= uth[3224] * c24;
    s -= uth[3225] * c25;
    s -= uth[3226] * c26;
    s -= uth[3227] * c27;
    s -= uth[3228] * c28;
    s -= uth[3229] * c29;
    s -= uth[3230] * c30;
    s -= uth[3231] * c31;
    s -= uth[3232] * c32;
    s -= uth[3233] * c33;
    s -= uth[3234] * c34;
    s -= uth[3235] * c35;
    s -= uth[3236] * c36;
    s -= uth[3237] * c37;
    s -= uth[3238] * c38;
    s -= uth[3239] * c39;
    s -= uth[3240] * c40;
    s -= uth[3241] * c41;
    s -= uth[3242] * c42;
    s -= uth[3243] * c43;
    s -= uth[3244] * c44;
    s -= uth[3245] * c45;
    s -= uth[3246] * c46;
    s -= uth[3247] * c47;
    s -= uth[3248] * c48;
    s -= uth[3249] * c49;
    c50 = s; nvh[(long)50 * GDN_DV + col] = s; }
  { float s = c51;
    s -= uth[3264] * c0;
    s -= uth[3265] * c1;
    s -= uth[3266] * c2;
    s -= uth[3267] * c3;
    s -= uth[3268] * c4;
    s -= uth[3269] * c5;
    s -= uth[3270] * c6;
    s -= uth[3271] * c7;
    s -= uth[3272] * c8;
    s -= uth[3273] * c9;
    s -= uth[3274] * c10;
    s -= uth[3275] * c11;
    s -= uth[3276] * c12;
    s -= uth[3277] * c13;
    s -= uth[3278] * c14;
    s -= uth[3279] * c15;
    s -= uth[3280] * c16;
    s -= uth[3281] * c17;
    s -= uth[3282] * c18;
    s -= uth[3283] * c19;
    s -= uth[3284] * c20;
    s -= uth[3285] * c21;
    s -= uth[3286] * c22;
    s -= uth[3287] * c23;
    s -= uth[3288] * c24;
    s -= uth[3289] * c25;
    s -= uth[3290] * c26;
    s -= uth[3291] * c27;
    s -= uth[3292] * c28;
    s -= uth[3293] * c29;
    s -= uth[3294] * c30;
    s -= uth[3295] * c31;
    s -= uth[3296] * c32;
    s -= uth[3297] * c33;
    s -= uth[3298] * c34;
    s -= uth[3299] * c35;
    s -= uth[3300] * c36;
    s -= uth[3301] * c37;
    s -= uth[3302] * c38;
    s -= uth[3303] * c39;
    s -= uth[3304] * c40;
    s -= uth[3305] * c41;
    s -= uth[3306] * c42;
    s -= uth[3307] * c43;
    s -= uth[3308] * c44;
    s -= uth[3309] * c45;
    s -= uth[3310] * c46;
    s -= uth[3311] * c47;
    s -= uth[3312] * c48;
    s -= uth[3313] * c49;
    s -= uth[3314] * c50;
    c51 = s; nvh[(long)51 * GDN_DV + col] = s; }
  { float s = c52;
    s -= uth[3328] * c0;
    s -= uth[3329] * c1;
    s -= uth[3330] * c2;
    s -= uth[3331] * c3;
    s -= uth[3332] * c4;
    s -= uth[3333] * c5;
    s -= uth[3334] * c6;
    s -= uth[3335] * c7;
    s -= uth[3336] * c8;
    s -= uth[3337] * c9;
    s -= uth[3338] * c10;
    s -= uth[3339] * c11;
    s -= uth[3340] * c12;
    s -= uth[3341] * c13;
    s -= uth[3342] * c14;
    s -= uth[3343] * c15;
    s -= uth[3344] * c16;
    s -= uth[3345] * c17;
    s -= uth[3346] * c18;
    s -= uth[3347] * c19;
    s -= uth[3348] * c20;
    s -= uth[3349] * c21;
    s -= uth[3350] * c22;
    s -= uth[3351] * c23;
    s -= uth[3352] * c24;
    s -= uth[3353] * c25;
    s -= uth[3354] * c26;
    s -= uth[3355] * c27;
    s -= uth[3356] * c28;
    s -= uth[3357] * c29;
    s -= uth[3358] * c30;
    s -= uth[3359] * c31;
    s -= uth[3360] * c32;
    s -= uth[3361] * c33;
    s -= uth[3362] * c34;
    s -= uth[3363] * c35;
    s -= uth[3364] * c36;
    s -= uth[3365] * c37;
    s -= uth[3366] * c38;
    s -= uth[3367] * c39;
    s -= uth[3368] * c40;
    s -= uth[3369] * c41;
    s -= uth[3370] * c42;
    s -= uth[3371] * c43;
    s -= uth[3372] * c44;
    s -= uth[3373] * c45;
    s -= uth[3374] * c46;
    s -= uth[3375] * c47;
    s -= uth[3376] * c48;
    s -= uth[3377] * c49;
    s -= uth[3378] * c50;
    s -= uth[3379] * c51;
    c52 = s; nvh[(long)52 * GDN_DV + col] = s; }
  { float s = c53;
    s -= uth[3392] * c0;
    s -= uth[3393] * c1;
    s -= uth[3394] * c2;
    s -= uth[3395] * c3;
    s -= uth[3396] * c4;
    s -= uth[3397] * c5;
    s -= uth[3398] * c6;
    s -= uth[3399] * c7;
    s -= uth[3400] * c8;
    s -= uth[3401] * c9;
    s -= uth[3402] * c10;
    s -= uth[3403] * c11;
    s -= uth[3404] * c12;
    s -= uth[3405] * c13;
    s -= uth[3406] * c14;
    s -= uth[3407] * c15;
    s -= uth[3408] * c16;
    s -= uth[3409] * c17;
    s -= uth[3410] * c18;
    s -= uth[3411] * c19;
    s -= uth[3412] * c20;
    s -= uth[3413] * c21;
    s -= uth[3414] * c22;
    s -= uth[3415] * c23;
    s -= uth[3416] * c24;
    s -= uth[3417] * c25;
    s -= uth[3418] * c26;
    s -= uth[3419] * c27;
    s -= uth[3420] * c28;
    s -= uth[3421] * c29;
    s -= uth[3422] * c30;
    s -= uth[3423] * c31;
    s -= uth[3424] * c32;
    s -= uth[3425] * c33;
    s -= uth[3426] * c34;
    s -= uth[3427] * c35;
    s -= uth[3428] * c36;
    s -= uth[3429] * c37;
    s -= uth[3430] * c38;
    s -= uth[3431] * c39;
    s -= uth[3432] * c40;
    s -= uth[3433] * c41;
    s -= uth[3434] * c42;
    s -= uth[3435] * c43;
    s -= uth[3436] * c44;
    s -= uth[3437] * c45;
    s -= uth[3438] * c46;
    s -= uth[3439] * c47;
    s -= uth[3440] * c48;
    s -= uth[3441] * c49;
    s -= uth[3442] * c50;
    s -= uth[3443] * c51;
    s -= uth[3444] * c52;
    c53 = s; nvh[(long)53 * GDN_DV + col] = s; }
  { float s = c54;
    s -= uth[3456] * c0;
    s -= uth[3457] * c1;
    s -= uth[3458] * c2;
    s -= uth[3459] * c3;
    s -= uth[3460] * c4;
    s -= uth[3461] * c5;
    s -= uth[3462] * c6;
    s -= uth[3463] * c7;
    s -= uth[3464] * c8;
    s -= uth[3465] * c9;
    s -= uth[3466] * c10;
    s -= uth[3467] * c11;
    s -= uth[3468] * c12;
    s -= uth[3469] * c13;
    s -= uth[3470] * c14;
    s -= uth[3471] * c15;
    s -= uth[3472] * c16;
    s -= uth[3473] * c17;
    s -= uth[3474] * c18;
    s -= uth[3475] * c19;
    s -= uth[3476] * c20;
    s -= uth[3477] * c21;
    s -= uth[3478] * c22;
    s -= uth[3479] * c23;
    s -= uth[3480] * c24;
    s -= uth[3481] * c25;
    s -= uth[3482] * c26;
    s -= uth[3483] * c27;
    s -= uth[3484] * c28;
    s -= uth[3485] * c29;
    s -= uth[3486] * c30;
    s -= uth[3487] * c31;
    s -= uth[3488] * c32;
    s -= uth[3489] * c33;
    s -= uth[3490] * c34;
    s -= uth[3491] * c35;
    s -= uth[3492] * c36;
    s -= uth[3493] * c37;
    s -= uth[3494] * c38;
    s -= uth[3495] * c39;
    s -= uth[3496] * c40;
    s -= uth[3497] * c41;
    s -= uth[3498] * c42;
    s -= uth[3499] * c43;
    s -= uth[3500] * c44;
    s -= uth[3501] * c45;
    s -= uth[3502] * c46;
    s -= uth[3503] * c47;
    s -= uth[3504] * c48;
    s -= uth[3505] * c49;
    s -= uth[3506] * c50;
    s -= uth[3507] * c51;
    s -= uth[3508] * c52;
    s -= uth[3509] * c53;
    c54 = s; nvh[(long)54 * GDN_DV + col] = s; }
  { float s = c55;
    s -= uth[3520] * c0;
    s -= uth[3521] * c1;
    s -= uth[3522] * c2;
    s -= uth[3523] * c3;
    s -= uth[3524] * c4;
    s -= uth[3525] * c5;
    s -= uth[3526] * c6;
    s -= uth[3527] * c7;
    s -= uth[3528] * c8;
    s -= uth[3529] * c9;
    s -= uth[3530] * c10;
    s -= uth[3531] * c11;
    s -= uth[3532] * c12;
    s -= uth[3533] * c13;
    s -= uth[3534] * c14;
    s -= uth[3535] * c15;
    s -= uth[3536] * c16;
    s -= uth[3537] * c17;
    s -= uth[3538] * c18;
    s -= uth[3539] * c19;
    s -= uth[3540] * c20;
    s -= uth[3541] * c21;
    s -= uth[3542] * c22;
    s -= uth[3543] * c23;
    s -= uth[3544] * c24;
    s -= uth[3545] * c25;
    s -= uth[3546] * c26;
    s -= uth[3547] * c27;
    s -= uth[3548] * c28;
    s -= uth[3549] * c29;
    s -= uth[3550] * c30;
    s -= uth[3551] * c31;
    s -= uth[3552] * c32;
    s -= uth[3553] * c33;
    s -= uth[3554] * c34;
    s -= uth[3555] * c35;
    s -= uth[3556] * c36;
    s -= uth[3557] * c37;
    s -= uth[3558] * c38;
    s -= uth[3559] * c39;
    s -= uth[3560] * c40;
    s -= uth[3561] * c41;
    s -= uth[3562] * c42;
    s -= uth[3563] * c43;
    s -= uth[3564] * c44;
    s -= uth[3565] * c45;
    s -= uth[3566] * c46;
    s -= uth[3567] * c47;
    s -= uth[3568] * c48;
    s -= uth[3569] * c49;
    s -= uth[3570] * c50;
    s -= uth[3571] * c51;
    s -= uth[3572] * c52;
    s -= uth[3573] * c53;
    s -= uth[3574] * c54;
    c55 = s; nvh[(long)55 * GDN_DV + col] = s; }
  { float s = c56;
    s -= uth[3584] * c0;
    s -= uth[3585] * c1;
    s -= uth[3586] * c2;
    s -= uth[3587] * c3;
    s -= uth[3588] * c4;
    s -= uth[3589] * c5;
    s -= uth[3590] * c6;
    s -= uth[3591] * c7;
    s -= uth[3592] * c8;
    s -= uth[3593] * c9;
    s -= uth[3594] * c10;
    s -= uth[3595] * c11;
    s -= uth[3596] * c12;
    s -= uth[3597] * c13;
    s -= uth[3598] * c14;
    s -= uth[3599] * c15;
    s -= uth[3600] * c16;
    s -= uth[3601] * c17;
    s -= uth[3602] * c18;
    s -= uth[3603] * c19;
    s -= uth[3604] * c20;
    s -= uth[3605] * c21;
    s -= uth[3606] * c22;
    s -= uth[3607] * c23;
    s -= uth[3608] * c24;
    s -= uth[3609] * c25;
    s -= uth[3610] * c26;
    s -= uth[3611] * c27;
    s -= uth[3612] * c28;
    s -= uth[3613] * c29;
    s -= uth[3614] * c30;
    s -= uth[3615] * c31;
    s -= uth[3616] * c32;
    s -= uth[3617] * c33;
    s -= uth[3618] * c34;
    s -= uth[3619] * c35;
    s -= uth[3620] * c36;
    s -= uth[3621] * c37;
    s -= uth[3622] * c38;
    s -= uth[3623] * c39;
    s -= uth[3624] * c40;
    s -= uth[3625] * c41;
    s -= uth[3626] * c42;
    s -= uth[3627] * c43;
    s -= uth[3628] * c44;
    s -= uth[3629] * c45;
    s -= uth[3630] * c46;
    s -= uth[3631] * c47;
    s -= uth[3632] * c48;
    s -= uth[3633] * c49;
    s -= uth[3634] * c50;
    s -= uth[3635] * c51;
    s -= uth[3636] * c52;
    s -= uth[3637] * c53;
    s -= uth[3638] * c54;
    s -= uth[3639] * c55;
    c56 = s; nvh[(long)56 * GDN_DV + col] = s; }
  { float s = c57;
    s -= uth[3648] * c0;
    s -= uth[3649] * c1;
    s -= uth[3650] * c2;
    s -= uth[3651] * c3;
    s -= uth[3652] * c4;
    s -= uth[3653] * c5;
    s -= uth[3654] * c6;
    s -= uth[3655] * c7;
    s -= uth[3656] * c8;
    s -= uth[3657] * c9;
    s -= uth[3658] * c10;
    s -= uth[3659] * c11;
    s -= uth[3660] * c12;
    s -= uth[3661] * c13;
    s -= uth[3662] * c14;
    s -= uth[3663] * c15;
    s -= uth[3664] * c16;
    s -= uth[3665] * c17;
    s -= uth[3666] * c18;
    s -= uth[3667] * c19;
    s -= uth[3668] * c20;
    s -= uth[3669] * c21;
    s -= uth[3670] * c22;
    s -= uth[3671] * c23;
    s -= uth[3672] * c24;
    s -= uth[3673] * c25;
    s -= uth[3674] * c26;
    s -= uth[3675] * c27;
    s -= uth[3676] * c28;
    s -= uth[3677] * c29;
    s -= uth[3678] * c30;
    s -= uth[3679] * c31;
    s -= uth[3680] * c32;
    s -= uth[3681] * c33;
    s -= uth[3682] * c34;
    s -= uth[3683] * c35;
    s -= uth[3684] * c36;
    s -= uth[3685] * c37;
    s -= uth[3686] * c38;
    s -= uth[3687] * c39;
    s -= uth[3688] * c40;
    s -= uth[3689] * c41;
    s -= uth[3690] * c42;
    s -= uth[3691] * c43;
    s -= uth[3692] * c44;
    s -= uth[3693] * c45;
    s -= uth[3694] * c46;
    s -= uth[3695] * c47;
    s -= uth[3696] * c48;
    s -= uth[3697] * c49;
    s -= uth[3698] * c50;
    s -= uth[3699] * c51;
    s -= uth[3700] * c52;
    s -= uth[3701] * c53;
    s -= uth[3702] * c54;
    s -= uth[3703] * c55;
    s -= uth[3704] * c56;
    c57 = s; nvh[(long)57 * GDN_DV + col] = s; }
  { float s = c58;
    s -= uth[3712] * c0;
    s -= uth[3713] * c1;
    s -= uth[3714] * c2;
    s -= uth[3715] * c3;
    s -= uth[3716] * c4;
    s -= uth[3717] * c5;
    s -= uth[3718] * c6;
    s -= uth[3719] * c7;
    s -= uth[3720] * c8;
    s -= uth[3721] * c9;
    s -= uth[3722] * c10;
    s -= uth[3723] * c11;
    s -= uth[3724] * c12;
    s -= uth[3725] * c13;
    s -= uth[3726] * c14;
    s -= uth[3727] * c15;
    s -= uth[3728] * c16;
    s -= uth[3729] * c17;
    s -= uth[3730] * c18;
    s -= uth[3731] * c19;
    s -= uth[3732] * c20;
    s -= uth[3733] * c21;
    s -= uth[3734] * c22;
    s -= uth[3735] * c23;
    s -= uth[3736] * c24;
    s -= uth[3737] * c25;
    s -= uth[3738] * c26;
    s -= uth[3739] * c27;
    s -= uth[3740] * c28;
    s -= uth[3741] * c29;
    s -= uth[3742] * c30;
    s -= uth[3743] * c31;
    s -= uth[3744] * c32;
    s -= uth[3745] * c33;
    s -= uth[3746] * c34;
    s -= uth[3747] * c35;
    s -= uth[3748] * c36;
    s -= uth[3749] * c37;
    s -= uth[3750] * c38;
    s -= uth[3751] * c39;
    s -= uth[3752] * c40;
    s -= uth[3753] * c41;
    s -= uth[3754] * c42;
    s -= uth[3755] * c43;
    s -= uth[3756] * c44;
    s -= uth[3757] * c45;
    s -= uth[3758] * c46;
    s -= uth[3759] * c47;
    s -= uth[3760] * c48;
    s -= uth[3761] * c49;
    s -= uth[3762] * c50;
    s -= uth[3763] * c51;
    s -= uth[3764] * c52;
    s -= uth[3765] * c53;
    s -= uth[3766] * c54;
    s -= uth[3767] * c55;
    s -= uth[3768] * c56;
    s -= uth[3769] * c57;
    c58 = s; nvh[(long)58 * GDN_DV + col] = s; }
  { float s = c59;
    s -= uth[3776] * c0;
    s -= uth[3777] * c1;
    s -= uth[3778] * c2;
    s -= uth[3779] * c3;
    s -= uth[3780] * c4;
    s -= uth[3781] * c5;
    s -= uth[3782] * c6;
    s -= uth[3783] * c7;
    s -= uth[3784] * c8;
    s -= uth[3785] * c9;
    s -= uth[3786] * c10;
    s -= uth[3787] * c11;
    s -= uth[3788] * c12;
    s -= uth[3789] * c13;
    s -= uth[3790] * c14;
    s -= uth[3791] * c15;
    s -= uth[3792] * c16;
    s -= uth[3793] * c17;
    s -= uth[3794] * c18;
    s -= uth[3795] * c19;
    s -= uth[3796] * c20;
    s -= uth[3797] * c21;
    s -= uth[3798] * c22;
    s -= uth[3799] * c23;
    s -= uth[3800] * c24;
    s -= uth[3801] * c25;
    s -= uth[3802] * c26;
    s -= uth[3803] * c27;
    s -= uth[3804] * c28;
    s -= uth[3805] * c29;
    s -= uth[3806] * c30;
    s -= uth[3807] * c31;
    s -= uth[3808] * c32;
    s -= uth[3809] * c33;
    s -= uth[3810] * c34;
    s -= uth[3811] * c35;
    s -= uth[3812] * c36;
    s -= uth[3813] * c37;
    s -= uth[3814] * c38;
    s -= uth[3815] * c39;
    s -= uth[3816] * c40;
    s -= uth[3817] * c41;
    s -= uth[3818] * c42;
    s -= uth[3819] * c43;
    s -= uth[3820] * c44;
    s -= uth[3821] * c45;
    s -= uth[3822] * c46;
    s -= uth[3823] * c47;
    s -= uth[3824] * c48;
    s -= uth[3825] * c49;
    s -= uth[3826] * c50;
    s -= uth[3827] * c51;
    s -= uth[3828] * c52;
    s -= uth[3829] * c53;
    s -= uth[3830] * c54;
    s -= uth[3831] * c55;
    s -= uth[3832] * c56;
    s -= uth[3833] * c57;
    s -= uth[3834] * c58;
    c59 = s; nvh[(long)59 * GDN_DV + col] = s; }
  { float s = c60;
    s -= uth[3840] * c0;
    s -= uth[3841] * c1;
    s -= uth[3842] * c2;
    s -= uth[3843] * c3;
    s -= uth[3844] * c4;
    s -= uth[3845] * c5;
    s -= uth[3846] * c6;
    s -= uth[3847] * c7;
    s -= uth[3848] * c8;
    s -= uth[3849] * c9;
    s -= uth[3850] * c10;
    s -= uth[3851] * c11;
    s -= uth[3852] * c12;
    s -= uth[3853] * c13;
    s -= uth[3854] * c14;
    s -= uth[3855] * c15;
    s -= uth[3856] * c16;
    s -= uth[3857] * c17;
    s -= uth[3858] * c18;
    s -= uth[3859] * c19;
    s -= uth[3860] * c20;
    s -= uth[3861] * c21;
    s -= uth[3862] * c22;
    s -= uth[3863] * c23;
    s -= uth[3864] * c24;
    s -= uth[3865] * c25;
    s -= uth[3866] * c26;
    s -= uth[3867] * c27;
    s -= uth[3868] * c28;
    s -= uth[3869] * c29;
    s -= uth[3870] * c30;
    s -= uth[3871] * c31;
    s -= uth[3872] * c32;
    s -= uth[3873] * c33;
    s -= uth[3874] * c34;
    s -= uth[3875] * c35;
    s -= uth[3876] * c36;
    s -= uth[3877] * c37;
    s -= uth[3878] * c38;
    s -= uth[3879] * c39;
    s -= uth[3880] * c40;
    s -= uth[3881] * c41;
    s -= uth[3882] * c42;
    s -= uth[3883] * c43;
    s -= uth[3884] * c44;
    s -= uth[3885] * c45;
    s -= uth[3886] * c46;
    s -= uth[3887] * c47;
    s -= uth[3888] * c48;
    s -= uth[3889] * c49;
    s -= uth[3890] * c50;
    s -= uth[3891] * c51;
    s -= uth[3892] * c52;
    s -= uth[3893] * c53;
    s -= uth[3894] * c54;
    s -= uth[3895] * c55;
    s -= uth[3896] * c56;
    s -= uth[3897] * c57;
    s -= uth[3898] * c58;
    s -= uth[3899] * c59;
    c60 = s; nvh[(long)60 * GDN_DV + col] = s; }
  { float s = c61;
    s -= uth[3904] * c0;
    s -= uth[3905] * c1;
    s -= uth[3906] * c2;
    s -= uth[3907] * c3;
    s -= uth[3908] * c4;
    s -= uth[3909] * c5;
    s -= uth[3910] * c6;
    s -= uth[3911] * c7;
    s -= uth[3912] * c8;
    s -= uth[3913] * c9;
    s -= uth[3914] * c10;
    s -= uth[3915] * c11;
    s -= uth[3916] * c12;
    s -= uth[3917] * c13;
    s -= uth[3918] * c14;
    s -= uth[3919] * c15;
    s -= uth[3920] * c16;
    s -= uth[3921] * c17;
    s -= uth[3922] * c18;
    s -= uth[3923] * c19;
    s -= uth[3924] * c20;
    s -= uth[3925] * c21;
    s -= uth[3926] * c22;
    s -= uth[3927] * c23;
    s -= uth[3928] * c24;
    s -= uth[3929] * c25;
    s -= uth[3930] * c26;
    s -= uth[3931] * c27;
    s -= uth[3932] * c28;
    s -= uth[3933] * c29;
    s -= uth[3934] * c30;
    s -= uth[3935] * c31;
    s -= uth[3936] * c32;
    s -= uth[3937] * c33;
    s -= uth[3938] * c34;
    s -= uth[3939] * c35;
    s -= uth[3940] * c36;
    s -= uth[3941] * c37;
    s -= uth[3942] * c38;
    s -= uth[3943] * c39;
    s -= uth[3944] * c40;
    s -= uth[3945] * c41;
    s -= uth[3946] * c42;
    s -= uth[3947] * c43;
    s -= uth[3948] * c44;
    s -= uth[3949] * c45;
    s -= uth[3950] * c46;
    s -= uth[3951] * c47;
    s -= uth[3952] * c48;
    s -= uth[3953] * c49;
    s -= uth[3954] * c50;
    s -= uth[3955] * c51;
    s -= uth[3956] * c52;
    s -= uth[3957] * c53;
    s -= uth[3958] * c54;
    s -= uth[3959] * c55;
    s -= uth[3960] * c56;
    s -= uth[3961] * c57;
    s -= uth[3962] * c58;
    s -= uth[3963] * c59;
    s -= uth[3964] * c60;
    c61 = s; nvh[(long)61 * GDN_DV + col] = s; }
  { float s = c62;
    s -= uth[3968] * c0;
    s -= uth[3969] * c1;
    s -= uth[3970] * c2;
    s -= uth[3971] * c3;
    s -= uth[3972] * c4;
    s -= uth[3973] * c5;
    s -= uth[3974] * c6;
    s -= uth[3975] * c7;
    s -= uth[3976] * c8;
    s -= uth[3977] * c9;
    s -= uth[3978] * c10;
    s -= uth[3979] * c11;
    s -= uth[3980] * c12;
    s -= uth[3981] * c13;
    s -= uth[3982] * c14;
    s -= uth[3983] * c15;
    s -= uth[3984] * c16;
    s -= uth[3985] * c17;
    s -= uth[3986] * c18;
    s -= uth[3987] * c19;
    s -= uth[3988] * c20;
    s -= uth[3989] * c21;
    s -= uth[3990] * c22;
    s -= uth[3991] * c23;
    s -= uth[3992] * c24;
    s -= uth[3993] * c25;
    s -= uth[3994] * c26;
    s -= uth[3995] * c27;
    s -= uth[3996] * c28;
    s -= uth[3997] * c29;
    s -= uth[3998] * c30;
    s -= uth[3999] * c31;
    s -= uth[4000] * c32;
    s -= uth[4001] * c33;
    s -= uth[4002] * c34;
    s -= uth[4003] * c35;
    s -= uth[4004] * c36;
    s -= uth[4005] * c37;
    s -= uth[4006] * c38;
    s -= uth[4007] * c39;
    s -= uth[4008] * c40;
    s -= uth[4009] * c41;
    s -= uth[4010] * c42;
    s -= uth[4011] * c43;
    s -= uth[4012] * c44;
    s -= uth[4013] * c45;
    s -= uth[4014] * c46;
    s -= uth[4015] * c47;
    s -= uth[4016] * c48;
    s -= uth[4017] * c49;
    s -= uth[4018] * c50;
    s -= uth[4019] * c51;
    s -= uth[4020] * c52;
    s -= uth[4021] * c53;
    s -= uth[4022] * c54;
    s -= uth[4023] * c55;
    s -= uth[4024] * c56;
    s -= uth[4025] * c57;
    s -= uth[4026] * c58;
    s -= uth[4027] * c59;
    s -= uth[4028] * c60;
    s -= uth[4029] * c61;
    c62 = s; nvh[(long)62 * GDN_DV + col] = s; }
  { float s = c63;
    s -= uth[4032] * c0;
    s -= uth[4033] * c1;
    s -= uth[4034] * c2;
    s -= uth[4035] * c3;
    s -= uth[4036] * c4;
    s -= uth[4037] * c5;
    s -= uth[4038] * c6;
    s -= uth[4039] * c7;
    s -= uth[4040] * c8;
    s -= uth[4041] * c9;
    s -= uth[4042] * c10;
    s -= uth[4043] * c11;
    s -= uth[4044] * c12;
    s -= uth[4045] * c13;
    s -= uth[4046] * c14;
    s -= uth[4047] * c15;
    s -= uth[4048] * c16;
    s -= uth[4049] * c17;
    s -= uth[4050] * c18;
    s -= uth[4051] * c19;
    s -= uth[4052] * c20;
    s -= uth[4053] * c21;
    s -= uth[4054] * c22;
    s -= uth[4055] * c23;
    s -= uth[4056] * c24;
    s -= uth[4057] * c25;
    s -= uth[4058] * c26;
    s -= uth[4059] * c27;
    s -= uth[4060] * c28;
    s -= uth[4061] * c29;
    s -= uth[4062] * c30;
    s -= uth[4063] * c31;
    s -= uth[4064] * c32;
    s -= uth[4065] * c33;
    s -= uth[4066] * c34;
    s -= uth[4067] * c35;
    s -= uth[4068] * c36;
    s -= uth[4069] * c37;
    s -= uth[4070] * c38;
    s -= uth[4071] * c39;
    s -= uth[4072] * c40;
    s -= uth[4073] * c41;
    s -= uth[4074] * c42;
    s -= uth[4075] * c43;
    s -= uth[4076] * c44;
    s -= uth[4077] * c45;
    s -= uth[4078] * c46;
    s -= uth[4079] * c47;
    s -= uth[4080] * c48;
    s -= uth[4081] * c49;
    s -= uth[4082] * c50;
    s -= uth[4083] * c51;
    s -= uth[4084] * c52;
    s -= uth[4085] * c53;
    s -= uth[4086] * c54;
    s -= uth[4087] * c55;
    s -= uth[4088] * c56;
    s -= uth[4089] * c57;
    s -= uth[4090] * c58;
    s -= uth[4091] * c59;
    s -= uth[4092] * c60;
    s -= uth[4093] * c61;
    s -= uth[4094] * c62;
    c63 = s; nvh[(long)63 * GDN_DV + col] = s; }
  {
    int t = t0 + 0;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[0];
    float ecd = eh[CS + 0];
    c0 = (Kv * B) * ec;
    qdh[(long)0 * GDN_DK + col] = Qv * ec;
    kdh[(long)0 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 1;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[1];
    float ecd = eh[CS + 1];
    c1 = (Kv * B) * ec;
    qdh[(long)1 * GDN_DK + col] = Qv * ec;
    kdh[(long)1 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 2;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[2];
    float ecd = eh[CS + 2];
    c2 = (Kv * B) * ec;
    qdh[(long)2 * GDN_DK + col] = Qv * ec;
    kdh[(long)2 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 3;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[3];
    float ecd = eh[CS + 3];
    c3 = (Kv * B) * ec;
    qdh[(long)3 * GDN_DK + col] = Qv * ec;
    kdh[(long)3 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 4;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[4];
    float ecd = eh[CS + 4];
    c4 = (Kv * B) * ec;
    qdh[(long)4 * GDN_DK + col] = Qv * ec;
    kdh[(long)4 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 5;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[5];
    float ecd = eh[CS + 5];
    c5 = (Kv * B) * ec;
    qdh[(long)5 * GDN_DK + col] = Qv * ec;
    kdh[(long)5 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 6;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[6];
    float ecd = eh[CS + 6];
    c6 = (Kv * B) * ec;
    qdh[(long)6 * GDN_DK + col] = Qv * ec;
    kdh[(long)6 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 7;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[7];
    float ecd = eh[CS + 7];
    c7 = (Kv * B) * ec;
    qdh[(long)7 * GDN_DK + col] = Qv * ec;
    kdh[(long)7 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 8;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[8];
    float ecd = eh[CS + 8];
    c8 = (Kv * B) * ec;
    qdh[(long)8 * GDN_DK + col] = Qv * ec;
    kdh[(long)8 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 9;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[9];
    float ecd = eh[CS + 9];
    c9 = (Kv * B) * ec;
    qdh[(long)9 * GDN_DK + col] = Qv * ec;
    kdh[(long)9 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 10;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[10];
    float ecd = eh[CS + 10];
    c10 = (Kv * B) * ec;
    qdh[(long)10 * GDN_DK + col] = Qv * ec;
    kdh[(long)10 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 11;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[11];
    float ecd = eh[CS + 11];
    c11 = (Kv * B) * ec;
    qdh[(long)11 * GDN_DK + col] = Qv * ec;
    kdh[(long)11 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 12;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[12];
    float ecd = eh[CS + 12];
    c12 = (Kv * B) * ec;
    qdh[(long)12 * GDN_DK + col] = Qv * ec;
    kdh[(long)12 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 13;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[13];
    float ecd = eh[CS + 13];
    c13 = (Kv * B) * ec;
    qdh[(long)13 * GDN_DK + col] = Qv * ec;
    kdh[(long)13 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 14;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[14];
    float ecd = eh[CS + 14];
    c14 = (Kv * B) * ec;
    qdh[(long)14 * GDN_DK + col] = Qv * ec;
    kdh[(long)14 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 15;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[15];
    float ecd = eh[CS + 15];
    c15 = (Kv * B) * ec;
    qdh[(long)15 * GDN_DK + col] = Qv * ec;
    kdh[(long)15 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 16;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[16];
    float ecd = eh[CS + 16];
    c16 = (Kv * B) * ec;
    qdh[(long)16 * GDN_DK + col] = Qv * ec;
    kdh[(long)16 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 17;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[17];
    float ecd = eh[CS + 17];
    c17 = (Kv * B) * ec;
    qdh[(long)17 * GDN_DK + col] = Qv * ec;
    kdh[(long)17 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 18;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[18];
    float ecd = eh[CS + 18];
    c18 = (Kv * B) * ec;
    qdh[(long)18 * GDN_DK + col] = Qv * ec;
    kdh[(long)18 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 19;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[19];
    float ecd = eh[CS + 19];
    c19 = (Kv * B) * ec;
    qdh[(long)19 * GDN_DK + col] = Qv * ec;
    kdh[(long)19 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 20;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[20];
    float ecd = eh[CS + 20];
    c20 = (Kv * B) * ec;
    qdh[(long)20 * GDN_DK + col] = Qv * ec;
    kdh[(long)20 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 21;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[21];
    float ecd = eh[CS + 21];
    c21 = (Kv * B) * ec;
    qdh[(long)21 * GDN_DK + col] = Qv * ec;
    kdh[(long)21 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 22;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[22];
    float ecd = eh[CS + 22];
    c22 = (Kv * B) * ec;
    qdh[(long)22 * GDN_DK + col] = Qv * ec;
    kdh[(long)22 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 23;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[23];
    float ecd = eh[CS + 23];
    c23 = (Kv * B) * ec;
    qdh[(long)23 * GDN_DK + col] = Qv * ec;
    kdh[(long)23 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 24;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[24];
    float ecd = eh[CS + 24];
    c24 = (Kv * B) * ec;
    qdh[(long)24 * GDN_DK + col] = Qv * ec;
    kdh[(long)24 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 25;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[25];
    float ecd = eh[CS + 25];
    c25 = (Kv * B) * ec;
    qdh[(long)25 * GDN_DK + col] = Qv * ec;
    kdh[(long)25 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 26;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[26];
    float ecd = eh[CS + 26];
    c26 = (Kv * B) * ec;
    qdh[(long)26 * GDN_DK + col] = Qv * ec;
    kdh[(long)26 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 27;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[27];
    float ecd = eh[CS + 27];
    c27 = (Kv * B) * ec;
    qdh[(long)27 * GDN_DK + col] = Qv * ec;
    kdh[(long)27 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 28;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[28];
    float ecd = eh[CS + 28];
    c28 = (Kv * B) * ec;
    qdh[(long)28 * GDN_DK + col] = Qv * ec;
    kdh[(long)28 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 29;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[29];
    float ecd = eh[CS + 29];
    c29 = (Kv * B) * ec;
    qdh[(long)29 * GDN_DK + col] = Qv * ec;
    kdh[(long)29 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 30;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[30];
    float ecd = eh[CS + 30];
    c30 = (Kv * B) * ec;
    qdh[(long)30 * GDN_DK + col] = Qv * ec;
    kdh[(long)30 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 31;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[31];
    float ecd = eh[CS + 31];
    c31 = (Kv * B) * ec;
    qdh[(long)31 * GDN_DK + col] = Qv * ec;
    kdh[(long)31 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 32;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[32];
    float ecd = eh[CS + 32];
    c32 = (Kv * B) * ec;
    qdh[(long)32 * GDN_DK + col] = Qv * ec;
    kdh[(long)32 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 33;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[33];
    float ecd = eh[CS + 33];
    c33 = (Kv * B) * ec;
    qdh[(long)33 * GDN_DK + col] = Qv * ec;
    kdh[(long)33 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 34;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[34];
    float ecd = eh[CS + 34];
    c34 = (Kv * B) * ec;
    qdh[(long)34 * GDN_DK + col] = Qv * ec;
    kdh[(long)34 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 35;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[35];
    float ecd = eh[CS + 35];
    c35 = (Kv * B) * ec;
    qdh[(long)35 * GDN_DK + col] = Qv * ec;
    kdh[(long)35 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 36;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[36];
    float ecd = eh[CS + 36];
    c36 = (Kv * B) * ec;
    qdh[(long)36 * GDN_DK + col] = Qv * ec;
    kdh[(long)36 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 37;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[37];
    float ecd = eh[CS + 37];
    c37 = (Kv * B) * ec;
    qdh[(long)37 * GDN_DK + col] = Qv * ec;
    kdh[(long)37 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 38;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[38];
    float ecd = eh[CS + 38];
    c38 = (Kv * B) * ec;
    qdh[(long)38 * GDN_DK + col] = Qv * ec;
    kdh[(long)38 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 39;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[39];
    float ecd = eh[CS + 39];
    c39 = (Kv * B) * ec;
    qdh[(long)39 * GDN_DK + col] = Qv * ec;
    kdh[(long)39 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 40;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[40];
    float ecd = eh[CS + 40];
    c40 = (Kv * B) * ec;
    qdh[(long)40 * GDN_DK + col] = Qv * ec;
    kdh[(long)40 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 41;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[41];
    float ecd = eh[CS + 41];
    c41 = (Kv * B) * ec;
    qdh[(long)41 * GDN_DK + col] = Qv * ec;
    kdh[(long)41 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 42;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[42];
    float ecd = eh[CS + 42];
    c42 = (Kv * B) * ec;
    qdh[(long)42 * GDN_DK + col] = Qv * ec;
    kdh[(long)42 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 43;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[43];
    float ecd = eh[CS + 43];
    c43 = (Kv * B) * ec;
    qdh[(long)43 * GDN_DK + col] = Qv * ec;
    kdh[(long)43 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 44;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[44];
    float ecd = eh[CS + 44];
    c44 = (Kv * B) * ec;
    qdh[(long)44 * GDN_DK + col] = Qv * ec;
    kdh[(long)44 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 45;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[45];
    float ecd = eh[CS + 45];
    c45 = (Kv * B) * ec;
    qdh[(long)45 * GDN_DK + col] = Qv * ec;
    kdh[(long)45 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 46;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[46];
    float ecd = eh[CS + 46];
    c46 = (Kv * B) * ec;
    qdh[(long)46 * GDN_DK + col] = Qv * ec;
    kdh[(long)46 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 47;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[47];
    float ecd = eh[CS + 47];
    c47 = (Kv * B) * ec;
    qdh[(long)47 * GDN_DK + col] = Qv * ec;
    kdh[(long)47 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 48;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[48];
    float ecd = eh[CS + 48];
    c48 = (Kv * B) * ec;
    qdh[(long)48 * GDN_DK + col] = Qv * ec;
    kdh[(long)48 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 49;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[49];
    float ecd = eh[CS + 49];
    c49 = (Kv * B) * ec;
    qdh[(long)49 * GDN_DK + col] = Qv * ec;
    kdh[(long)49 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 50;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[50];
    float ecd = eh[CS + 50];
    c50 = (Kv * B) * ec;
    qdh[(long)50 * GDN_DK + col] = Qv * ec;
    kdh[(long)50 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 51;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[51];
    float ecd = eh[CS + 51];
    c51 = (Kv * B) * ec;
    qdh[(long)51 * GDN_DK + col] = Qv * ec;
    kdh[(long)51 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 52;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[52];
    float ecd = eh[CS + 52];
    c52 = (Kv * B) * ec;
    qdh[(long)52 * GDN_DK + col] = Qv * ec;
    kdh[(long)52 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 53;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[53];
    float ecd = eh[CS + 53];
    c53 = (Kv * B) * ec;
    qdh[(long)53 * GDN_DK + col] = Qv * ec;
    kdh[(long)53 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 54;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[54];
    float ecd = eh[CS + 54];
    c54 = (Kv * B) * ec;
    qdh[(long)54 * GDN_DK + col] = Qv * ec;
    kdh[(long)54 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 55;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[55];
    float ecd = eh[CS + 55];
    c55 = (Kv * B) * ec;
    qdh[(long)55 * GDN_DK + col] = Qv * ec;
    kdh[(long)55 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 56;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[56];
    float ecd = eh[CS + 56];
    c56 = (Kv * B) * ec;
    qdh[(long)56 * GDN_DK + col] = Qv * ec;
    kdh[(long)56 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 57;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[57];
    float ecd = eh[CS + 57];
    c57 = (Kv * B) * ec;
    qdh[(long)57 * GDN_DK + col] = Qv * ec;
    kdh[(long)57 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 58;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[58];
    float ecd = eh[CS + 58];
    c58 = (Kv * B) * ec;
    qdh[(long)58 * GDN_DK + col] = Qv * ec;
    kdh[(long)58 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 59;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[59];
    float ecd = eh[CS + 59];
    c59 = (Kv * B) * ec;
    qdh[(long)59 * GDN_DK + col] = Qv * ec;
    kdh[(long)59 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 60;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[60];
    float ecd = eh[CS + 60];
    c60 = (Kv * B) * ec;
    qdh[(long)60 * GDN_DK + col] = Qv * ec;
    kdh[(long)60 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 61;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[61];
    float ecd = eh[CS + 61];
    c61 = (Kv * B) * ec;
    qdh[(long)61 * GDN_DK + col] = Qv * ec;
    kdh[(long)61 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 62;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[62];
    float ecd = eh[CS + 62];
    c62 = (Kv * B) * ec;
    qdh[(long)62 * GDN_DK + col] = Qv * ec;
    kdh[(long)62 * GDN_DK + col] = Kv * ecd;
  }
  {
    int t = t0 + 63;
    float B = (t < T) ? gateB[(long)t * GDN_HV + hh] : 0.f;
    float Kv = (t < T) ? Kh[(long)t * GDN_DK + col] : 0.f;
    float Qv = (t < T) ? Qh[(long)t * GDN_DK + col] : 0.f;
    float ec = eh[63];
    float ecd = eh[CS + 63];
    c63 = (Kv * B) * ec;
    qdh[(long)63 * GDN_DK + col] = Qv * ec;
    kdh[(long)63 * GDN_DK + col] = Kv * ecd;
  }
  { float s = c0;
    c0 = s; kcdh[(long)0 * GDN_DK + col] = s; }
  { float s = c1;
    s -= uth[64] * c0;
    c1 = s; kcdh[(long)1 * GDN_DK + col] = s; }
  { float s = c2;
    s -= uth[128] * c0;
    s -= uth[129] * c1;
    c2 = s; kcdh[(long)2 * GDN_DK + col] = s; }
  { float s = c3;
    s -= uth[192] * c0;
    s -= uth[193] * c1;
    s -= uth[194] * c2;
    c3 = s; kcdh[(long)3 * GDN_DK + col] = s; }
  { float s = c4;
    s -= uth[256] * c0;
    s -= uth[257] * c1;
    s -= uth[258] * c2;
    s -= uth[259] * c3;
    c4 = s; kcdh[(long)4 * GDN_DK + col] = s; }
  { float s = c5;
    s -= uth[320] * c0;
    s -= uth[321] * c1;
    s -= uth[322] * c2;
    s -= uth[323] * c3;
    s -= uth[324] * c4;
    c5 = s; kcdh[(long)5 * GDN_DK + col] = s; }
  { float s = c6;
    s -= uth[384] * c0;
    s -= uth[385] * c1;
    s -= uth[386] * c2;
    s -= uth[387] * c3;
    s -= uth[388] * c4;
    s -= uth[389] * c5;
    c6 = s; kcdh[(long)6 * GDN_DK + col] = s; }
  { float s = c7;
    s -= uth[448] * c0;
    s -= uth[449] * c1;
    s -= uth[450] * c2;
    s -= uth[451] * c3;
    s -= uth[452] * c4;
    s -= uth[453] * c5;
    s -= uth[454] * c6;
    c7 = s; kcdh[(long)7 * GDN_DK + col] = s; }
  { float s = c8;
    s -= uth[512] * c0;
    s -= uth[513] * c1;
    s -= uth[514] * c2;
    s -= uth[515] * c3;
    s -= uth[516] * c4;
    s -= uth[517] * c5;
    s -= uth[518] * c6;
    s -= uth[519] * c7;
    c8 = s; kcdh[(long)8 * GDN_DK + col] = s; }
  { float s = c9;
    s -= uth[576] * c0;
    s -= uth[577] * c1;
    s -= uth[578] * c2;
    s -= uth[579] * c3;
    s -= uth[580] * c4;
    s -= uth[581] * c5;
    s -= uth[582] * c6;
    s -= uth[583] * c7;
    s -= uth[584] * c8;
    c9 = s; kcdh[(long)9 * GDN_DK + col] = s; }
  { float s = c10;
    s -= uth[640] * c0;
    s -= uth[641] * c1;
    s -= uth[642] * c2;
    s -= uth[643] * c3;
    s -= uth[644] * c4;
    s -= uth[645] * c5;
    s -= uth[646] * c6;
    s -= uth[647] * c7;
    s -= uth[648] * c8;
    s -= uth[649] * c9;
    c10 = s; kcdh[(long)10 * GDN_DK + col] = s; }
  { float s = c11;
    s -= uth[704] * c0;
    s -= uth[705] * c1;
    s -= uth[706] * c2;
    s -= uth[707] * c3;
    s -= uth[708] * c4;
    s -= uth[709] * c5;
    s -= uth[710] * c6;
    s -= uth[711] * c7;
    s -= uth[712] * c8;
    s -= uth[713] * c9;
    s -= uth[714] * c10;
    c11 = s; kcdh[(long)11 * GDN_DK + col] = s; }
  { float s = c12;
    s -= uth[768] * c0;
    s -= uth[769] * c1;
    s -= uth[770] * c2;
    s -= uth[771] * c3;
    s -= uth[772] * c4;
    s -= uth[773] * c5;
    s -= uth[774] * c6;
    s -= uth[775] * c7;
    s -= uth[776] * c8;
    s -= uth[777] * c9;
    s -= uth[778] * c10;
    s -= uth[779] * c11;
    c12 = s; kcdh[(long)12 * GDN_DK + col] = s; }
  { float s = c13;
    s -= uth[832] * c0;
    s -= uth[833] * c1;
    s -= uth[834] * c2;
    s -= uth[835] * c3;
    s -= uth[836] * c4;
    s -= uth[837] * c5;
    s -= uth[838] * c6;
    s -= uth[839] * c7;
    s -= uth[840] * c8;
    s -= uth[841] * c9;
    s -= uth[842] * c10;
    s -= uth[843] * c11;
    s -= uth[844] * c12;
    c13 = s; kcdh[(long)13 * GDN_DK + col] = s; }
  { float s = c14;
    s -= uth[896] * c0;
    s -= uth[897] * c1;
    s -= uth[898] * c2;
    s -= uth[899] * c3;
    s -= uth[900] * c4;
    s -= uth[901] * c5;
    s -= uth[902] * c6;
    s -= uth[903] * c7;
    s -= uth[904] * c8;
    s -= uth[905] * c9;
    s -= uth[906] * c10;
    s -= uth[907] * c11;
    s -= uth[908] * c12;
    s -= uth[909] * c13;
    c14 = s; kcdh[(long)14 * GDN_DK + col] = s; }
  { float s = c15;
    s -= uth[960] * c0;
    s -= uth[961] * c1;
    s -= uth[962] * c2;
    s -= uth[963] * c3;
    s -= uth[964] * c4;
    s -= uth[965] * c5;
    s -= uth[966] * c6;
    s -= uth[967] * c7;
    s -= uth[968] * c8;
    s -= uth[969] * c9;
    s -= uth[970] * c10;
    s -= uth[971] * c11;
    s -= uth[972] * c12;
    s -= uth[973] * c13;
    s -= uth[974] * c14;
    c15 = s; kcdh[(long)15 * GDN_DK + col] = s; }
  { float s = c16;
    s -= uth[1024] * c0;
    s -= uth[1025] * c1;
    s -= uth[1026] * c2;
    s -= uth[1027] * c3;
    s -= uth[1028] * c4;
    s -= uth[1029] * c5;
    s -= uth[1030] * c6;
    s -= uth[1031] * c7;
    s -= uth[1032] * c8;
    s -= uth[1033] * c9;
    s -= uth[1034] * c10;
    s -= uth[1035] * c11;
    s -= uth[1036] * c12;
    s -= uth[1037] * c13;
    s -= uth[1038] * c14;
    s -= uth[1039] * c15;
    c16 = s; kcdh[(long)16 * GDN_DK + col] = s; }
  { float s = c17;
    s -= uth[1088] * c0;
    s -= uth[1089] * c1;
    s -= uth[1090] * c2;
    s -= uth[1091] * c3;
    s -= uth[1092] * c4;
    s -= uth[1093] * c5;
    s -= uth[1094] * c6;
    s -= uth[1095] * c7;
    s -= uth[1096] * c8;
    s -= uth[1097] * c9;
    s -= uth[1098] * c10;
    s -= uth[1099] * c11;
    s -= uth[1100] * c12;
    s -= uth[1101] * c13;
    s -= uth[1102] * c14;
    s -= uth[1103] * c15;
    s -= uth[1104] * c16;
    c17 = s; kcdh[(long)17 * GDN_DK + col] = s; }
  { float s = c18;
    s -= uth[1152] * c0;
    s -= uth[1153] * c1;
    s -= uth[1154] * c2;
    s -= uth[1155] * c3;
    s -= uth[1156] * c4;
    s -= uth[1157] * c5;
    s -= uth[1158] * c6;
    s -= uth[1159] * c7;
    s -= uth[1160] * c8;
    s -= uth[1161] * c9;
    s -= uth[1162] * c10;
    s -= uth[1163] * c11;
    s -= uth[1164] * c12;
    s -= uth[1165] * c13;
    s -= uth[1166] * c14;
    s -= uth[1167] * c15;
    s -= uth[1168] * c16;
    s -= uth[1169] * c17;
    c18 = s; kcdh[(long)18 * GDN_DK + col] = s; }
  { float s = c19;
    s -= uth[1216] * c0;
    s -= uth[1217] * c1;
    s -= uth[1218] * c2;
    s -= uth[1219] * c3;
    s -= uth[1220] * c4;
    s -= uth[1221] * c5;
    s -= uth[1222] * c6;
    s -= uth[1223] * c7;
    s -= uth[1224] * c8;
    s -= uth[1225] * c9;
    s -= uth[1226] * c10;
    s -= uth[1227] * c11;
    s -= uth[1228] * c12;
    s -= uth[1229] * c13;
    s -= uth[1230] * c14;
    s -= uth[1231] * c15;
    s -= uth[1232] * c16;
    s -= uth[1233] * c17;
    s -= uth[1234] * c18;
    c19 = s; kcdh[(long)19 * GDN_DK + col] = s; }
  { float s = c20;
    s -= uth[1280] * c0;
    s -= uth[1281] * c1;
    s -= uth[1282] * c2;
    s -= uth[1283] * c3;
    s -= uth[1284] * c4;
    s -= uth[1285] * c5;
    s -= uth[1286] * c6;
    s -= uth[1287] * c7;
    s -= uth[1288] * c8;
    s -= uth[1289] * c9;
    s -= uth[1290] * c10;
    s -= uth[1291] * c11;
    s -= uth[1292] * c12;
    s -= uth[1293] * c13;
    s -= uth[1294] * c14;
    s -= uth[1295] * c15;
    s -= uth[1296] * c16;
    s -= uth[1297] * c17;
    s -= uth[1298] * c18;
    s -= uth[1299] * c19;
    c20 = s; kcdh[(long)20 * GDN_DK + col] = s; }
  { float s = c21;
    s -= uth[1344] * c0;
    s -= uth[1345] * c1;
    s -= uth[1346] * c2;
    s -= uth[1347] * c3;
    s -= uth[1348] * c4;
    s -= uth[1349] * c5;
    s -= uth[1350] * c6;
    s -= uth[1351] * c7;
    s -= uth[1352] * c8;
    s -= uth[1353] * c9;
    s -= uth[1354] * c10;
    s -= uth[1355] * c11;
    s -= uth[1356] * c12;
    s -= uth[1357] * c13;
    s -= uth[1358] * c14;
    s -= uth[1359] * c15;
    s -= uth[1360] * c16;
    s -= uth[1361] * c17;
    s -= uth[1362] * c18;
    s -= uth[1363] * c19;
    s -= uth[1364] * c20;
    c21 = s; kcdh[(long)21 * GDN_DK + col] = s; }
  { float s = c22;
    s -= uth[1408] * c0;
    s -= uth[1409] * c1;
    s -= uth[1410] * c2;
    s -= uth[1411] * c3;
    s -= uth[1412] * c4;
    s -= uth[1413] * c5;
    s -= uth[1414] * c6;
    s -= uth[1415] * c7;
    s -= uth[1416] * c8;
    s -= uth[1417] * c9;
    s -= uth[1418] * c10;
    s -= uth[1419] * c11;
    s -= uth[1420] * c12;
    s -= uth[1421] * c13;
    s -= uth[1422] * c14;
    s -= uth[1423] * c15;
    s -= uth[1424] * c16;
    s -= uth[1425] * c17;
    s -= uth[1426] * c18;
    s -= uth[1427] * c19;
    s -= uth[1428] * c20;
    s -= uth[1429] * c21;
    c22 = s; kcdh[(long)22 * GDN_DK + col] = s; }
  { float s = c23;
    s -= uth[1472] * c0;
    s -= uth[1473] * c1;
    s -= uth[1474] * c2;
    s -= uth[1475] * c3;
    s -= uth[1476] * c4;
    s -= uth[1477] * c5;
    s -= uth[1478] * c6;
    s -= uth[1479] * c7;
    s -= uth[1480] * c8;
    s -= uth[1481] * c9;
    s -= uth[1482] * c10;
    s -= uth[1483] * c11;
    s -= uth[1484] * c12;
    s -= uth[1485] * c13;
    s -= uth[1486] * c14;
    s -= uth[1487] * c15;
    s -= uth[1488] * c16;
    s -= uth[1489] * c17;
    s -= uth[1490] * c18;
    s -= uth[1491] * c19;
    s -= uth[1492] * c20;
    s -= uth[1493] * c21;
    s -= uth[1494] * c22;
    c23 = s; kcdh[(long)23 * GDN_DK + col] = s; }
  { float s = c24;
    s -= uth[1536] * c0;
    s -= uth[1537] * c1;
    s -= uth[1538] * c2;
    s -= uth[1539] * c3;
    s -= uth[1540] * c4;
    s -= uth[1541] * c5;
    s -= uth[1542] * c6;
    s -= uth[1543] * c7;
    s -= uth[1544] * c8;
    s -= uth[1545] * c9;
    s -= uth[1546] * c10;
    s -= uth[1547] * c11;
    s -= uth[1548] * c12;
    s -= uth[1549] * c13;
    s -= uth[1550] * c14;
    s -= uth[1551] * c15;
    s -= uth[1552] * c16;
    s -= uth[1553] * c17;
    s -= uth[1554] * c18;
    s -= uth[1555] * c19;
    s -= uth[1556] * c20;
    s -= uth[1557] * c21;
    s -= uth[1558] * c22;
    s -= uth[1559] * c23;
    c24 = s; kcdh[(long)24 * GDN_DK + col] = s; }
  { float s = c25;
    s -= uth[1600] * c0;
    s -= uth[1601] * c1;
    s -= uth[1602] * c2;
    s -= uth[1603] * c3;
    s -= uth[1604] * c4;
    s -= uth[1605] * c5;
    s -= uth[1606] * c6;
    s -= uth[1607] * c7;
    s -= uth[1608] * c8;
    s -= uth[1609] * c9;
    s -= uth[1610] * c10;
    s -= uth[1611] * c11;
    s -= uth[1612] * c12;
    s -= uth[1613] * c13;
    s -= uth[1614] * c14;
    s -= uth[1615] * c15;
    s -= uth[1616] * c16;
    s -= uth[1617] * c17;
    s -= uth[1618] * c18;
    s -= uth[1619] * c19;
    s -= uth[1620] * c20;
    s -= uth[1621] * c21;
    s -= uth[1622] * c22;
    s -= uth[1623] * c23;
    s -= uth[1624] * c24;
    c25 = s; kcdh[(long)25 * GDN_DK + col] = s; }
  { float s = c26;
    s -= uth[1664] * c0;
    s -= uth[1665] * c1;
    s -= uth[1666] * c2;
    s -= uth[1667] * c3;
    s -= uth[1668] * c4;
    s -= uth[1669] * c5;
    s -= uth[1670] * c6;
    s -= uth[1671] * c7;
    s -= uth[1672] * c8;
    s -= uth[1673] * c9;
    s -= uth[1674] * c10;
    s -= uth[1675] * c11;
    s -= uth[1676] * c12;
    s -= uth[1677] * c13;
    s -= uth[1678] * c14;
    s -= uth[1679] * c15;
    s -= uth[1680] * c16;
    s -= uth[1681] * c17;
    s -= uth[1682] * c18;
    s -= uth[1683] * c19;
    s -= uth[1684] * c20;
    s -= uth[1685] * c21;
    s -= uth[1686] * c22;
    s -= uth[1687] * c23;
    s -= uth[1688] * c24;
    s -= uth[1689] * c25;
    c26 = s; kcdh[(long)26 * GDN_DK + col] = s; }
  { float s = c27;
    s -= uth[1728] * c0;
    s -= uth[1729] * c1;
    s -= uth[1730] * c2;
    s -= uth[1731] * c3;
    s -= uth[1732] * c4;
    s -= uth[1733] * c5;
    s -= uth[1734] * c6;
    s -= uth[1735] * c7;
    s -= uth[1736] * c8;
    s -= uth[1737] * c9;
    s -= uth[1738] * c10;
    s -= uth[1739] * c11;
    s -= uth[1740] * c12;
    s -= uth[1741] * c13;
    s -= uth[1742] * c14;
    s -= uth[1743] * c15;
    s -= uth[1744] * c16;
    s -= uth[1745] * c17;
    s -= uth[1746] * c18;
    s -= uth[1747] * c19;
    s -= uth[1748] * c20;
    s -= uth[1749] * c21;
    s -= uth[1750] * c22;
    s -= uth[1751] * c23;
    s -= uth[1752] * c24;
    s -= uth[1753] * c25;
    s -= uth[1754] * c26;
    c27 = s; kcdh[(long)27 * GDN_DK + col] = s; }
  { float s = c28;
    s -= uth[1792] * c0;
    s -= uth[1793] * c1;
    s -= uth[1794] * c2;
    s -= uth[1795] * c3;
    s -= uth[1796] * c4;
    s -= uth[1797] * c5;
    s -= uth[1798] * c6;
    s -= uth[1799] * c7;
    s -= uth[1800] * c8;
    s -= uth[1801] * c9;
    s -= uth[1802] * c10;
    s -= uth[1803] * c11;
    s -= uth[1804] * c12;
    s -= uth[1805] * c13;
    s -= uth[1806] * c14;
    s -= uth[1807] * c15;
    s -= uth[1808] * c16;
    s -= uth[1809] * c17;
    s -= uth[1810] * c18;
    s -= uth[1811] * c19;
    s -= uth[1812] * c20;
    s -= uth[1813] * c21;
    s -= uth[1814] * c22;
    s -= uth[1815] * c23;
    s -= uth[1816] * c24;
    s -= uth[1817] * c25;
    s -= uth[1818] * c26;
    s -= uth[1819] * c27;
    c28 = s; kcdh[(long)28 * GDN_DK + col] = s; }
  { float s = c29;
    s -= uth[1856] * c0;
    s -= uth[1857] * c1;
    s -= uth[1858] * c2;
    s -= uth[1859] * c3;
    s -= uth[1860] * c4;
    s -= uth[1861] * c5;
    s -= uth[1862] * c6;
    s -= uth[1863] * c7;
    s -= uth[1864] * c8;
    s -= uth[1865] * c9;
    s -= uth[1866] * c10;
    s -= uth[1867] * c11;
    s -= uth[1868] * c12;
    s -= uth[1869] * c13;
    s -= uth[1870] * c14;
    s -= uth[1871] * c15;
    s -= uth[1872] * c16;
    s -= uth[1873] * c17;
    s -= uth[1874] * c18;
    s -= uth[1875] * c19;
    s -= uth[1876] * c20;
    s -= uth[1877] * c21;
    s -= uth[1878] * c22;
    s -= uth[1879] * c23;
    s -= uth[1880] * c24;
    s -= uth[1881] * c25;
    s -= uth[1882] * c26;
    s -= uth[1883] * c27;
    s -= uth[1884] * c28;
    c29 = s; kcdh[(long)29 * GDN_DK + col] = s; }
  { float s = c30;
    s -= uth[1920] * c0;
    s -= uth[1921] * c1;
    s -= uth[1922] * c2;
    s -= uth[1923] * c3;
    s -= uth[1924] * c4;
    s -= uth[1925] * c5;
    s -= uth[1926] * c6;
    s -= uth[1927] * c7;
    s -= uth[1928] * c8;
    s -= uth[1929] * c9;
    s -= uth[1930] * c10;
    s -= uth[1931] * c11;
    s -= uth[1932] * c12;
    s -= uth[1933] * c13;
    s -= uth[1934] * c14;
    s -= uth[1935] * c15;
    s -= uth[1936] * c16;
    s -= uth[1937] * c17;
    s -= uth[1938] * c18;
    s -= uth[1939] * c19;
    s -= uth[1940] * c20;
    s -= uth[1941] * c21;
    s -= uth[1942] * c22;
    s -= uth[1943] * c23;
    s -= uth[1944] * c24;
    s -= uth[1945] * c25;
    s -= uth[1946] * c26;
    s -= uth[1947] * c27;
    s -= uth[1948] * c28;
    s -= uth[1949] * c29;
    c30 = s; kcdh[(long)30 * GDN_DK + col] = s; }
  { float s = c31;
    s -= uth[1984] * c0;
    s -= uth[1985] * c1;
    s -= uth[1986] * c2;
    s -= uth[1987] * c3;
    s -= uth[1988] * c4;
    s -= uth[1989] * c5;
    s -= uth[1990] * c6;
    s -= uth[1991] * c7;
    s -= uth[1992] * c8;
    s -= uth[1993] * c9;
    s -= uth[1994] * c10;
    s -= uth[1995] * c11;
    s -= uth[1996] * c12;
    s -= uth[1997] * c13;
    s -= uth[1998] * c14;
    s -= uth[1999] * c15;
    s -= uth[2000] * c16;
    s -= uth[2001] * c17;
    s -= uth[2002] * c18;
    s -= uth[2003] * c19;
    s -= uth[2004] * c20;
    s -= uth[2005] * c21;
    s -= uth[2006] * c22;
    s -= uth[2007] * c23;
    s -= uth[2008] * c24;
    s -= uth[2009] * c25;
    s -= uth[2010] * c26;
    s -= uth[2011] * c27;
    s -= uth[2012] * c28;
    s -= uth[2013] * c29;
    s -= uth[2014] * c30;
    c31 = s; kcdh[(long)31 * GDN_DK + col] = s; }
  { float s = c32;
    s -= uth[2048] * c0;
    s -= uth[2049] * c1;
    s -= uth[2050] * c2;
    s -= uth[2051] * c3;
    s -= uth[2052] * c4;
    s -= uth[2053] * c5;
    s -= uth[2054] * c6;
    s -= uth[2055] * c7;
    s -= uth[2056] * c8;
    s -= uth[2057] * c9;
    s -= uth[2058] * c10;
    s -= uth[2059] * c11;
    s -= uth[2060] * c12;
    s -= uth[2061] * c13;
    s -= uth[2062] * c14;
    s -= uth[2063] * c15;
    s -= uth[2064] * c16;
    s -= uth[2065] * c17;
    s -= uth[2066] * c18;
    s -= uth[2067] * c19;
    s -= uth[2068] * c20;
    s -= uth[2069] * c21;
    s -= uth[2070] * c22;
    s -= uth[2071] * c23;
    s -= uth[2072] * c24;
    s -= uth[2073] * c25;
    s -= uth[2074] * c26;
    s -= uth[2075] * c27;
    s -= uth[2076] * c28;
    s -= uth[2077] * c29;
    s -= uth[2078] * c30;
    s -= uth[2079] * c31;
    c32 = s; kcdh[(long)32 * GDN_DK + col] = s; }
  { float s = c33;
    s -= uth[2112] * c0;
    s -= uth[2113] * c1;
    s -= uth[2114] * c2;
    s -= uth[2115] * c3;
    s -= uth[2116] * c4;
    s -= uth[2117] * c5;
    s -= uth[2118] * c6;
    s -= uth[2119] * c7;
    s -= uth[2120] * c8;
    s -= uth[2121] * c9;
    s -= uth[2122] * c10;
    s -= uth[2123] * c11;
    s -= uth[2124] * c12;
    s -= uth[2125] * c13;
    s -= uth[2126] * c14;
    s -= uth[2127] * c15;
    s -= uth[2128] * c16;
    s -= uth[2129] * c17;
    s -= uth[2130] * c18;
    s -= uth[2131] * c19;
    s -= uth[2132] * c20;
    s -= uth[2133] * c21;
    s -= uth[2134] * c22;
    s -= uth[2135] * c23;
    s -= uth[2136] * c24;
    s -= uth[2137] * c25;
    s -= uth[2138] * c26;
    s -= uth[2139] * c27;
    s -= uth[2140] * c28;
    s -= uth[2141] * c29;
    s -= uth[2142] * c30;
    s -= uth[2143] * c31;
    s -= uth[2144] * c32;
    c33 = s; kcdh[(long)33 * GDN_DK + col] = s; }
  { float s = c34;
    s -= uth[2176] * c0;
    s -= uth[2177] * c1;
    s -= uth[2178] * c2;
    s -= uth[2179] * c3;
    s -= uth[2180] * c4;
    s -= uth[2181] * c5;
    s -= uth[2182] * c6;
    s -= uth[2183] * c7;
    s -= uth[2184] * c8;
    s -= uth[2185] * c9;
    s -= uth[2186] * c10;
    s -= uth[2187] * c11;
    s -= uth[2188] * c12;
    s -= uth[2189] * c13;
    s -= uth[2190] * c14;
    s -= uth[2191] * c15;
    s -= uth[2192] * c16;
    s -= uth[2193] * c17;
    s -= uth[2194] * c18;
    s -= uth[2195] * c19;
    s -= uth[2196] * c20;
    s -= uth[2197] * c21;
    s -= uth[2198] * c22;
    s -= uth[2199] * c23;
    s -= uth[2200] * c24;
    s -= uth[2201] * c25;
    s -= uth[2202] * c26;
    s -= uth[2203] * c27;
    s -= uth[2204] * c28;
    s -= uth[2205] * c29;
    s -= uth[2206] * c30;
    s -= uth[2207] * c31;
    s -= uth[2208] * c32;
    s -= uth[2209] * c33;
    c34 = s; kcdh[(long)34 * GDN_DK + col] = s; }
  { float s = c35;
    s -= uth[2240] * c0;
    s -= uth[2241] * c1;
    s -= uth[2242] * c2;
    s -= uth[2243] * c3;
    s -= uth[2244] * c4;
    s -= uth[2245] * c5;
    s -= uth[2246] * c6;
    s -= uth[2247] * c7;
    s -= uth[2248] * c8;
    s -= uth[2249] * c9;
    s -= uth[2250] * c10;
    s -= uth[2251] * c11;
    s -= uth[2252] * c12;
    s -= uth[2253] * c13;
    s -= uth[2254] * c14;
    s -= uth[2255] * c15;
    s -= uth[2256] * c16;
    s -= uth[2257] * c17;
    s -= uth[2258] * c18;
    s -= uth[2259] * c19;
    s -= uth[2260] * c20;
    s -= uth[2261] * c21;
    s -= uth[2262] * c22;
    s -= uth[2263] * c23;
    s -= uth[2264] * c24;
    s -= uth[2265] * c25;
    s -= uth[2266] * c26;
    s -= uth[2267] * c27;
    s -= uth[2268] * c28;
    s -= uth[2269] * c29;
    s -= uth[2270] * c30;
    s -= uth[2271] * c31;
    s -= uth[2272] * c32;
    s -= uth[2273] * c33;
    s -= uth[2274] * c34;
    c35 = s; kcdh[(long)35 * GDN_DK + col] = s; }
  { float s = c36;
    s -= uth[2304] * c0;
    s -= uth[2305] * c1;
    s -= uth[2306] * c2;
    s -= uth[2307] * c3;
    s -= uth[2308] * c4;
    s -= uth[2309] * c5;
    s -= uth[2310] * c6;
    s -= uth[2311] * c7;
    s -= uth[2312] * c8;
    s -= uth[2313] * c9;
    s -= uth[2314] * c10;
    s -= uth[2315] * c11;
    s -= uth[2316] * c12;
    s -= uth[2317] * c13;
    s -= uth[2318] * c14;
    s -= uth[2319] * c15;
    s -= uth[2320] * c16;
    s -= uth[2321] * c17;
    s -= uth[2322] * c18;
    s -= uth[2323] * c19;
    s -= uth[2324] * c20;
    s -= uth[2325] * c21;
    s -= uth[2326] * c22;
    s -= uth[2327] * c23;
    s -= uth[2328] * c24;
    s -= uth[2329] * c25;
    s -= uth[2330] * c26;
    s -= uth[2331] * c27;
    s -= uth[2332] * c28;
    s -= uth[2333] * c29;
    s -= uth[2334] * c30;
    s -= uth[2335] * c31;
    s -= uth[2336] * c32;
    s -= uth[2337] * c33;
    s -= uth[2338] * c34;
    s -= uth[2339] * c35;
    c36 = s; kcdh[(long)36 * GDN_DK + col] = s; }
  { float s = c37;
    s -= uth[2368] * c0;
    s -= uth[2369] * c1;
    s -= uth[2370] * c2;
    s -= uth[2371] * c3;
    s -= uth[2372] * c4;
    s -= uth[2373] * c5;
    s -= uth[2374] * c6;
    s -= uth[2375] * c7;
    s -= uth[2376] * c8;
    s -= uth[2377] * c9;
    s -= uth[2378] * c10;
    s -= uth[2379] * c11;
    s -= uth[2380] * c12;
    s -= uth[2381] * c13;
    s -= uth[2382] * c14;
    s -= uth[2383] * c15;
    s -= uth[2384] * c16;
    s -= uth[2385] * c17;
    s -= uth[2386] * c18;
    s -= uth[2387] * c19;
    s -= uth[2388] * c20;
    s -= uth[2389] * c21;
    s -= uth[2390] * c22;
    s -= uth[2391] * c23;
    s -= uth[2392] * c24;
    s -= uth[2393] * c25;
    s -= uth[2394] * c26;
    s -= uth[2395] * c27;
    s -= uth[2396] * c28;
    s -= uth[2397] * c29;
    s -= uth[2398] * c30;
    s -= uth[2399] * c31;
    s -= uth[2400] * c32;
    s -= uth[2401] * c33;
    s -= uth[2402] * c34;
    s -= uth[2403] * c35;
    s -= uth[2404] * c36;
    c37 = s; kcdh[(long)37 * GDN_DK + col] = s; }
  { float s = c38;
    s -= uth[2432] * c0;
    s -= uth[2433] * c1;
    s -= uth[2434] * c2;
    s -= uth[2435] * c3;
    s -= uth[2436] * c4;
    s -= uth[2437] * c5;
    s -= uth[2438] * c6;
    s -= uth[2439] * c7;
    s -= uth[2440] * c8;
    s -= uth[2441] * c9;
    s -= uth[2442] * c10;
    s -= uth[2443] * c11;
    s -= uth[2444] * c12;
    s -= uth[2445] * c13;
    s -= uth[2446] * c14;
    s -= uth[2447] * c15;
    s -= uth[2448] * c16;
    s -= uth[2449] * c17;
    s -= uth[2450] * c18;
    s -= uth[2451] * c19;
    s -= uth[2452] * c20;
    s -= uth[2453] * c21;
    s -= uth[2454] * c22;
    s -= uth[2455] * c23;
    s -= uth[2456] * c24;
    s -= uth[2457] * c25;
    s -= uth[2458] * c26;
    s -= uth[2459] * c27;
    s -= uth[2460] * c28;
    s -= uth[2461] * c29;
    s -= uth[2462] * c30;
    s -= uth[2463] * c31;
    s -= uth[2464] * c32;
    s -= uth[2465] * c33;
    s -= uth[2466] * c34;
    s -= uth[2467] * c35;
    s -= uth[2468] * c36;
    s -= uth[2469] * c37;
    c38 = s; kcdh[(long)38 * GDN_DK + col] = s; }
  { float s = c39;
    s -= uth[2496] * c0;
    s -= uth[2497] * c1;
    s -= uth[2498] * c2;
    s -= uth[2499] * c3;
    s -= uth[2500] * c4;
    s -= uth[2501] * c5;
    s -= uth[2502] * c6;
    s -= uth[2503] * c7;
    s -= uth[2504] * c8;
    s -= uth[2505] * c9;
    s -= uth[2506] * c10;
    s -= uth[2507] * c11;
    s -= uth[2508] * c12;
    s -= uth[2509] * c13;
    s -= uth[2510] * c14;
    s -= uth[2511] * c15;
    s -= uth[2512] * c16;
    s -= uth[2513] * c17;
    s -= uth[2514] * c18;
    s -= uth[2515] * c19;
    s -= uth[2516] * c20;
    s -= uth[2517] * c21;
    s -= uth[2518] * c22;
    s -= uth[2519] * c23;
    s -= uth[2520] * c24;
    s -= uth[2521] * c25;
    s -= uth[2522] * c26;
    s -= uth[2523] * c27;
    s -= uth[2524] * c28;
    s -= uth[2525] * c29;
    s -= uth[2526] * c30;
    s -= uth[2527] * c31;
    s -= uth[2528] * c32;
    s -= uth[2529] * c33;
    s -= uth[2530] * c34;
    s -= uth[2531] * c35;
    s -= uth[2532] * c36;
    s -= uth[2533] * c37;
    s -= uth[2534] * c38;
    c39 = s; kcdh[(long)39 * GDN_DK + col] = s; }
  { float s = c40;
    s -= uth[2560] * c0;
    s -= uth[2561] * c1;
    s -= uth[2562] * c2;
    s -= uth[2563] * c3;
    s -= uth[2564] * c4;
    s -= uth[2565] * c5;
    s -= uth[2566] * c6;
    s -= uth[2567] * c7;
    s -= uth[2568] * c8;
    s -= uth[2569] * c9;
    s -= uth[2570] * c10;
    s -= uth[2571] * c11;
    s -= uth[2572] * c12;
    s -= uth[2573] * c13;
    s -= uth[2574] * c14;
    s -= uth[2575] * c15;
    s -= uth[2576] * c16;
    s -= uth[2577] * c17;
    s -= uth[2578] * c18;
    s -= uth[2579] * c19;
    s -= uth[2580] * c20;
    s -= uth[2581] * c21;
    s -= uth[2582] * c22;
    s -= uth[2583] * c23;
    s -= uth[2584] * c24;
    s -= uth[2585] * c25;
    s -= uth[2586] * c26;
    s -= uth[2587] * c27;
    s -= uth[2588] * c28;
    s -= uth[2589] * c29;
    s -= uth[2590] * c30;
    s -= uth[2591] * c31;
    s -= uth[2592] * c32;
    s -= uth[2593] * c33;
    s -= uth[2594] * c34;
    s -= uth[2595] * c35;
    s -= uth[2596] * c36;
    s -= uth[2597] * c37;
    s -= uth[2598] * c38;
    s -= uth[2599] * c39;
    c40 = s; kcdh[(long)40 * GDN_DK + col] = s; }
  { float s = c41;
    s -= uth[2624] * c0;
    s -= uth[2625] * c1;
    s -= uth[2626] * c2;
    s -= uth[2627] * c3;
    s -= uth[2628] * c4;
    s -= uth[2629] * c5;
    s -= uth[2630] * c6;
    s -= uth[2631] * c7;
    s -= uth[2632] * c8;
    s -= uth[2633] * c9;
    s -= uth[2634] * c10;
    s -= uth[2635] * c11;
    s -= uth[2636] * c12;
    s -= uth[2637] * c13;
    s -= uth[2638] * c14;
    s -= uth[2639] * c15;
    s -= uth[2640] * c16;
    s -= uth[2641] * c17;
    s -= uth[2642] * c18;
    s -= uth[2643] * c19;
    s -= uth[2644] * c20;
    s -= uth[2645] * c21;
    s -= uth[2646] * c22;
    s -= uth[2647] * c23;
    s -= uth[2648] * c24;
    s -= uth[2649] * c25;
    s -= uth[2650] * c26;
    s -= uth[2651] * c27;
    s -= uth[2652] * c28;
    s -= uth[2653] * c29;
    s -= uth[2654] * c30;
    s -= uth[2655] * c31;
    s -= uth[2656] * c32;
    s -= uth[2657] * c33;
    s -= uth[2658] * c34;
    s -= uth[2659] * c35;
    s -= uth[2660] * c36;
    s -= uth[2661] * c37;
    s -= uth[2662] * c38;
    s -= uth[2663] * c39;
    s -= uth[2664] * c40;
    c41 = s; kcdh[(long)41 * GDN_DK + col] = s; }
  { float s = c42;
    s -= uth[2688] * c0;
    s -= uth[2689] * c1;
    s -= uth[2690] * c2;
    s -= uth[2691] * c3;
    s -= uth[2692] * c4;
    s -= uth[2693] * c5;
    s -= uth[2694] * c6;
    s -= uth[2695] * c7;
    s -= uth[2696] * c8;
    s -= uth[2697] * c9;
    s -= uth[2698] * c10;
    s -= uth[2699] * c11;
    s -= uth[2700] * c12;
    s -= uth[2701] * c13;
    s -= uth[2702] * c14;
    s -= uth[2703] * c15;
    s -= uth[2704] * c16;
    s -= uth[2705] * c17;
    s -= uth[2706] * c18;
    s -= uth[2707] * c19;
    s -= uth[2708] * c20;
    s -= uth[2709] * c21;
    s -= uth[2710] * c22;
    s -= uth[2711] * c23;
    s -= uth[2712] * c24;
    s -= uth[2713] * c25;
    s -= uth[2714] * c26;
    s -= uth[2715] * c27;
    s -= uth[2716] * c28;
    s -= uth[2717] * c29;
    s -= uth[2718] * c30;
    s -= uth[2719] * c31;
    s -= uth[2720] * c32;
    s -= uth[2721] * c33;
    s -= uth[2722] * c34;
    s -= uth[2723] * c35;
    s -= uth[2724] * c36;
    s -= uth[2725] * c37;
    s -= uth[2726] * c38;
    s -= uth[2727] * c39;
    s -= uth[2728] * c40;
    s -= uth[2729] * c41;
    c42 = s; kcdh[(long)42 * GDN_DK + col] = s; }
  { float s = c43;
    s -= uth[2752] * c0;
    s -= uth[2753] * c1;
    s -= uth[2754] * c2;
    s -= uth[2755] * c3;
    s -= uth[2756] * c4;
    s -= uth[2757] * c5;
    s -= uth[2758] * c6;
    s -= uth[2759] * c7;
    s -= uth[2760] * c8;
    s -= uth[2761] * c9;
    s -= uth[2762] * c10;
    s -= uth[2763] * c11;
    s -= uth[2764] * c12;
    s -= uth[2765] * c13;
    s -= uth[2766] * c14;
    s -= uth[2767] * c15;
    s -= uth[2768] * c16;
    s -= uth[2769] * c17;
    s -= uth[2770] * c18;
    s -= uth[2771] * c19;
    s -= uth[2772] * c20;
    s -= uth[2773] * c21;
    s -= uth[2774] * c22;
    s -= uth[2775] * c23;
    s -= uth[2776] * c24;
    s -= uth[2777] * c25;
    s -= uth[2778] * c26;
    s -= uth[2779] * c27;
    s -= uth[2780] * c28;
    s -= uth[2781] * c29;
    s -= uth[2782] * c30;
    s -= uth[2783] * c31;
    s -= uth[2784] * c32;
    s -= uth[2785] * c33;
    s -= uth[2786] * c34;
    s -= uth[2787] * c35;
    s -= uth[2788] * c36;
    s -= uth[2789] * c37;
    s -= uth[2790] * c38;
    s -= uth[2791] * c39;
    s -= uth[2792] * c40;
    s -= uth[2793] * c41;
    s -= uth[2794] * c42;
    c43 = s; kcdh[(long)43 * GDN_DK + col] = s; }
  { float s = c44;
    s -= uth[2816] * c0;
    s -= uth[2817] * c1;
    s -= uth[2818] * c2;
    s -= uth[2819] * c3;
    s -= uth[2820] * c4;
    s -= uth[2821] * c5;
    s -= uth[2822] * c6;
    s -= uth[2823] * c7;
    s -= uth[2824] * c8;
    s -= uth[2825] * c9;
    s -= uth[2826] * c10;
    s -= uth[2827] * c11;
    s -= uth[2828] * c12;
    s -= uth[2829] * c13;
    s -= uth[2830] * c14;
    s -= uth[2831] * c15;
    s -= uth[2832] * c16;
    s -= uth[2833] * c17;
    s -= uth[2834] * c18;
    s -= uth[2835] * c19;
    s -= uth[2836] * c20;
    s -= uth[2837] * c21;
    s -= uth[2838] * c22;
    s -= uth[2839] * c23;
    s -= uth[2840] * c24;
    s -= uth[2841] * c25;
    s -= uth[2842] * c26;
    s -= uth[2843] * c27;
    s -= uth[2844] * c28;
    s -= uth[2845] * c29;
    s -= uth[2846] * c30;
    s -= uth[2847] * c31;
    s -= uth[2848] * c32;
    s -= uth[2849] * c33;
    s -= uth[2850] * c34;
    s -= uth[2851] * c35;
    s -= uth[2852] * c36;
    s -= uth[2853] * c37;
    s -= uth[2854] * c38;
    s -= uth[2855] * c39;
    s -= uth[2856] * c40;
    s -= uth[2857] * c41;
    s -= uth[2858] * c42;
    s -= uth[2859] * c43;
    c44 = s; kcdh[(long)44 * GDN_DK + col] = s; }
  { float s = c45;
    s -= uth[2880] * c0;
    s -= uth[2881] * c1;
    s -= uth[2882] * c2;
    s -= uth[2883] * c3;
    s -= uth[2884] * c4;
    s -= uth[2885] * c5;
    s -= uth[2886] * c6;
    s -= uth[2887] * c7;
    s -= uth[2888] * c8;
    s -= uth[2889] * c9;
    s -= uth[2890] * c10;
    s -= uth[2891] * c11;
    s -= uth[2892] * c12;
    s -= uth[2893] * c13;
    s -= uth[2894] * c14;
    s -= uth[2895] * c15;
    s -= uth[2896] * c16;
    s -= uth[2897] * c17;
    s -= uth[2898] * c18;
    s -= uth[2899] * c19;
    s -= uth[2900] * c20;
    s -= uth[2901] * c21;
    s -= uth[2902] * c22;
    s -= uth[2903] * c23;
    s -= uth[2904] * c24;
    s -= uth[2905] * c25;
    s -= uth[2906] * c26;
    s -= uth[2907] * c27;
    s -= uth[2908] * c28;
    s -= uth[2909] * c29;
    s -= uth[2910] * c30;
    s -= uth[2911] * c31;
    s -= uth[2912] * c32;
    s -= uth[2913] * c33;
    s -= uth[2914] * c34;
    s -= uth[2915] * c35;
    s -= uth[2916] * c36;
    s -= uth[2917] * c37;
    s -= uth[2918] * c38;
    s -= uth[2919] * c39;
    s -= uth[2920] * c40;
    s -= uth[2921] * c41;
    s -= uth[2922] * c42;
    s -= uth[2923] * c43;
    s -= uth[2924] * c44;
    c45 = s; kcdh[(long)45 * GDN_DK + col] = s; }
  { float s = c46;
    s -= uth[2944] * c0;
    s -= uth[2945] * c1;
    s -= uth[2946] * c2;
    s -= uth[2947] * c3;
    s -= uth[2948] * c4;
    s -= uth[2949] * c5;
    s -= uth[2950] * c6;
    s -= uth[2951] * c7;
    s -= uth[2952] * c8;
    s -= uth[2953] * c9;
    s -= uth[2954] * c10;
    s -= uth[2955] * c11;
    s -= uth[2956] * c12;
    s -= uth[2957] * c13;
    s -= uth[2958] * c14;
    s -= uth[2959] * c15;
    s -= uth[2960] * c16;
    s -= uth[2961] * c17;
    s -= uth[2962] * c18;
    s -= uth[2963] * c19;
    s -= uth[2964] * c20;
    s -= uth[2965] * c21;
    s -= uth[2966] * c22;
    s -= uth[2967] * c23;
    s -= uth[2968] * c24;
    s -= uth[2969] * c25;
    s -= uth[2970] * c26;
    s -= uth[2971] * c27;
    s -= uth[2972] * c28;
    s -= uth[2973] * c29;
    s -= uth[2974] * c30;
    s -= uth[2975] * c31;
    s -= uth[2976] * c32;
    s -= uth[2977] * c33;
    s -= uth[2978] * c34;
    s -= uth[2979] * c35;
    s -= uth[2980] * c36;
    s -= uth[2981] * c37;
    s -= uth[2982] * c38;
    s -= uth[2983] * c39;
    s -= uth[2984] * c40;
    s -= uth[2985] * c41;
    s -= uth[2986] * c42;
    s -= uth[2987] * c43;
    s -= uth[2988] * c44;
    s -= uth[2989] * c45;
    c46 = s; kcdh[(long)46 * GDN_DK + col] = s; }
  { float s = c47;
    s -= uth[3008] * c0;
    s -= uth[3009] * c1;
    s -= uth[3010] * c2;
    s -= uth[3011] * c3;
    s -= uth[3012] * c4;
    s -= uth[3013] * c5;
    s -= uth[3014] * c6;
    s -= uth[3015] * c7;
    s -= uth[3016] * c8;
    s -= uth[3017] * c9;
    s -= uth[3018] * c10;
    s -= uth[3019] * c11;
    s -= uth[3020] * c12;
    s -= uth[3021] * c13;
    s -= uth[3022] * c14;
    s -= uth[3023] * c15;
    s -= uth[3024] * c16;
    s -= uth[3025] * c17;
    s -= uth[3026] * c18;
    s -= uth[3027] * c19;
    s -= uth[3028] * c20;
    s -= uth[3029] * c21;
    s -= uth[3030] * c22;
    s -= uth[3031] * c23;
    s -= uth[3032] * c24;
    s -= uth[3033] * c25;
    s -= uth[3034] * c26;
    s -= uth[3035] * c27;
    s -= uth[3036] * c28;
    s -= uth[3037] * c29;
    s -= uth[3038] * c30;
    s -= uth[3039] * c31;
    s -= uth[3040] * c32;
    s -= uth[3041] * c33;
    s -= uth[3042] * c34;
    s -= uth[3043] * c35;
    s -= uth[3044] * c36;
    s -= uth[3045] * c37;
    s -= uth[3046] * c38;
    s -= uth[3047] * c39;
    s -= uth[3048] * c40;
    s -= uth[3049] * c41;
    s -= uth[3050] * c42;
    s -= uth[3051] * c43;
    s -= uth[3052] * c44;
    s -= uth[3053] * c45;
    s -= uth[3054] * c46;
    c47 = s; kcdh[(long)47 * GDN_DK + col] = s; }
  { float s = c48;
    s -= uth[3072] * c0;
    s -= uth[3073] * c1;
    s -= uth[3074] * c2;
    s -= uth[3075] * c3;
    s -= uth[3076] * c4;
    s -= uth[3077] * c5;
    s -= uth[3078] * c6;
    s -= uth[3079] * c7;
    s -= uth[3080] * c8;
    s -= uth[3081] * c9;
    s -= uth[3082] * c10;
    s -= uth[3083] * c11;
    s -= uth[3084] * c12;
    s -= uth[3085] * c13;
    s -= uth[3086] * c14;
    s -= uth[3087] * c15;
    s -= uth[3088] * c16;
    s -= uth[3089] * c17;
    s -= uth[3090] * c18;
    s -= uth[3091] * c19;
    s -= uth[3092] * c20;
    s -= uth[3093] * c21;
    s -= uth[3094] * c22;
    s -= uth[3095] * c23;
    s -= uth[3096] * c24;
    s -= uth[3097] * c25;
    s -= uth[3098] * c26;
    s -= uth[3099] * c27;
    s -= uth[3100] * c28;
    s -= uth[3101] * c29;
    s -= uth[3102] * c30;
    s -= uth[3103] * c31;
    s -= uth[3104] * c32;
    s -= uth[3105] * c33;
    s -= uth[3106] * c34;
    s -= uth[3107] * c35;
    s -= uth[3108] * c36;
    s -= uth[3109] * c37;
    s -= uth[3110] * c38;
    s -= uth[3111] * c39;
    s -= uth[3112] * c40;
    s -= uth[3113] * c41;
    s -= uth[3114] * c42;
    s -= uth[3115] * c43;
    s -= uth[3116] * c44;
    s -= uth[3117] * c45;
    s -= uth[3118] * c46;
    s -= uth[3119] * c47;
    c48 = s; kcdh[(long)48 * GDN_DK + col] = s; }
  { float s = c49;
    s -= uth[3136] * c0;
    s -= uth[3137] * c1;
    s -= uth[3138] * c2;
    s -= uth[3139] * c3;
    s -= uth[3140] * c4;
    s -= uth[3141] * c5;
    s -= uth[3142] * c6;
    s -= uth[3143] * c7;
    s -= uth[3144] * c8;
    s -= uth[3145] * c9;
    s -= uth[3146] * c10;
    s -= uth[3147] * c11;
    s -= uth[3148] * c12;
    s -= uth[3149] * c13;
    s -= uth[3150] * c14;
    s -= uth[3151] * c15;
    s -= uth[3152] * c16;
    s -= uth[3153] * c17;
    s -= uth[3154] * c18;
    s -= uth[3155] * c19;
    s -= uth[3156] * c20;
    s -= uth[3157] * c21;
    s -= uth[3158] * c22;
    s -= uth[3159] * c23;
    s -= uth[3160] * c24;
    s -= uth[3161] * c25;
    s -= uth[3162] * c26;
    s -= uth[3163] * c27;
    s -= uth[3164] * c28;
    s -= uth[3165] * c29;
    s -= uth[3166] * c30;
    s -= uth[3167] * c31;
    s -= uth[3168] * c32;
    s -= uth[3169] * c33;
    s -= uth[3170] * c34;
    s -= uth[3171] * c35;
    s -= uth[3172] * c36;
    s -= uth[3173] * c37;
    s -= uth[3174] * c38;
    s -= uth[3175] * c39;
    s -= uth[3176] * c40;
    s -= uth[3177] * c41;
    s -= uth[3178] * c42;
    s -= uth[3179] * c43;
    s -= uth[3180] * c44;
    s -= uth[3181] * c45;
    s -= uth[3182] * c46;
    s -= uth[3183] * c47;
    s -= uth[3184] * c48;
    c49 = s; kcdh[(long)49 * GDN_DK + col] = s; }
  { float s = c50;
    s -= uth[3200] * c0;
    s -= uth[3201] * c1;
    s -= uth[3202] * c2;
    s -= uth[3203] * c3;
    s -= uth[3204] * c4;
    s -= uth[3205] * c5;
    s -= uth[3206] * c6;
    s -= uth[3207] * c7;
    s -= uth[3208] * c8;
    s -= uth[3209] * c9;
    s -= uth[3210] * c10;
    s -= uth[3211] * c11;
    s -= uth[3212] * c12;
    s -= uth[3213] * c13;
    s -= uth[3214] * c14;
    s -= uth[3215] * c15;
    s -= uth[3216] * c16;
    s -= uth[3217] * c17;
    s -= uth[3218] * c18;
    s -= uth[3219] * c19;
    s -= uth[3220] * c20;
    s -= uth[3221] * c21;
    s -= uth[3222] * c22;
    s -= uth[3223] * c23;
    s -= uth[3224] * c24;
    s -= uth[3225] * c25;
    s -= uth[3226] * c26;
    s -= uth[3227] * c27;
    s -= uth[3228] * c28;
    s -= uth[3229] * c29;
    s -= uth[3230] * c30;
    s -= uth[3231] * c31;
    s -= uth[3232] * c32;
    s -= uth[3233] * c33;
    s -= uth[3234] * c34;
    s -= uth[3235] * c35;
    s -= uth[3236] * c36;
    s -= uth[3237] * c37;
    s -= uth[3238] * c38;
    s -= uth[3239] * c39;
    s -= uth[3240] * c40;
    s -= uth[3241] * c41;
    s -= uth[3242] * c42;
    s -= uth[3243] * c43;
    s -= uth[3244] * c44;
    s -= uth[3245] * c45;
    s -= uth[3246] * c46;
    s -= uth[3247] * c47;
    s -= uth[3248] * c48;
    s -= uth[3249] * c49;
    c50 = s; kcdh[(long)50 * GDN_DK + col] = s; }
  { float s = c51;
    s -= uth[3264] * c0;
    s -= uth[3265] * c1;
    s -= uth[3266] * c2;
    s -= uth[3267] * c3;
    s -= uth[3268] * c4;
    s -= uth[3269] * c5;
    s -= uth[3270] * c6;
    s -= uth[3271] * c7;
    s -= uth[3272] * c8;
    s -= uth[3273] * c9;
    s -= uth[3274] * c10;
    s -= uth[3275] * c11;
    s -= uth[3276] * c12;
    s -= uth[3277] * c13;
    s -= uth[3278] * c14;
    s -= uth[3279] * c15;
    s -= uth[3280] * c16;
    s -= uth[3281] * c17;
    s -= uth[3282] * c18;
    s -= uth[3283] * c19;
    s -= uth[3284] * c20;
    s -= uth[3285] * c21;
    s -= uth[3286] * c22;
    s -= uth[3287] * c23;
    s -= uth[3288] * c24;
    s -= uth[3289] * c25;
    s -= uth[3290] * c26;
    s -= uth[3291] * c27;
    s -= uth[3292] * c28;
    s -= uth[3293] * c29;
    s -= uth[3294] * c30;
    s -= uth[3295] * c31;
    s -= uth[3296] * c32;
    s -= uth[3297] * c33;
    s -= uth[3298] * c34;
    s -= uth[3299] * c35;
    s -= uth[3300] * c36;
    s -= uth[3301] * c37;
    s -= uth[3302] * c38;
    s -= uth[3303] * c39;
    s -= uth[3304] * c40;
    s -= uth[3305] * c41;
    s -= uth[3306] * c42;
    s -= uth[3307] * c43;
    s -= uth[3308] * c44;
    s -= uth[3309] * c45;
    s -= uth[3310] * c46;
    s -= uth[3311] * c47;
    s -= uth[3312] * c48;
    s -= uth[3313] * c49;
    s -= uth[3314] * c50;
    c51 = s; kcdh[(long)51 * GDN_DK + col] = s; }
  { float s = c52;
    s -= uth[3328] * c0;
    s -= uth[3329] * c1;
    s -= uth[3330] * c2;
    s -= uth[3331] * c3;
    s -= uth[3332] * c4;
    s -= uth[3333] * c5;
    s -= uth[3334] * c6;
    s -= uth[3335] * c7;
    s -= uth[3336] * c8;
    s -= uth[3337] * c9;
    s -= uth[3338] * c10;
    s -= uth[3339] * c11;
    s -= uth[3340] * c12;
    s -= uth[3341] * c13;
    s -= uth[3342] * c14;
    s -= uth[3343] * c15;
    s -= uth[3344] * c16;
    s -= uth[3345] * c17;
    s -= uth[3346] * c18;
    s -= uth[3347] * c19;
    s -= uth[3348] * c20;
    s -= uth[3349] * c21;
    s -= uth[3350] * c22;
    s -= uth[3351] * c23;
    s -= uth[3352] * c24;
    s -= uth[3353] * c25;
    s -= uth[3354] * c26;
    s -= uth[3355] * c27;
    s -= uth[3356] * c28;
    s -= uth[3357] * c29;
    s -= uth[3358] * c30;
    s -= uth[3359] * c31;
    s -= uth[3360] * c32;
    s -= uth[3361] * c33;
    s -= uth[3362] * c34;
    s -= uth[3363] * c35;
    s -= uth[3364] * c36;
    s -= uth[3365] * c37;
    s -= uth[3366] * c38;
    s -= uth[3367] * c39;
    s -= uth[3368] * c40;
    s -= uth[3369] * c41;
    s -= uth[3370] * c42;
    s -= uth[3371] * c43;
    s -= uth[3372] * c44;
    s -= uth[3373] * c45;
    s -= uth[3374] * c46;
    s -= uth[3375] * c47;
    s -= uth[3376] * c48;
    s -= uth[3377] * c49;
    s -= uth[3378] * c50;
    s -= uth[3379] * c51;
    c52 = s; kcdh[(long)52 * GDN_DK + col] = s; }
  { float s = c53;
    s -= uth[3392] * c0;
    s -= uth[3393] * c1;
    s -= uth[3394] * c2;
    s -= uth[3395] * c3;
    s -= uth[3396] * c4;
    s -= uth[3397] * c5;
    s -= uth[3398] * c6;
    s -= uth[3399] * c7;
    s -= uth[3400] * c8;
    s -= uth[3401] * c9;
    s -= uth[3402] * c10;
    s -= uth[3403] * c11;
    s -= uth[3404] * c12;
    s -= uth[3405] * c13;
    s -= uth[3406] * c14;
    s -= uth[3407] * c15;
    s -= uth[3408] * c16;
    s -= uth[3409] * c17;
    s -= uth[3410] * c18;
    s -= uth[3411] * c19;
    s -= uth[3412] * c20;
    s -= uth[3413] * c21;
    s -= uth[3414] * c22;
    s -= uth[3415] * c23;
    s -= uth[3416] * c24;
    s -= uth[3417] * c25;
    s -= uth[3418] * c26;
    s -= uth[3419] * c27;
    s -= uth[3420] * c28;
    s -= uth[3421] * c29;
    s -= uth[3422] * c30;
    s -= uth[3423] * c31;
    s -= uth[3424] * c32;
    s -= uth[3425] * c33;
    s -= uth[3426] * c34;
    s -= uth[3427] * c35;
    s -= uth[3428] * c36;
    s -= uth[3429] * c37;
    s -= uth[3430] * c38;
    s -= uth[3431] * c39;
    s -= uth[3432] * c40;
    s -= uth[3433] * c41;
    s -= uth[3434] * c42;
    s -= uth[3435] * c43;
    s -= uth[3436] * c44;
    s -= uth[3437] * c45;
    s -= uth[3438] * c46;
    s -= uth[3439] * c47;
    s -= uth[3440] * c48;
    s -= uth[3441] * c49;
    s -= uth[3442] * c50;
    s -= uth[3443] * c51;
    s -= uth[3444] * c52;
    c53 = s; kcdh[(long)53 * GDN_DK + col] = s; }
  { float s = c54;
    s -= uth[3456] * c0;
    s -= uth[3457] * c1;
    s -= uth[3458] * c2;
    s -= uth[3459] * c3;
    s -= uth[3460] * c4;
    s -= uth[3461] * c5;
    s -= uth[3462] * c6;
    s -= uth[3463] * c7;
    s -= uth[3464] * c8;
    s -= uth[3465] * c9;
    s -= uth[3466] * c10;
    s -= uth[3467] * c11;
    s -= uth[3468] * c12;
    s -= uth[3469] * c13;
    s -= uth[3470] * c14;
    s -= uth[3471] * c15;
    s -= uth[3472] * c16;
    s -= uth[3473] * c17;
    s -= uth[3474] * c18;
    s -= uth[3475] * c19;
    s -= uth[3476] * c20;
    s -= uth[3477] * c21;
    s -= uth[3478] * c22;
    s -= uth[3479] * c23;
    s -= uth[3480] * c24;
    s -= uth[3481] * c25;
    s -= uth[3482] * c26;
    s -= uth[3483] * c27;
    s -= uth[3484] * c28;
    s -= uth[3485] * c29;
    s -= uth[3486] * c30;
    s -= uth[3487] * c31;
    s -= uth[3488] * c32;
    s -= uth[3489] * c33;
    s -= uth[3490] * c34;
    s -= uth[3491] * c35;
    s -= uth[3492] * c36;
    s -= uth[3493] * c37;
    s -= uth[3494] * c38;
    s -= uth[3495] * c39;
    s -= uth[3496] * c40;
    s -= uth[3497] * c41;
    s -= uth[3498] * c42;
    s -= uth[3499] * c43;
    s -= uth[3500] * c44;
    s -= uth[3501] * c45;
    s -= uth[3502] * c46;
    s -= uth[3503] * c47;
    s -= uth[3504] * c48;
    s -= uth[3505] * c49;
    s -= uth[3506] * c50;
    s -= uth[3507] * c51;
    s -= uth[3508] * c52;
    s -= uth[3509] * c53;
    c54 = s; kcdh[(long)54 * GDN_DK + col] = s; }
  { float s = c55;
    s -= uth[3520] * c0;
    s -= uth[3521] * c1;
    s -= uth[3522] * c2;
    s -= uth[3523] * c3;
    s -= uth[3524] * c4;
    s -= uth[3525] * c5;
    s -= uth[3526] * c6;
    s -= uth[3527] * c7;
    s -= uth[3528] * c8;
    s -= uth[3529] * c9;
    s -= uth[3530] * c10;
    s -= uth[3531] * c11;
    s -= uth[3532] * c12;
    s -= uth[3533] * c13;
    s -= uth[3534] * c14;
    s -= uth[3535] * c15;
    s -= uth[3536] * c16;
    s -= uth[3537] * c17;
    s -= uth[3538] * c18;
    s -= uth[3539] * c19;
    s -= uth[3540] * c20;
    s -= uth[3541] * c21;
    s -= uth[3542] * c22;
    s -= uth[3543] * c23;
    s -= uth[3544] * c24;
    s -= uth[3545] * c25;
    s -= uth[3546] * c26;
    s -= uth[3547] * c27;
    s -= uth[3548] * c28;
    s -= uth[3549] * c29;
    s -= uth[3550] * c30;
    s -= uth[3551] * c31;
    s -= uth[3552] * c32;
    s -= uth[3553] * c33;
    s -= uth[3554] * c34;
    s -= uth[3555] * c35;
    s -= uth[3556] * c36;
    s -= uth[3557] * c37;
    s -= uth[3558] * c38;
    s -= uth[3559] * c39;
    s -= uth[3560] * c40;
    s -= uth[3561] * c41;
    s -= uth[3562] * c42;
    s -= uth[3563] * c43;
    s -= uth[3564] * c44;
    s -= uth[3565] * c45;
    s -= uth[3566] * c46;
    s -= uth[3567] * c47;
    s -= uth[3568] * c48;
    s -= uth[3569] * c49;
    s -= uth[3570] * c50;
    s -= uth[3571] * c51;
    s -= uth[3572] * c52;
    s -= uth[3573] * c53;
    s -= uth[3574] * c54;
    c55 = s; kcdh[(long)55 * GDN_DK + col] = s; }
  { float s = c56;
    s -= uth[3584] * c0;
    s -= uth[3585] * c1;
    s -= uth[3586] * c2;
    s -= uth[3587] * c3;
    s -= uth[3588] * c4;
    s -= uth[3589] * c5;
    s -= uth[3590] * c6;
    s -= uth[3591] * c7;
    s -= uth[3592] * c8;
    s -= uth[3593] * c9;
    s -= uth[3594] * c10;
    s -= uth[3595] * c11;
    s -= uth[3596] * c12;
    s -= uth[3597] * c13;
    s -= uth[3598] * c14;
    s -= uth[3599] * c15;
    s -= uth[3600] * c16;
    s -= uth[3601] * c17;
    s -= uth[3602] * c18;
    s -= uth[3603] * c19;
    s -= uth[3604] * c20;
    s -= uth[3605] * c21;
    s -= uth[3606] * c22;
    s -= uth[3607] * c23;
    s -= uth[3608] * c24;
    s -= uth[3609] * c25;
    s -= uth[3610] * c26;
    s -= uth[3611] * c27;
    s -= uth[3612] * c28;
    s -= uth[3613] * c29;
    s -= uth[3614] * c30;
    s -= uth[3615] * c31;
    s -= uth[3616] * c32;
    s -= uth[3617] * c33;
    s -= uth[3618] * c34;
    s -= uth[3619] * c35;
    s -= uth[3620] * c36;
    s -= uth[3621] * c37;
    s -= uth[3622] * c38;
    s -= uth[3623] * c39;
    s -= uth[3624] * c40;
    s -= uth[3625] * c41;
    s -= uth[3626] * c42;
    s -= uth[3627] * c43;
    s -= uth[3628] * c44;
    s -= uth[3629] * c45;
    s -= uth[3630] * c46;
    s -= uth[3631] * c47;
    s -= uth[3632] * c48;
    s -= uth[3633] * c49;
    s -= uth[3634] * c50;
    s -= uth[3635] * c51;
    s -= uth[3636] * c52;
    s -= uth[3637] * c53;
    s -= uth[3638] * c54;
    s -= uth[3639] * c55;
    c56 = s; kcdh[(long)56 * GDN_DK + col] = s; }
  { float s = c57;
    s -= uth[3648] * c0;
    s -= uth[3649] * c1;
    s -= uth[3650] * c2;
    s -= uth[3651] * c3;
    s -= uth[3652] * c4;
    s -= uth[3653] * c5;
    s -= uth[3654] * c6;
    s -= uth[3655] * c7;
    s -= uth[3656] * c8;
    s -= uth[3657] * c9;
    s -= uth[3658] * c10;
    s -= uth[3659] * c11;
    s -= uth[3660] * c12;
    s -= uth[3661] * c13;
    s -= uth[3662] * c14;
    s -= uth[3663] * c15;
    s -= uth[3664] * c16;
    s -= uth[3665] * c17;
    s -= uth[3666] * c18;
    s -= uth[3667] * c19;
    s -= uth[3668] * c20;
    s -= uth[3669] * c21;
    s -= uth[3670] * c22;
    s -= uth[3671] * c23;
    s -= uth[3672] * c24;
    s -= uth[3673] * c25;
    s -= uth[3674] * c26;
    s -= uth[3675] * c27;
    s -= uth[3676] * c28;
    s -= uth[3677] * c29;
    s -= uth[3678] * c30;
    s -= uth[3679] * c31;
    s -= uth[3680] * c32;
    s -= uth[3681] * c33;
    s -= uth[3682] * c34;
    s -= uth[3683] * c35;
    s -= uth[3684] * c36;
    s -= uth[3685] * c37;
    s -= uth[3686] * c38;
    s -= uth[3687] * c39;
    s -= uth[3688] * c40;
    s -= uth[3689] * c41;
    s -= uth[3690] * c42;
    s -= uth[3691] * c43;
    s -= uth[3692] * c44;
    s -= uth[3693] * c45;
    s -= uth[3694] * c46;
    s -= uth[3695] * c47;
    s -= uth[3696] * c48;
    s -= uth[3697] * c49;
    s -= uth[3698] * c50;
    s -= uth[3699] * c51;
    s -= uth[3700] * c52;
    s -= uth[3701] * c53;
    s -= uth[3702] * c54;
    s -= uth[3703] * c55;
    s -= uth[3704] * c56;
    c57 = s; kcdh[(long)57 * GDN_DK + col] = s; }
  { float s = c58;
    s -= uth[3712] * c0;
    s -= uth[3713] * c1;
    s -= uth[3714] * c2;
    s -= uth[3715] * c3;
    s -= uth[3716] * c4;
    s -= uth[3717] * c5;
    s -= uth[3718] * c6;
    s -= uth[3719] * c7;
    s -= uth[3720] * c8;
    s -= uth[3721] * c9;
    s -= uth[3722] * c10;
    s -= uth[3723] * c11;
    s -= uth[3724] * c12;
    s -= uth[3725] * c13;
    s -= uth[3726] * c14;
    s -= uth[3727] * c15;
    s -= uth[3728] * c16;
    s -= uth[3729] * c17;
    s -= uth[3730] * c18;
    s -= uth[3731] * c19;
    s -= uth[3732] * c20;
    s -= uth[3733] * c21;
    s -= uth[3734] * c22;
    s -= uth[3735] * c23;
    s -= uth[3736] * c24;
    s -= uth[3737] * c25;
    s -= uth[3738] * c26;
    s -= uth[3739] * c27;
    s -= uth[3740] * c28;
    s -= uth[3741] * c29;
    s -= uth[3742] * c30;
    s -= uth[3743] * c31;
    s -= uth[3744] * c32;
    s -= uth[3745] * c33;
    s -= uth[3746] * c34;
    s -= uth[3747] * c35;
    s -= uth[3748] * c36;
    s -= uth[3749] * c37;
    s -= uth[3750] * c38;
    s -= uth[3751] * c39;
    s -= uth[3752] * c40;
    s -= uth[3753] * c41;
    s -= uth[3754] * c42;
    s -= uth[3755] * c43;
    s -= uth[3756] * c44;
    s -= uth[3757] * c45;
    s -= uth[3758] * c46;
    s -= uth[3759] * c47;
    s -= uth[3760] * c48;
    s -= uth[3761] * c49;
    s -= uth[3762] * c50;
    s -= uth[3763] * c51;
    s -= uth[3764] * c52;
    s -= uth[3765] * c53;
    s -= uth[3766] * c54;
    s -= uth[3767] * c55;
    s -= uth[3768] * c56;
    s -= uth[3769] * c57;
    c58 = s; kcdh[(long)58 * GDN_DK + col] = s; }
  { float s = c59;
    s -= uth[3776] * c0;
    s -= uth[3777] * c1;
    s -= uth[3778] * c2;
    s -= uth[3779] * c3;
    s -= uth[3780] * c4;
    s -= uth[3781] * c5;
    s -= uth[3782] * c6;
    s -= uth[3783] * c7;
    s -= uth[3784] * c8;
    s -= uth[3785] * c9;
    s -= uth[3786] * c10;
    s -= uth[3787] * c11;
    s -= uth[3788] * c12;
    s -= uth[3789] * c13;
    s -= uth[3790] * c14;
    s -= uth[3791] * c15;
    s -= uth[3792] * c16;
    s -= uth[3793] * c17;
    s -= uth[3794] * c18;
    s -= uth[3795] * c19;
    s -= uth[3796] * c20;
    s -= uth[3797] * c21;
    s -= uth[3798] * c22;
    s -= uth[3799] * c23;
    s -= uth[3800] * c24;
    s -= uth[3801] * c25;
    s -= uth[3802] * c26;
    s -= uth[3803] * c27;
    s -= uth[3804] * c28;
    s -= uth[3805] * c29;
    s -= uth[3806] * c30;
    s -= uth[3807] * c31;
    s -= uth[3808] * c32;
    s -= uth[3809] * c33;
    s -= uth[3810] * c34;
    s -= uth[3811] * c35;
    s -= uth[3812] * c36;
    s -= uth[3813] * c37;
    s -= uth[3814] * c38;
    s -= uth[3815] * c39;
    s -= uth[3816] * c40;
    s -= uth[3817] * c41;
    s -= uth[3818] * c42;
    s -= uth[3819] * c43;
    s -= uth[3820] * c44;
    s -= uth[3821] * c45;
    s -= uth[3822] * c46;
    s -= uth[3823] * c47;
    s -= uth[3824] * c48;
    s -= uth[3825] * c49;
    s -= uth[3826] * c50;
    s -= uth[3827] * c51;
    s -= uth[3828] * c52;
    s -= uth[3829] * c53;
    s -= uth[3830] * c54;
    s -= uth[3831] * c55;
    s -= uth[3832] * c56;
    s -= uth[3833] * c57;
    s -= uth[3834] * c58;
    c59 = s; kcdh[(long)59 * GDN_DK + col] = s; }
  { float s = c60;
    s -= uth[3840] * c0;
    s -= uth[3841] * c1;
    s -= uth[3842] * c2;
    s -= uth[3843] * c3;
    s -= uth[3844] * c4;
    s -= uth[3845] * c5;
    s -= uth[3846] * c6;
    s -= uth[3847] * c7;
    s -= uth[3848] * c8;
    s -= uth[3849] * c9;
    s -= uth[3850] * c10;
    s -= uth[3851] * c11;
    s -= uth[3852] * c12;
    s -= uth[3853] * c13;
    s -= uth[3854] * c14;
    s -= uth[3855] * c15;
    s -= uth[3856] * c16;
    s -= uth[3857] * c17;
    s -= uth[3858] * c18;
    s -= uth[3859] * c19;
    s -= uth[3860] * c20;
    s -= uth[3861] * c21;
    s -= uth[3862] * c22;
    s -= uth[3863] * c23;
    s -= uth[3864] * c24;
    s -= uth[3865] * c25;
    s -= uth[3866] * c26;
    s -= uth[3867] * c27;
    s -= uth[3868] * c28;
    s -= uth[3869] * c29;
    s -= uth[3870] * c30;
    s -= uth[3871] * c31;
    s -= uth[3872] * c32;
    s -= uth[3873] * c33;
    s -= uth[3874] * c34;
    s -= uth[3875] * c35;
    s -= uth[3876] * c36;
    s -= uth[3877] * c37;
    s -= uth[3878] * c38;
    s -= uth[3879] * c39;
    s -= uth[3880] * c40;
    s -= uth[3881] * c41;
    s -= uth[3882] * c42;
    s -= uth[3883] * c43;
    s -= uth[3884] * c44;
    s -= uth[3885] * c45;
    s -= uth[3886] * c46;
    s -= uth[3887] * c47;
    s -= uth[3888] * c48;
    s -= uth[3889] * c49;
    s -= uth[3890] * c50;
    s -= uth[3891] * c51;
    s -= uth[3892] * c52;
    s -= uth[3893] * c53;
    s -= uth[3894] * c54;
    s -= uth[3895] * c55;
    s -= uth[3896] * c56;
    s -= uth[3897] * c57;
    s -= uth[3898] * c58;
    s -= uth[3899] * c59;
    c60 = s; kcdh[(long)60 * GDN_DK + col] = s; }
  { float s = c61;
    s -= uth[3904] * c0;
    s -= uth[3905] * c1;
    s -= uth[3906] * c2;
    s -= uth[3907] * c3;
    s -= uth[3908] * c4;
    s -= uth[3909] * c5;
    s -= uth[3910] * c6;
    s -= uth[3911] * c7;
    s -= uth[3912] * c8;
    s -= uth[3913] * c9;
    s -= uth[3914] * c10;
    s -= uth[3915] * c11;
    s -= uth[3916] * c12;
    s -= uth[3917] * c13;
    s -= uth[3918] * c14;
    s -= uth[3919] * c15;
    s -= uth[3920] * c16;
    s -= uth[3921] * c17;
    s -= uth[3922] * c18;
    s -= uth[3923] * c19;
    s -= uth[3924] * c20;
    s -= uth[3925] * c21;
    s -= uth[3926] * c22;
    s -= uth[3927] * c23;
    s -= uth[3928] * c24;
    s -= uth[3929] * c25;
    s -= uth[3930] * c26;
    s -= uth[3931] * c27;
    s -= uth[3932] * c28;
    s -= uth[3933] * c29;
    s -= uth[3934] * c30;
    s -= uth[3935] * c31;
    s -= uth[3936] * c32;
    s -= uth[3937] * c33;
    s -= uth[3938] * c34;
    s -= uth[3939] * c35;
    s -= uth[3940] * c36;
    s -= uth[3941] * c37;
    s -= uth[3942] * c38;
    s -= uth[3943] * c39;
    s -= uth[3944] * c40;
    s -= uth[3945] * c41;
    s -= uth[3946] * c42;
    s -= uth[3947] * c43;
    s -= uth[3948] * c44;
    s -= uth[3949] * c45;
    s -= uth[3950] * c46;
    s -= uth[3951] * c47;
    s -= uth[3952] * c48;
    s -= uth[3953] * c49;
    s -= uth[3954] * c50;
    s -= uth[3955] * c51;
    s -= uth[3956] * c52;
    s -= uth[3957] * c53;
    s -= uth[3958] * c54;
    s -= uth[3959] * c55;
    s -= uth[3960] * c56;
    s -= uth[3961] * c57;
    s -= uth[3962] * c58;
    s -= uth[3963] * c59;
    s -= uth[3964] * c60;
    c61 = s; kcdh[(long)61 * GDN_DK + col] = s; }
  { float s = c62;
    s -= uth[3968] * c0;
    s -= uth[3969] * c1;
    s -= uth[3970] * c2;
    s -= uth[3971] * c3;
    s -= uth[3972] * c4;
    s -= uth[3973] * c5;
    s -= uth[3974] * c6;
    s -= uth[3975] * c7;
    s -= uth[3976] * c8;
    s -= uth[3977] * c9;
    s -= uth[3978] * c10;
    s -= uth[3979] * c11;
    s -= uth[3980] * c12;
    s -= uth[3981] * c13;
    s -= uth[3982] * c14;
    s -= uth[3983] * c15;
    s -= uth[3984] * c16;
    s -= uth[3985] * c17;
    s -= uth[3986] * c18;
    s -= uth[3987] * c19;
    s -= uth[3988] * c20;
    s -= uth[3989] * c21;
    s -= uth[3990] * c22;
    s -= uth[3991] * c23;
    s -= uth[3992] * c24;
    s -= uth[3993] * c25;
    s -= uth[3994] * c26;
    s -= uth[3995] * c27;
    s -= uth[3996] * c28;
    s -= uth[3997] * c29;
    s -= uth[3998] * c30;
    s -= uth[3999] * c31;
    s -= uth[4000] * c32;
    s -= uth[4001] * c33;
    s -= uth[4002] * c34;
    s -= uth[4003] * c35;
    s -= uth[4004] * c36;
    s -= uth[4005] * c37;
    s -= uth[4006] * c38;
    s -= uth[4007] * c39;
    s -= uth[4008] * c40;
    s -= uth[4009] * c41;
    s -= uth[4010] * c42;
    s -= uth[4011] * c43;
    s -= uth[4012] * c44;
    s -= uth[4013] * c45;
    s -= uth[4014] * c46;
    s -= uth[4015] * c47;
    s -= uth[4016] * c48;
    s -= uth[4017] * c49;
    s -= uth[4018] * c50;
    s -= uth[4019] * c51;
    s -= uth[4020] * c52;
    s -= uth[4021] * c53;
    s -= uth[4022] * c54;
    s -= uth[4023] * c55;
    s -= uth[4024] * c56;
    s -= uth[4025] * c57;
    s -= uth[4026] * c58;
    s -= uth[4027] * c59;
    s -= uth[4028] * c60;
    s -= uth[4029] * c61;
    c62 = s; kcdh[(long)62 * GDN_DK + col] = s; }
  { float s = c63;
    s -= uth[4032] * c0;
    s -= uth[4033] * c1;
    s -= uth[4034] * c2;
    s -= uth[4035] * c3;
    s -= uth[4036] * c4;
    s -= uth[4037] * c5;
    s -= uth[4038] * c6;
    s -= uth[4039] * c7;
    s -= uth[4040] * c8;
    s -= uth[4041] * c9;
    s -= uth[4042] * c10;
    s -= uth[4043] * c11;
    s -= uth[4044] * c12;
    s -= uth[4045] * c13;
    s -= uth[4046] * c14;
    s -= uth[4047] * c15;
    s -= uth[4048] * c16;
    s -= uth[4049] * c17;
    s -= uth[4050] * c18;
    s -= uth[4051] * c19;
    s -= uth[4052] * c20;
    s -= uth[4053] * c21;
    s -= uth[4054] * c22;
    s -= uth[4055] * c23;
    s -= uth[4056] * c24;
    s -= uth[4057] * c25;
    s -= uth[4058] * c26;
    s -= uth[4059] * c27;
    s -= uth[4060] * c28;
    s -= uth[4061] * c29;
    s -= uth[4062] * c30;
    s -= uth[4063] * c31;
    s -= uth[4064] * c32;
    s -= uth[4065] * c33;
    s -= uth[4066] * c34;
    s -= uth[4067] * c35;
    s -= uth[4068] * c36;
    s -= uth[4069] * c37;
    s -= uth[4070] * c38;
    s -= uth[4071] * c39;
    s -= uth[4072] * c40;
    s -= uth[4073] * c41;
    s -= uth[4074] * c42;
    s -= uth[4075] * c43;
    s -= uth[4076] * c44;
    s -= uth[4077] * c45;
    s -= uth[4078] * c46;
    s -= uth[4079] * c47;
    s -= uth[4080] * c48;
    s -= uth[4081] * c49;
    s -= uth[4082] * c50;
    s -= uth[4083] * c51;
    s -= uth[4084] * c52;
    s -= uth[4085] * c53;
    s -= uth[4086] * c54;
    s -= uth[4087] * c55;
    s -= uth[4088] * c56;
    s -= uth[4089] * c57;
    s -= uth[4090] * c58;
    s -= uth[4091] * c59;
    s -= uth[4092] * c60;
    s -= uth[4093] * c61;
    s -= uth[4094] * c62;
    c63 = s; kcdh[(long)63 * GDN_DK + col] = s; }
}

__global__ void k_gdn_gemv(const float* __restrict__ kcd, const float* __restrict__ qd,
                           const float* __restrict__ nv, const float* __restrict__ S,
                           float* __restrict__ vn, float* __restrict__ s2, int chunk, int n) {
  int id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (id >= n) return;
  int j = id % GDN_DV;
  int a = (id / GDN_DV) % GDN_CS;
  int hh = id / (GDN_DV * GDN_CS);
  long slot = ((long)chunk * GDN_HV + hh) * GDN_CS + a;
  const float* krow = kcd + slot * GDN_DK;
  const float* qrow = qd + slot * GDN_DK;
  const float* Sh = S + ((long)hh * GDN_DK) * GDN_DV;
  float s1 = 0.f, s2v = 0.f;
  for (int i = 0; i < GDN_DK; i++) {
    float Si = Sh[(long)i * GDN_DV + j];
    s1 += krow[i] * Si;
    s2v += qrow[i] * Si;
  }
  long o = slot * GDN_DV + j;
  vn[o] = nv[o] - s1;
  s2[o] = s2v;
}

// One thread per (head, row, column). A warp shares the row, so the b loop
// is uniform and vn loads of consecutive j coalesce. No LDS: the old staging
// buffer was 33 KB and limited the kernel to one block per processor.
__global__ void k_gdn_out(const float* __restrict__ vn, const float* __restrict__ s2,
                          const float* __restrict__ intra, uint16_t* __restrict__ out,
                          int chunk, int T, int n) {
  int id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (id >= n) return;
  int j = id % GDN_DV;
  int a = (id / GDN_DV) % GDN_CS;
  int hh = id / (GDN_DV * GDN_CS);
  int t = chunk * GDN_CS + a;
  long base = ((long)chunk * GDN_HV + hh) * GDN_CS;
  const float* wrow = intra + (base + a) * GDN_CS;
  const float* vnh = vn + base * GDN_DV;
  float inter = s2[(base + a) * GDN_DV + j];
  for (int b = 0; b <= a; b++) {
    float w = wrow[b];
    if (w == 0.f) continue;
    inter += w * vnh[(long)b * GDN_DV + j];
  }
  if (t < T) out[(long)t * (GDN_HV * GDN_DV) + hh * GDN_DV + j] = gdn_f32bf_host(inter);
}

// One warp owns one state row and four columns per lane. kd is broadcast,
// vn arrives as an aligned float4, and the four FMAs are independent.
__global__ void k_gdn_state(float* __restrict__ S, const float* __restrict__ kd,
                            const float* __restrict__ vn, const float* __restrict__ exps,
                            int chunk, int T) {
  const int i = (int)blockIdx.x;
  const int hh = (int)blockIdx.y;
  const int lane = (int)threadIdx.x;
  const int j = lane * 4;
  const int CS = GDN_CS;
  const int t0 = chunk * CS;
  float decay = exps[((long)chunk * GDN_HV + hh) * (2 * CS + 1) + 2 * CS];
  long sbase = ((long)hh * GDN_DK + i) * GDN_DV + j;
  float s0 = S[sbase] * decay;
  float s1 = S[sbase + 1] * decay;
  float s2v = S[sbase + 2] * decay;
  float s3 = S[sbase + 3] * decay;
  long base = ((long)chunk * GDN_HV + hh) * CS;
  for (int a = 0; a < CS; a++) {
    float kk = kd[(base + a) * GDN_DK + i];
    const float* vp = vn + (base + a) * GDN_DV + j;
    s0 += kk * vp[0];
    s1 += kk * vp[1];
    s2v += kk * vp[2];
    s3 += kk * vp[3];
  }
  S[sbase] = s0;
  S[sbase + 1] = s1;
  S[sbase + 2] = s2v;
  S[sbase + 3] = s3;
}



// One workgroup owns 32 columns of a head's recurrent state for the entire
// scan. All reductions retain the host's i/b/a order. No inter-workgroup
// synchronization is needed: output columns and state columns are disjoint.
// This replaces three launches per chunk with one launch for all chunks.
template<int NC>
__global__ void __launch_bounds__(NC * 32)
k_gdn_scan_persistent(const float* __restrict__ kcd, const float* __restrict__ qd,
                      const float* __restrict__ nv, const float* __restrict__ kd,
                      const float* __restrict__ intra, const float* __restrict__ exps,
                      float* __restrict__ S, uint16_t* __restrict__ out, int T) {
  constexpr int NT = NC * 32, ROWS_PER_THREAD = GDN_CS * NC / NT;
  const int tid = threadIdx.x, hh = blockIdx.x, col0 = blockIdx.y * NC;
  const int j = tid % NC;
  __shared__ float st[GDN_DK][NC];
  __shared__ float vn[GDN_CS][NC];
  for (int e = tid; e < GDN_DK * NC; e += NT) {
    int i = e / NC, col = e % NC;
    st[i][col] = S[((long)hh * GDN_DK + i) * GDN_DV + col0 + col];
  }
  __syncthreads();
  const int nch = (T + GDN_CS - 1) / GDN_CS;
  for (int c = 0; c < nch; c++) {
    const long base = ((long)c * GDN_HV + hh) * GDN_CS;
    float inter[ROWS_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
      const int a = tid / NC + r * (NT / NC);
      const float* kr = kcd + (base + a) * GDN_DK;
      const float* qr = qd + (base + a) * GDN_DK;
      float s1 = 0.f, s2 = 0.f;
      for (int i = 0; i < GDN_DK; i++) {
        const float si = st[i][j];
        s1 += kr[i] * si;
        s2 += qr[i] * si;
      }
      vn[a][j] = nv[(base + a) * GDN_DV + col0 + j] - s1;
      inter[r] = s2;
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
      const int a = tid / NC + r * (NT / NC);
      const float* wr = intra + (base + a) * GDN_CS;
      float value = inter[r];
      for (int b = 0; b <= a; b++) {
        const float w = wr[b];
        if (w != 0.f) value += w * vn[b][j];
      }
      const int t = c * GDN_CS + a;
      if (t < T) out[(long)t * GDN_HV * GDN_DV + hh * GDN_DV + col0 + j] = gdn_f32bf_host(value);
    }
    const float decay = exps[((long)c * GDN_HV + hh) * (2 * GDN_CS + 1) + 2 * GDN_CS];
    for (int e = tid; e < GDN_DK * NC; e += NT) {
      const int i = e / NC;
      float value = st[i][j] * decay;
      for (int a = 0; a < GDN_CS; a++)
        value += kd[(base + a) * GDN_DK + i] * vn[a][j];
      st[i][j] = value;
    }
    // State is complete and all readers of vn finished before the next chunk.
    __syncthreads();
  }
  for (int e = tid; e < GDN_DK * NC; e += NT) {
    const int i = e / NC;
    S[((long)hh * GDN_DK + i) * GDN_DV + col0 + j] = st[i][j];
  }
}

static float* g_blob = nullptr;

// Single-token step (T == 1): one launch per layer instead of prep, grams,
// solve and one chunk of gemv/out/state over a zero-padded 64-row chunk.
// Every expression is the chunk path's expression for row a == 0 with the
// padded rows contributing nothing: cum[0] = 0 + g, cum[63] - cum[0] = 0,
// nv = V*B (no substitution terms), kcd = (K*B)*exp(cum), qd = Q*exp(cum),
// kd = K*exp(0), W[0][0] = (q.k)*exp(0), then the same i-ordered gemv sums,
// the same out expression and the same state update. Bit-exact with the
// chunk path by construction; bench/gdn_gpu_check.hip's T=1 case is the test.
__global__ void __launch_bounds__(128)
k_gdn_step1(const uint16_t* __restrict__ qkv, const float* __restrict__ gateG,
            const float* __restrict__ gateB, const float* Sin, float* Sout,
            uint16_t* __restrict__ out, int dnprep, float qscale) {
  // Sin == Sout is the in-place serial step. A distinct Sout writes the new
  // state to another ring slot (speculative verify) with identical values:
  // thread j reads and writes only column j of each head's state.
  const int hh = (int)blockIdx.x;
  const int j = (int)threadIdx.x;               // 0..127: key dim i for prep, value column j after
  __shared__ float sq[GDN_DK], sk[GDN_DK];
  __shared__ float Qs[GDN_DK], Ks[GDN_DK], Vs[GDN_DV];
  __shared__ float sc[4];
  const uint16_t* qp = qkv + (hh / 3) * GDN_DK;
  const uint16_t* kp = qkv + GDN_HK * GDN_DK + (hh / 3) * GDN_DK;
  const uint16_t* vp = qkv + GDN_HK * GDN_DK * 2 + hh * GDN_DV;
  {
    float a = gdn_bf16f(qp[j]);
    float b = gdn_bf16f(kp[j]);
    sq[j] = a * a;
    sk[j] = b * b;
  }
  __syncthreads();
  if (j == 0) {
    float qs = 0.f, ks = 0.f;
    for (int i = 0; i < GDN_DK; i++) {
      qs += sq[i];
      ks += sk[i];
    }
    sc[0] = 1.0f / sqrtf(qs + GDN_EPS) * qscale;
    sc[1] = 1.0f / sqrtf(ks + GDN_EPS);
  }
  __syncthreads();
  {
    float qv = gdn_bf16f(qp[j]) * sc[0];
    float kv = gdn_bf16f(kp[j]) * sc[1];
    if (dnprep) {
      qv = gdn_bf16f(gdn_f32bf_host(qv));
      kv = gdn_bf16f(gdn_f32bf_host(kv));
    }
    Qs[j] = qv;
    Ks[j] = kv;
    Vs[j] = gdn_bf16f(vp[j]);
  }
  __syncthreads();
  // gates and exponentials exactly as k_gdn_grams computes them for chunk 0
  float cum0 = 0.f;
  cum0 += gateG[hh];                 // cum[0]; the 63 padded rows add 0 and leave cum[63] == cum[0]
  const float B = gateB[hh];
  const float ec = gdn_expf(cum0);                 // eh[0]
  const float ecd = gdn_expf(cum0 - cum0);         // eh[CS + 0]
  const float decay = gdn_expf(cum0);              // eh[2 * CS]
  const float decay00 = gdn_expf(cum0 - cum0);     // grams' decay for the (0,0) pair
  float qkd = 0.f;
  if (decay00 != 0.f) {
    for (int i = 0; i < GDN_DK; i++) {
      float kv = Ks[i];
      qkd += Qs[i] * kv;
    }
  }
  const float W = qkd * decay00;                   // intra[0][0]
  // solve row 0
  const float nv = Vs[j] * B;
  // gemv: i-ordered sums against the state column j
  const float* Sh = Sin + ((long)hh * GDN_DK) * GDN_DV;
  float s1 = 0.f, s2v = 0.f;
  for (int i = 0; i < GDN_DK; i++) {
    float Si = Sh[(long)i * GDN_DV + j];
    float krow = (Ks[i] * B) * ec;                 // kcd[0][i]
    float qrow = Qs[i] * ec;                       // qd[0][i]
    s1 += krow * Si;
    s2v += qrow * Si;
  }
  const float vn = nv - s1;
  // out
  float inter = s2v;
  if (W != 0.f) inter += W * vn;
  out[hh * GDN_DV + j] = gdn_f32bf_host(inter);
  // state update: S = S*decay + kd*vn, kd[0][i] = K[i]*ecd
  for (int i = 0; i < GDN_DK; i++) {
    long sbase = ((long)hh * GDN_DK + i) * GDN_DV + j;
    float s = Sh[(long)i * GDN_DV + j] * decay;
    float kk = Ks[i] * ecd;
    s += kk * vn;
    Sout[sbase] = s;
  }
}

static int g_tpad = 0;
static float *g_Q, *g_K, *g_V, *g_nv, *g_kcd, *g_qd, *g_kd, *g_vn, *g_s2, *g_ut, *g_intra, *g_exps;

static int gdn_ck(hipError_t e, const char* what) {
  if (e == hipSuccess) return 0;
  fprintf(stderr, "gdn gpu: %s: %s\n", what, hipGetErrorString(e));
  return -2;
}

static int gdn_ensure(int T) {
  if (T < 1) return -3;
  int tpad = (T + GDN_CS - 1) & ~(GDN_CS - 1);
  if (g_blob && tpad <= g_tpad) return 0;
  if (g_blob) {
    (void)hipFree(g_blob);
    g_blob = nullptr;
    g_tpad = 0;
  }
  int nch = tpad / GDN_CS;
  size_t nQ = (size_t)GDN_HV * (size_t)tpad * GDN_DK;
  size_t nC = (size_t)nch * GDN_HV * GDN_CS * GDN_DK;
  size_t nU = (size_t)nch * GDN_HV * GDN_CS * GDN_CS;
  size_t nE = (size_t)nch * GDN_HV * (2 * GDN_CS + 1);
  size_t n = nQ * 3 + nC * 6 + nU * 2 + nE;
  if (gdn_ck(hipMalloc(&g_blob, n * sizeof(float)), "alloc")) return -1;
  float* p = g_blob;
  g_Q = p; p += nQ;
  g_K = p; p += nQ;
  g_V = p; p += nQ;
  g_nv = p; p += nC;
  g_kcd = p; p += nC;
  g_qd = p; p += nC;
  g_kd = p; p += nC;
  g_vn = p; p += nC;
  g_s2 = p; p += nC;
  g_ut = p; p += nU;
  g_intra = p; p += nU;
  g_exps = p;
  g_tpad = tpad;
  fprintf(stderr, "gdn workspace: reserved Tpad=%d bytes=%zu Q/K/V_head_stride=%ld\n",
          g_tpad, n * sizeof(float), (long)g_tpad * GDN_DK);
  return 0;
}

static int gdn_gpu_impl(const uint16_t* qkv, const float* gateG, const float* gateB,
                                float* S, uint16_t* out, int T, int dnprep, float qscale, bool sync) {
  int rc = gdn_ensure(T);
  if (rc) return rc;
  static int step1 = -1;
  if (step1 < 0) { const char* e = getenv("HALO_GDN_STEP1"); step1 = (e && atoi(e) == 1) ? 1 : 0; }
  if (T == 1 && step1) {
    k_gdn_step1<<<GDN_HV, 128>>>(qkv, gateG, gateB, S, S, out, dnprep, qscale);
    if (gdn_ck(hipGetLastError(), "step1")) return -2;
    if (sync && gdn_ck(hipDeviceSynchronize(), "sync")) return -2;
    return 0;
  }
  int tpad = g_tpad;
  int nch = (T + GDN_CS - 1) / GDN_CS;
  int ngroups = T * GDN_HV;
  int nprep = (ngroups + 7) >> 3;
  k_gdn_prep<<<nprep, 256>>>(qkv, g_Q, g_K, g_V, T, tpad, dnprep, qscale);
  if (gdn_ck(hipGetLastError(), "prep")) return -2;
  dim3 grams(nch, GDN_HV);
  k_gdn_grams<<<grams, 256>>>(g_Q, g_K, gateG, gateB, g_ut, g_intra, g_exps, T, tpad);
  if (gdn_ck(hipGetLastError(), "grams")) return -2;
  dim3 solve(nch, GDN_HV, 4);
  const char* lds_solve = getenv("HALO_GDN_SOLVE_LDS");
  if (lds_solve && atoi(lds_solve) == 1)
    k_gdn_solve<true><<<dim3(nch, GDN_HV), 128>>>(g_Q, g_K, g_V, gateG, gateB, g_ut, g_nv, g_kcd, g_qd, g_kd, g_exps, T, tpad);
  else
    k_gdn_solve<false><<<solve, 32>>>(g_Q, g_K, g_V, gateG, gateB, g_ut, g_nv, g_kcd, g_qd, g_kd, g_exps, T, tpad);
  if (gdn_ck(hipGetLastError(), "solve")) return -2;
  int nvec = GDN_HV * GDN_CS * GDN_DV;
  dim3 state(GDN_DK, GDN_HV);
  const char* persistent = getenv("HALO_GDN_PERSISTENT");
  const int strip = persistent ? atoi(persistent) : 0;
  if (strip) {
    if (strip == 32)
      k_gdn_scan_persistent<32><<<dim3(GDN_HV, GDN_DV / 32), 1024>>>(g_kcd, g_qd, g_nv, g_kd, g_intra, g_exps, S, out, T);
    else if (strip == 8)
      k_gdn_scan_persistent<8><<<dim3(GDN_HV, GDN_DV / 8), 256>>>(g_kcd, g_qd, g_nv, g_kd, g_intra, g_exps, S, out, T);
    else
      k_gdn_scan_persistent<16><<<dim3(GDN_HV, GDN_DV / 16), 512>>>(g_kcd, g_qd, g_nv, g_kd, g_intra, g_exps, S, out, T);
  } else for (int c = 0; c < nch; c++) {
    k_gdn_gemv<<<(nvec + 255) / 256, 256>>>(g_kcd, g_qd, g_nv, S, g_vn, g_s2, c, nvec);
    k_gdn_out<<<(nvec + 255) / 256, 256>>>(g_vn, g_s2, g_intra, out, c, T, nvec);
    k_gdn_state<<<state, 32>>>(S, g_kd, g_vn, g_exps, c, T);
  }
  if (gdn_ck(hipGetLastError(), "scan")) return -2;
  if (sync && gdn_ck(hipDeviceSynchronize(), "sync")) return -2;
  return 0;
}

// The standalone correctness API remains synchronous. Only the engine's
// stream-ordered prefill may defer completion until the final logits fence.
extern "C" int chlorine_gdn_gpu(const uint16_t* qkv, const float* G, const float* B,
                                float* S, uint16_t* out, int T, int dnprep, float qscale) {
  return gdn_gpu_impl(qkv,G,B,S,out,T,dnprep,qscale,true);
}
extern "C" int chlorine_gdn_gpu_async(const uint16_t* qkv, const float* G, const float* B,
                                      float* S, uint16_t* out, int T, int dnprep, float qscale) {
  return gdn_gpu_impl(qkv,G,B,S,out,T,dnprep,qscale,false);
}

extern "C" int chlorine_gdn_reserve(int T) {
  return gdn_ensure(T);
}

// One token of recurrence from state Sin into Sout (may be equal). Used by the
// speculative verify pass, one call per verified row; bit-exact with the
// single-token step and therefore with the chunk path.
extern "C" int chlorine_gdn_step_io(const uint16_t* qkv, const float* G, const float* B,
                                    const float* Sin, float* Sout, uint16_t* out,
                                    int dnprep, float qscale, int sync) {
  k_gdn_step1<<<GDN_HV, 128>>>(qkv, G, B, Sin, Sout, out, dnprep, qscale);
  if (gdn_ck(hipGetLastError(), "step_io")) return -2;
  if (sync && gdn_ck(hipDeviceSynchronize(), "sync")) return -2;
  return 0;
}
