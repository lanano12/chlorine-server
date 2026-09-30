// dequant_common.h -- bf16 and fp8 E4M3 helpers shared by generator.hip and the
// isolated GEMV harness (variant-27b/bench/gemv_check.hip). Moved verbatim from
// generator.hip; the numerical rules are unchanged.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

// ---------------- bf16 helpers ----------------
__device__ inline float bf16f(uint16_t h) { uint32_t u = (uint32_t)h << 16; return __uint_as_float(u); }
__device__ inline uint16_t f32bf(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lo = u & 0xFFFFu;
  u = (u >> 16) + (lo > 0x8000u) + ((lo == 0x8000u) & ((u >> 16) & 1u));
  return (uint16_t)(u & 0xFFFFu);
}
static inline float h_bf16f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t h_f32bf(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  uint32_t lo = u & 0xFFFFu;
  u = (u >> 16) + ((lo > 0x7FFFu) | ((lo == 0x7FFFu) & ((u >> 16) & 1u)));
  return (uint16_t)(u & 0xFFFFu);
}

// ---------------- dequant (from dequant_test.hip, validated) ----------------
__device__ inline float dequant_fp8_e4m3(uint32_t b) {
  float s = (b & 0x80u) ? -1.0f : 1.0f;
  uint32_t e = (b >> 3) & 0xFu, m = b & 0x7u;
  if (e == 0) return s * (float)m * 0.125f * ldexpf(1.0f, -6);
  return s * (1.0f + (float)m * 0.125f) * ldexpf(1.0f, (int)e - 7);
}

// Branch-free E4M3 decode. Produces the same float as dequant_fp8_e4m3 for every
// byte: normal values are (1 + m/8) * 2^(e-7) built directly from the bit
// pattern, subnormal values are m * 2^-9 (exact), and the sign bit is copied so
// a zero keeps its sign. No branch, no ldexpf.
__device__ inline float fp8_e4m3_nobranch(uint32_t b) {
  uint32_t s = (b & 0x80u) << 24, e = (b >> 3) & 0xFu, m = b & 0x7u;
  float normal = __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
  float sub = __uint_as_float(s | __float_as_uint((float)m * 0.001953125f));
  return e ? normal : sub;
}
