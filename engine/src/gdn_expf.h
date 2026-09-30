// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2017-2026 Free Software Foundation, Inc.
// HIP adaptation, 2026. See ../COPYING.LIB for the license.
// Derived from glibc sysdeps/ieee754/flt-32/e_expf.c and e_exp2f_data.c.
// The explicit double FMAs reproduce this boot's glibc 2.43 FMA expf.
// Other arithmetic must not contract. This is a value contract under RNE;
// device execution does not reproduce host errno or floating exceptions.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#define GDN_EXP_HD __host__ __device__
#else
#define GDN_EXP_HD
#endif
GDN_EXP_HD static inline double gdn_exp_u2d(uint64_t u) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __longlong_as_double((long long)u);
#else
  double d; memcpy(&d, &u, 8); return d;
#endif
}
GDN_EXP_HD static inline uint64_t gdn_exp_d2u(double d) {
#if defined(__HIP_DEVICE_COMPILE__)
  return (uint64_t)__double_as_longlong(d);
#else
  uint64_t u; memcpy(&u, &d, 8); return u;
#endif
}
GDN_EXP_HD static inline float gdn_expf(float x) {
#pragma clang fp contract(off)
  static const uint64_t tab[32] = { 0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL, 0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL, 0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL, 0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL, 0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL, 0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL };
  const double InvLn2N = 0x1.71547652b82fep+0 * 32;
  const double SHIFT = 0x1.8p+52;
  const double C0 = 0x1.c6af84b912394p-5 / 32 / 32 / 32, C1 = 0x1.ebfce50fac4f3p-3 / 32 / 32, C2 = 0x1.62e42ff0c52d6p-1 / 32;
  // The original scratch port truncated this region at -103 and +88.
  // Preserve NaNs, infinities, and gradual underflow in round-to-nearest.
  if (std::isnan(x)) return x + x;
  if (x == -INFINITY) return 0.0f;
  if (x > 0x1.62e42ep6f) return INFINITY;
  if (x < -0x1.9fe368p6f) return 0.0f;
  double xd = (double)x;
  double z = InvLn2N * xd;
  double kd = z + SHIFT;
  uint64_t ki = gdn_exp_d2u(kd);
  kd -= SHIFT;
  double r = z - kd;
  uint64_t t = tab[ki % 32];
  t += ki << (52 - 5);
  double s = gdn_exp_u2d(t);
  double zz = fma(C0, r, C1);
  double r2 = r * r;
  double y = fma(C2, r, 1.0);
  y = fma(zz, r2, y);
  y = y * s;
  return (float)y;
}

#undef GDN_EXP_HD
