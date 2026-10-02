// ---------------- Lightning indexer: fused score + top-512 select kernels ----
// Self-contained (HIP only, no engine globals) so tools/index_fast_test.cu can
// include it next to the production kernels it is checked against.
//
// Old (reference) kernels, moved verbatim from 10_ple_io.inc:
//   k_index_select_rs      n <= 8192, LDS row cache, 4 MSD radix passes
//   k_index_select_stream  n  > 8192, 5 global passes over the score row
//   k_index_scores_tiled   16q x 32k tile, fp32 FMA, rocBLAS K-loop start
// Fast bit-exact replacements (2026-09-28, 128K prefill work):
//   k_index_scores_t64     64q x 128k tile, 4q x 4h x 8k per thread
//   k_index_select_2p      2 global passes (8192-bin histogram + LDS candidates)

// Exact top-512 selection, bit-identical output to k_index_select (same
// combined-score formula and the same total order: score desc, then smaller
// block id), but sort-free. Four MSD radix passes over the fp32 bit pattern
// (scores are nonnegative, so bits are monotone) resolve the exact 512th
// score threshold s*; the selected set is {s > s*} plus the smallest-id
// elements of {s == s*}, which is exactly the old kernel's uint64-key order.
// Emission scans b ascending, so ids come out in the same ascending order the
// old kernel's final IdSort produced. Requires n <= 8192 (LDS combined-score
// cache); larger n keeps the old kernel. n <= 512 fills directly (same as old).
template <bool Combined = false>
__global__ void k_index_select_rs(const float* __restrict__ scores, int stride,
                                  int* __restrict__ selected, int first,
                                  const int* __restrict__ pos = nullptr) {
  __shared__ float sbuf[8192];
  __shared__ unsigned hist[256];
  __shared__ unsigned shr_cnt[8];
  __shared__ unsigned shr_eq[8];
  __shared__ int s_passcnt, s_passbyte, s_cntgt;
  const int q = blockIdx.x, t = pos ? *pos : first + q, n = (t + 1) / 4;
  const int tid = threadIdx.x;
  if (n <= 512) {
    for (int i = tid; i < 512; i += 256)
      selected[(size_t)q * 512 + i] = i < n ? i : -1;
    return;
  }
  // combine (ReLU per head, ordered sum, scale) into LDS + top-byte histogram
  hist[tid] = 0;
  __syncthreads();
  const float* qb = scores + (size_t)q * 4 * stride;
  for (int b = tid; b < n; b += 256) {
    float s = 0.f;
    if constexpr (Combined) {
      s = scores[(size_t)q * stride + b];
    } else {
#pragma unroll
      for (int h = 0; h < 4; h++) s += fmaxf(0.f, qb[(size_t)h * stride + b]);
      s *= 0.08838834764831845f;
    }
    sbuf[b] = s;
    atomicAdd(&hist[__float_as_uint(s) >> 24], 1u);
  }
  __syncthreads();
  // 4 MSD radix passes -> exact 512th-largest score bits in prefix
  unsigned prefix = 0;
  int K = 512;
  if (tid == 0) s_cntgt = 0;
  for (int p = 0; p < 4; p++) {
    const int shift = (3 - p) * 8;
    if (p) {
      __syncthreads();  // hist reuse
      hist[tid] = 0;
      __syncthreads();
      for (int b = tid; b < n; b += 256) {
        unsigned bits = __float_as_uint(sbuf[b]);
        if ((bits >> (shift + 8)) == prefix)
          atomicAdd(&hist[(bits >> shift) & 0xffu], 1u);
      }
      __syncthreads();
    }
    if (tid == 0) {
      int acc = 0, chosen = 255;
      for (int i = 255; i >= 0; i--) {
        int c = (int)hist[i];
        if (acc + c >= K) {
          chosen = i;
          break;
        }
        acc += c;
      }
      s_passcnt = acc;
      s_passbyte = chosen;
    }
    __syncthreads();
    K -= s_passcnt;
    if (tid == 0) s_cntgt += s_passcnt;
    prefix = (prefix << 8) | (unsigned)s_passbyte;
    __syncthreads();
  }
  const unsigned sstar = prefix;
  const int cnt_gt = s_cntgt;  // < 512 by construction
  const int need = 512 - cnt_gt;  // tie slots taken by smallest ids
  // Single ascending-b emit: sel = (s > s*) || (s == s* && tie-rank < need).
  // The old kernel's final IdSort yields ascending ids, so ties and strict
  // greaters must interleave in one id-ordered scan (not two phases).
  int outbase = 0, eqbase = 0;
  const int lane = tid % warpSize, w = tid / warpSize, nw = 256 / warpSize;
#pragma unroll 1
  for (int c0 = 0; c0 < n; c0 += 256) {
    const int b = c0 + tid;
    unsigned bits = b < n ? __float_as_uint(sbuf[b]) : 0u;
    bool gt = b < n && bits > sstar;
    bool eq = b < n && bits == sstar;
    unsigned long long bal_eq = __ballot_sync(~0ull, eq);
    int eq_pre = __popcll(bal_eq & (lane ? (~0ull) >> (64 - lane) : 0ull));
    if (lane == 0) shr_eq[w] = __popcll(bal_eq);
    __syncthreads();
    int eq_woff = 0;
    for (int i = 0; i < nw; i++) eq_woff += i < w ? (int)shr_eq[i] : 0;
    bool sel = gt || (eq && eqbase + eq_woff + eq_pre < need);
    unsigned long long bal_sel = __ballot_sync(~0ull, sel);
    if (lane == 0) shr_cnt[w] = __popcll(bal_sel);
    __syncthreads();
    int woff = 0, wtot = 0, eqtot = 0;
    for (int i = 0; i < nw; i++) {
      int c = (int)shr_cnt[i];
      woff += i < w ? c : 0;
      wtot += c;
      eqtot += (int)shr_eq[i];
    }
    if (sel)
      selected[(size_t)q * 512 + outbase + woff +
               __popcll(bal_sel & (lane ? (~0ull) >> (64 - lane) : 0ull))] = b;
    outbase += wtot;
    eqbase = min(eqbase + eqtot, need);
    __syncthreads();  // shr reuse
  }
}


// Stream the existing combined scores instead of storing an unbounded row in LDS.
// Four exact radix passes find the threshold, then an ascending scan emits IDs.
// The total ordering and tie handling are identical to the existing selectors.
template <bool Combined>
__device__ __forceinline__ float index_stream_score(const float* scores, int stride, int q, int b) {
  if constexpr (Combined) return scores[(size_t)q * stride + b];
  float s = 0.f;
  #pragma unroll
  for (int h = 0; h < 4; h++) s += fmaxf(0.f, scores[((size_t)q * 4 + h) * stride + b]);
  return s * 0.08838834764831845f;
}
template <bool Combined = false>
__global__ void k_index_select_stream(const float* __restrict__ scores, int stride,
                                  int* __restrict__ selected, int first,
                                  const int* __restrict__ pos = nullptr) {
  __shared__ unsigned hist[256];
  __shared__ unsigned shr_cnt[8];
  __shared__ unsigned shr_eq[8];
  __shared__ int s_passcnt, s_passbyte, s_cntgt;
  const int q = blockIdx.x, t = pos ? *pos : first + q, n = (t + 1) / 4;
  const int tid = threadIdx.x;
  if (n <= 512) {
    for (int i = tid; i < 512; i += 256)
      selected[(size_t)q * 512 + i] = i < n ? i : -1;
    return;
  }
  // First pass: combine scores and build the top-byte histogram.
  hist[tid] = 0;
  __syncthreads();
  const float* qb = scores + (size_t)q * 4 * stride;
  for (int b = tid; b < n; b += 256) {
    float s = 0.f;
    if constexpr (Combined) {
      s = scores[(size_t)q * stride + b];
    } else {
#pragma unroll
      for (int h = 0; h < 4; h++) s += fmaxf(0.f, qb[(size_t)h * stride + b]);
      s *= 0.08838834764831845f;
    }
    atomicAdd(&hist[__float_as_uint(s) >> 24], 1u);
  }
  __syncthreads();
  // 4 MSD radix passes -> exact 512th-largest score bits in prefix
  unsigned prefix = 0;
  int K = 512;
  if (tid == 0) s_cntgt = 0;
  for (int p = 0; p < 4; p++) {
    const int shift = (3 - p) * 8;
    if (p) {
      __syncthreads();  // hist reuse
      hist[tid] = 0;
      __syncthreads();
      for (int b = tid; b < n; b += 256) {
        unsigned bits = __float_as_uint(index_stream_score<Combined>(scores, stride, q, b));
        if ((bits >> (shift + 8)) == prefix)
          atomicAdd(&hist[(bits >> shift) & 0xffu], 1u);
      }
      __syncthreads();
    }
    if (tid == 0) {
      int acc = 0, chosen = 255;
      for (int i = 255; i >= 0; i--) {
        int c = (int)hist[i];
        if (acc + c >= K) {
          chosen = i;
          break;
        }
        acc += c;
      }
      s_passcnt = acc;
      s_passbyte = chosen;
    }
    __syncthreads();
    K -= s_passcnt;
    if (tid == 0) s_cntgt += s_passcnt;
    prefix = (prefix << 8) | (unsigned)s_passbyte;
    __syncthreads();
  }
  const unsigned sstar = prefix;
  const int cnt_gt = s_cntgt;  // < 512 by construction
  const int need = 512 - cnt_gt;  // tie slots taken by smallest ids
  // Single ascending-b emit: sel = (s > s*) || (s == s* && tie-rank < need).
  // The old kernel's final IdSort yields ascending ids, so ties and strict
  // greaters must interleave in one id-ordered scan (not two phases).
  int outbase = 0, eqbase = 0;
  const int lane = tid % warpSize, w = tid / warpSize, nw = 256 / warpSize;
#pragma unroll 1
  for (int c0 = 0; c0 < n; c0 += 256) {
    const int b = c0 + tid;
    unsigned bits = b < n ? __float_as_uint(index_stream_score<Combined>(scores, stride, q, b)) : 0u;
    bool gt = b < n && bits > sstar;
    bool eq = b < n && bits == sstar;
    unsigned long long bal_eq = __ballot_sync(~0ull, eq);
    int eq_pre = __popcll(bal_eq & (lane ? (~0ull) >> (64 - lane) : 0ull));
    if (lane == 0) shr_eq[w] = __popcll(bal_eq);
    __syncthreads();
    int eq_woff = 0;
    for (int i = 0; i < nw; i++) eq_woff += i < w ? (int)shr_eq[i] : 0;
    bool sel = gt || (eq && eqbase + eq_woff + eq_pre < need);
    unsigned long long bal_sel = __ballot_sync(~0ull, sel);
    if (lane == 0) shr_cnt[w] = __popcll(bal_sel);
    __syncthreads();
    int woff = 0, wtot = 0, eqtot = 0;
    for (int i = 0; i < nw; i++) {
      int c = (int)shr_cnt[i];
      woff += i < w ? c : 0;
      wtot += c;
      eqtot += (int)shr_eq[i];
    }
    if (sel)
      selected[(size_t)q * 512 + outbase + woff +
               __popcll(bal_sel & (lane ? (~0ull) >> (64 - lane) : 0ull))] = b;
    outbase += wtot;
    eqbase = min(eqbase + eqtot, need);
    __syncthreads();  // shr reuse
  }
}



// 16 queries x 32 keys: fuse FP32 heads, ReLU and combine, skipping
// causally unreachable tiles. rocBLAS gfx1151 MT32x32x8 SU32 SUS256
// starts the K loop at ((key / 32) & 1) * 64 and wraps at 128.
// Keep one FMA chain per head and the same ordered head reduction.
__global__ void __launch_bounds__(256)
k_index_scores_tiled(const float* __restrict__ q,
                     const float* __restrict__ keys,
                     float* __restrict__ out, int nb, int count, int first) {
  const int q0 = (int)blockIdx.y * 16, k0 = (int)blockIdx.x * 32;
  if (q0 >= count || k0 >= min(nb, (first + min(q0 + 16, count)) / 4)) return;
  __shared__ float qs[32][65];
  __shared__ float ks[32][33];
  const int tid = threadIdx.x, qr = tid / 16, kr = (tid % 16) * 2;
  float a[4][2] = {};
  for (int stage = 0; stage < 4; stage++) {
    const int d0 = (stage * 32 + ((int)blockIdx.x & 1) * 64) & 127;
    for (int j = tid; j < 512; j += 256) {
      int row = j / 8, d = (j % 8) * 4;
      float4 qv = make_float4(0, 0, 0, 0), kv = qv;
      if (q0 + row / 4 < count)
        qv = *reinterpret_cast<const float4*>(q + (size_t)(q0 * 4 + row) * 128 + d0 + d);
      if (row < 32 && k0 + row < nb)
        kv = *reinterpret_cast<const float4*>(keys + (size_t)(k0 + row) * 128 + d0 + d);
      qs[d][row] = qv.x; qs[d + 1][row] = qv.y;
      qs[d + 2][row] = qv.z; qs[d + 3][row] = qv.w;
      if (row < 32) {
        ks[d][row] = kv.x; ks[d + 1][row] = kv.y;
        ks[d + 2][row] = kv.z; ks[d + 3][row] = kv.w;
      }
    }
    __syncthreads();
#pragma unroll
    for (int d = 0; d < 32; d++) {
      float qv[4], kv[2];
#pragma unroll
      for (int h = 0; h < 4; h++) qv[h] = qs[d][qr * 4 + h];
#pragma unroll
      for (int c = 0; c < 2; c++) kv[c] = ks[d][kr + c];
#pragma unroll
      for (int h = 0; h < 4; h++)
#pragma unroll
        for (int c = 0; c < 2; c++) a[h][c] = fmaf(qv[h], kv[c], a[h][c]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int c = 0; c < 2; c++) {
    const int t = q0 + qr, b = k0 + kr + c;
    if (t < count && b < nb && b < (first + t + 1) / 4) {
      float s = 0.f;
#pragma unroll
      for (int h = 0; h < 4; h++) s += fmaxf(0.f, a[h][c]);
      out[(size_t)t * nb + b] = s * 0.08838834764831845f;
    }
  }
}

// ---------------------------------------------------------------------------
// k_index_scores_t64: bit-exact replacement for k_index_scores_tiled.
//
// The old kernel is LDS-bandwidth bound: 6 scalar LDS reads per 8 FMA = 3 B
// per FMA, which is exactly the ~128 B/clk/WGP LDS limit at its measured
// 4.5 TFLOPS. Here a block covers 64 queries x 128 keys and a thread
// 4 queries x 4 heads x 8 keys (128 accumulators): per d it issues 4 + 2
// float4 LDS reads for 128 FMA (0.75 B/FMA).
// All 128 keys of a block share one K-loop parity: they are the four 32-key
// groups of parity p inside a 256-key super tile,
//   blockIdx.x = 2*m + p  ->  keys 256m + 32p + 64g + [0, 32),  g = 0..3
// so every key has the same rocBLAS K-loop start ((b / 32) & 1) * 64, and
// every (query, head, key) keeps exactly the old single FMA chain
//   a = fmaf(q[d], k[d], a),  d = start, start+1, ... (mod 128),  a0 = 0
// (16-d stages here vs 32-d there: same d sequence). The epilogue is the
// same expression: s = 0; s += fmaxf(0, a_h) for h = 0..3; s * 1/sqrt(128).
// The next stage's global loads are prefetched into registers while the
// current stage computes. Output / skip rules are unchanged:
// out[t * nb + b] written only for t < count, b < nb, b < (first + t + 1) / 4.
// Tile key j (0..127) -> key S + 64 * (j >> 5) + (j & 31), S = 256m + 32p.
// grid = (2 * ceil(nb / 256), ceil(count / 64)), block = 256.
__global__ void __launch_bounds__(256)
k_index_scores_t64(const float* __restrict__ q, const float* __restrict__ keys,
                   float* __restrict__ out, int nb, int count, int first) {
  const int par = (int)blockIdx.x & 1;
  const int S = ((int)blockIdx.x >> 1) * 256 + par * 32;
  const int q0 = (int)blockIdx.y * 64;
  if (q0 >= count || S >= min(nb, (first + min(q0 + 64, count)) / 4)) return;
  __shared__ __align__(16) float qs[16][260];  // [d][query*4 + head]
  __shared__ __align__(16) float ks[16][132];  // [d][tile key]
  const int tid = threadIdx.x;
  const int qg = tid >> 4, kg = tid & 15;  // thread keys: kg*4 + c + 64*e
  // global -> LDS: q = 256 rows x 4 float4 (4 per thread),
  //                k = 128 rows x 4 float4 (2 per thread)
  const float* kp[2];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int j = (tid + 256 * i) >> 2, key = S + 64 * (j >> 5) + (j & 31);
    kp[i] = key < nb ? keys + (size_t)key * 128 + (tid & 3) * 4 : nullptr;
  }
  float4 pq[4], pk[2];
  auto load = [&](int d0) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int j = tid + 256 * i, row = j >> 2, dd = (j & 3) * 4;
      pq[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (q0 + (row >> 2) < count)
        pq[i] = *reinterpret_cast<const float4*>(q + (size_t)(q0 * 4 + row) * 128 + d0 + dd);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      pk[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (kp[i]) pk[i] = *reinterpret_cast<const float4*>(kp[i] + d0);
    }
  };
  float acc[4][4][8];  // [query][head][key]
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
      for (int c = 0; c < 8; c++) acc[a][h][c] = 0.f;
  load((par * 64) & 127);
#pragma unroll 1
  for (int s = 0; s < 8; s++) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int j = tid + 256 * i, row = j >> 2, dd = (j & 3) * 4;
      qs[dd][row] = pq[i].x; qs[dd + 1][row] = pq[i].y;
      qs[dd + 2][row] = pq[i].z; qs[dd + 3][row] = pq[i].w;
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int j = (tid + 256 * i) >> 2, dd = (tid & 3) * 4;
      ks[dd][j] = pk[i].x; ks[dd + 1][j] = pk[i].y;
      ks[dd + 2][j] = pk[i].z; ks[dd + 3][j] = pk[i].w;
    }
    __syncthreads();
    if (s < 7) load(((s + 1) * 16 + par * 64) & 127);
#pragma unroll
    for (int d = 0; d < 16; d++) {
      const float4 a0 = *reinterpret_cast<const float4*>(&qs[d][qg * 16]);
      const float4 a1 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 4]);
      const float4 a2 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 8]);
      const float4 a3 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 12]);
      const float4 k0 = *reinterpret_cast<const float4*>(&ks[d][kg * 4]);
      const float4 k1 = *reinterpret_cast<const float4*>(&ks[d][64 + kg * 4]);
      const float qv[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                            a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
      const float kk[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int h = 0; h < 4; h++)
#pragma unroll
          for (int c = 0; c < 8; c++)
            acc[a][h][c] = fmaf(qv[a * 4 + h], kk[c], acc[a][h][c]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < 4; a++) {
    const int t = q0 + qg * 4 + a;
    if (t >= count) continue;
    const int lim = min(nb, (first + t + 1) / 4);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const int j = kg * 4 + (c & 3) + 64 * (c >> 2);
      const int b = S + 64 * (j >> 5) + (j & 31);
      if (b < lim) {
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < 4; h++) s += fmaxf(0.f, acc[a][h][c]);
        out[(size_t)t * nb + b] = s * 0.08838834764831845f;
      }
    }
  }
}

static inline dim3 index_t64_grid(int nb, int count) {
  return dim3(2 * ((nb + 255) / 256), (count + 63) / 64);
}

// ---------------------------------------------------------------------------
// k_index_select_2p: exact top-512 over combined score rows, bit-identical to
// k_index_select_stream<true> / k_index_select_rs<true> / k_index_select
// (score desc, ties -> smaller block id; output = the 512 ids ascending;
// n <= 512 -> 0..n-1 then -1). Two global passes instead of five:
//   1. 8192-bin histogram of bits >> 18 (scores are >= 0, so fp32 bits are
//      monotone and bit 31 is 0) -> threshold bin T, cnt_gt = #(bin > T)
//   2. bin > T -> selected directly; bin == T -> (bits, id) candidates in LDS
//   then, all in LDS: exact s* = (512 - cnt_gt)-th largest candidate (4 radix
//   passes), ties at s* resolved by the smallest ids (radix on ~id), and a
//   512-entry bitonic sort for the ascending output.
// If the threshold bin holds more than `cap` (<= ISEL_CAP) candidates the
// block falls back to exact streaming radix passes over the remaining 18
// bits + one ascending emit scan (same algorithm as k_index_select_stream).
// grid = count, block = 256. Row q: scores[q * stride + b], b < n.
constexpr int ISEL_BINS = 8192;
constexpr int ISEL_CAP = 4096;
// Histogram bin of a combined score: bits >> 18 (< 8192 for any score >= 0,
// which the engine always produces: 0 + ReLU terms). Clamped so a NaN /
// negative input can never index outside the LDS histogram.
__device__ __forceinline__ unsigned isel_bin(unsigned bits) { return min(bits >> 18, 8191u); }

// Block-wide: with hist[0 .. 256*PER) counting items per bin (ascending
// values), find the bin T holding the K-th largest item:
//   cum(bins > T) < K <= cum(bins >= T).
// Result: sh[0] = cum(bins > T), sh[1] = T. All 256 threads must call; the
// function ends with a barrier.
template <int PER>
__device__ __forceinline__ void isel_find_desc(const unsigned* hist, int K, int* sh,
                                               unsigned* wsum) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int hi = 256 * PER - 1 - tid * PER;  // this thread: bins hi, hi-1, ... (desc)
  unsigned part = 0;
#pragma unroll
  for (int i = 0; i < PER; i++) part += hist[hi - i];
  unsigned x = part;  // inclusive scan in tid order (= descending bins)
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned y = __shfl_up(x, o, 32);
    if (lane >= o) x += y;
  }
  if (lane == 31) wsum[w] = x;
  __syncthreads();
  unsigned off = 0;
  for (int i = 0; i < w; i++) off += wsum[i];
  const unsigned incl = x + off, excl = incl - part;
  if (excl < (unsigned)K && (unsigned)K <= incl) {  // exactly one thread
    unsigned acc = excl;
    for (int i = 0; i < PER; i++) {
      const unsigned c = hist[hi - i];
      if (acc + c >= (unsigned)K) {
        sh[0] = (int)acc;
        sh[1] = hi - i;
        break;
      }
      acc += c;
    }
  }
  __syncthreads();
}

// K-th largest (1-based, with multiplicity) 32-bit key among LDS items
// i < cnt for which get(i, key) is true (K must be <= that count). Returns the
// key; gt = number of those items with a strictly greater key.
template <class Get>
__device__ __forceinline__ unsigned isel_kth(int cnt, int K, Get get, unsigned* h256,
                                             int* sh, unsigned* wsum, int& gt) {
  unsigned prefix = 0;
  gt = 0;
#pragma unroll 1
  for (int p = 0; p < 4; p++) {
    const int shift = 24 - 8 * p;
    h256[threadIdx.x] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < cnt; i += 256) {
      unsigned key;
      if (get(i, key) && (p == 0 || (key >> (shift + 8)) == prefix))
        atomicAdd(&h256[(key >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    isel_find_desc<1>(h256, K, sh, wsum);
    K -= sh[0];
    gt += sh[0];
    prefix = (prefix << 8) | (unsigned)sh[1];
    __syncthreads();  // sh / h256 reuse
  }
  return prefix;
}

__global__ void __launch_bounds__(256)
k_index_select_2p(const float* __restrict__ scores, int stride,
                  int* __restrict__ selected, int first, int cap = ISEL_CAP) {
  __shared__ unsigned big[ISEL_BINS];  // pass 1 histogram; then cand bits | cand ids
  __shared__ int sel[512];
  __shared__ unsigned h256[256];
  __shared__ unsigned wsum[8];
  __shared__ unsigned shr_cnt[8], shr_eq[8];
  __shared__ int sh[2];
  __shared__ int s_nsel, s_ncand, s_eq;
  const int q = blockIdx.x, t = first + q, n = (t + 1) / 4;
  const int tid = threadIdx.x;
  if (n <= 512) {
    for (int i = tid; i < 512; i += 256)
      selected[(size_t)q * 512 + i] = i < n ? i : -1;
    return;
  }
  cap = min(cap, ISEL_CAP);
  const float* row = scores + (size_t)q * stride;
  for (int i = tid; i < ISEL_BINS; i += 256) big[i] = 0;
  if (tid == 0) s_nsel = s_ncand = s_eq = 0;
  __syncthreads();
  // ---- pass 1: top-13-bit histogram (bit 31 is 0 -> 8192 bins)
  // 8 loads in flight per thread before the LDS atomics (latency bound)
  for (int b0 = tid; b0 < n; b0 += 256 * 8) {
    unsigned v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = b0 + 256 * k < n ? __float_as_uint(row[b0 + 256 * k]) : 0u;
#pragma unroll
    for (int k = 0; k < 8; k++)
      if (b0 + 256 * k < n) atomicAdd(&big[isel_bin(v[k])], 1u);
  }
  __syncthreads();
  isel_find_desc<ISEL_BINS / 256>(big, 512, sh, wsum);
  const int cnt_gt = sh[0];
  const unsigned T = (unsigned)sh[1];
  const int ncand = (int)big[T];
  __syncthreads();  // everyone has big[T] before the candidates overwrite big
  if (ncand > cap) {
    // ---- fallback: exact streaming radix over the remaining 18 bits
    // (bits 17..10, 9..2, 1..0), then one ascending emit scan (as
    // k_index_select_stream).
    unsigned prefix = T;
    int K = 512 - cnt_gt, cgt = cnt_gt;
    const int shifts[3] = {10, 2, 0}, widths[3] = {8, 8, 2};
#pragma unroll 1
    for (int p = 0; p < 3; p++) {
      const int shift = shifts[p], wdt = widths[p];
      h256[tid] = 0;
      __syncthreads();
      for (int b = tid; b < n; b += 256) {
        const unsigned bits = __float_as_uint(row[b]);
        if ((p == 0 ? isel_bin(bits) : bits >> (shift + wdt)) == prefix)
          atomicAdd(&h256[(bits >> shift) & ((1u << wdt) - 1u)], 1u);
      }
      __syncthreads();
      isel_find_desc<1>(h256, K, sh, wsum);
      K -= sh[0];
      cgt += sh[0];
      prefix = (prefix << wdt) | (unsigned)sh[1];
      __syncthreads();
    }
    const unsigned sstar = prefix;
    const int need = 512 - cgt;
    int outbase = 0, eqbase = 0;
    const int lane = tid % warpSize, w = tid / warpSize, nw = 256 / warpSize;
#pragma unroll 1
    for (int c0 = 0; c0 < n; c0 += 256) {
      const int b = c0 + tid;
      unsigned bits = b < n ? __float_as_uint(row[b]) : 0u;
      bool gt = b < n && bits > sstar;
      bool eq = b < n && bits == sstar;
      unsigned long long bal_eq = __ballot_sync(~0ull, eq);
      int eq_pre = __popcll(bal_eq & (lane ? (~0ull) >> (64 - lane) : 0ull));
      if (lane == 0) shr_eq[w] = __popcll(bal_eq);
      __syncthreads();
      int eq_woff = 0;
      for (int i = 0; i < nw; i++) eq_woff += i < w ? (int)shr_eq[i] : 0;
      bool pick = gt || (eq && eqbase + eq_woff + eq_pre < need);
      unsigned long long bal_sel = __ballot_sync(~0ull, pick);
      if (lane == 0) shr_cnt[w] = __popcll(bal_sel);
      __syncthreads();
      int woff = 0, wtot = 0, eqtot = 0;
      for (int i = 0; i < nw; i++) {
        int c = (int)shr_cnt[i];
        woff += i < w ? c : 0;
        wtot += c;
        eqtot += (int)shr_eq[i];
      }
      if (pick)
        selected[(size_t)q * 512 + outbase + woff +
                 __popcll(bal_sel & (lane ? (~0ull) >> (64 - lane) : 0ull))] = b;
      outbase += wtot;
      eqbase = min(eqbase + eqtot, need);
      __syncthreads();
    }
    return;
  }
  // ---- pass 2: definite ids (bin > T) + candidates (bin == T)
  unsigned* cbits = big;
  int* cid = reinterpret_cast<int*>(big + ISEL_CAP);
  for (int b0 = tid; b0 < n; b0 += 256 * 8) {
    unsigned v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = b0 + 256 * k < n ? __float_as_uint(row[b0 + 256 * k]) : 0u;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int b = b0 + 256 * k;
      const unsigned bin = isel_bin(v[k]);
      if (b >= n) continue;
      if (bin > T) {
        sel[atomicAdd(&s_nsel, 1)] = b;
      } else if (bin == T) {
        const int i = atomicAdd(&s_ncand, 1);
        cbits[i] = v[k];
        cid[i] = b;
      }
    }
  }
  __syncthreads();
  // ---- exact s* among the candidates
  int gt2;
  const unsigned sstar = isel_kth(
      ncand, 512 - cnt_gt, [&](int i, unsigned& k) { k = cbits[i]; return true; },
      h256, sh, wsum, gt2);
  const int need = 512 - cnt_gt - gt2;  // >= 1 slots for ties at s*
  for (int i = tid; i < ncand; i += 256)
    if (cbits[i] == sstar) atomicAdd(&s_eq, 1);
  __syncthreads();
  const int E = s_eq;
  unsigned idk = 0u;  // eq ids with ~id >= idk are taken (all when E == need)
  if (E > need) {
    int g;
    idk = isel_kth(
        ncand, need,
        [&](int i, unsigned& k) { k = ~(unsigned)cid[i]; return cbits[i] == sstar; },
        h256, sh, wsum, g);
  }
  for (int i = tid; i < ncand; i += 256) {
    const unsigned bits = cbits[i];
    if (bits > sstar || (bits == sstar && ~(unsigned)cid[i] >= idk))
      sel[atomicAdd(&s_nsel, 1)] = cid[i];
  }
  __syncthreads();
  // ---- ascending ids: bitonic sort of the 512 selected (distinct) ids
#pragma unroll 1
  for (int k = 2; k <= 512; k <<= 1) {
#pragma unroll 1
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int i = (tid / j) * 2 * j + (tid % j), p = i + j;
      const int a = sel[i], b = sel[p];
      if ((a > b) == ((i & k) == 0)) {
        sel[i] = b;
        sel[p] = a;
      }
      __syncthreads();
    }
  }
  selected[(size_t)q * 512 + tid] = sel[tid];
  selected[(size_t)q * 512 + tid + 256] = sel[tid + 256];
}
