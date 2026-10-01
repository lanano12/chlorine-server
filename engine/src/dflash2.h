// dflash2.h -- DFlash2 block drafter (drafter id 2, HALO_DFLASH=1) for the
// Chlorine 27B engine. Included by generator.hip after the MTP drafter; it
// uses the MTP loaders, the trunk's embedding and lm_head planes, and the
// speculative verify pass (rings, multi-row GEMVs).
//
// Structure, from the checkpoint's drafter.config and the public z-lab
// DFlash2DraftModel (dflash/model.py): five Qwen3 layers (hidden 5120,
// 32 query / 8 key-value heads of 128, SwiGLU 17408, RMSNorm eps 1e-6,
// full 128-dim neox RoPE with theta 1e7, sliding window 2048, non-causal
// inside the block), a grouped dynamic causal convolution (kernel 2, groups of
// 16 channels) before and after each attention and MLP sublayer, a candidate
// selector (rank 256, top-16 over the draft vocabulary).
//
// One round: the block [anchor, MASK x 7] at positions P..P+7 attends to the
// drafter's context K/V (one entry per verified position p < P: the target's
// residual stream after layers 5/19/33/47/61 at p, concatenated, through fc
// and hidden_norm, then each layer's k/v projection, k_norm and RoPE(p)) and
// to all eight block rows; rows 1..7 go through the final norm, the NF4
// draft_lm_head (98,304 draft-vocabulary rows), a top-16 per row and the
// selector walk (score = logit + <pre[prev] * h_t, suc[cand]>, greedy from
// the anchor) -> seven drafts. The target verifies [anchor, d1..d7] as eight
// decode rows (the MTP verify pass), so the emitted tokens are exactly the
// serial ones; the drafter only decides how many are accepted per round. The
// accepted rows' residuals (captured during the verify pass) are ingested
// into the context K/V, the bonus token becomes the next anchor.
//
// Weights: NF4 (q4c variant 1) blobs read directly by the multi-row GEMV
// (k_gemv_dq_vec_mr<3, P>) for the block forward and the per-round ingest;
// fc and the k/v projections also keep a bf16 copy for the prompt ingest
// GEMMs (hipBLASLt over up to 4096 rows per piece).
#pragma once

static const int DF_L = 5, DF_B = 8, DF_HD = 128, DF_NQ = 32, DF_NKV = 8;
static const int DF_QW = DF_NQ * DF_HD, DF_KW = DF_NKV * DF_HD;        // 4096, 1024
static const int DF_RANK = 256, DF_TOPK = 16, DF_KP = 1280, DF_FCIN = DF_L * HID;
static const int DF_PIECE = 4096;                                       // ingest GEMM rows per piece
// g_df_capture, d_df_aux, g_df_tl and df_layer_index() are declared in
// generator.hip before forward(), which captures the residual stream.
static int g_df_ok = 0, g_df_mask = 248070, g_df_win = 2048, g_df_zc = 0;
static float g_df_theta = 1e7f;
struct DfLayer {
  MtpQ4 q, k, v, o, g, u, d, kpa, kpm;
  uint16_t *kbf = nullptr, *vbf = nullptr;
  uint16_t *ln_in = nullptr, *ln_post = nullptr, *qn = nullptr, *kn = nullptr, *base_a = nullptr, *base_m = nullptr;
};
static DfLayer g_df[DF_L];
static MtpQ4 g_df_fc, g_df_dlm;
static uint16_t *d_df_fc_bf = nullptr, *d_df_hnorm = nullptr, *d_df_norm = nullptr, *d_df_hproj = nullptr;
static uint16_t *d_df_pre = nullptr, *d_df_suc = nullptr;              // selector codebooks [VOC][256] bf16
static uint16_t *d_df_kc = nullptr, *d_df_vc = nullptr;                // context K/V [L][CXT][NKV][HD]
static uint16_t *d_df_cat = nullptr, *d_df_ht = nullptr, *d_df_kt = nullptr, *d_df_vt = nullptr, *d_df_kn2 = nullptr;
static uint16_t *d_df_x, *d_df_xn, *d_df_xc, *d_df_dyn, *d_df_q, *d_df_qn, *d_df_kb, *d_df_knb, *d_df_vb, *d_df_att, *d_df_o, *d_df_g, *d_df_u, *d_df_act, *d_df_xf, *d_df_h;
static float *d_df_logits = nullptr, *d_df_unary = nullptr;
static int *d_df_cand = nullptr, *d_df_ids = nullptr, *d_df_out = nullptr, *d_df_map = nullptr;
static double g_df_ingest_ms = 0;
static int g_df_nv = 0;                                                 // draft vocabulary size

// ---- kernels ----
// [L][T][HID] captures -> [T][L*HID] (config order)
__global__ void k_df_concat(const uint16_t* __restrict__ aux, uint16_t* __restrict__ cat, int T, int t0) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * DF_FCIN) return;
  int t = (int)(i / DF_FCIN), c = (int)(i % DF_FCIN);
  int k = c / HID, cc = c % HID;
  cat[i] = aux[((long)k * TMAX + t0 + t) * HID + cc];
}
// per-head RMSNorm with a plain gain (zc = 1: zero-centred, gain 1 + w)
__global__ void k_df_headnorm(const uint16_t* __restrict__ X, const uint16_t* __restrict__ w,
                              uint16_t* __restrict__ O, int T, int heads, int D, float eps, int zc) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * heads) return;
  const uint16_t* xp = X + i * D;
  float ss = 0.f;
  for (int d = 0; d < D; d++) { float v = bf16f(xp[d]); ss += v * v; }
  float inv = rsqrtf(ss / (float)D + eps);
  uint16_t* op = O + i * D;
  for (int d = 0; d < D; d++) {
    float gain = zc ? 1.0f + bf16f(w[d]) : bf16f(w[d]);
    op[d] = f32bf(bf16f(xp[d]) * inv * gain);
  }
}
// full 128-dim neox RoPE (pairs d, d+64), HF bf16 rounding of cos/sin and products
__global__ void k_df_rope(uint16_t* __restrict__ X, int T, int heads, int pos0, float theta) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const int half = DF_HD / 2;
  if (i >= (long)T * heads * half) return;
  int p = (int)(i % half);
  int h = (int)((i / half) % heads);
  int t = (int)(i / ((long)half * heads));
  uint16_t* xp = X + ((long)t * heads + h) * DF_HD;
  float inv = powf(theta, -(float)(2 * p) / (float)DF_HD);
  float ang = (float)(pos0 + t) * inv;
  float cf = bf16f(f32bf(cosf(ang))), sf = bf16f(f32bf(sinf(ang)));
  float a1 = bf16f(xp[p]), a2 = bf16f(xp[p + half]);
  float t1 = bf16f(f32bf(a1 * cf)), t2 = bf16f(f32bf(a2 * sf));
  float t3 = bf16f(f32bf(a2 * cf)), t4 = bf16f(f32bf(a1 * sf));
  xp[p] = f32bf(t1 - t2);
  xp[p + half] = f32bf(t3 + t4);
}
__global__ void k_df_kvwrite(const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
                             uint16_t* __restrict__ kc, uint16_t* __restrict__ vc, int T, int pos0) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * DF_KW) return;
  int t = (int)(i / DF_KW), c = (int)(i % DF_KW);
  kc[((long)pos0 + t) * DF_KW + c] = k[i];
  vc[((long)pos0 + t) * DF_KW + c] = v[i];
}
// grouped dynamic causal convolution over the block rows:
//   out[t][c] = sum_{tap<2} (base[side][tap][c] + dyn[t][side][tap][c/16]) * x[t-tap][c], x[-1] = 0
__global__ void k_df_conv(const uint16_t* __restrict__ x, const uint16_t* __restrict__ dyn,
                          const uint16_t* __restrict__ base, uint16_t* __restrict__ out, int T, int side) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * HID) return;
  int t = (int)(i / HID), c = (int)(i % HID), g = c / 16;
  float acc = 0.f;
#pragma unroll
  for (int tap = 0; tap < 2; tap++) {
    if (t - tap < 0) continue;
    float coef = bf16f(base[(side * 2 + tap) * HID + c]) + bf16f(dyn[(long)t * DF_KP + side * 640 + tap * 320 + g]);
    acc = fmaf(coef, bf16f(x[(long)(t - tap) * HID + c]), acc);
  }
  out[i] = f32bf(acc);
}
// block attention: query (head h, row j) over the context window and the 8 block rows
__global__ void __launch_bounds__(256)
k_df_attn(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kb, const uint16_t* __restrict__ vb,
          const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc, int P, int W, int T,
          uint16_t* __restrict__ out) {
  const int h = blockIdx.x, j = blockIdx.y, kvh = h / (DF_NQ / DF_NKV), tid = threadIdx.x;
  __shared__ float qf[DF_HD];
  __shared__ float sc[2048 + DF_B];
  __shared__ float red[256];
  const int lo = max(0, P + j - (W - 1)), nctx = P - lo, nk = nctx + T;
  if (tid < DF_HD) qf[tid] = bf16f(q[((long)j * DF_NQ + h) * DF_HD + tid]);
  __syncthreads();
  const float scale = rsqrtf((float)DF_HD);
  float mx = -3.0e38f;
  for (int idx = tid; idx < nk; idx += 256) {
    const uint16_t* kp = idx < nctx ? kc + ((long)(lo + idx) * DF_NKV + kvh) * DF_HD
                                    : kb + ((long)(idx - nctx) * DF_NKV + kvh) * DF_HD;
    float acc = 0.f;
    for (int d = 0; d < DF_HD; d += 2) {
      const uint32_t pr = *reinterpret_cast<const uint32_t*>(kp + d);
      acc = fmaf(qf[d], bf16f((uint16_t)(pr & 0xFFFFu)), acc);
      acc = fmaf(qf[d + 1], bf16f((uint16_t)(pr >> 16)), acc);
    }
    acc *= scale;
    sc[idx] = acc;
    mx = fmaxf(mx, acc);
  }
  red[tid] = mx; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]); __syncthreads(); }
  const float gmx = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int idx = tid; idx < nk; idx += 256) { float e = expf(sc[idx] - gmx); sc[idx] = e; sum += e; }
  red[tid] = sum; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid + off]; __syncthreads(); }
  const float inv = 1.0f / red[0];
  if (tid < DF_HD) {
    float acc = 0.f;
    for (int idx = 0; idx < nk; idx++) {
      const uint16_t* vp = idx < nctx ? vc + ((long)(lo + idx) * DF_NKV + kvh) * DF_HD
                                      : vb + ((long)(idx - nctx) * DF_NKV + kvh) * DF_HD;
      acc = fmaf(sc[idx], bf16f(vp[tid]), acc);
    }
    out[((long)j * DF_NQ + h) * DF_HD + tid] = f32bf(acc * inv);
  }
}
// top-16 of each logits row over the draft vocabulary; candidates mapped to trunk ids
__global__ void __launch_bounds__(256)
k_df_topk(const float* __restrict__ L, int NV, const int* __restrict__ map,
          int* __restrict__ cand, float* __restrict__ unary) {
  const int row = blockIdx.x, tid = threadIdx.x;
  const float* Lr = L + (long)row * NV;
  float lv[DF_TOPK]; int li[DF_TOPK];
#pragma unroll
  for (int i = 0; i < DF_TOPK; i++) { lv[i] = -3.0e38f; li[i] = -1; }
  for (int v = tid; v < NV; v += 256) {
    float x = Lr[v];
    if (x > lv[DF_TOPK - 1]) {
      int i = DF_TOPK - 1;
      while (i > 0 && lv[i - 1] < x) { lv[i] = lv[i - 1]; li[i] = li[i - 1]; i--; }
      lv[i] = x; li[i] = v;
    }
  }
  __shared__ float sv[256]; __shared__ int si[256];
  int head = 0;
  for (int k = 0; k < DF_TOPK; k++) {
    sv[tid] = head < DF_TOPK ? lv[head] : -3.0e38f; si[tid] = tid;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
      if (tid < off && (sv[tid + off] > sv[tid] || (sv[tid + off] == sv[tid] && si[tid + off] < si[tid]))) { sv[tid] = sv[tid + off]; si[tid] = si[tid + off]; }
      __syncthreads();
    }
    const int win = si[0];
    if (tid == win) {
      const int v = li[head];
      cand[row * DF_TOPK + k] = map ? map[v] : v;
      unary[row * DF_TOPK + k] = lv[head];
      head++;
    }
    __syncthreads();
  }
}
// greedy selector walk: for each slot t, score_b = unary[t][b] + <pre[prev] * h[t], suc[cand[t][b]]>
__global__ void __launch_bounds__(256)
k_df_select(const int* __restrict__ cand, const float* __restrict__ unary, const uint16_t* __restrict__ h,
            const uint16_t* __restrict__ pre, const uint16_t* __restrict__ suc, const int* __restrict__ anchor,
            int nslots, int* __restrict__ out) {
  const int tid = threadIdx.x, b = tid >> 4, s = tid & 15;     // 16 candidates x 16 lanes
  __shared__ float part[256];
  __shared__ int prev;
  if (tid == 0) prev = anchor[0];
  __syncthreads();
  for (int t = 0; t < nslots; t++) {
    const int c = cand[t * DF_TOPK + b];
    const uint16_t* a = pre + (long)prev * DF_RANK;
    const uint16_t* bb = suc + (long)c * DF_RANK;
    const uint16_t* hh = h + (long)t * DF_RANK;
    float acc = 0.f;
    for (int r = s * 16; r < s * 16 + 16; r++) acc = fmaf(bf16f(a[r]) * bf16f(hh[r]), bf16f(bb[r]), acc);
    part[tid] = acc;
    __syncthreads();
    if (tid < DF_TOPK) {
      float sc = unary[t * DF_TOPK + tid];
      for (int i = 0; i < 16; i++) sc += part[tid * 16 + i];
      part[tid] = sc;
    }
    __syncthreads();
    if (tid == 0) {
      int best = 0; float bv = part[0];
      for (int i = 1; i < DF_TOPK; i++) if (part[i] > bv) { bv = part[i]; best = i; }
      prev = cand[t * DF_TOPK + best];
      out[t] = prev;
    }
    __syncthreads();
  }
}
__global__ void k_df_block_ids(int* __restrict__ ids, const int* __restrict__ anchor, int mask, int T) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < T) ids[i] = i == 0 ? anchor[0] : mask;
}

// ---- host ----
template<int P>
static void df_mr(const uint16_t* A, const MtpQ4& q, void* C, int N, int K, bool f32) {
  const long rs = (long)K / 2 + K / 16;
  if (f32) k_gemv_dq_vec_mr<3, P, 0, 4, true><<<dim3((unsigned)N / 4), 256>>>(A, q.blob + 64, nullptr, (const float*)q.blob, rs, (uint16_t*)C, K, N);
  else     k_gemv_dq_vec_mr<3, P, 0, 4, false><<<dim3((unsigned)N / 4), 256>>>(A, q.blob + 64, nullptr, (const float*)q.blob, rs, (uint16_t*)C, K, N);
}
// C[T][N] = A[T][K] . W[N][K]^T from the NF4 blob (T <= 8) or the bf16 copy (T > 8)
static void df_proj(const uint16_t* A, const MtpQ4& q, const uint16_t* bf, void* C, int T, int N, int K, bool f32 = false) {
  if (T > 8) {
    if (!bf || f32) { fprintf(stderr, "dflash: no bf16 copy for a %d-row projection\n", T); exit(1); }
    gemm(A, bf, (uint16_t*)C, T, N, K);
    return;
  }
  if (T == 1 && !f32) { k_gemv_q4c1_vec<<<N, 256>>>(A, q.blob + 64, (const float*)q.blob, (long)(K / 2 + K / 16), (uint16_t*)C, K); return; }
  switch (T) {
    case 1: df_mr<1>(A, q, C, N, K, f32); break;
    case 2: df_mr<2>(A, q, C, N, K, f32); break;
    case 3: df_mr<3>(A, q, C, N, K, f32); break;
    case 4: df_mr<4>(A, q, C, N, K, f32); break;
    case 5: df_mr<5>(A, q, C, N, K, f32); break;
    case 6: df_mr<6>(A, q, C, N, K, f32); break;
    case 7: df_mr<7>(A, q, C, N, K, f32); break;
    default: df_mr<8>(A, q, C, N, K, f32); break;
  }
}
static void df_hproj(const uint16_t* A, uint16_t* C, int T) {   // [T][256] = A[T][5120] . hidden_projection^T (bf16)
  switch (T) {
    case 1: k_gemv_bf16_bf16out_vec<<<DF_RANK, 256>>>(A, d_df_hproj, C, HID); break;
    case 2: k_gemv_bf16_bf16out_rows<2><<<DF_RANK, 256>>>(A, d_df_hproj, C, HID, DF_RANK); break;
    case 3: k_gemv_bf16_bf16out_rows<3><<<DF_RANK, 256>>>(A, d_df_hproj, C, HID, DF_RANK); break;
    case 4: k_gemv_bf16_bf16out_rows<4><<<DF_RANK, 256>>>(A, d_df_hproj, C, HID, DF_RANK); break;
    case 5: k_gemv_bf16_bf16out_rows<5><<<DF_RANK, 256>>>(A, d_df_hproj, C, HID, DF_RANK); break;
    case 6: k_gemv_bf16_bf16out_rows<6><<<DF_RANK, 256>>>(A, d_df_hproj, C, HID, DF_RANK); break;
    default: k_gemv_bf16_bf16out_rows<7><<<DF_RANK, 256>>>(A, d_df_hproj, C, HID, DF_RANK); break;
  }
}
static void df_rmsnorm(const uint16_t* X, const uint16_t* w, uint16_t* O, int T) {
  k_rmsnorm<<<T, 256>>>(X, w, O, HID, T, g_df_zc);
}
// Ingest T captured rows (positions pos0..pos0+T-1, rows 0..T-1 of d_df_aux)
// into the drafter's context K/V.
static void df_ingest(int pos0, int T) {
  double t0 = now_s();
  for (int off = 0; off < T; off += DF_PIECE) {
    const int n = std::min(DF_PIECE, T - off);
    k_df_concat<<<GRID1((long)n * DF_FCIN), 256>>>(d_df_aux, d_df_cat, n, off);
    df_proj(d_df_cat, g_df_fc, d_df_fc_bf, d_df_ht, n, HID, DF_FCIN);
    df_rmsnorm(d_df_ht, d_df_hnorm, d_df_ht, n);
    for (int l = 0; l < DF_L; l++) {
      df_proj(d_df_ht, g_df[l].k, g_df[l].kbf, d_df_kt, n, DF_KW, HID);
      df_proj(d_df_ht, g_df[l].v, g_df[l].vbf, d_df_vt, n, DF_KW, HID);
      k_df_headnorm<<<GRID1((long)n * DF_NKV), 256>>>(d_df_kt, g_df[l].kn, d_df_kn2, n, DF_NKV, DF_HD, EPS, g_df_zc);
      k_df_rope<<<GRID1((long)n * DF_NKV * (DF_HD / 2)), 256>>>(d_df_kn2, n, DF_NKV, pos0 + off, g_df_theta);
      k_df_kvwrite<<<GRID1((long)n * DF_KW), 256>>>(d_df_kn2, d_df_vt, d_df_kc + (size_t)l * CXT * DF_KW, d_df_vc + (size_t)l * CXT * DF_KW, n, pos0 + off);
    }
  }
  g_df_ingest_ms += (now_s() - t0) * 1000.0;
}
// One block forward from the anchor (device int) at position P; writes the
// seven drafts (trunk ids) to d_df_out.
static void df_draft(const int* d_anchor, int P) {
  const int T = DF_B;
  k_df_block_ids<<<1, 32>>>(d_df_ids, d_anchor, g_df_mask, T);
  k_embed<<<GRID1((long)T * HID), 256>>>(d_df_ids, d_embed, d_df_x, HID);
  for (int l = 0; l < DF_L; l++) {
    const DfLayer& L = g_df[l];
    df_rmsnorm(d_df_x, L.ln_in, d_df_xn, T);
    df_proj(d_df_xn, L.kpa, nullptr, d_df_dyn, T, DF_KP, HID);
    k_df_conv<<<GRID1((long)T * HID), 256>>>(d_df_xn, d_df_dyn, L.base_a, d_df_xc, T, 0);
    df_proj(d_df_xc, L.q, nullptr, d_df_q, T, DF_QW, HID);
    df_proj(d_df_xc, L.k, L.kbf, d_df_kb, T, DF_KW, HID);
    df_proj(d_df_xc, L.v, L.vbf, d_df_vb, T, DF_KW, HID);
    k_df_headnorm<<<GRID1((long)T * DF_NQ), 256>>>(d_df_q, L.qn, d_df_qn, T, DF_NQ, DF_HD, EPS, g_df_zc);
    k_df_headnorm<<<GRID1((long)T * DF_NKV), 256>>>(d_df_kb, L.kn, d_df_knb, T, DF_NKV, DF_HD, EPS, g_df_zc);
    k_df_rope<<<GRID1((long)T * DF_NQ * (DF_HD / 2)), 256>>>(d_df_qn, T, DF_NQ, P, g_df_theta);
    k_df_rope<<<GRID1((long)T * DF_NKV * (DF_HD / 2)), 256>>>(d_df_knb, T, DF_NKV, P, g_df_theta);
    k_df_attn<<<dim3(DF_NQ, T), 256>>>(d_df_qn, d_df_knb, d_df_vb, d_df_kc + (size_t)l * CXT * DF_KW, d_df_vc + (size_t)l * CXT * DF_KW, P, g_df_win, T, d_df_att);
    df_proj(d_df_att, L.o, nullptr, d_df_o, T, HID, DF_QW);
    k_df_conv<<<GRID1((long)T * HID), 256>>>(d_df_o, d_df_dyn, L.base_a, d_df_xc, T, 1);
    k_add<<<GRID1((long)T * HID), 256>>>(d_df_x, d_df_xc, d_df_x, (long)T * HID);
    df_rmsnorm(d_df_x, L.ln_post, d_df_xn, T);
    df_proj(d_df_xn, L.kpm, nullptr, d_df_dyn, T, DF_KP, HID);
    k_df_conv<<<GRID1((long)T * HID), 256>>>(d_df_xn, d_df_dyn, L.base_m, d_df_xc, T, 0);
    df_proj(d_df_xc, L.g, nullptr, d_df_g, T, INTM, HID);
    df_proj(d_df_xc, L.u, nullptr, d_df_u, T, INTM, HID);
    k_silu_mul<<<GRID1((long)T * INTM), 256>>>(d_df_g, d_df_u, d_df_act, (long)T * INTM);
    df_proj(d_df_act, L.d, nullptr, d_df_o, T, HID, INTM);
    k_df_conv<<<GRID1((long)T * HID), 256>>>(d_df_o, d_df_dyn, L.base_m, d_df_xc, T, 1);
    k_add<<<GRID1((long)T * HID), 256>>>(d_df_x, d_df_xc, d_df_x, (long)T * HID);
  }
  df_rmsnorm(d_df_x, d_df_norm, d_df_xf, T);
  const int S = DF_B - 1;                                  // slots 1..7
  const uint16_t* xs = d_df_xf + (size_t)HID;
  df_proj(xs, g_df_dlm, nullptr, d_df_logits, S, g_df_nv, HID, true);
  k_df_topk<<<S, 256>>>(d_df_logits, g_df_nv, d_df_map, d_df_cand, d_df_unary);
  df_hproj(xs, d_df_h, S);
  k_df_select<<<1, 256>>>(d_df_cand, d_df_unary, d_df_h, d_df_pre, d_df_suc, d_anchor, S, d_df_out);
}
static float df_mean_gain(const uint16_t* d, long n) {
  std::vector<uint16_t> h(n);
  CK(hipMemcpy(h.data(), d, n * 2, hipMemcpyDeviceToHost));
  double s = 0; for (long i = 0; i < n; i++) s += h_bf16f(h[i]);
  return (float)(s / n);
}
static void df_load() {
  const char* ev = getenv("HALO_DFLASH");
  if (!ev || atoi(ev) != 1) return;
  if (!find("drafter.fc.weight") || !find("draft_lm_head.weight") || !find("drafter.candidate_selector.predecessor_codebook")) {
    fprintf(stderr, "dflash: checkpoint has no DFlash2 drafter\n"); return;
  }
  if (!d_lmhead_q || !g_gemv_vec || !g_gdn_gpu) { fprintf(stderr, "dflash: needs HALO_LMHEAD_FP8=1, HALO_GEMV_VEC>=1 and the device GDN scan\n"); return; }
  double t0 = now_s();
  {
    const Ent* c = find("drafter.config");
    if (c && c->dt == 3 && c->size == 21 * 4) {
      int cfg[21]; rd(c->off, cfg, sizeof cfg);
      if (cfg[1] != DF_L || cfg[2] != HID || cfg[3] != INTM || cfg[4] != DF_HD || cfg[5] != DF_NQ || cfg[6] != DF_NKV || cfg[7] != DF_B ||
          cfg[8] != 2 || cfg[9] != 16 || cfg[10] != DF_RANK || cfg[11] != DF_TOPK || cfg[15] != DF_L) {
        fprintf(stderr, "dflash: drafter.config does not match the compiled structure\n"); return;
      }
      g_df_mask = cfg[12]; g_df_win = cfg[13];
      for (int k = 0; k < DF_L; k++) g_df_tl[k] = cfg[16 + k];
    }
    const Ent* f = find("drafter.config_f32");
    if (f && f->size == 8) { float v[2]; rd(f->off, v, 8); g_df_theta = v[0]; }
  }
  {
    const Ent* dm = find("draft_vocab_map");
    if (!dm) { fprintf(stderr, "dflash: no draft_vocab_map\n"); return; }
    g_df_nv = (int)(dm->size / 4);
    if (d_draft_map && g_draft_nv == g_df_nv) d_df_map = d_draft_map;     // shared with the MTP draft vocabulary
    else {
      CK(hipMalloc((void**)&d_df_map, dm->size));
      rd(dm->off, h_stage, dm->size); up(d_df_map, h_stage, dm->size);
    }
  }
  mtp_keep_q4("drafter.fc.weight", HID, DF_FCIN, g_df_fc);
  d_df_fc_bf = mtp_load_q4c1("drafter.fc.weight", HID, DF_FCIN);
  mtp_keep_q4("draft_lm_head.weight", g_df_nv, HID, g_df_dlm);
  d_df_hnorm = mtp_load_bf16("drafter.hidden_norm.weight", HID);
  d_df_norm = mtp_load_bf16("drafter.norm.weight", HID);
  d_df_hproj = mtp_load_bf16("drafter.candidate_selector.hidden_projection.weight", (long)DF_RANK * HID);
  d_df_pre = mtp_load_bf16("drafter.candidate_selector.predecessor_codebook", (long)VOC * DF_RANK);
  d_df_suc = mtp_load_bf16("drafter.candidate_selector.successor_codebook", (long)VOC * DF_RANK);
  bool ok = d_df_hnorm && d_df_norm && d_df_hproj && d_df_pre && d_df_suc && d_df_fc_bf;
  char n[176];
  for (int l = 0; l < DF_L && ok; l++) {
    DfLayer& L = g_df[l];
    auto q4 = [&](const char* suffix, long rows, long cols, MtpQ4& q) {
      snprintf(n, sizeof n, "drafter.layers.%d.%s", l, suffix);
      if (!find(n)) { fprintf(stderr, "dflash: %s missing\n", n); ok = false; return; }
      mtp_keep_q4(n, rows, cols, q);
    };
    auto bf = [&](const char* suffix, long count) -> uint16_t* {
      snprintf(n, sizeof n, "drafter.layers.%d.%s", l, suffix);
      uint16_t* d = mtp_load_bf16(n, count); if (!d) ok = false; return d;
    };
    q4("self_attn.q_proj.weight", DF_QW, HID, L.q); q4("self_attn.k_proj.weight", DF_KW, HID, L.k); q4("self_attn.v_proj.weight", DF_KW, HID, L.v);
    q4("self_attn.o_proj.weight", HID, DF_QW, L.o);
    q4("mlp.gate_proj.weight", INTM, HID, L.g); q4("mlp.up_proj.weight", INTM, HID, L.u); q4("mlp.down_proj.weight", HID, INTM, L.d);
    q4("attention_conv.kernel_projection.weight", DF_KP, HID, L.kpa); q4("mlp_conv.kernel_projection.weight", DF_KP, HID, L.kpm);
    snprintf(n, sizeof n, "drafter.layers.%d.self_attn.k_proj.weight", l); L.kbf = mtp_load_q4c1(n, DF_KW, HID);
    snprintf(n, sizeof n, "drafter.layers.%d.self_attn.v_proj.weight", l); L.vbf = mtp_load_q4c1(n, DF_KW, HID);
    L.ln_in = bf("input_layernorm.weight", HID); L.ln_post = bf("post_attention_layernorm.weight", HID);
    L.qn = bf("self_attn.q_norm.weight", DF_HD); L.kn = bf("self_attn.k_norm.weight", DF_HD);
    L.base_a = bf("attention_conv.base_kernel", 2L * 2 * HID); L.base_m = bf("mlp_conv.base_kernel", 2L * 2 * HID);
    if (!L.kbf || !L.vbf) ok = false;
  }
  if (!ok) { fprintf(stderr, "dflash: drafter load failed\n"); return; }
  // Gain convention: HF Qwen3 RMSNorm gains are plain (values near 1); a
  // zero-centred store (GGUF value minus 1, as the trunk's) clusters near 0.
  const float mg = df_mean_gain(d_df_hnorm, HID);
  g_df_zc = fabsf(mg) < 0.5f ? 1 : 0;
  CK(hipMalloc((void**)&d_df_kc, (size_t)DF_L * CXT * DF_KW * 2));
  CK(hipMalloc((void**)&d_df_vc, (size_t)DF_L * CXT * DF_KW * 2));
  CK(hipMalloc((void**)&d_df_aux, (size_t)DF_L * TMAX * HID * 2));
  CK(hipMalloc((void**)&d_df_cat, (size_t)DF_PIECE * DF_FCIN * 2));
  CK(hipMalloc((void**)&d_df_ht, (size_t)DF_PIECE * HID * 2));
  CK(hipMalloc((void**)&d_df_kt, (size_t)DF_PIECE * DF_KW * 2));
  CK(hipMalloc((void**)&d_df_vt, (size_t)DF_PIECE * DF_KW * 2));
  CK(hipMalloc((void**)&d_df_kn2, (size_t)DF_PIECE * DF_KW * 2));
  const size_t rows = DF_B;
  CK(hipMalloc((void**)&d_df_x, rows * HID * 2));   CK(hipMalloc((void**)&d_df_xn, rows * HID * 2));  CK(hipMalloc((void**)&d_df_xc, rows * HID * 2));
  CK(hipMalloc((void**)&d_df_dyn, rows * DF_KP * 2)); CK(hipMalloc((void**)&d_df_q, rows * DF_QW * 2)); CK(hipMalloc((void**)&d_df_qn, rows * DF_QW * 2));
  CK(hipMalloc((void**)&d_df_kb, rows * DF_KW * 2)); CK(hipMalloc((void**)&d_df_knb, rows * DF_KW * 2)); CK(hipMalloc((void**)&d_df_vb, rows * DF_KW * 2));
  CK(hipMalloc((void**)&d_df_att, rows * DF_QW * 2)); CK(hipMalloc((void**)&d_df_o, rows * HID * 2));
  CK(hipMalloc((void**)&d_df_g, rows * INTM * 2));  CK(hipMalloc((void**)&d_df_u, rows * INTM * 2));  CK(hipMalloc((void**)&d_df_act, rows * INTM * 2));
  CK(hipMalloc((void**)&d_df_xf, rows * HID * 2));  CK(hipMalloc((void**)&d_df_h, rows * DF_RANK * 2));
  CK(hipMalloc((void**)&d_df_logits, rows * (size_t)g_df_nv * 4));
  CK(hipMalloc((void**)&d_df_unary, rows * DF_TOPK * 4)); CK(hipMalloc((void**)&d_df_cand, rows * DF_TOPK * 4));
  CK(hipMalloc((void**)&d_df_ids, 64)); CK(hipMalloc((void**)&d_df_out, 64));
  g_df_ok = 1;
  fprintf(stderr, "dflash: DFlash2 drafter loaded (%.1fs): %d layers, block %d, window %d, mask %d, target layers %d/%d/%d/%d/%d, theta %.0f, gains %s (hidden_norm mean %.3f), draft vocabulary %d\n",
          now_s() - t0, DF_L, DF_B, g_df_win, g_df_mask, g_df_tl[0], g_df_tl[1], g_df_tl[2], g_df_tl[3], g_df_tl[4], (double)g_df_theta,
          g_df_zc ? "zero-centred" : "plain", (double)mg, g_df_nv);
}
