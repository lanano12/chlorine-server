// bf16 GEMM for the prefill path. C[m,n] = sum_k A[m,k] * W[n,k], both
// row-major. hipBLASLt sees those buffers as column-major (K,N) and (K,M):
// C_col(N,M) = W^T * A, which is the same addresses as C_row(M,N).
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

namespace {

struct Key {
  int M, N, K, f32;
  bool operator<(const Key& o) const {
    if (M != o.M) return M < o.M;
    if (N != o.N) return N < o.N;
    if (K != o.K) return K < o.K;
    return f32 < o.f32;
  }
};

struct Entry {
  hipblasLtMatmulDesc_t desc{};
  hipblasLtMatrixLayout_t Wd{};
  hipblasLtMatrixLayout_t Ad{};
  hipblasLtMatrixLayout_t Cd{};
  hipblasLtMatmulHeuristicResult_t heur{};
};

hipblasLtHandle_t g_handle = nullptr;
hipblasLtMatmulPreference_t g_pref = nullptr;
void* g_ws = nullptr;
size_t g_ws_bytes = 128ull << 20;
std::map<Key, Entry> g_cache;
int g_state = 0;  // 0 unready, 1 ok, -1 dead
float g_check_maxabs = -1.f;

uint16_t f32bf(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return (uint16_t)(u >> 16);
}
float bf16f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

int ensure_handle() {
  if (g_state == 1) return 0;
  if (g_state < 0) return -1;
  if (hipblasLtCreate(&g_handle) != HIPBLAS_STATUS_SUCCESS) {
    g_state = -1;
    return -1;
  }
  if (hipblasLtMatmulPreferenceCreate(&g_pref) != HIPBLAS_STATUS_SUCCESS) {
    g_state = -1;
    return -1;
  }
  uint64_t ws = g_ws_bytes;
  hipblasLtMatmulPreferenceSetAttribute(g_pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  if (hipMalloc(&g_ws, g_ws_bytes) != hipSuccess) {
    g_state = -1;
    return -1;
  }
  g_state = 1;
  return 0;
}

const Entry* cached(int M, int N, int K, int f32) {
  Key key{M, N, K, f32};
  auto it = g_cache.find(key);
  if (it != g_cache.end()) return &it->second;
  Entry e;
  if (hipblasLtMatmulDescCreate(&e.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS)
    return nullptr;
  hipblasOperation_t opW = HIPBLAS_OP_T, opA = HIPBLAS_OP_N;
  hipblasLtMatmulDescSetAttribute(e.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opW, sizeof(opW));
  hipblasLtMatmulDescSetAttribute(e.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA));
  hipDataType ct = f32 ? HIP_R_32F : HIP_R_16BF;
  if (hipblasLtMatrixLayoutCreate(&e.Wd, HIP_R_16BF, K, N, K) != HIPBLAS_STATUS_SUCCESS) return nullptr;
  if (hipblasLtMatrixLayoutCreate(&e.Ad, HIP_R_16BF, K, M, K) != HIPBLAS_STATUS_SUCCESS) return nullptr;
  if (hipblasLtMatrixLayoutCreate(&e.Cd, ct, N, M, N) != HIPBLAS_STATUS_SUCCESS) return nullptr;
  int nalgo = 0;
  hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(
      g_handle, e.desc, e.Wd, e.Ad, e.Cd, e.Cd, g_pref, 1, &e.heur, &nalgo);
  if (st != HIPBLAS_STATUS_SUCCESS || nalgo < 1 || e.heur.state != HIPBLAS_STATUS_SUCCESS) {
    hipblasLtMatrixLayoutDestroy(e.Wd);
    hipblasLtMatrixLayoutDestroy(e.Ad);
    hipblasLtMatrixLayoutDestroy(e.Cd);
    hipblasLtMatmulDescDestroy(e.desc);
    return nullptr;
  }
  auto ins = g_cache.emplace(key, e);
  return &ins.first->second;
}

int matmul(const void* A, const void* W, void* C, int M, int N, int K, int f32) {
  if (M < 1 || N < 1 || K < 1) return -1;
  if (ensure_handle() != 0) return -2;
  const Entry* e = cached(M, N, K, f32);
  if (!e) return -1;
  float alpha = 1.f, beta = 0.f;
  hipblasStatus_t st = hipblasLtMatmul(
      g_handle, e->desc, &alpha, W, e->Wd, A, e->Ad, &beta, C, e->Cd, C, e->Cd,
      &e->heur.algo, g_ws, g_ws_bytes, nullptr);
  return st == HIPBLAS_STATUS_SUCCESS ? 0 : -1;
}

int self_check() {
  const int M = 8, N = 32, K = 128;
  std::vector<uint16_t> hA(M * K), hW(N * K), hC(M * N);
  for (int i = 0; i < M * K; i++) hA[i] = f32bf(((i * 17) % 100) / 50.f - 1.f);
  for (int i = 0; i < N * K; i++) hW[i] = f32bf(((i * 13) % 100) / 50.f - 1.f);
  uint16_t *A = nullptr, *W = nullptr, *C = nullptr;
  if (hipMalloc(&A, hA.size() * 2) || hipMalloc(&W, hW.size() * 2) || hipMalloc(&C, hC.size() * 2))
    return -1;
  hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(W, hW.data(), hW.size() * 2, hipMemcpyHostToDevice);
  int rc = matmul(A, W, C, M, N, K, 0);
  hipDeviceSynchronize();
  if (rc == 0) hipMemcpy(hC.data(), C, hC.size() * 2, hipMemcpyDeviceToHost);
  hipFree(A);
  hipFree(W);
  hipFree(C);
  if (rc != 0) return rc;
  double maxabs = 0;
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      double acc = 0;
      for (int k = 0; k < K; k++) acc += (double)bf16f(hA[m * K + k]) * (double)bf16f(hW[n * K + k]);
      maxabs = std::max(maxabs, fabs(bf16f(hC[m * N + n]) - acc));
    }
  }
  g_check_maxabs = (float)maxabs;
  if (maxabs > 0.5) {
    fprintf(stderr, "gemm: hipBLASLt self-check failed maxabs %.4g\n", maxabs);
    g_state = -1;
    return -2;
  }
  fprintf(stderr, "gemm: hipBLASLt bf16 on (self-check maxabs %.4g)\n", maxabs);
  return 0;
}

}  // namespace

extern "C" int chlorine_gemm_lt(const void* A, const void* W, void* C, int M, int N, int K, int f32) {
  if (g_state < 0) return -2;
  if (g_check_maxabs < 0.f) {
    int rc = self_check();
    if (rc != 0) return rc;
  }
  return matmul(A, W, C, M, N, K, f32);
}

extern "C" float chlorine_gemm_lt_check_maxabs(void) { return g_check_maxabs; }
