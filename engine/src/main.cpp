// main.cpp — chlorine-server engine entry point.
// Modes: --checkpoint/--build-info/--serve. With a GPU trunk available the GEN
// handler runs the real model (prefill + greedy argmax decode); the sampler
// (temp/top-k/top-p) is the next phase and is rejected until then. Without a
// trunk (no GPU / no checkpoint) a deterministic stub keeps the wire protocol
// testable.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <string>
#include <vector>

#include "hgn.h"
#include "serve.h"

using namespace chlorine;

namespace {

// HIP trunk backend (engine/src/generator.hip), compiled with hipcc when present.
extern "C" {
int chlorine_trunk_init(const char* ckpt_path);
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
int chlorine_trunk_generate2(const int* prompt, int n_prompt, long max_tokens,
                             const int* eos, int n_eos,
                             void (*emit)(void*, int, float), void* ctx,
                             double* prefill_ms, double* decode_ms, int* stop_hit,
                             const chlorine_sample_opts* opts);
struct chlorine_spec_stats {
  int drafter, rounds, proposed, accepted, committed;
  double draft_ms, verify_ms, ingest_ms, mtp_prefill_ms;
  int n_cached;
};
int chlorine_trunk_cstat(char* buf, int cap);
int chlorine_trunk_generate3(const int* prompt, int n_prompt, long max_tokens,
                             const int* eos, int n_eos,
                             void (*emit)(void*, int, float), void* ctx,
                             double* prefill_ms, double* decode_ms, int* stop_hit,
                             const chlorine_sample_opts* opts, int drafter,
                             chlorine_spec_stats* st);
int chlorine_trunk_score(const int* ids, int n, int n_prompt, float* nll_out, int* pred_out);
int chlorine_trunk_score_decode(const int* ids, int n, int n_prompt, float* nll_out, int* pred_out);
int chlorine_trunk_ctx_cap(void);
int chlorine_trunk_tmax(void);
void chlorine_trunk_shutdown(void);
}

bool g_trunk_ready = false;

struct EmitCtx {
  long id;
  int fd;
  bool logprobs;
};

static void trunk_emit(void* ctx, int tok, float logp) {
  EmitCtx* e = (EmitCtx*)ctx;
  char buf[64];
  int n = e->logprobs ? snprintf(buf, sizeof buf, "T %ld %d %.9g\n", e->id, tok, double(logp))
                      : snprintf(buf, sizeof buf, "T %ld %d\n", e->id, tok);
  size_t off = 0;
  while (off < (size_t)n) {
    ssize_t w = send(e->fd, buf + off, size_t(n) - off, MSG_NOSIGNAL);
    if (w <= 0) return;
    off += size_t(w);
  }
}

void real_score(void* ctx, long id, int n_prompt, const std::vector<int>& ids, int fd) {
  (void)ctx;
  std::vector<float> nll(ids.size());
  std::vector<int> pred(ids.size());
  static int score_decode = -1;
  if (score_decode < 0) { const char* e = getenv("HALO_SCORE_DECODE"); score_decode = (e && atoi(e) == 1) ? 1 : 0; }
  int n = !g_trunk_ready ? -1
        : score_decode ? chlorine_trunk_score_decode(ids.data(), int(ids.size()), n_prompt, nll.data(), pred.data())
                       : chlorine_trunk_score(ids.data(), int(ids.size()), n_prompt, nll.data(), pred.data());
  std::string out;
  if (n < 0) {
    out = "S " + std::to_string(id) + " error " + std::to_string(n) + "\n";
  } else {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += nll[size_t(i)];
    char buf[96];
    snprintf(buf, sizeof buf, "S %ld %d %.6f\n", id, n, n ? sum / n : 0.0);
    out = buf;
    out += "N " + std::to_string(id);
    for (int i = 0; i < n; i++) { snprintf(buf, sizeof buf, " %.5f", double(nll[size_t(i)])); out += buf; }
    out += "\nP " + std::to_string(id);
    for (int i = 0; i < n; i++) { snprintf(buf, sizeof buf, " %d", pred[size_t(i)]); out += buf; }
    out += "\n";
  }
  send(fd, out.data(), out.size(), MSG_NOSIGNAL);
}

void real_generate(void* ctx, long id, int max_tokens, const std::vector<int>& eos,
                   const std::vector<int>& prompt, const chlorine::GenOpts& go, int fd) {
  (void)ctx;
  if (!g_trunk_ready) {
    fprintf(stderr, "serve: request %ld trunk unavailable\n", id);
    char buf[64];
    snprintf(buf, sizeof buf, "D %ld error 0 0 0.0 0.0\n", id);
    send(fd, buf, strlen(buf), MSG_NOSIGNAL);
    return;
  }
  chlorine_sample_opts so = {};
  so.has_sample = go.has_sample ? 1 : 0;
  so.temp = float(go.temp);
  so.top_k = go.top_k;
  so.top_p = float(go.top_p);
  so.min_p = float(go.min_p);
  so.seed = go.seed;
  so.presence = float(go.presence);
  so.frequency = float(go.frequency);
  so.n_bias = int(go.bias_ids.size());
  so.bias_ids = go.bias_ids.data();
  so.bias_vals = go.bias_vals.data();
  so.logprobs = go.logprobs ? 1 : 0;
  EmitCtx ec{id, fd, go.logprobs};
  double prefill_ms = 0, decode_ms = 0;
  int stop_hit = 0;
  chlorine_spec_stats st{};
  int n_gen = chlorine_trunk_generate3(prompt.data(), int(prompt.size()), max_tokens,
                                       eos.data(), int(eos.size()), trunk_emit, &ec,
                                       &prefill_ms, &decode_ms, &stop_hit, &so, go.drafter, &st);
  if (n_gen < 0) {
    fprintf(stderr, "serve: request %ld prompt/max_tokens out of trunk capacity (%d)\n",
            id, n_gen);
    char buf[64];
    snprintf(buf, sizeof buf, "D %ld error 0 0 0.0 0.0\n", id);
    send(fd, buf, strlen(buf), MSG_NOSIGNAL);
    return;
  }
  char buf[160];
  // drafter / rounds / committed-in-rounds, as WIRE-PROTOCOL.md specifies
  snprintf(buf, sizeof buf, "D %ld %s %zu %d %.1f %.1f %d %d %d %d\n", id,
           stop_hit ? "stop" : "length", prompt.size(), n_gen, prefill_ms, decode_ms,
           st.drafter, st.rounds, st.committed, st.n_cached);
  size_t off = 0;
  size_t len = strlen(buf);
  while (off < len) {
    ssize_t w = send(fd, buf + off, len - off, MSG_NOSIGNAL);
    if (w <= 0) return;
    off += size_t(w);
  }
}

void stub_generate(void* ctx, long id, int max_tokens, const std::vector<int>& eos,
                   const std::vector<int>& prompt, const chlorine::GenOpts& o, int fd) {
  (void)ctx; (void)o;
  // Deterministic stub: token ids derived from the prompt. Never emits an eos
  // unless the prompt itself ends with one (then 0 tokens).
  std::string out;
  char buf[128];
  int n_gen = 0;
  for (int i = 0; i < max_tokens && n_gen < 8; i++) {
    int tok = int((size_t(prompt.empty() ? 0 : prompt.back()) + 7919u * (i + 1)) % 248320u);
    bool stop = false;
    for (int e : eos)
      if (e == tok) { stop = true; break; }
    if (stop) { snprintf(buf, sizeof buf, "D %ld stop %zu %d 0.0 0.0\n", id, prompt.size(), n_gen); out += buf; break; }
    snprintf(buf, sizeof buf, "T %ld %d\n", id, tok);
    out += buf;
    n_gen++;
  }
  if (out.find('D') == std::string::npos) {
    snprintf(buf, sizeof buf, "D %ld length %zu %d 0.1 0.2 0 0 0 0\n", id, prompt.size(), n_gen);
    out += buf;
  }
  size_t off = 0;
  while (off < out.size()) {
    ssize_t n = send(fd, out.data() + off, out.size() - off, MSG_NOSIGNAL);
    if (n <= 0) return;
    off += size_t(n);
  }
}

void build_info(const Checkpoint& ckpt) {
  printf("unit         compiled   kCtxCap   kSdLen\n");
  printf("chlorine     skeleton   262144    683\n");
  printf("checkpoint: %s - %llu tensors, %.1f GB\n", ckpt.model_name().c_str(),
         (unsigned long long)ckpt.n_tensors(), double(ckpt.file_size()) / 1e9);
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt_path, bind = "127.0.0.1";
  int port = 8730;
  bool do_serve = false, do_info = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(2); }
      return argv[++i];
    };
    if (a == "--checkpoint") ckpt_path = next("--checkpoint");
    else if (a == "--serve") do_serve = true;
    else if (a == "--port") port = atoi(next("--port"));
    else if (a == "--bind") bind = next("--bind");
    else if (a == "--build-info") do_info = true;
    else { fprintf(stderr, "unknown flag %s\nusage: chlorine --checkpoint FILE.hgn [--build-info] [--serve [--port N] [--bind ADDR]]\n", a.c_str()); return 2; }
  }
  if (!do_serve && !do_info) {
    fprintf(stderr, "usage: chlorine --checkpoint FILE.hgn [--build-info] [--serve [--port N] [--bind ADDR]]\n");
    return 2;
  }
  if (ckpt_path.empty() && do_serve) {
    fprintf(stderr, "--serve needs --checkpoint\n");
    return 2;
  }
  if (ckpt_path.empty()) {
    printf("unit         compiled   kCtxCap   kSdLen\n");
    printf("chlorine     skeleton   262144    683\n");
    if (do_serve) { fprintf(stderr, "--serve needs --checkpoint\n"); return 2; }
    return 0;
  }
  Checkpoint ckpt(ckpt_path);
  if (do_info) build_info(ckpt);
  if (do_serve) {
    Server::Options o;
    o.port = port;
    o.bind = bind;
    o.default_drafter = ckpt.has_dflash2() ? 2 : (ckpt.has_mtp() ? 1 : 0);
    // CHLORINE_STUB=1 forces the deterministic stub generator (wire-protocol
    // conformance runs; the real trunk takes minutes per GEN while weights
    // stream from the checkpoint).
    if (!getenv("CHLORINE_STUB") && !g_trunk_ready && !ckpt_path.empty())
      g_trunk_ready = chlorine_trunk_init(ckpt_path.c_str()) == 0;
    if (g_trunk_ready && !getenv("CHLORINE_STUB")) {
      o.slot_ctx = chlorine_trunk_ctx_cap();
      fprintf(stderr, "trunk: ctx cap %d tmax %d\n", o.slot_ctx, chlorine_trunk_tmax());
    }
    Server svr(ckpt, o, g_trunk_ready && !getenv("CHLORINE_STUB") ? real_generate
                                                                  : stub_generate,
               nullptr, g_trunk_ready ? real_score : nullptr, g_trunk_ready ? chlorine_trunk_cstat : nullptr);
    svr.serve();
  }
  if (g_trunk_ready) chlorine_trunk_shutdown();
  return 0;
}
