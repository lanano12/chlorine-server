# chlorine-server

An independent, AGPL-3.0 reimplementation of the halogen 0.1.3 inference
engine for **Qwen3.8-27B on AMD Strix Halo (gfx1151)** — clean-room rebuilt
from reverse-engineering notes in `docs/halogen/`.

- The token engine implements `PING/INFO/CSTAT/GEN/X`. The OpenAI-compatible
  API is a separate frontend scaffold; it is not started by the engine command below.
- Loads `.hgn` checkpoints (see `converter/` and `docs/halogen/CHECKPOINT-FORMAT.md`).
- Greedy serving is validated against frozen local baseline tokens; full
  Halogen performance and output parity are not established.

**Not** affiliated with Peonist or halogen. The original engine is closed
source and lives at [peonist-ai/halogen-server](https://github.com/peonist-ai/halogen-server);
its EULA permits reverse engineering and interoperability. This codebase is
built from the behavioral specifications, HIP kernels and attributed
open-source numerical routines.

## Status

Work in progress — see [docs/halogen/REBUILD-PLAN.md](docs/halogen/REBUILD-PLAN.md).
The 2026-09-30 speed audit ranks this engine **56 / 100** against a
halogen-equivalent 27B server. The weights and the upgrade order are in the
parent repository at
[`../variant-27b/AUDIT.md`](../variant-27b/AUDIT.md).

| Component | State |
|---|---|
| `docs/halogen/` RE notes + specs | in progress |
| `engine/` C++20 host and token server | implemented greedy trunk |
| `engine/src/` HIP gfx1151 | hipBLASLt bf16 prefill, compressed-weight decode, strict GDN, opt-in f16 WMMA attention |
| `api/` OpenAI front-end | scaffolded |
| `converter/` safetensors → `.hgn` | scaffolded |

## Build and serving configuration

```bash
cmake -S engine -B build-32k -DHIPCC_EXECUTABLE=/opt/rocm/bin/hipcc \
  -DCHLORINE_TMAX=32768 -DCHLORINE_CXT=33024
cmake --build build-32k -j4
export OMP_WAIT_POLICY=passive OMP_NUM_THREADS=16
export HALO_FLASH_WMMA=1 HALO_FLASH_OPT=2 HALO_WEIGHT_CACHE=1
export HALO_GDN_SOLVE_LDS=1 HALO_PREFILL_ASYNC=1
# phase-3 decode path (validated: 9.46 tok/s at 1K, 9.14 at 8K context)
export HALO_GEMV_VEC=1 HALO_LMHEAD_FP8=1 HALO_GDN_STEP1=1
export HALO_DECODE_ASYNC=1 HALO_ATTN_SPLITK=1
exec build-32k/chlorine --checkpoint /path/to/model.hgn \
  --serve --bind 127.0.0.1 --port 8740
```

Use a new build directory for each measured binary. `TMAX` is the maximum
prefill row count; `CXT` covers prompt plus generation positions. The engine
requires prompt length <= `TMAX` and <= `CXT-2`. A 32,768-token prompt plus
256 generated tokens needs `CXT >= 33024`. These caps make the shape runnable;
they do not establish its throughput. Validate a short real request first:
attention initialization and the first real-layer comparison run before serving
its first token.

| Switch | Default | Behavior |
| --- | --- | --- |
| `HALO_FLASH_WMMA` | 0 | 0 retains global-bf16 `attn_tiled`; 1 uses f16 WMMA with online fp32 softmax and f16 tile probabilities. The rounding order differs. |
| `HALO_FLASH_OPT` | 0 | 0 selects the phase-2E kernel; 1 adds DPP/head-major scheduling; 2 adds a bounded, coalesced V-transpose workspace. Single-token attention keeps phase 2E. |
| `HALO_WEIGHT_CACHE` | 0 | Optionally stores 400 exact bf16 projection tensors (48,653,926,400 bytes). Requires allocation plus 8 GiB headroom in both HIP free memory and Linux MemAvailable. Decode retains compressed weights. |
| `HALO_GDN_SOLVE_LDS` | 0 | 1 stages the GDN triangular matrix in LDS, retaining address-order substitution. |
| `HALO_GDN_PERSISTENT` | 0 | Experimental 8/16/32-column persistent scans. Correctness tested; leave off because the measured layouts did not improve scan time. |
| `HALO_PREFILL_ASYNC` | 0 | 1 defers intermediate prefill fences on the default stream. Final logits remain fenced. Debug, profiling and host-GDN fallback remain synchronous. |
| `HALO_GEMV_VEC` | 0 | 1 selects the 16-byte-load decode GEMVs in `gemv_kernels.h` (same strips, accumulators and reduction tree: bit-identical). 2 additionally issues the weight loads as streaming loads. Falls back per tensor if K % 16 or alignment fails. |
| `HALO_LMHEAD_FP8` | 0 | 1 keeps the stored fp8r `lm_head` (1.27 GB) and skips the bf16 copy (2.54 GB); weights are rounded to bf16 in registers exactly as the copy was, so logits are bit-identical. |
| `HALO_GDN_STEP1` | 0 | 1 runs one GDN launch per layer at T=1 instead of the chunk-64 machinery. Bit-exact with the chunk path; the strict harness passes with it. |
| `HALO_DECODE_ASYNC` | 0 | With `HALO_PREFILL_ASYNC=1`, also defers the per-layer fences of the single-token step. Timing only. |
| `HALO_ATTN_SPLITK` | 0 | 1 uses flash-decoding for T=1: KV-head workgroups over key splits, the six query heads of a KV head read K/V once, fp32 online softmax, one combine. Declared numerics change, guarded at init and on the first real step. |
| `HALO_MTP` | 0 | 1 loads the MTP drafter and enables speculative decoding for requests with drafter id 1. Needs `HALO_GEMV_VEC>=1`, `HALO_LMHEAD_FP8=1` and the device GDN scan. Output is bit-identical to serial decode (verify rows use the serial kernels; see TECHNIQUES.md). |
| `HALO_SPEC_MAXG` | 4 | Maximum drafts per round (1–7). The GDN and conv state rings hold this + 2 slots (151 MB each for GDN). |
| `HALO_SPEC_PMIN` | 0 | Stop drafting when the drafter's probability for its token falls below this value. Drafter-only. |
| `HALO_MTP_VOCAB` | 0 | 1 takes the draft argmax over the checkpoint's 98,304-token `draft_vocab_map`. Drafter-only. |
| `HALO_MTP_Q4` | 0 | 1 makes the drafter's single-row steps read its NF4 weights directly instead of a bf16 copy. Drafter-only. |
| `HALO_LOGIT_HASH` | 0 | 1 prints `LH <pos> <hash>` for the full logits row behind each committed token: the serial-vs-speculative identity test. Debug only. |
| `HALO_AB_GEMV` | 0 | 1 computes the 48-row GDN a/b gate projections with a GEMV at T=1 instead of two hipBLASLt M=1 calls. Declared numerics change (accumulation order differs from hipBLASLt). |

The GDN translation unit disables FMA contraction and uses a glibc-derived
`expf` value implementation. Host scan arithmetic is unchanged. Initialization
requires output maxabs zero and state maxabs < 1e-5, and rejects nonfinite values.
Standalone correctness APIs remain synchronous. The exp value contract is
validated against this boot's host libm under round-to-nearest; a different host
libm/compiler configuration requires renewed validation.

How each switch achieves its speedup, with evidence, is in
[`variant-27b/TECHNIQUES.md`](https://github.com/lanano12/cpu-performance-engineering/blob/develop/variant-27b/TECHNIQUES.md).
The `QUIT` wire verb stops the server cleanly (used by the profiling script so rocprofv3 flushes).

The bf16 weight cache is independent of prompt-prefix reuse. Speculative
decoding with the MTP drafter (drafter id 1) is implemented; DFlash2 (drafter 2)
falls back to serial decode and prefix caching is not implemented. The D line reports
`<drafter> <rounds> <committed>`. `HALO_ACTQ` must remain unset
for the validated configuration; native activation-int4 is behind separate
correctness, quality and real 32K performance gates.

Full benchmark evidence and the regularly generated throughput table live in
[`variant-27b/benchmarks.md`](https://github.com/lanano12/cpu-performance-engineering/blob/develop/variant-27b/benchmarks.md)
in the parent repository. Isolated kernel gains are separate from server tok/s.

## Weights

Never in this repo. Bring your own: the halogen `.hgn` checkpoints are
published separately (`peonist-ai/halogen-qwen3.8-27b` on Hugging Face), or
convert your own Qwen3.8-27B safetensors with `converter/`.

## License

AGPL-3.0 — see [LICENSE](LICENSE). Model weights are governed by their own
licenses and are explicitly excluded from this repository's terms.
The attributed glibc-derived `engine/src/gdn_expf.h` is LGPL-2.1-or-later;
see [engine/COPYING.LIB](engine/COPYING.LIB).
