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
| `HALO_PREFILL_ASYNC` | 0 | 1 defers intermediate prefill fences on the default stream. Final logits remain fenced. Debug, profiling, host-GDN fallback and T=1 remain synchronous. |

The GDN translation unit disables FMA contraction and uses a glibc-derived
`expf` value implementation. Host scan arithmetic is unchanged. Initialization
requires output maxabs zero and state maxabs < 1e-5, and rejects nonfinite values.
Standalone correctness APIs remain synchronous. The exp value contract is
validated against this boot's host libm under round-to-nearest; a different host
libm/compiler configuration requires renewed validation.

The bf16 weight cache is independent of prompt-prefix reuse. Speculation and
prefix caching are not implemented in this trunk. `HALO_ACTQ` must remain unset
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
