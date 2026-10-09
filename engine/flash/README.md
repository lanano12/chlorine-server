# Chlorine Flash backend

Source-compiled Qwen3.8-Flash-Next for gfx1151. This is the first checkpoint-compatibility implementation, **not a qualified model release**. The engine and HTTP API compile, CPU format/mapping/template checks pass, all 22 GPU operator tests pass on gfx1151, and the local UD-IQ4_XS GGUF loads and generates (2026-10-03). A full-precision model reference, long context, speculation and admitted throughput are unverified.

The complete 48-layer Flash architecture is imported from the pinned AGPL gfx1151-engine, independently of the existing 27B engine. Chlorine additions cover the local mixed IQ3_S/IQ4_XS/IQ4_NL/Q8_0 experts, stronger file validation, preserved F32 small projections, independently checked PLE/tokenizer semantics, build/tests and controlled service defaults. See the parent repository's [`variant-flash/STATUS.md`](../../../variant-flash/STATUS.md) and [`IMPLEMENTATION.md`](../../../variant-flash/IMPLEMENTATION.md) for acceptance gates. Independent CPU checks do not qualify GPU/model behavior.

## Build

From the parent workspace:

```bash
cmake -S chlorine-server/engine/flash -B variant-flash/build-local -DCMAKE_BUILD_TYPE=Release -DFLASH_PLE_CAUSAL=ON -DFLASH_GGUF_DIRECT_DECAY=ON
cmake --build variant-flash/build-local -j2
ctest --test-dir variant-flash/build-local --output-on-failure
```

Requirements: CMake 3.20+, C++17, Python 3, ROCm hipcc with gfx1151 support, rocBLAS, hipBLASLt, libpng, libjpeg and libwebp. Python fixture rendering additionally needs Jinja2. Configure `-DFLASH_BUILD_GPU=OFF -DFLASH_BUILD_API=OFF` for host-only reader/quantization/mapping tests. GPU tests return skip code 77 when no HIP device exists; a skipped test never qualifies a kernel.

Main outputs: `flash-q6-test`, `flash-mapping-test`, `flash-index-reference`, `flash-attention-reference`, `flash-trace-test`, `chlorine-flash`, `chlorine-flash-api`, `chlorine-flash-inspect`, `flash-tok`, `flash-template` and `flash-tokenizer-metadata`. Builds belong outside the source directory and are not committed.

The pinned CPU index/attention oracles and bounded trace comparator are described in the parent [Flash diagnostic guide](../../../variant-flash/DIAGNOSTICS.md). Two GPU tests compare actual indexer and attention operators against those references. Return code 77 means device unavailable; neither these tests nor the host references qualify a complete model. The parent's `variant-flash/bench/model_reference.py` compares the loaded model with llama.cpp's `qwen4exp` on the same GGUF. The current candidate and test results are in [Flash STATUS](../../../variant-flash/STATUS.md).

`FLASH_PLE_CAUSAL` defaults **OFF** to preserve the separately named `legacy-nine-slot-alias-v1` behavior. The example explicitly selects the candidate `causal-dilation3-v1`: ten ring slots preserve the back-nine tap, zero gate dots produce 0.5, and n-gram predecessors reset at EOS. Both CPU modes are built for comparison. The causal operator matches frozen independent Qwen4-Exp/PyTorch fixtures; GPU and complete model qualification are pending. Cache fingerprints and all ring allocation/snapshot/restore paths carry the selected semantics. The imported `upstream/src/ref.cpp` remains a legacy reference, not an independent oracle for the new mode.

Each successful engine build writes `chlorine-flash.build.json`, binding its SHA256 to the compiler command, PLE/decay modes and source-file hashes. Measurement requires this record; missing or mismatched provenance is an error. Preserve existing `build-f1`, `build-f2-causal` and `build-f2-mapping` artifacts and use a new directory for later candidates.

## Speed paths and switches (2026-10-04)

**Always on, bit-identical to what they replace.** Each has a retained reference kernel and a test that requires zero differing bits (`moe_test`, `gemv_test`, `mapping_gpu_test`).

- IQ3_S experts run on the staged lookup-table prefill kernel (`k_moe_lut<kIQ3S>`), decoded in packed FP16 as `(magnitude × (1+2s)) × d`: the first product is an integer below 2,048, so one rounding remains. `GDEC_FLASH_IQ3_SCALAR=1` selects the original kernel.
- Expert token tiles hold 160 slots instead of 64, so a busy expert's weights are decoded once per tile of 160 tokens. `GDEC_FLASH_MOE_TILE=64|128|192|256` selects another size.
- The shared-expert add is folded into the routed reduce in batched prefill (`GDEC_MOE_SG_FUSE=0` restores the separate pass).
- The expert decode GEMV loads IQ3_S/IQ4_NL/IQ4_XS units a block at a time, and the Q6_K head fetches a row's ten blocks before decoding (`GDEC_FLASH_Q6_LOOP=1` restores the block loop).
- Router top-k (`k_router_topk`) uses 12 block barriers instead of about 130; the tree kernel's tie rule is kept exactly (`k_router_topk_tree` is the reference).
- Serial decode fuses neighbouring single-token launches (GDN gates, hyperconnection read and write, shared expert) and reduces the F32 row dot with one barrier in the reference tree's pairing. `GDEC_FLASH_DECODE_FUSE=0` restores the separate launches.

**Opt-in, arithmetic changes** (the fast set), passed through the supervisor's `--engine-env`:

- `GDEC_F32_GEMM_BF16=1`: batched GEMMs (more than 8 rows) on F32 GGUF matrices take the bf16 paths. A matrix qualifies only if every value is exact in bf16, checked on device at first use; the weights then lose nothing and the activations are rounded to bf16 as for the quantized matrices. With `GDEC_GR_BF16=1` the F32 inject matrices also join the fused scatter-norm-inject kernel through the same exact copy.
- `GDEC_GR_BF16=1`: bf16 residual stream in batched prefill. Rejected for GGUF unless `GDEC_F32_GEMM_BF16=1` is also set.
- `GDEC_QSA_UNION=1` (imported): sparse prefill attention over the union of four neighbouring queries' selected blocks. Two corrections to the imported path: the per-token launch for a batch's trailing one to three queries now receives those queries' own selection rows (it was given the first sparse tokens' rows, so prompts that did not end on a 4-token boundary attended wrongly at their end), and the group owning block 65,535 (positions 262,140 to 262,143) stays on the per-token kernel because the union lists use 16-bit block numbers with `0xFFFF` as padding.
- `GDEC_QSA_DENSE_UNION=1` (needs `GDEC_QSA_UNION`): the dense attention prefix (positions below 2,051) runs on the same union WMMA kernel with "every earlier block" lists, instead of the per-row dense kernel.
- `GDEC_FLASH_PAIRS_F16=1`: expert pair rows are stored as FP16 between the down kernel and the reduce.
- The imported production switches (`GDEC_QSA_KV_BF16`, `GDEC_QSA_WMMA`, `GDEC_QSA_WMMA_BTV`, `GDEC_GDN_STREAM`, `GDEC_GDN_WAVE`, `GDEC_GDN_FUSED`, `GDEC_GEMM_WMMA`, `GDEC_INDEX_FUSED2`, `GDEC_INDEX_STREAM_SELECT`, `GDEC_PP_MOE_OUT`).

**Opt-in, scheduling only.** `GDEC_GDN_ZLAP=1` issues the GDN `in_proj_z` GEMM on a side stream beside the fused GDN scan (which leaves the GPU partly idle). Kernels and operands are unchanged, so results are those of the same switch set without it.

Diagnostics: `GDEC_PHASE=3` prints named sub-layer prefill times per chunk, `GDEC_PERF=1` prints decode time per token by phase, and both together add per-step decode timers. They synchronise the stream and are never throughput measurements. Mechanisms, measurements and the quality-gate results are in the parent's [`variant-flash/TECHNIQUES.md`](../../../variant-flash/TECHNIQUES.md).

## Runtime boundaries

**Unqualified verification experiments (2026-10-09).** `GDEC_FLASH_VERIFY_F32=1` selects the serial fixed-order F32 GEMV for all batches of 1–9 rows, including prompt tails and the final mixer; larger batches retain their existing paths. It rejects `GDEC_GR_BF16`, which does not maintain the required FP32 residual inputs. `GDEC_FLASH_VERIFY_NO_FINAL=1` skips the discarded single-row final head before MTP/chain batch logits. Both default off. The expanded `mapping_gpu_test` requires zero bit differences for the F32 route but has only compiled, not executed on device in the implementation session. Neither switch proves complete verifier identity or qualifies drafting. The parent [handoff](../../../variant-flash/IMPLEMENTATION-2026-10-09.md) records build/test evidence and the remaining recurrent/rollback checks.

Union-attention dispatch also now uses `qsa_union_partition.hpp` to handle short unaligned continuations without out-of-range query or selection rows. `flash-qsa-union-partition` checks exact interval coverage and the native-context padding boundary; it does not replace GPU output checks.

Use the parent `variant-flash/bench/serve.py` supervisor. It defaults to preflight only, clears inherited experimental settings, estimates memory admission, checks the two ports and other model processes, and owns only its own children. Add `--execute` on a device-enabled host to start it; Ctrl-C stops both children. The memory allowance is conservative, not a measured peak. See [`variant-flash/bench/README.md`](../../../variant-flash/bench/README.md).

Default engine: `127.0.0.1:8742`. Default API: `127.0.0.1:8733`. Model id: `chlorine-qwen3.8-flash-next`. Serial greedy qualification precedes any drafting qualification. The API still uses the checkpoint's sampling defaults unless temperature is explicitly zero. Vision input is rejected in this initial text release.

`CAPS` returns `C` followed by versioned JSON (protocol `chlorine-flash-engine`, version 1), with context, chunk, slots, KV precision, drafter, checkpoint/binary identity from the supervisor, PLE/decay semantics, head dtype and explicit pending qualification. Historical `INFO`, `GEN`, `T`, `D`, `X` and `MEM` remain available. `/health` validates the engine's capability reply and reports its identity and PLE mode, with readiness separate from release qualification. An absent/malformed reply produces 503 with unknown identity, not a guessed mode. The Flash D grammar has twelve fields; do not parse it as the 27B ten-field variant.

By default F32 GGUF matrices stay F32. Scalar rows use a fixed-order FP32 GEMV; prefill uses rocBLAS SGEMM. Accumulation order can differ between these operations, so this is not a serial-versus-batch equivalence claim. `GDEC_GR_BF16` is rejected with GGUF unless `GDEC_F32_GEMM_BF16=1` is set (see above). Other upstream experimental modes remain opt-in; the supervisor clears them. IQ3_S WMMA rounds operands to f16 and accumulates in f32; the host format decoder's bit equality does not prove WMMA output equality.

## Packed head and GGUF decay

The local vocabulary head is Q6_K. The imported mapper required Q8_0 and had no Q6_K decoder, so earlier builds could not complete loading this checkpoint; prior documentation describing an inherited head conversion was incorrect. The current mapper borrows the exact Q6_K bytes as internal dtype 12, then copies only the packed 521,472,000-byte head into the device arena. Dtype 12 remains invalid in on-disk HGN files. FP32 accumulation consumes decoded weights directly for single-row and multi-row logits, including row subsets and BF16 input rows. No full BF16/F32 head copy or requantization is created. GPU output checks are still pending.

`FLASH_GGUF_DIRECT_DECAY=ON` explicitly selects stored GGUF `ssm_a` coefficients for every GDN gate call. The default OFF preserves the compatibility `log(-ssm_a)` followed by device `-exp`; native HGN continues to use its A_log representation in either build. The mapper retains both named representations and rejects nonfinite/nonnegative coefficients before publishing them. Decay mode and head dtype enter the persisted cache fingerprint, `CAPS`, build record and measurement handshake.

`flash-q6-test FIRST.gguf` compares the decoder to pinned GGML code, including 40 actual head rows. `flash-mapping-test FIRST.gguf` checks all 36 actual GDN coefficient/bias vectors and 184 norm tensors without allocating a model. Host evidence reports 15 changed coefficients from the compatibility log/exp round trip versus zero from direct copying; all 1,037,824 norm values recover exactly through their configured folding rule. These are host mapping facts, not model or throughput results. The GPU mapping test additionally checks real dispatch, batch/single identity, row offsets, in-place gates and softplus boundaries.

## Provenance and licenses

`upstream/` begins at gfx1151-engine commit `78a41cc87289dc759ace2f9c030ad5483b30673c`, then carries the local changes described above. `UPSTREAM.json` records the **original imported file hashes**, not current patched hashes. Benchmark/validation artifacts record current source-tree hashes. Historical speed statements in imported comments describe upstream experiments, not Chlorine Flash measurements.

The derivative engine retains AGPL-3.0 under [`upstream/LICENSE`](upstream/LICENSE). GGML IQ lookup/decoder reference material retains its MIT notice in [`quant/LICENSE.ggml`](quant/LICENSE.ggml); [`quant/SOURCES.json`](quant/SOURCES.json) pins the inspected source. See [`NOTICE.md`](NOTICE.md). The parent's MIT license does not relicense these components.
