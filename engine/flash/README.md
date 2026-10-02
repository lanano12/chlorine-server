# Chlorine Flash backend

Source-compiled Qwen3.8-Flash-Next for gfx1151. This is the first checkpoint-compatibility implementation, **not a qualified model release**. The engine and HTTP API compile. CPU format/mapping/template checks pass; GPU operators, real model tokens, long context, speculation and throughput are unverified.

The complete 48-layer Flash architecture is imported from the pinned AGPL gfx1151-engine, independently of the existing 27B engine. Chlorine additions cover the local mixed IQ3_S/IQ4_XS/IQ4_NL/Q8_0 experts, stronger file validation, preserved F32 small projections, independently checked PLE/tokenizer semantics, build/tests and controlled service defaults. See the parent repository's [`variant-flash/STATUS.md`](../../../variant-flash/STATUS.md) and [`IMPLEMENTATION.md`](../../../variant-flash/IMPLEMENTATION.md) for acceptance gates. Independent CPU checks do not qualify GPU/model behavior.

## Build

From the parent workspace:

```bash
cmake -S chlorine-server/engine/flash -B variant-flash/build-local -DCMAKE_BUILD_TYPE=Release -DFLASH_PLE_CAUSAL=ON
cmake --build variant-flash/build-local -j2
ctest --test-dir variant-flash/build-local --output-on-failure
```

Requirements: CMake 3.20+, C++17, Python 3, ROCm hipcc with gfx1151 support, rocBLAS, hipBLASLt, libpng, libjpeg and libwebp. Python fixture rendering additionally needs Jinja2. Configure `-DFLASH_BUILD_GPU=OFF -DFLASH_BUILD_API=OFF` for host-only reader/quantization/mapping tests. GPU tests return skip code 77 when no HIP device exists; a skipped test never qualifies a kernel.

Main outputs: `chlorine-flash`, `chlorine-flash-api`, `chlorine-flash-inspect`, `flash-tok`, `flash-template`, `flash-tokenizer-metadata`. Builds belong outside the source directory and are not committed.

`FLASH_PLE_CAUSAL` defaults **OFF** to preserve the separately named `legacy-nine-slot-alias-v1` behavior. The example explicitly selects the candidate `causal-dilation3-v1`: ten ring slots preserve the back-nine tap, zero gate dots produce 0.5, and n-gram predecessors reset at EOS. Both CPU modes are built for comparison. The causal operator matches frozen independent Qwen4-Exp/PyTorch fixtures; GPU and complete model qualification are pending. Cache fingerprints and all ring allocation/snapshot/restore paths carry the selected semantics. The imported `upstream/src/ref.cpp` remains a legacy reference, not an independent oracle for the new mode.

Each successful engine build writes `chlorine-flash.build.json`, binding its SHA256 to the compiler command, PLE mode and source-file hashes. Measurement requires this record; missing or mismatched provenance is an error. Preserve existing `build-f1` and `build-f2-causal` artifacts and use a new directory for later candidates.

## Runtime boundaries

Use the parent `variant-flash/bench/serve.py` supervisor. It defaults to preflight only, clears inherited experimental settings, estimates memory admission, checks the two ports and other model processes, and owns only its own children. Add `--execute` on a device-enabled host to start it; Ctrl-C stops both children. The memory allowance is conservative, not a measured peak. See [`variant-flash/bench/README.md`](../../../variant-flash/bench/README.md).

Default engine: `127.0.0.1:8742`. Default API: `127.0.0.1:8733`. Model id: `chlorine-qwen3.8-flash-next`. Serial greedy qualification precedes any drafting qualification. The API still uses the checkpoint's sampling defaults unless temperature is explicitly zero. Vision input is rejected in this initial text release.

`CAPS` returns `C` followed by versioned JSON (protocol `chlorine-flash-engine`, version 1), with context, chunk, slots, KV precision, drafter, checkpoint/binary identity from the supervisor, PLE semantics and explicit pending qualification. Historical `INFO`, `GEN`, `T`, `D`, `X` and `MEM` remain available. `/health` validates the engine's capability reply and reports its identity and PLE mode, with readiness separate from release qualification. An absent/malformed reply produces 503 with unknown identity, not a guessed mode. The Flash D grammar has twelve fields; do not parse it as the 27B ten-field variant.

F32 GGUF matrices now stay F32. Scalar rows use a fixed-tree FP32 GEMV; prefill uses rocBLAS SGEMM. Accumulation order can differ between these operations, so this is not a serial-versus-batch equivalence claim. `GDEC_GR_BF16` is rejected with GGUF until its F32-input path is qualified. Other upstream experimental modes remain opt-in; the supervisor clears them. IQ3_S WMMA rounds operands to f16 and accumulates in f32; the host format decoder's bit equality does not prove WMMA output equality.

## Provenance and licenses

`upstream/` begins at gfx1151-engine commit `78a41cc87289dc759ace2f9c030ad5483b30673c`, then carries the local changes described above. `UPSTREAM.json` records the **original imported file hashes**, not current patched hashes. Benchmark/validation artifacts record current source-tree hashes. Historical speed statements in imported comments describe upstream experiments, not Chlorine Flash measurements.

The derivative engine retains AGPL-3.0 under [`upstream/LICENSE`](upstream/LICENSE). GGML IQ lookup/decoder reference material retains its MIT notice in [`quant/LICENSE.ggml`](quant/LICENSE.ggml); [`quant/SOURCES.json`](quant/SOURCES.json) pins the inspected source. See [`NOTICE.md`](NOTICE.md). The parent's MIT license does not relicense these components.
