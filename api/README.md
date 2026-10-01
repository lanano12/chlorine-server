# chlorine-server API

OpenAI-compatible front end for the chlorine engine: `serve_api.py` (one
file, Python 3 stdlib plus `regex` for the tokenizer). It renders the Qwen
chat template, tokenizes, speaks the token protocol of
`docs/halogen/WIRE-PROTOCOL.md` to the engine (one connection per request)
and turns the `T`/`D` replies into OpenAI JSON or SSE chunks. The engine
never sees text.

## Usage

```bash
# engine first (port 8740, see ../README.md), then:
python3 api/serve_api.py                      # 127.0.0.1:8731 -> engine 127.0.0.1:8740
python3 api/serve_api.py --port 8731 --engine-port 8740 --default-drafter 1 --threads 8
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--bind`, `--port` | `127.0.0.1`, `8731` | HTTP listen address |
| `--engine-host`, `--engine-port` | `127.0.0.1`, `8740` | the token engine |
| `--tokenizer` | `/ML_AI/LocalModels/halogen-27b/tokenizer` (env `CHLORINE_TOKENIZER`) | dir holding `tokenizer.json` |
| `--default-drafter` | `1` | 0 serial, 1 MTP, 2 DFlash2; used when a request names none |
| `--threads` | `8` | HTTP worker threads; engine requests are still one at a time |
| `--default-max-tokens`, `--max-tokens-cap` | `512`, `65536` | when `max_tokens` is absent / the hard cap |
| `--queue-timeout` | `7200` s | wait for the engine lock before a 503 |
| `--first-token-timeout`, `--token-timeout` | `1800` s, `300` s | engine silence budgets (as halogen) |
| `--verbose` | off | log every HTTP request |

At start the process logs its CPU affinity (`api: pid N cpu affinity 16 cpus
[0-15]`), the tokenizer load time and the engine `INFO` line (a busy or
absent engine only produces a warning; requests still try).

## Endpoints

- `GET /v1/models` - one model, `qwen3.8-27b`.
- `GET /health` - `{status, model, context, drafter_default, engine:{state, info}, queue, busy}`;
  `INFO` is probed under the engine lock with a 1 s wait, so a running
  generation reports `state: busy` instead of blocking.
- `POST /v1/chat/completions` - `messages`, `max_tokens` (or
  `max_completion_tokens`), `stream` (`stream_options` accepted; usage is
  always on the final chunk), `reasoning_effort` (`low`/`medium`/`xhigh`, the
  template's vocabulary; `minimal` maps to `low` and `high` to `xhigh`;
  omitted means `xhigh` as in the template), `enable_thinking`,
  `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`,
  `preserve_thinking`), `tools`, `tool_choice` (see *Tool calling*),
  `drafter`, `stop`, sampling fields below. Thinking is on by default; the
  model's `<think>...</think>` block is returned as `message.reasoning_content`
  and the rest as `message.content` (the newlines after `</think>` are
  dropped). `chat_template_kwargs: {"enable_thinking": false}` (or top-level
  `enable_thinking: false`, the halogen-bench shape) pre-closes the think block
  so everything is content.
- `POST /v1/completions` - `prompt` as raw text (no template) or as a list
  of token ids; eos is `<|endoftext|>` only. Same `stream`, `max_tokens`,
  `drafter`, `stop` and sampling fields.

Drafter selection: body field `drafter` (or `x-drafter`) or the `x-drafter`
header, as `0/1/2` or `serial/mtp/dflash2`; default `--default-drafter`. A
drafter the checkpoint cannot serve makes the engine answer `D error`, which
comes back as `400 {"error": {"code": "engine_rejected"}}`.

Sampling: `temperature` (default **0 = greedy**, no `SAMPLE` clause),
`top_p`, `top_k`, `min_p`, `seed` (generated and echoed under `sampling` when
absent; the engine RNG is stateless so a seed is mandatory on the wire),
`presence_penalty`/`frequency_penalty` (`PENALTY`), `logit_bias` (`BIAS`,
<= 20480 ids < 248320), `logprobs` (`LOGPROBS`; returned in the non-streaming
response and in the final SSE chunk). The last four need `temperature > 0`
and are refused with 400 otherwise, the same rule the engine applies. Floats
go on the wire as `%.9g`.

`finish_reason` is `stop` when the engine's `D` line says `stop`, when an eos
id arrives as a `T` line (never detokenized), or when a `stop` string
matched; `length` otherwise. `usage` = `prompt_tokens` (rendered ids),
`completion_tokens` (the `D` line's `n_gen`), `total_tokens`.

Streaming (`stream: true`) sends `text/event-stream` chunks
(`chat.completion.chunk` / `text_completion`), the first delta carrying
`role`, `delta.tool_calls` chunks for generated tool calls, then a final
chunk with `finish_reason`, `usage` and `timings`, then `data: [DONE]`. Headers are sent only when the first token or the `D` line
arrives, so an engine rejection on a streaming request is still a plain 400.
Detokenization is incremental and UTF-8 safe: a token that ends in an
incomplete multi-byte sequence is held until the next token completes it;
a partial `</think>` or `stop` string is held the same way.

Concurrency: HTTP threads queue on one engine lock (the engine decodes one
request at a time). `timings.queue_ms` is the wait.

## Timings and the ledger

Every response (and the final SSE chunk) carries `timings`:

| Field | Source |
| --- | --- |
| `engine_prefill_ms`, `engine_decode_ms` | the engine's `D` line |
| `engine_ms_total` | wall time in the API from sending `GEN` to receiving `D` (includes socket and engine overhead; prefill + decode is the engine's own view) |
| `tokenize_ms` | chat-template render + BPE encode |
| `detokenize_ms` | incremental decode + think/stop splitting, summed over tokens |
| `queue_ms` | wait for the engine lock |
| `first_token_ms` | request received -> first `T` line |
| `http_ms` | request received -> response serialized (non-streaming) / final chunk written (streaming) |
| `drafter`, `spec_rounds`, `spec_committed`, `n_cached` | the `D` line (`drafter` is the one the engine ran; `drafter_requested` is what the request asked for) |
| `decode_tok_s` | `completion_tokens / (engine_decode_ms / 1000)` |
| `engine_reason` | the raw `D` reason |

Sampled requests also get a top-level `sampling` object (temperature, top_p,
top_k, min_p, seed, penalties, bias count, logprobs).

Per request, stderr gets three lines:

```
gen req=12 kind=chat ids=23 max_tokens=256 eos=248046,248044 drafter=1 (mtp) sample=greedy
ledger req=12 prompt=23 gen=256 prefill_ms=512.0 decode_ms=12345.0 drafter=1 rounds=64 commit=256 tok_s=20.74
serve_api: mtp 256 tok in 12.345s = 20.74 t/s | 64 rounds, commit 4.00/round | prompt 23, prefill 0.512s
```

`ledger` is the record of this API: `req` is the wire request id, `prompt`/`gen`
the `D` line counts, `drafter`/`rounds`/`commit` the engine's speculative
stats, `tok_s = gen / (decode_ms / 1000)`. The `serve_api:` line is
halogen's shape, matched verbatim by the `LEDGER` regex in
`halogen-server/tools/bench-serving.py` (`commit` there is per round; the
optional `(N cached)` group appears when the `D` line reports `n_cached`);
its drafter *name* is the one the request asked for, because that script
aligns ledger lines with requests by name. Unlike halogen, a serial request
also emits the line (bench-serving consumes it by name, so nothing is left
over). The `gen` line echoes what went on the `GEN` line (drafter id, eos
ids, sampling clause) without the ids.

## Tool calling

OpenAI-style function calling, the shape the Pi coding agent sends
(`tools` with JSON-schema `parameters`, `tool_choice`, assistant
`tool_calls` and `tool` messages in the history, SSE with
`stream_options: {include_usage: true}`):

- **Request.** `tools: [{type: "function", function: {name, description,
  parameters}}]` is rendered into the system turn exactly as
  `chat_template.jinja` does it: `# Tools ... <tools>` with one JSON object
  per tool, `</tools>`, the template's call-format instructions, then the
  client's system content. `tool_choice`: `auto` (default) and `none` (no
  tools rendered, no parsing); `required` and `{type: "function", function:
  {name}}` are accepted best effort (the full list is rendered; the template
  has no way to force a call, so the model may still answer in text; an
  unknown name is a 400). `tools: []` renders nothing (the Pi shape when the
  history has tool calls but no tools are declared). Assistant messages with
  `tool_calls` render as the template's `<tool_call>\n<function=NAME>\n
  <parameter=K>\nV\n</parameter>...</function>\n</tool_call>` blocks
  (`function.arguments` as a JSON string, the OpenAI form, or as an object;
  string values verbatim, other values as JSON; a string that is not JSON is
  a 400) and `tool` messages as `<tool_response>` blocks inside one user turn
  per run of consecutive results. `parallel_tool_calls` is accepted and
  ignored.
- **Response.** The template teaches the model the XML form above, so the
  parser reads `<tool_call>` blocks in that form (parameter values get the
  JSON type their schema declares: `integer`, `number`, `boolean`, `object`,
  `array`, `null`; anything else, or a value that does not parse, stays a
  string) and also accepts the JSON form `<tool_call>{"name": ..,
  "arguments": {..}}</tool_call>`. Each block becomes
  `{id: "call_<n>", type: "function", function: {name, arguments: <JSON
  string>}}`; `message.tool_calls` in a non-streaming response (`content` is
  the text before the blocks, or `null` when there is none), `finish_reason`
  `tool_calls`. Reasoning still comes back as `reasoning_content`. A block
  that parses as neither form, or is never closed, is returned as plain
  content with its tags. Ids count up per process.
- **Streaming.** A block is buffered until `</tool_call>` arrives and then
  sent as one `delta.tool_calls: [{index, id, type, function: {name,
  arguments}}]` chunk, so content deltas never show the raw tags (a partial
  `<tool_call>` at a chunk boundary is held back like a partial `</think>`);
  arguments are not streamed incrementally. Whitespace between the text and
  the first block, between blocks and after the last one is dropped, as the
  template's own `trim` would on the next turn.
- **Parser scope.** The parser only runs when tools were rendered; without
  `tools` (or with `tool_choice: none`) a `<tool_call>` in the output is
  ordinary content, so requests without tools behave exactly as before.

`reasoning_effort` mapping: the template knows `xhigh`, `medium` and `low`
(and `medium` renders no instruction line). OpenAI clients also send
`minimal` and `high`, which map to `low` and `xhigh`; an omitted value is the
template's default `xhigh`; anything else is the template's own 400
(`Unexpected reasoning effort ...`).

## Chat template

`render_chat()` is a hand port of the checkpoint's `chat_template.jinja`,
byte-identical to a jinja2 rendering (`test_api.py` phase T, 28 cases when
jinja2 is importable) on single-turn, multi-turn, system/no-system, empty
system, list content, `enable_thinking`, `reasoning_effort`,
`preserve_thinking`, `tools`, assistant `tool_calls` and `tool` result
cases, including the template's `raise_exception` messages, which surface
as 400s. Covered: the reasoning-effort system line (`medium` has no text, so
no line unless the client sent a system message), system-message merging,
the tools system turn, user turns, assistant turns with their `<think>` block
(kept unless `preserve_thinking: false`, which strips it from turns before
the last user query) and their `<tool_call>` blocks, `<tool_response>`
turns for `tool` messages, the `<tool_response>` user-turn rule for the last
query, and the generation prompt (`<think>\n` or `<think>\n\n</think>\n\n`).
The template's `tojson` follows transformers (`json.dumps(ensure_ascii=False)`,
keys in request order, no HTML escaping), which is what vLLM and
transformers feed the model; plain jinja2's `tojson` would sort keys and
escape `<>&'`. Not covered: images/video (400).

## Tests

```bash
python3 api/test_api.py        # ~5 s, no GPU, ports 8742 (engine stub), 8731 (API), 8743 (recorder)
```

Phase T runs without an engine: `render_chat()` against
`chat_template.jinja` rendered by jinja2 with transformers' filter semantics
(skipped, with a note, when jinja2 is not importable; fixed expected strings
for the tools turns run regardless), sha256 digests of the no-tools renders
taken before tools support (byte-identity with the previous behaviour), the
`<tool_call>` parser (schema typing, JSON form, malformed blocks), the
streaming splitter fed 1/3/7 characters at a time, and the
`reasoning_effort` aliases.
Phase A compiles a CPU-only copy of `engine/src/{main,hgn,serve}.cpp` with a
trunk shim (cached under `$TMPDIR/chlorine-api-stub-<uid>`, or set
`CHLORINE_STUB_BIN` to a built `chlorine` to run it with `CHLORINE_STUB=1`),
feeds it a fabricated `.hgn` header that advertises only the MTP drafter, and
checks `/v1/models`, `/health`, chat and completions (non-streaming and SSE),
usage, timings, the think split, drafter selection on the `GEN` line, the
`D error` -> 400 path, malformed requests, the Pi request shape (`tools`,
`tool_choice`, tool-call history, `max_completion_tokens`, `stream_options`),
the effort aliases and 6 parallel requests. Phase B swaps in a recorder
engine to check the exact `GEN` line with `SAMPLE`/`PENALTY`/`BIAS`/`LOGPROBS`,
logprob `T` lines, an eos `T` line, speculative stats, stop strings, split
UTF-8, generated tool calls (two blocks after reasoning and text,
non-streaming and SSE, tags split into single-byte tokens, a malformed JSON
block, the JSON form, an unterminated block at `max_tokens`, a stop string
before a block, and the same tokens without `tools`), and engine failures.
The stub cannot emit a tool call by itself (its output is a fixed arithmetic
sequence), so the recorder's scripted `T` lines are the test seam. Both
engines are stopped with `QUIT`.

## Limits

- No cancel: the engine handles one connection at a time and ignores `X`, so
  a client disconnect or a `stop` string only stops the output; the engine
  runs to `max_tokens`. Prefer `max_tokens` for bounded runs.
- `n > 1`, images and prompt lists (more than one prompt) are refused with
  400. Any `model` value is served as `qwen3.8-27b`.
- Tool calls: `tool_choice: required` / a named function cannot be enforced
  (the template has no forced-call mode); arguments arrive in one chunk per
  call, not token by token; a `</parameter>` inside a parameter value ends
  the value (as in vLLM's parser); `type: "custom"` (grammar) tools are
  refused with 400.
- `logprobs` are not attached per chunk while streaming (final chunk only).
- SIGTERM/SIGINT stop the process at once; in-flight requests are cut, the
  engine finishes its current `GEN` on its own.
- The sampling clauses (`SAMPLE`, `PENALTY`, `BIAS`, `LOGPROBS`) are checked
  for wire format against the spec and the engine parser, not for effect:
  the stub ignores them.
