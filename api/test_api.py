#!/usr/bin/env python3
"""test_api.py - end-to-end tests for serve_api.py.  No GPU, no real engine.

    python3 api/test_api.py

Phase T needs no engine: render_chat against chat_template.jinja through
jinja2 (transformers' filter semantics; skipped without jinja2) plus fixed
expected strings, pre-change digests for the no-tools cases, the tool-call
parser/splitter and the reasoning_effort aliases.
Phase A talks to the engine's own deterministic stub (CHLORINE_STUB=1): a
CPU-only copy of engine/src/{main,hgn,serve}.cpp plus a trunk shim is compiled
into a cache dir (set CHLORINE_STUB_BIN to use an existing chlorine binary in
stub mode instead) and served a fabricated .hgn header that advertises the MTP
drafter only, so drafter 2 is rejected with a real engine `D error`.
Phase B swaps in a 60-line recorder engine (this file) that captures the exact
GEN line and replays scripted T/D lines: sampling clauses, the think split,
stop strings, split UTF-8, an eos delivered as a T line and generated
<tool_call> blocks (non-streaming, SSE, byte-split tags, malformed blocks).

Ports: stub 8742, API 8731, recorder 8743 (env STUB_PORT / API_PORT / FAKE_PORT).
Never 8740.  Everything started here is stopped at the end (QUIT for the engine).
"""
import json
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_api  # noqa: E402  (tokenizer, render_chat, build_gen_line)

REPO = HERE.parent
ENGINE_SRC = REPO / 'engine' / 'src'
STUB_PORT = int(os.environ.get('STUB_PORT', 8742))
API_PORT = int(os.environ.get('API_PORT', 8731))
FAKE_PORT = int(os.environ.get('FAKE_PORT', 8743))
API = f'http://127.0.0.1:{API_PORT}'
FORBIDDEN_PORT = 8740
IM_END, EOT, THINK_END_ID = 248046, 248044, 248069
TOOL_CALL_ID, TOOL_CALL_END_ID = 248058, 248059
# halogen-server/tools/bench-serving.py LEDGER regex, verbatim
HALOGEN_LEDGER = re.compile(
    r"serve_api: (\w+) (\d+) tok in ([\d.]+)s = ([\d.]+) t/s \| "
    r"(\d+) rounds, commit ([\d.]+)/round \| prompt (\d+)"
    r"(?: \((\d+) cached\))?, prefill ([\d.]+)s")
LEDGER = re.compile(r"ledger req=(\d+) prompt=(\d+) gen=(\d+) prefill_ms=([\d.]+) decode_ms=([\d.]+) "
                    r"drafter=(\d+) rounds=(\d+) commit=(\d+) tok_s=([\d.]+)")
GEN_ECHO = re.compile(r"gen req=(\d+) kind=(\w+) ids=(\d+) max_tokens=(\d+) eos=([\d,]+) drafter=(\d) \((\w+)\)")
TIMING_FIELDS = ['engine_prefill_ms', 'engine_decode_ms', 'engine_ms_total', 'tokenize_ms', 'detokenize_ms',
                 'http_ms', 'drafter', 'spec_rounds', 'spec_committed', 'queue_ms', 'first_token_ms']

PASS, FAIL = 0, 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  ok    {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}  {str(detail)[:600]}')
    return cond


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
TRUNK_SHIM = '''// Test-only shim: satisfies main.cpp's chlorine_trunk_* references so the
// engine's wire-protocol stub (CHLORINE_STUB=1) links without HIP. Never called.
extern "C" {
int chlorine_trunk_init(const char*) { return -1; }
int chlorine_trunk_generate2() { return -1; }
int chlorine_trunk_generate3() { return -1; }
int chlorine_trunk_score() { return -1; }
int chlorine_trunk_score_decode() { return -1; }
int chlorine_trunk_cstat() { return 0; }
int chlorine_trunk_ctx_cap(void) { return 262144; }
int chlorine_trunk_tmax(void) { return 0; }
void chlorine_trunk_shutdown(void) {}
}
'''


def build_stub_engine():
    override = os.environ.get('CHLORINE_STUB_BIN')
    if override:
        return Path(override)
    bdir = Path(os.environ.get('CHLORINE_STUB_BUILD', Path(tempfile.gettempdir()) / f'chlorine-api-stub-{os.getuid()}'))
    bdir.mkdir(parents=True, exist_ok=True)
    srcs = [ENGINE_SRC / n for n in ('main.cpp', 'hgn.cpp', 'serve.cpp')]
    deps = srcs + [ENGINE_SRC / 'serve.h', ENGINE_SRC / 'hgn.h']
    out = bdir / 'chlorine-stub'
    shim = bdir / 'trunk_shim.cpp'
    shim_fresh = shim.exists() and shim.read_text() == TRUNK_SHIM
    if shim_fresh and out.exists() and out.stat().st_mtime > max(p.stat().st_mtime for p in deps):
        return out
    shim.write_text(TRUNK_SHIM)
    t0 = time.time()
    cmd = ['g++', '-std=c++20', '-O1', '-o', str(out)] + [str(s) for s in srcs] + [str(shim),
           '-Wl,--unresolved-symbols=ignore-all', '-Wl,-z,lazy']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f'stub engine build failed:\n{r.stderr}')
    print(f'built {out} in {time.time() - t0:.1f} s')
    return out


def write_fake_checkpoint(path, tensors=('mtp.fc.weight',)):
    """Minimal .hgn: header (0x68 bytes) + 0xA0-byte table entries, no data.
    Only mtp.fc.weight -> has_mtp, no drafter.fc.weight -> drafter 2 rejected."""
    n = len(tensors)
    size = 0x68 + 0xA0 * n
    b = bytearray(size)
    struct.pack_into('<I', b, 0, 0x314E4748)
    struct.pack_into('<I', b, 4, 1)
    struct.pack_into('<Q', b, 8, n)
    struct.pack_into('<Q', b, 0x10, 0x68)
    struct.pack_into('<Q', b, 0x18, size)
    struct.pack_into('<Q', b, 0x20, size)
    b[0x28:0x28 + 9] = b'fake-stub'
    for i, name in enumerate(tensors):
        e = 0x68 + i * 0xA0
        b[e:e + len(name)] = name.encode()
    path.write_bytes(bytes(b))


def port_open(port):
    try:
        socket.create_connection(('127.0.0.1', port), timeout=0.5).close()
        return True
    except OSError:
        return False


def wait_port(port, proc, what, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            sys.exit(f'{what} exited early (rc {proc.returncode})')
        if port_open(port):
            return
        time.sleep(0.05)
    sys.exit(f'{what} did not open port {port} within {timeout} s')


def stub_tokens(last_id, n):
    """The stub's deterministic output: (last prompt id + 7919*(i+1)) % 248320."""
    return [(last_id + 7919 * (i + 1)) % 248320 for i in range(min(n, 8))]


def post(path, body, headers=None):
    req = urllib.request.Request(API + path, json.dumps(body).encode(),
                                 {'content-type': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {'raw': raw.decode('utf-8', 'replace')}


def post_raw(path, data, headers=None):
    req = urllib.request.Request(API + path, data, {'content-type': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(path):
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return r.status, json.loads(r.read())


def stream(path, body):
    """Returns (content_type, events, raw_lines): events are the parsed `data:` payloads, '[DONE]' kept as a string."""
    req = urllib.request.Request(API + path, json.dumps(body).encode(), {'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        ctype = r.headers.get('content-type', '')
        lines = [ln.decode('utf-8') for ln in r]
    events = []
    for ln in lines:
        if ln.startswith('data: '):
            payload = ln[6:].strip()
            events.append(payload if payload == '[DONE]' else json.loads(payload))
    return ctype, events, lines


def quit_engine(port):
    try:
        s = socket.create_connection(('127.0.0.1', port), timeout=5)
        s.sendall(b'QUIT\n')
        s.settimeout(5)
        reply = s.recv(64)
        s.close()
        return reply.strip() == b'BYE'
    except OSError:
        return False


def stop_proc(proc, what):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f'  note: {what} needed SIGKILL')


class FakeEngine(threading.Thread):
    """Recorder engine: answers PING/INFO/QUIT, records every GEN line and
    replies with whatever `self.reply(req, fields)` returns (a list of lines)."""

    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.gen_lines = []
        self.reply = lambda req, f: [f'D {req} length 1 0 0.0 0.0']
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('127.0.0.1', port))
        self.sock.listen(8)
        self.stopped = False

    def run(self):
        while not self.stopped:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            with c:
                buf = b''
                while b'\n' not in buf:
                    d = c.recv(1 << 20)
                    if not d:
                        break
                    buf += d
                if b'\n' not in buf:
                    continue
                line = buf.split(b'\n', 1)[0].decode()
                f = line.split()
                if not f:
                    continue
                if f[0] == 'PING':
                    c.sendall(b'PONG\n')
                elif f[0] == 'INFO':
                    c.sendall(b'I 1 1 262144 8 1 1 1 0 2048 1 262144\n')
                elif f[0] == 'QUIT':
                    c.sendall(b'BYE\n')
                    self.stopped = True
                elif f[0] == 'GEN':
                    self.gen_lines.append(line)
                    out = self.reply(int(f[1]), f)
                    if out is not None:
                        c.sendall(''.join(x + '\n' for x in out).encode())
        self.sock.close()


def read_log(path):
    return Path(path).read_text(errors='replace')


# ---------------------------------------------------------------------------
# phase T: chat template (tools branch) and tool-call parsing, no engine
# ---------------------------------------------------------------------------
TOOLS = [  # the Pi coding agent's shape: function definitions with JSON-schema parameters
    {'type': 'function', 'function': {'name': 'read', 'description': 'Read a file <b> & "q"',
                                      'parameters': {'type': 'object', 'properties': {
                                          'path': {'type': 'string', 'description': 'file path'},
                                          'offset': {'type': 'number'}, 'limit': {'type': 'number'}},
                                          'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'bash', 'description': 'Run a command',
                                      'parameters': {'type': 'object', 'properties': {
                                          'command': {'type': 'string'}, 'timeout': {'type': 'integer'}},
                                          'required': ['command']}}},
]
TOOLS_INSTRUCTIONS = (  # chat_template.jinja, verbatim
    "# Tools\n\nYou have access to the following functions:\n\n<tools>{TOOLS}\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide "
    "optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is "
    "no function call available, answer the question like normal with your current knowledge and do not tell the user "
    "about function calls\n</IMPORTANT>")
# sha256[:16] of render_chat() before tools support, per no-tools case: the byte-identity fixture
PRE_TOOLS_DIGESTS = {
    'single user, defaults': '9c7135d227f1fa32', 'system + low': 'e87401fae3cf84fb',
    'system + medium (no effort line)': 'bde9c33f8012b959', 'no system + medium (no system block)': 'a5b037bd82a31283',
    'empty system + xhigh': '9c7135d227f1fa32', 'empty system + medium': 'a5b037bd82a31283',
    'list content': 'b3d12ac382d7e0b1', 'thinking off': 'aa485751c513179d', 'thinking off + low': '2df911111668774e',
    'multi-turn, preserve default': 'bf6ed47ffaa32de9', 'multi-turn, preserve False': '553d0d2ad780092f',
    'multi-turn, preserve True, thinking off': 'd0244e3b285f079f',
    'assistant after last user, preserve False': 'f6411ebe62d04b45', 'assistant None content/reasoning': 'e19c5e02560d6795',
    'tool_response-wrapped user is not the last query': '3713b3bceb15b51c', 'unicode': '1d2d13882e60a5ec',
}


def template_cases():
    """(name, messages, render kwargs).  The first group renders no tools (the
    pre-change matrix, plus `tool` turns that used to be refused); the second
    group exercises the tools branch: system tool block, assistant tool_calls
    (dict and JSON-string arguments), consecutive tool results."""
    u = {'role': 'user', 'content': 'What is 2+2?'}
    multi = [{'role': 'system', 'content': 'Be brief.'}, u,
             {'role': 'assistant', 'content': '4', 'reasoning_content': 'Simple arithmetic.'},
             {'role': 'user', 'content': 'And times 3?'},
             {'role': 'assistant', 'content': '12', 'reasoning_content': ' more \n'},
             {'role': 'user', 'content': 'Thanks.'}]
    args = {'path': 'foo.py', 'limit': 20, 'flag': True, 'n': None, 'obj': {'a': [1, 'ü']}, 'f': 1.5, 's': 'multi\nline'}
    call = {'id': 'call_1', 'type': 'function', 'function': {'name': 'read', 'arguments': args}}
    call_str = {'id': 'call_1', 'type': 'function', 'function': {'name': 'read', 'arguments': json.dumps(args)}}
    return [
        ('single user, defaults', [u], {}),
        ('system + low', [{'role': 'system', 'content': ' Be brief. '}, u], {'reasoning_effort': 'low'}),
        ('system + medium (no effort line)', [{'role': 'system', 'content': 'Be brief.'}, u], {'reasoning_effort': 'medium'}),
        ('no system + medium (no system block)', [u], {'reasoning_effort': 'medium'}),
        ('empty system + xhigh', [{'role': 'system', 'content': ''}, u], {'reasoning_effort': 'xhigh'}),
        ('empty system + medium', [{'role': 'system', 'content': '  '}, u], {'reasoning_effort': 'medium'}),
        ('list content', [{'role': 'system', 'content': [{'type': 'text', 'text': 'Sys '}, {'text': 'two'}]},
                          {'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': ' b '}]}], {}),
        ('thinking off', [u], {'enable_thinking': False}),
        ('thinking off + low', [{'role': 'system', 'content': 'S'}, u], {'enable_thinking': False, 'reasoning_effort': 'low'}),
        ('multi-turn, preserve default', multi, {}),
        ('multi-turn, preserve False', multi, {'preserve_thinking': False}),
        ('multi-turn, preserve True, thinking off', multi, {'preserve_thinking': True, 'enable_thinking': False}),
        ('assistant after last user, preserve False',
         multi + [{'role': 'assistant', 'content': 'Welcome.', 'reasoning_content': 'polite'}], {'preserve_thinking': False}),
        ('assistant None content/reasoning', [u, {'role': 'assistant', 'content': None, 'reasoning_content': None}, u], {}),
        ('tool_response-wrapped user is not the last query',
         [u, {'role': 'assistant', 'content': 'x', 'reasoning_content': 'r'},
          {'role': 'user', 'content': '<tool_response>\nout\n</tool_response>'}], {'preserve_thinking': False}),
        ('unicode', [{'role': 'system', 'content': 'Ünïcode 中文 <b> & "q"'}, {'role': 'user', 'content': 'é́ 　'}], {}),
        ('tool role, no tools', [u, {'role': 'tool', 'content': 'r'}, {'role': 'user', 'content': 'again'}], {}),
        ('tool role first', [{'role': 'tool', 'content': 'r'}, {'role': 'user', 'content': 'again'}], {}),
        ('two tool roles then user', [u, {'role': 'tool', 'content': 'r1'}, {'role': 'tool', 'content': [{'text': 'r2'}]},
                                      {'role': 'user', 'content': 'again'}], {}),
        # --- tools branch ---
        ('tools + system + low', [{'role': 'system', 'content': 'Be brief.'}, u], {'tools': TOOLS, 'reasoning_effort': 'low'}),
        ('tools, no system, medium', [u], {'tools': TOOLS, 'reasoning_effort': 'medium'}),
        ('tools, empty system, thinking off', [{'role': 'system', 'content': ' '}, u], {'tools': TOOLS, 'enable_thinking': False}),
        ('tools: [] renders none', [u], {'tools': []}),
        ('tool_calls (dict args) + two tool results + assistant + user',
         [u, {'role': 'assistant', 'content': '', 'reasoning_content': 'think', 'tool_calls': [call]},
          {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'line1\nline2'},
          {'role': 'tool', 'tool_call_id': 'call_2', 'content': 'second'},
          {'role': 'assistant', 'content': 'ok'}, u], {'tools': TOOLS}),
        ('tool_calls (JSON-string args, the OpenAI shape)',
         [u, {'role': 'assistant', 'content': '', 'reasoning_content': 'think', 'tool_calls': [call_str]},
          {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'line1\nline2'}], {'tools': TOOLS}),
        ('content + two calls, {} and "" arguments, thinking off',
         [u, {'role': 'assistant', 'content': 'Reading.',
              'tool_calls': [{'function': {'name': 'read', 'arguments': {}}}, {'name': 'bash', 'arguments': ''}]},
          {'role': 'tool', 'content': 'x'}], {'tools': TOOLS, 'enable_thinking': False}),
        ('preserve False with tool_calls',
         [u, {'role': 'assistant', 'content': None, 'reasoning_content': 'r',
              'tool_calls': [{'function': {'name': 'bash', 'arguments': {'command': 'ls'}}}]},
          {'role': 'tool', 'content': 'x'}, {'role': 'assistant', 'content': 'done', 'reasoning_content': 'r2'},
          {'role': 'user', 'content': 'next'}], {'tools': TOOLS, 'preserve_thinking': False}),
        ('tool_calls without tools rendered (Pi sends tools: [] with tool history)',
         [u, {'role': 'assistant', 'content': '', 'tool_calls': [call_str]}, {'role': 'tool', 'content': 'x'}], {'tools': []}),
    ]


def jinja_render():
    """chat_template.jinja through jinja2 with transformers' filter semantics
    (tojson = json.dumps(ensure_ascii=False), unsorted, no HTML escaping; that
    is what vLLM/transformers feed the model).  None without jinja2 or the file."""
    try:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        return None
    path = Path(serve_api.DEFAULT_TOKENIZER) / 'chat_template.jinja'
    if not path.exists():
        return None

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters['tojson'] = tojson
    env.globals['raise_exception'] = raise_exception
    tpl = env.from_string(path.read_text())

    def render(messages, **kw):
        kw = {k: v for k, v in kw.items() if v is not None}   # None -> undefined in the template
        if 'tools' in kw:   # the template sees dict arguments (vLLM parses the JSON string first)
            messages = json.loads(json.dumps(messages))
            for m in messages:
                for tc in (m.get('tool_calls') or []):
                    fn = tc.get('function', tc)
                    if isinstance(fn.get('arguments'), str) and fn['arguments']:
                        fn['arguments'] = json.loads(fn['arguments'])
        try:
            return tpl.render(add_generation_prompt=True, messages=messages, **kw)
        except jinja2.exceptions.TemplateError as e:
            return 'ERR ' + str(e)
    return render


def our_render(messages, **kw):
    try:
        return serve_api.render_chat(messages, **kw)
    except serve_api.ApiError as e:
        return 'ERR ' + e.message


def phase_t():
    import hashlib
    print('phase T: chat template and tool-call parsing (no engine)')
    render = jinja_render()
    cases = template_cases()
    if render is None:
        print('  note: jinja2 or chat_template.jinja unavailable; template cases use the fixed strings only')
    else:
        bad = [(n, our_render(m, **kw)[-300:], render(m, **kw)[-300:]) for n, m, kw in cases if our_render(m, **kw) != render(m, **kw)]
        check(f'template: {len(cases)} cases byte-identical to jinja2 (transformers tojson semantics)', not bad, bad[:2])
    bad = [(n, hashlib.sha256(our_render(m, **kw).encode()).hexdigest()[:16]) for n, m, kw in cases
           if n in PRE_TOOLS_DIGESTS and hashlib.sha256(our_render(m, **kw).encode()).hexdigest()[:16] != PRE_TOOLS_DIGESTS[n]]
    check(f'template: {len(PRE_TOOLS_DIGESTS)} no-tools renders byte-identical to before tools support', not bad, bad)
    u = {'role': 'user', 'content': 'hi'}
    want = ('<|im_start|>system\n' + TOOLS_INSTRUCTIONS.replace('{TOOLS}', ''.join('\n' + json.dumps(t) for t in TOOLS))
            + '<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n')
    check('template: tools system turn (fixed string, medium = no effort line)',
          our_render([u], tools=TOOLS, reasoning_effort='medium') == want, our_render([u], tools=TOOLS, reasoning_effort='medium')[:200])
    got = our_render([{'role': 'system', 'content': 'Be brief.'}, u], tools=TOOLS, reasoning_effort='low')
    check('template: tools + effort line + system content order',
          got.startswith('<|im_start|>system\n' + serve_api.EFFORT_TEXT['low'] + '\n\n# Tools\n')
          and got.endswith('\n</IMPORTANT>\n\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n'), got[-200:])
    hist = [{'role': 'user', 'content': 'read foo'},
            {'role': 'assistant', 'content': '', 'reasoning_content': 'think',
             'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path": "foo.py", "limit": 20}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'line1\nline2'},
            {'role': 'tool', 'tool_call_id': 'call_2', 'content': 'second'},
            {'role': 'user', 'content': 'thanks'}]
    tail = ('<|im_start|>user\nread foo<|im_end|>\n<|im_start|>assistant\n<think>\nthink\n</think>\n\n<tool_call>\n<function=read>\n'
            '<parameter=path>\nfoo.py\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call><|im_end|>\n'
            '<|im_start|>user\n<tool_response>\nline1\nline2\n</tool_response>\n<tool_response>\nsecond\n</tool_response><|im_end|>\n'
            '<|im_start|>user\nthanks<|im_end|>\n<|im_start|>assistant\n<think>\n')
    check('template: tool_calls + tool results turns (fixed string)', our_render(hist, tools=TOOLS).endswith(tail), our_render(hist, tools=TOOLS)[-500:])
    check('template: tool_calls render the same without tools (tools: [] / None)',
          our_render(hist, tools=[]) == our_render(hist) and our_render(hist).endswith(tail), our_render(hist)[-300:])
    for name, msgs, needle in [('non-JSON string arguments', [u, {'role': 'assistant', 'tool_calls': [{'function': {'name': 'read', 'arguments': '{oops'}}]}], 'not valid JSON'),
                               ('tool_call without a name', [u, {'role': 'assistant', 'tool_calls': [{'function': {}}]}], 'function name'),
                               ('array arguments', [u, {'role': 'assistant', 'tool_calls': [{'function': {'name': 'r', 'arguments': [1]}}]}], 'JSON object')]:
        got = our_render(msgs, tools=TOOLS)
        check(f'template: {name} -> 400', got.startswith('ERR') and needle in got, got)

    # tool-call parsing
    P = serve_api.parse_tool_call_block
    sch = serve_api.tool_schemas(TOOLS + [{'type': 'function', 'function': {'name': 't', 'parameters': {'type': 'object', 'properties': {
        'b': {'type': 'boolean'}, 'o': {'type': 'object'}, 'a': {'type': 'array'}, 'i': {'type': ['integer', 'null']}, 'n': {'type': 'number'}}}}}])
    got = P('\n<function=read>\n<parameter=path>\nfoo.py\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n', sch)
    check('parse: <function=> block -> name + JSON arguments, number typed by the schema',
          got == {'name': 'read', 'arguments': '{"path": "foo.py", "limit": 20}'}, got)
    got = P('<function=t>\n<parameter=b>\ntrue\n</parameter>\n<parameter=o>\n{"k": [1]}\n</parameter>\n<parameter=a>\n[1, "x"]\n</parameter>\n'
            '<parameter=i>\nnull\n</parameter>\n<parameter=n>\n1.5\n</parameter>\n<parameter=extra>\n7\n</parameter>\n</function>', sch)
    check('parse: boolean/object/array/null/number/unknown-param typing',
          got and json.loads(got['arguments']) == {'b': True, 'o': {'k': [1]}, 'a': [1, 'x'], 'i': None, 'n': 1.5, 'extra': '7'}, got)
    got = P('<function=read>\n<parameter=path>\nline1\n\nline3\n</parameter>\n<parameter=limit>\nabc\n</parameter>\n</function>', sch)
    check('parse: multi-line value kept, untypable number stays a string',
          got and json.loads(got['arguments']) == {'path': 'line1\n\nline3', 'limit': 'abc'}, got)
    got = P('\n{"name": "read", "arguments": {"path": "a.py", "limit": 2}}\n', sch)
    check('parse: JSON form accepted', got == {'name': 'read', 'arguments': '{"path": "a.py", "limit": 2}'}, got)
    check('parse: malformed JSON / no function tag -> None (falls back to content)',
          P('\n{"name": "read", "arguments": {oops}}\n', sch) is None and P('\njust text\n', sch) is None and P('{"arguments": {}}', sch) is None)

    def run(text, step):
        sp, out = serve_api.ToolCallSplitter(TOOLS), []
        for i in range(0, len(text), step):
            out += sp.feed(text[i:i + step])
        out += sp.flush()
        merged = []
        for k, v in out:
            if k == 'content' and merged and merged[-1][0] == 'content':
                merged[-1] = ('content', merged[-1][1] + v)
            else:
                merged.append((k, v))
        return merged
    text = ('I will read it.\n\n<tool_call>\n<function=read>\n<parameter=path>\nfoo.py\n</parameter>\n</function>\n</tool_call>\n'
            '<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>')
    want = [('content', 'I will read it.'), ('tool_call', {'name': 'read', 'arguments': '{"path": "foo.py"}'}),
            ('tool_call', {'name': 'bash', 'arguments': '{"command": "ls -la"}'})]
    check('splitter: text + two blocks, identical for 1/3/7/1000-char feeds (partial tags held back)',
          all(run(text, k) == want for k in (1, 3, 7, 1000)), run(text, 1))
    for name, text in [('text without tool calls is unchanged', 'a < b, <tool is fine, <tool_ca\n\n'),
                       ('malformed block passes through with its tags', 'x\n<tool_call>\nnope\n</tool_call>\nmore'),
                       ('unterminated block passes through', 'text\n<tool_call>\n<function=read>\n<parameter=path>\nx')]:
        check(f'splitter: {name}', all(run(text, k) == [('content', text)] for k in (1, 4, 1000)), run(text, 1))
    check('splitter: whitespace after the last block is dropped, text after it is content',
          run('<tool_call>\n<function=read>\n</function>\n</tool_call>\n\n', 2) == [('tool_call', {'name': 'read', 'arguments': '{}'})]
          and run('<tool_call>\n<function=read>\n</function>\n</tool_call>\n\nDone.', 2)[1] == ('content', 'Done.'))

    N = serve_api.normalize_effort
    check('reasoning_effort aliases: minimal->low, high->xhigh, others untouched',
          N('minimal') == 'low' and N('high') == 'xhigh' and [N(x) for x in ('low', 'medium', 'xhigh', 'ultra', None)] == ['low', 'medium', 'xhigh', 'ultra', None])


# ---------------------------------------------------------------------------
# phase A: the engine's stub
# ---------------------------------------------------------------------------
def phase_a(tok, api_log):
    print('phase A: engine stub')
    st, d = get('/v1/models')
    check('GET /v1/models', st == 200 and d['object'] == 'list' and [m['id'] for m in d['data']] == ['qwen3.8-27b'], d)
    st, d = get('/health')
    check('GET /health probes INFO', st == 200 and d['status'] == 'ok' and d['engine']['state'] == 'ok'
          and d['engine']['info']['ctx'] == 262144 and d['drafter_default'] == 'mtp' and d['context'] == 262144, d)

    # --- non-streaming chat ---------------------------------------------------
    msgs = [{'role': 'user', 'content': 'Say hello in one word.'}]
    ids = tok.encode(serve_api.render_chat(msgs, reasoning_effort='low'))
    expect = tok.decode(stub_tokens(ids[-1], 8))
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 8, 'reasoning_effort': 'low'})
    ok = check('chat: 200 chat.completion', st == 200 and d.get('object') == 'chat.completion', d)
    if ok:
        m = d['choices'][0]['message']
        u = d['usage']
        check('chat: usage prompt_tokens == rendered ids', u['prompt_tokens'] == len(ids), (u, len(ids)))
        check('chat: usage completion/total', u['completion_tokens'] == 8 and u['total_tokens'] == len(ids) + 8, u)
        check('chat: finish_reason length (stub never stops)', d['choices'][0]['finish_reason'] == 'length', d['choices'][0])
        check('chat: stub text lands in reasoning_content (open <think>)',
              m['role'] == 'assistant' and '</think>' not in expect and m.get('reasoning_content') == expect and m['content'] == '',
              (m, expect))
        t = d['timings']
        check('chat: timings fields present', all(k in t for k in TIMING_FIELDS), sorted(t))
        check('chat: timings values from the D line', t['engine_prefill_ms'] == 0.1 and t['engine_decode_ms'] == 0.2
              and t['drafter'] == 0 and t['spec_rounds'] == 0 and t['spec_committed'] == 0 and t['drafter_requested'] == 1, t)
        check('chat: http_ms > tokenize_ms and engine_ms_total > 0', t['http_ms'] > t['tokenize_ms'] >= 0 and t['engine_ms_total'] > 0, t)
        check('chat: id/model/created', d['id'].startswith('chatcmpl-') and d['model'] == 'qwen3.8-27b' and isinstance(d['created'], int), d)
    log = read_log(api_log)
    led = LEDGER.findall(log)
    check('ledger line format', bool(led) and led[-1][1] == str(len(ids)) and led[-1][2] == '8' and led[-1][5] == '0', led[-1:] or log[-500:])
    hal = HALOGEN_LEDGER.findall(log)
    check('halogen serve_api: line matches bench-serving.py regex', bool(hal) and hal[-1][0] == 'mtp' and hal[-1][1] == '8'
          and hal[-1][6] == str(len(ids)), hal[-1:] or log[-500:])
    gen = GEN_ECHO.findall(log)
    check('gen echo: default drafter 1 on the GEN line', bool(gen) and gen[-1][5] == '1' and gen[-1][6] == 'mtp'
          and gen[-1][4] == '248046,248044', gen[-1:])
    check('api logs cpu affinity at start', re.search(r'api: pid \d+ cpu affinity \d+ cpus \[', log) is not None, log[:300])

    # --- streaming chat -------------------------------------------------------
    ctype, ev, lines = stream('/v1/chat/completions', {'messages': msgs, 'max_tokens': 8, 'reasoning_effort': 'low', 'stream': True})
    check('stream: content-type text/event-stream', ctype.startswith('text/event-stream'), ctype)
    check('stream: ends with data: [DONE]', ev and ev[-1] == '[DONE]' and lines[-2:] == ['data: [DONE]\n', '\n'] or lines[-1] == 'data: [DONE]\n', lines[-3:])
    chunks = [e for e in ev if isinstance(e, dict)]
    check('stream: chunks are chat.completion.chunk', chunks and all(c['object'] == 'chat.completion.chunk' for c in chunks), chunks[:1])
    check('stream: first delta carries the role', chunks and chunks[0]['choices'][0]['delta'].get('role') == 'assistant', chunks[:1])
    reasoning = ''.join(c['choices'][0]['delta'].get('reasoning_content', '') for c in chunks)
    content = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks)
    check('stream: deltas concatenate to the non-streaming text', reasoning == expect and content == '', (reasoning, expect))
    last = chunks[-1] if chunks else {}
    check('stream: final chunk has finish_reason, usage, timings',
          last.get('choices', [{}])[0].get('finish_reason') == 'length' and last.get('usage', {}).get('completion_tokens') == 8
          and all(k in last.get('timings', {}) for k in TIMING_FIELDS), last)
    check('stream: SSE framing (blank line after every event)', all(lines[i + 1] == '\n' for i, ln in enumerate(lines[:-1]) if ln.startswith('data: ')), lines[:4])

    # --- thinking off, multi-turn ---------------------------------------------
    ids_off = tok.encode(serve_api.render_chat(msgs, enable_thinking=False))
    expect_off = tok.decode(stub_tokens(ids_off[-1], 8))
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 8, 'chat_template_kwargs': {'enable_thinking': False}})
    m = d['choices'][0]['message'] if st == 200 else {}
    check('thinking off: prompt ids match the closed-think render', st == 200 and d['usage']['prompt_tokens'] == len(ids_off)
          and ids_off != ids, (st, d.get('usage'), len(ids_off)))
    check('thinking off: everything is content', m.get('content') == expect_off and not m.get('reasoning_content'), (m, expect_off))
    st2, d2 = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 8, 'enable_thinking': False})
    check('thinking off: top-level enable_thinking (halogen-bench shape)', st2 == 200 and d2['usage']['prompt_tokens'] == len(ids_off), d2.get('usage'))
    multi = [{'role': 'system', 'content': 'Be brief.'},
             {'role': 'user', 'content': 'What is 2+2?'},
             {'role': 'assistant', 'content': '4', 'reasoning_content': 'Simple arithmetic.'},
             {'role': 'user', 'content': 'And times 3?'}]
    ids_multi = tok.encode(serve_api.render_chat(multi, reasoning_effort='xhigh'))
    st, d = post('/v1/chat/completions', {'messages': multi, 'max_tokens': 4})
    check('multi-turn: system/user/assistant render (xhigh default)', st == 200 and d['usage']['prompt_tokens'] == len(ids_multi)
          and len(ids_multi) > len(ids), (st, d.get('usage'), len(ids_multi)))

    # --- tools and reasoning_effort aliases (the Pi agent's request shape) ------------
    ids_tools = tok.encode(serve_api.render_chat(msgs, tools=TOOLS))
    pi_body = {'messages': msgs, 'max_completion_tokens': 4, 'tools': TOOLS, 'tool_choice': 'auto', 'temperature': 0,
               'stream_options': {'include_usage': True}, 'reasoning_effort': 'xhigh'}
    st, d = post('/v1/chat/completions', pi_body)
    check('tools: accepted, prompt is the tools render', st == 200 and d['usage']['prompt_tokens'] == len(ids_tools)
          and len(ids_tools) > len(ids) + 100, (st, d.get('usage'), len(ids_tools)))
    gen = GEN_ECHO.findall(read_log(api_log))
    check('tools: gen line echoes tools=2', gen and 'tools=2' in read_log(api_log).rsplit('gen req=', 1)[-1].split('\n')[0], read_log(api_log)[-300:])
    ids_x = tok.encode(serve_api.render_chat(msgs))
    st, d = post('/v1/chat/completions', {**pi_body, 'tool_choice': 'none'})
    check('tools: tool_choice none renders no tools', st == 200 and d['usage']['prompt_tokens'] == len(ids_x), (st, d.get('usage'), len(ids_x)))
    for label, choice in [('required', 'required'), ('named function', {'type': 'function', 'function': {'name': 'read'}})]:
        st, d = post('/v1/chat/completions', {**pi_body, 'tool_choice': choice})
        check(f'tools: tool_choice {label} accepted (best effort, full tool list)', st == 200 and d['usage']['prompt_tokens'] == len(ids_tools), (st, d.get('usage')))
    hist = [{'role': 'system', 'content': 'You are a coding agent.'}, {'role': 'user', 'content': 'read foo.py'},
            {'role': 'assistant', 'content': '', 'reasoning_content': 'Need the file.',
             'tool_calls': [{'id': 'call_7', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path": "foo.py"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_7', 'content': 'print(1)\n'}]
    ids_hist = tok.encode(serve_api.render_chat(hist, tools=TOOLS))
    st, d = post('/v1/chat/completions', {'messages': hist, 'max_tokens': 4, 'tools': TOOLS})
    check('tools: assistant tool_calls + tool result history accepted', st == 200 and d['usage']['prompt_tokens'] == len(ids_hist), (st, d.get('usage'), len(ids_hist)))
    st, d = post('/v1/chat/completions', {'messages': hist, 'max_tokens': 4, 'tools': []})
    check('tools: tools [] with tool history (Pi shape) renders the calls without a tool block',
          st == 200 and d['usage']['prompt_tokens'] == len(tok.encode(serve_api.render_chat(hist))) < len(ids_hist), (st, d.get('usage')))
    ids_low = tok.encode(serve_api.render_chat(msgs, reasoning_effort='low'))
    for label, body, want in [('minimal -> low', {'reasoning_effort': 'minimal'}, ids_low),
                              ('high -> xhigh', {'reasoning_effort': 'high'}, ids_x),
                              ('omitted -> xhigh', {}, ids_x),
                              ('chat_template_kwargs high -> xhigh', {'chat_template_kwargs': {'reasoning_effort': 'high'}}, ids_x),
                              ('low unchanged', {'reasoning_effort': 'low'}, ids_low)]:
        st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 2, **body})
        check(f'reasoning_effort {label}', st == 200 and d['usage']['prompt_tokens'] == len(want), (st, d.get('usage'), len(want)))

    # --- completions ------------------------------------------------------------
    pids = tok.encode('The quick brown fox')
    st, d = post('/v1/completions', {'prompt': 'The quick brown fox', 'max_tokens': 5})
    check('completions: text_completion with the stub text', st == 200 and d['object'] == 'text_completion'
          and d['choices'][0]['text'] == tok.decode(stub_tokens(pids[-1], 5)) and d['usage'] == {'prompt_tokens': len(pids), 'completion_tokens': 5, 'total_tokens': len(pids) + 5}
          and d['choices'][0]['finish_reason'] == 'length' and 'timings' in d, d)
    gen = GEN_ECHO.findall(read_log(api_log))
    check('completions: eos is <|endoftext|> only', gen and gen[-1][1] == 'completions' and gen[-1][4] == '248044', gen[-1:])
    # prompt as ids ending in 240125: the stub's first token is 240125+7919 = 248044 = eos -> `D stop` (6-field form)
    st, d = post('/v1/completions', {'prompt': [1000, 2000, 240125], 'max_tokens': 4})
    check('completions: id-list prompt, engine `stop` -> finish_reason stop', st == 200 and d['choices'][0]['finish_reason'] == 'stop'
          and d['usage']['completion_tokens'] == 0 and d['choices'][0]['text'] == '' and d['usage']['prompt_tokens'] == 3, d)
    ctype, ev, lines = stream('/v1/completions', {'prompt': 'The quick brown fox', 'max_tokens': 5, 'stream': True})
    chunks = [e for e in ev if isinstance(e, dict)]
    check('completions stream: text deltas + [DONE]', ev and ev[-1] == '[DONE]' and all(c['object'] == 'text_completion' for c in chunks)
          and ''.join(c['choices'][0]['text'] for c in chunks) == tok.decode(stub_tokens(pids[-1], 5)), (ev[:2], ev[-1:]))

    # --- drafter selection ----------------------------------------------------
    for label, body, hdr, want in [('body drafter=0', {'drafter': 0}, None, '0'),
                                   ('body drafter="mtp"', {'drafter': 'mtp'}, None, '1'),
                                   ('header x-drafter: serial', {}, {'x-drafter': 'serial'}, '0'),
                                   ('body x-drafter=1', {'x-drafter': 1}, None, '1')]:
        st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 2, **body}, hdr)
        gen = GEN_ECHO.findall(read_log(api_log))
        check(f'drafter: {label} -> GEN drafter {want}', st == 200 and gen and gen[-1][5] == want and str(d['timings']['drafter_requested']) == want,
              (st, gen[-1:]))
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 2, 'drafter': 2})
    check('drafter 2: engine D error -> 400 engine_rejected', st == 400 and d['error']['code'] == 'engine_rejected', (st, d))
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 2, 'drafter': 'turbo'})
    check('drafter "turbo" -> 400', st == 400 and d['error']['param'] == 'drafter', (st, d))

    # --- bad requests ---------------------------------------------------------
    st, d = post_raw('/v1/chat/completions', b'{"messages": [')
    check('bad JSON -> 400', st == 400 and 'JSON' in d['error']['message'], (st, d))
    cases = [('no messages', {'max_tokens': 2}, 'No messages provided.'),
             ('empty messages', {'messages': []}, 'No messages provided.'),
             ('system not first', {'messages': [msgs[0], {'role': 'system', 'content': 'x'}]}, 'System message must be at the beginning.'),
             ('unknown role', {'messages': msgs + [{'role': 'robot', 'content': 'x'}]}, 'Unexpected message role.'),
             ('assistant only', {'messages': [{'role': 'assistant', 'content': 'x'}]}, 'No user query found in messages.'),
             ('bad reasoning_effort', {'messages': msgs, 'reasoning_effort': 'ultra'}, 'Unexpected reasoning effort ultra.'),
             ('n=2', {'messages': msgs, 'n': 2}, 'n > 1'),
             ('max_tokens 0', {'messages': msgs, 'max_tokens': 0}, 'max_tokens'),
             ('penalty without temperature', {'messages': msgs, 'presence_penalty': 0.5}, 'temperature > 0'),
             ('logprobs without temperature', {'messages': msgs, 'logprobs': True}, 'temperature > 0'),
             ('tools not a list', {'messages': msgs, 'tools': {'name': 'f'}}, 'tools must be a list'),
             ('tool without a name', {'messages': msgs, 'tools': [{'type': 'function', 'function': {}}]}, 'tools entries'),
             ('non-function tool', {'messages': msgs, 'tools': [{'type': 'custom', 'custom': {'name': 'g'}}]}, 'tools entries'),
             ('tool_choice value', {'messages': msgs, 'tools': TOOLS, 'tool_choice': 'always'}, 'tool_choice must be'),
             ('tool_choice unknown function', {'messages': msgs, 'tools': TOOLS, 'tool_choice': {'type': 'function', 'function': {'name': 'nope'}}}, 'not in tools'),
             ('tool_calls with bad JSON arguments', {'messages': msgs + [{'role': 'assistant', 'tool_calls': [{'function': {'name': 'read', 'arguments': '{'}}]}, msgs[0]]}, 'not valid JSON'),
             ('image content', {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'x'}}]}]}, 'text only'),
             ('bad seed', {'messages': msgs, 'temperature': 0.5, 'seed': -1}, 'seed')]
    for name, body, needle in cases:
        st, d = post('/v1/chat/completions', body)
        check(f'400: {name}', st == 400 and needle in d.get('error', {}).get('message', ''), (st, d))
    st, d = post('/v1/completions', {'prompt': ['a', 'b']})
    check('400: completions with two prompts', st == 400, (st, d))
    st, d = post('/v1/nope', {})
    check('404: unknown route (OpenAI error envelope)', st == 404 and set(d['error']) == {'message', 'type', 'param', 'code'}, (st, d))
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 2, 'stop': ['x', 'y', 'z', 'w', 'v']})
    check('400: more than 4 stop strings', st == 400, (st, d))

    # --- concurrency: parallel requests are serialized on the engine ---------------
    results = []
    def worker(i):
        results.append(post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': f'req {i}'}], 'max_tokens': 8}))
    before = len(LEDGER.findall(read_log(api_log)))
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    after = len(LEDGER.findall(read_log(api_log)))
    check('concurrency: 6 parallel requests all 200 with 6 ledger lines', all(s == 200 for s, _ in results) and after - before == 6,
          ([s for s, _ in results], after - before))
    check('concurrency: every response has queue_ms', all('queue_ms' in d['timings'] for _, d in results), results[:1])


# ---------------------------------------------------------------------------
# phase B: recorder engine
# ---------------------------------------------------------------------------
def phase_b(tok, fake, api_log):
    print('phase B: recorder engine')
    msgs = [{'role': 'user', 'content': 'What is 2+2?'}]
    ids = tok.encode(serve_api.render_chat(msgs))
    answer_ids = tok.encode('Thinking.\n</think>\n\nFour.')
    check('recorder: </think> is one added token', THINK_END_ID in answer_ids, answer_ids)

    # B1: sampling clause + logprobs + eos as a T line + speculative stats
    def reply(req, f):
        out = [f'T {req} {t} {-0.25 * (i + 1):.9g}' for i, t in enumerate(answer_ids)]
        out.append(f'T {req} {IM_END} -0.01')
        out.append(f'D {req} stop {len(ids)} {len(answer_ids) + 1} 12.5 100.0 1 3 4')
        return out
    fake.reply = reply
    body = {'messages': msgs, 'max_tokens': 64, 'temperature': 0.7, 'top_p': 0.9, 'top_k': 40, 'min_p': 0.05, 'seed': 42,
            'presence_penalty': 0.5, 'frequency_penalty': 0.25, 'logit_bias': {'123': -1.5, '456': 2}, 'logprobs': True,
            'drafter': 'mtp'}
    st, d = post('/v1/chat/completions', body)
    ok = check('B1: sampled request 200', st == 200, (st, d))
    if ok:
        req = int(LEDGER.findall(read_log(api_log))[-1][0])
        want = serve_api.build_gen_line(req, 64, [IM_END, EOT], ids, 1, serve_api.Sampling(0.7, 40, 0.9, 0.05, 42, 0.5, 0.25, [(123, -1.5), (456, 2.0)], True)).strip()
        tail = ' '.join(want.split()[-17:])
        check('B1: GEN line exactly as WIRE-PROTOCOL.md (SAMPLE/PENALTY/BIAS/LOGPROBS)', fake.gen_lines[-1] == want
              and tail == '1 SAMPLE 0.7 40 0.9 0.05 42 PENALTY 0.5 0.25 BIAS 2 123 -1.5 456 2 LOGPROBS', (fake.gen_lines[-1][-120:], tail))
        m = d['choices'][0]['message']
        check('B1: think split reasoning/content', m['reasoning_content'] == 'Thinking.\n' and m['content'] == 'Four.', m)
        check('B1: eos T line -> finish_reason stop, not detokenized', d['choices'][0]['finish_reason'] == 'stop' and '<|im_end|>' not in m['content'], d['choices'][0])
        t = d['timings']
        check('B1: speculative stats from the D line', t['drafter'] == 1 and t['spec_rounds'] == 3 and t['spec_committed'] == 4
              and t['engine_prefill_ms'] == 12.5 and t['engine_decode_ms'] == 100.0 and t['engine_reason'] == 'stop', t)
        check('B1: usage.completion_tokens from the D line', d['usage']['completion_tokens'] == len(answer_ids) + 1, d['usage'])
        check('B1: decode_tok_s = gen / decode_s', abs(t['decode_tok_s'] - (len(answer_ids) + 1) / 0.1) < 1e-6, t['decode_tok_s'])
        lp = d['choices'][0]['logprobs']
        check('B1: logprobs parsed from T lines', lp and len(lp['content']) == len(answer_ids) and lp['content'][0]['logprob'] == -0.25
              and lp['content'][0]['token'] == tok.decode([answer_ids[0]]), lp['content'][:1] if lp else lp)
        check('B1: sampling echoed with the seed', d.get('sampling', {}).get('seed') == 42 and d['sampling']['temperature'] == 0.7, d.get('sampling'))
        log = read_log(api_log)
        led = LEDGER.findall(log)[-1]
        check('B1: ledger drafter/rounds/commit/tok_s', led[5:] == ('1', '3', '4', f'{(len(answer_ids) + 1) / 0.1:.2f}'), led)
        hal = HALOGEN_LEDGER.findall(log)[-1]
        check('B1: halogen line: name mtp, 3 rounds, commit 1.33/round, prefill 0.013s', hal[0] == 'mtp' and hal[4] == '3' and hal[5] == '1.33'
              and hal[8] == '0.013' and hal[2] == '0.100', hal)
    # seed omitted -> generated and echoed
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 8, 'temperature': 1.0})
    check('B1: omitted seed is generated and echoed', st == 200 and isinstance(d.get('sampling', {}).get('seed'), int)
          and f" {d['sampling']['seed']} " in fake.gen_lines[-1] + ' ', (st, d.get('sampling'), fake.gen_lines[-1][-60:]))

    # B2: streaming think split + partial '</think>' hold-back across tokens
    def reply2(req, f):
        return [f'T {req} {t}' for t in answer_ids] + [f'D {req} length {len(ids)} {len(answer_ids)} 1.0 2.0 1 2 2']
    fake.reply = reply2
    ctype, ev, lines = stream('/v1/chat/completions', {'messages': msgs, 'max_tokens': 64, 'stream': True})
    chunks = [e for e in ev if isinstance(e, dict)]
    reasoning = ''.join(c['choices'][0]['delta'].get('reasoning_content', '') for c in chunks)
    content = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks)
    check('B2: streamed think split', reasoning == 'Thinking.\n' and content == 'Four.' and ev[-1] == '[DONE]', (reasoning, content))
    check('B2: no chunk leaks </think> or leading newlines into content',
          all('</think>' not in c['choices'][0]['delta'].get('content', '') for c in chunks)
          and not any(c['choices'][0]['delta'].get('content', 'x').startswith('\n') for c in chunks), [c['choices'][0]['delta'] for c in chunks])
    check('B2: final chunk finish_reason length with spec stats', chunks[-1]['choices'][0]['finish_reason'] == 'length'
          and chunks[-1]['timings']['spec_rounds'] == 2, chunks[-1])
    check('B2: greedy request has no SAMPLE clause and ends with the drafter id', 'SAMPLE' not in fake.gen_lines[-1]
          and fake.gen_lines[-1].endswith(' 1'), fake.gen_lines[-1][-40:])

    # B3: stop strings (content only) and split UTF-8 bytes
    words = tok.encode('one two three four five')
    fake.reply = lambda req, f: [f'T {req} {t}' for t in words] + [f'D {req} length {len(ids)} {len(words)} 1.0 2.0']
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 64, 'stop': 'three', 'chat_template_kwargs': {'enable_thinking': False}})
    check('B3: stop string truncates content, finish_reason stop', st == 200 and d['choices'][0]['message']['content'] == 'one two '
          and d['choices'][0]['finish_reason'] == 'stop', d.get('choices'))
    euro = [tok.vocab[tok.b2u[b]] for b in '€'.encode('utf-8')]  # three single-byte tokens
    fake.reply = lambda req, f: [f'T {req} {t}' for t in euro + tok.encode(' ok')] + [f'D {req} length {len(ids)} 4 1.0 2.0']
    ctype, ev, lines = stream('/v1/chat/completions', {'messages': msgs, 'max_tokens': 8, 'stream': True, 'chat_template_kwargs': {'enable_thinking': False}})
    chunks = [e for e in ev if isinstance(e, dict)]
    pieces = [c['choices'][0]['delta'].get('content', '') for c in chunks]
    check('B3: split UTF-8 bytes are held until complete', ''.join(pieces) == '€ ok' and all('�' not in p for p in pieces), pieces)
    fake.reply = lambda req, f: [f'T {req} {t}' for t in euro[:2]] + [f'D {req} length {len(ids)} 2 1.0 2.0']
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 8, 'chat_template_kwargs': {'enable_thinking': False}})
    check('B3: truncated UTF-8 at the end flushes as U+FFFD', st == 200 and d['choices'][0]['message']['content'] == '�', d.get('choices'))

    # B5: generated tool calls, non-streaming: reasoning, text, then two <function=> blocks
    call_text = ('Thinking.\n</think>\n\nI will read it.\n\n<tool_call>\n<function=read>\n<parameter=path>\nfoo.py\n</parameter>\n'
                 '<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=bash>\n<parameter=command>\n'
                 'ls -la\n</parameter>\n</function>\n</tool_call>')
    call_ids = tok.encode(call_text)
    check('B5: <tool_call> and </tool_call> are single added tokens', TOOL_CALL_ID in call_ids and TOOL_CALL_END_ID in call_ids, call_ids[:8])
    ids_t = tok.encode(serve_api.render_chat(msgs, tools=TOOLS))
    fake.reply = lambda req, f: [f'T {req} {t}' for t in call_ids] + [f'T {req} {IM_END}', f'D {req} stop {len(ids_t)} {len(call_ids) + 1} 1.0 2.0 1 4 8']
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 256, 'tools': TOOLS})
    m = d['choices'][0]['message'] if st == 200 else {}
    tcs = m.get('tool_calls') or []
    check('B5: finish_reason tool_calls, two tool_calls', st == 200 and d['choices'][0]['finish_reason'] == 'tool_calls' and len(tcs) == 2, (st, d.get('choices')))
    check('B5: OpenAI shape: id call_<n>, type function, function.name, arguments a JSON string typed by the schema',
          tcs and all(re.fullmatch(r'call_\d+', tc['id']) and tc['type'] == 'function' and set(tc) == {'id', 'type', 'function'} for tc in tcs)
          and tcs[0]['id'] != tcs[1]['id'] and tcs[0]['function']['name'] == 'read'
          and json.loads(tcs[0]['function']['arguments']) == {'path': 'foo.py', 'limit': 20}
          and tcs[1]['function'] == {'name': 'bash', 'arguments': '{"command": "ls -la"}'}, tcs)
    check('B5: content is the text before the calls, reasoning_content intact, no raw tags',
          m.get('content') == 'I will read it.' and m.get('reasoning_content') == 'Thinking.\n', m)
    check('B5: usage/timings unchanged by tool parsing', d['usage']['completion_tokens'] == len(call_ids) + 1 and d['timings']['spec_rounds'] == 4, (d.get('usage'), d.get('timings')))
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 256})
    m = d['choices'][0]['message'] if st == 200 else {}
    check('B5: without tools the same output is plain content (parser off), finish stop',
          st == 200 and m.get('content') == call_text.split('</think>\n\n', 1)[1] and 'tool_calls' not in m and d['choices'][0]['finish_reason'] == 'stop', (st, m))

    # B6: the same, streamed: tool_calls deltas with index, content never shows the tags
    ctype, ev, lines = stream('/v1/chat/completions', {'messages': msgs, 'max_tokens': 256, 'tools': TOOLS, 'stream': True,
                                                        'stream_options': {'include_usage': True}})
    chunks = [e for e in ev if isinstance(e, dict)]
    deltas = [c['choices'][0]['delta'] for c in chunks]
    content = ''.join(x.get('content') or '' for x in deltas)
    reasoning = ''.join(x.get('reasoning_content') or '' for x in deltas)
    tcs = [tc for x in deltas for tc in x.get('tool_calls', [])]
    check('B6: streamed content == non-streaming content, no chunk carries a tag',
          content == 'I will read it.' and reasoning == 'Thinking.\n' and all('<tool_call' not in (x.get('content') or '') for x in deltas), (content, reasoning))
    check('B6: delta.tool_calls chunks with index 0 and 1, id, name and full arguments',
          [tc['index'] for tc in tcs] == [0, 1] and tcs[0]['function']['name'] == 'read' and re.fullmatch(r'call_\d+', tcs[0]['id'])
          and json.loads(tcs[0]['function']['arguments']) == {'path': 'foo.py', 'limit': 20} and tcs[1]['function']['name'] == 'bash', tcs)
    check('B6: tool_calls chunks are chat.completion.chunk with role only on the first delta',
          all(c['object'] == 'chat.completion.chunk' for c in chunks) and deltas[0].get('role') == 'assistant'
          and sum(1 for x in deltas if 'role' in x) == 1, deltas[:2])
    check('B6: final chunk finish_reason tool_calls with usage and timings, then [DONE]',
          chunks[-1]['choices'][0]['finish_reason'] == 'tool_calls' and chunks[-1]['usage']['completion_tokens'] == len(call_ids) + 1
          and 'timings' in chunks[-1] and ev[-1] == '[DONE]', chunks[-1])

    # B7: tags split into single-byte tokens are held back until complete (thinking off)
    byte_ids = lambda s: [tok.vocab[tok.b2u[b]] for b in s.encode('utf-8')]
    seq = tok.encode('Sure.\n') + byte_ids('<tool_call>') + tok.encode('\n<function=read>\n<parameter=path>\nx.py\n</parameter>\n</function>\n') + byte_ids('</tool_call>')
    fake.reply = lambda req, f: [f'T {req} {t}' for t in seq] + [f'D {req} stop {len(ids_t)} {len(seq)} 1.0 2.0']
    ctype, ev, lines = stream('/v1/chat/completions', {'messages': msgs, 'max_tokens': 256, 'tools': TOOLS, 'stream': True,
                                                        'chat_template_kwargs': {'enable_thinking': False}})
    deltas = [e['choices'][0]['delta'] for e in ev if isinstance(e, dict)]
    pieces = [x.get('content') or '' for x in deltas]
    tcs = [tc for x in deltas for tc in x.get('tool_calls', [])]
    check('B7: byte-split tags never leak, content "Sure." and one parsed call', ''.join(pieces) == 'Sure.' and all('<' not in p for p in pieces)
          and len(tcs) == 1 and tcs[0]['function'] == {'name': 'read', 'arguments': '{"path": "x.py"}'}, (pieces, tcs))

    # B8: malformed JSON block -> plain content; JSON form -> parsed; unterminated -> content + length
    bad = tok.encode('<tool_call>\n{"name": "read", "arguments": {oops}}\n</tool_call>')
    fake.reply = lambda req, f: [f'T {req} {t}' for t in bad] + [f'D {req} stop {len(ids_t)} {len(bad)} 1.0 2.0']
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 256, 'tools': TOOLS, 'chat_template_kwargs': {'enable_thinking': False}})
    m = d['choices'][0]['message']
    check('B8: malformed JSON block falls back to plain content, finish stop',
          m['content'] == tok.decode(bad) and 'tool_calls' not in m and d['choices'][0]['finish_reason'] == 'stop', (m, d['choices'][0]['finish_reason']))
    js = tok.encode('<tool_call>\n{"name": "read", "arguments": {"path": "a.py"}}\n</tool_call>')
    fake.reply = lambda req, f: [f'T {req} {t}' for t in js] + [f'D {req} stop {len(ids_t)} {len(js)} 1.0 2.0']
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 256, 'tools': TOOLS, 'chat_template_kwargs': {'enable_thinking': False}})
    m = d['choices'][0]['message']
    check('B8: JSON-form block parsed; content null when there is only a call',
          m.get('content') is None and m['tool_calls'][0]['function'] == {'name': 'read', 'arguments': '{"path": "a.py"}'}, m)
    unt = tok.encode('Let me look.\n<tool_call>\n<function=read>\n<parameter=path>\nx')
    fake.reply = lambda req, f: [f'T {req} {t}' for t in unt] + [f'D {req} length {len(ids_t)} {len(unt)} 1.0 2.0']
    ctype, ev, lines = stream('/v1/chat/completions', {'messages': msgs, 'max_tokens': 256, 'tools': TOOLS, 'stream': True,
                                                        'chat_template_kwargs': {'enable_thinking': False}})
    chunks = [e for e in ev if isinstance(e, dict)]
    content = ''.join(c['choices'][0]['delta'].get('content') or '' for c in chunks)
    check('B8: unterminated block at max_tokens is flushed as content, finish length',
          content == tok.decode(unt) and chunks[-1]['choices'][0]['finish_reason'] == 'length'
          and not any(c['choices'][0]['delta'].get('tool_calls') for c in chunks), (content, chunks[-1]['choices'][0]))
    stopped = tok.encode('one two three\n<tool_call>\n<function=read>\n</function>\n</tool_call>')
    fake.reply = lambda req, f: [f'T {req} {t}' for t in stopped] + [f'D {req} stop {len(ids_t)} {len(stopped)} 1.0 2.0']
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 256, 'tools': TOOLS, 'stop': 'three', 'chat_template_kwargs': {'enable_thinking': False}})
    m = d['choices'][0]['message']
    check('B8: a stop string before the block suppresses the call', m['content'] == 'one two ' and 'tool_calls' not in m
          and d['choices'][0]['finish_reason'] == 'stop', (m, d['choices'][0]['finish_reason']))

    # B4: engine failures
    fake.reply = lambda req, f: None  # close without a D line
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 8})
    check('B4: engine closes without D -> 502 server_error', st == 502 and d['error']['type'] == 'server_error', (st, d))
    fake.reply = lambda req, f: [f'D {req} error 0 0 0 0']
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 8, 'stream': True})
    check('B4: D error before any token on a stream -> plain 400', st == 400 and d['error']['code'] == 'engine_rejected', (st, d))
    fake.reply = lambda req, f: [f'D {req} cancel 0 0 0.0 0.0']
    st, d = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 8})
    check('B4: D cancel -> 500', st == 500, (st, d))


def main():
    for p in (STUB_PORT, API_PORT, FAKE_PORT):
        if p == FORBIDDEN_PORT:
            sys.exit(f'refusing to use port {FORBIDDEN_PORT} (the real engine)')
        if port_open(p):
            sys.exit(f'port {p} is already in use; set STUB_PORT/API_PORT/FAKE_PORT')
    t_start = time.time()
    phase_t()
    tok = serve_api.Tokenizer(Path(serve_api.DEFAULT_TOKENIZER) / 'tokenizer.json')
    engine_bin = build_stub_engine()
    work = Path(tempfile.mkdtemp(prefix='chlorine-api-test-'))
    ckpt = work / 'fake.hgn'
    write_fake_checkpoint(ckpt)
    stub_log = open(work / 'stub.log', 'w')
    api_log_path = work / 'api.log'
    stub = api = fake = None
    try:
        stub = subprocess.Popen([str(engine_bin), '--checkpoint', str(ckpt), '--serve', '--bind', '127.0.0.1', '--port', str(STUB_PORT)],
                                env={**os.environ, 'CHLORINE_STUB': '1'}, stdout=stub_log, stderr=subprocess.STDOUT)
        wait_port(STUB_PORT, stub, 'stub engine', 10)
        api_cmd = [sys.executable, str(HERE / 'serve_api.py'), '--port', str(API_PORT), '--default-drafter', '1']
        api_log = open(api_log_path, 'w')
        api = subprocess.Popen(api_cmd + ['--engine-port', str(STUB_PORT)], stdout=api_log, stderr=subprocess.STDOUT)
        wait_port(API_PORT, api, 'api')
        phase_a(tok, api_log_path)

        stop_proc(api, 'api')
        api_log.close()
        fake = FakeEngine(FAKE_PORT)
        fake.start()
        api_log = open(api_log_path, 'a')
        api = subprocess.Popen(api_cmd + ['--engine-port', str(FAKE_PORT)], stdout=api_log, stderr=subprocess.STDOUT)
        wait_port(API_PORT, api, 'api (recorder)')
        phase_b(tok, fake, api_log_path)
    finally:
        stop_proc(api, 'api')
        if stub is not None:
            check('teardown: stub engine answers QUIT with BYE', quit_engine(STUB_PORT))
            try:
                stub.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stub.kill()
        if fake is not None:
            quit_engine(FAKE_PORT)
            fake.join(timeout=3)
        stub_log.close()
        time.sleep(0.2)
        check('teardown: ports released', not port_open(STUB_PORT) and not port_open(API_PORT) and not port_open(FAKE_PORT))
    print(f'\n{PASS} passed, {FAIL} failed in {time.time() - t_start:.1f} s  (logs: {work})')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
