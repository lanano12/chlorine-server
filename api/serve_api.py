#!/usr/bin/env python3
"""serve_api.py - OpenAI-compatible HTTP front end for the chlorine engine.

    python3 api/serve_api.py [--bind 127.0.0.1] [--port 8731]
                             [--engine-host 127.0.0.1] [--engine-port 8740]
                             [--tokenizer DIR] [--default-drafter 1] [--threads 8]

Endpoints
    GET  /v1/models              one model, ``qwen3.8-27b``
    GET  /health                 engine INFO probe (never blocks longer than ~1 s)
    POST /v1/chat/completions    chat template -> ids -> GEN, ``stream`` true/false
    POST /v1/completions         raw text (or an id list) -> GEN, ``stream`` true/false

The engine only sees token ids (docs/halogen/WIRE-PROTOCOL.md): this process
renders the Qwen chat template (a hand port of the checkpoint's
chat_template.jinja: system/user/assistant turns, enable_thinking,
reasoning_effort, preserve_thinking, ``tools`` in the system turn, assistant
``tool_calls`` and ``tool`` results), tokenizes, opens ONE engine connection
per request, sends a GEN line and turns the T/D replies into OpenAI JSON or
SSE chunks.  Generated ``<tool_call>`` blocks come back as OpenAI
``tool_calls`` (finish_reason ``tool_calls``), buffered so a stream never
shows the raw tags.  Engine requests are serialized with a
lock (the engine decodes one request at a time); HTTP threads queue on it.

Every response carries a ``timings`` object (engine_prefill_ms,
engine_decode_ms, engine_ms_total, tokenize_ms, detokenize_ms, queue_ms,
first_token_ms, http_ms, drafter, spec_rounds, spec_committed, ...) and every
request logs one ``ledger`` line plus one halogen-style ``serve_api:`` line to
stderr (see api/README.md).  Stdlib only, plus ``regex`` for the tokenizer.
"""
import argparse
import itertools
import json
import os
import random
import re
import signal
import socket
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import regex

MODEL_ID = 'qwen3.8-27b'
IM_START, IM_END, ENDOFTEXT = 248045, 248046, 248044
THINK_END = '</think>'
VOCAB_LIMIT = 248320      # BIAS tid < 248320 (WIRE-PROTOCOL.md)
MAX_BIAS = 20480          # BIAS n <= 20480
MAX_PROMPT_IDS = 0x40000  # the engine rejects n_prompt >= 0x40000
DRAFTER_NAMES = {0: 'serial', 1: 'mtp', 2: 'dflash2'}
DRAFTER_IDS = {v: k for k, v in DRAFTER_NAMES.items()}
DEFAULT_TOKENIZER = os.environ.get('CHLORINE_TOKENIZER', '/ML_AI/LocalModels/halogen-27b/tokenizer')
# chat_template.jinja: the reasoning-effort system line. `medium` deliberately
# renders no instruction (the template only has text for xhigh and low).
EFFORT_TEXT = {
    'xhigh': 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key '
             'assumptions, consider plausible alternatives, and prioritize correctness, consistency, and '
             'clarity in the final answer.',
    'medium': '',
    'low': 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the '
           'conclusion without unnecessary elaboration.',
}
# OpenAI reasoning_effort values the template has no text for, mapped onto the
# nearest one it knows (the Pi agent sends minimal/low/medium/high/xhigh).
EFFORT_ALIASES = {'minimal': 'low', 'high': 'xhigh'}
# chat_template.jinja, tools branch: the system turn's tool block (verbatim).
TOOLS_HEAD = '# Tools\n\nYou have access to the following functions:\n\n<tools>'
TOOLS_TAIL = (
    '\n</tools>'
    '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n'
    '<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n'
    '<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n'
    '</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n'
    '- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested '
    'within <tool_call></tool_call> XML tags\n'
    '- Required parameters MUST be specified\n'
    '- You may provide optional reasoning for your function call in natural language BEFORE the function call, '
    'but NOT after\n'
    '- If there is no function call available, answer the question like normal with your current knowledge and '
    'do not tell the user about function calls\n</IMPORTANT>')
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = '<tool_call>', '</tool_call>'
MAX_TOOL_CALL_CHARS = 1 << 20   # an unterminated <tool_call> longer than this is flushed as content
_call_ids = itertools.count(1)  # OpenAI tool_call ids: call_<n>, unique per process

_log_lock = threading.Lock()


def log(line):
    with _log_lock:
        sys.stderr.write(line + '\n')
        sys.stderr.flush()


def g9(x):
    """%.9g, the float format the engine's SAMPLE/PENALTY/BIAS clauses expect."""
    return '%.9g' % float(x)


class ApiError(Exception):
    def __init__(self, status, message, etype='invalid_request_error', param=None, code=None):
        super().__init__(message)
        self.status, self.message, self.etype, self.param, self.code = status, message, etype, param, code

    def body(self):
        return {'error': {'message': self.message, 'type': self.etype, 'param': self.param, 'code': self.code}}


# ---------------------------------------------------------------------------
# Tokenizer: byte-level BPE straight from tokenizer.json (port of
# variant-27b/bench/qwen_tok.py; NFC, the Split regex, GPT-2 byte map, merges
# by rank, added tokens).  Only `regex` is needed.
# ---------------------------------------------------------------------------
def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    def __init__(self, path):
        t = json.loads(Path(path).read_text())
        m = t['model']
        self.vocab = m['vocab']
        self.inv = {v: k for k, v in self.vocab.items()}
        self.ranks = {}
        for i, mg in enumerate(m['merges']):
            a, b = mg.split(' ') if isinstance(mg, str) else mg
            self.ranks[(a, b)] = i
        self.special = {a['content']: a['id'] for a in t['added_tokens']}
        self.special_ids = set(self.special.values())
        for a in t['added_tokens']:
            self.inv[a['id']] = a['content']
        pre = t['pre_tokenizer']['pretokenizers'][0]['pattern']['Regex']
        self.pat = regex.compile(pre)
        self.special_pat = regex.compile('(' + '|'.join(regex.escape(s) for s in sorted(self.special, key=len, reverse=True)) + ')')
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

    @lru_cache(maxsize=65536)
    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            best = None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            i = best[1]
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
        return tuple(parts)

    def encode(self, text):
        ids = []
        for piece in self.special_pat.split(text):
            if not piece:
                continue
            if piece in self.special:
                ids.append(self.special[piece])
                continue
            piece = unicodedata.normalize('NFC', piece)
            for m in self.pat.findall(piece):
                word = ''.join(self.b2u[b] for b in m.encode('utf-8'))
                for tok in self._bpe(word):
                    ids.append(self.vocab[tok])
        return ids

    def decode_bytes(self, ids):
        """Raw bytes of a token run (special tokens contribute their text)."""
        out = bytearray()
        for i in ids:
            s = self.inv.get(i, '')
            if i in self.special_ids:
                out += s.encode('utf-8')
            else:
                out += bytes(self.u2b[c] for c in s)
        return bytes(out)

    def decode(self, ids):
        return self.decode_bytes(ids).decode('utf-8', 'replace')


# ---------------------------------------------------------------------------
# Chat template: hand port of chat_template.jinja (no jinja at runtime).
# ---------------------------------------------------------------------------
def _content_text(content, is_system=False):
    """render_content(): strings as-is, lists of {text} items joined, None -> ''."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                raise ApiError(400, 'Unexpected item type in content.')
            if 'image' in item or 'image_url' in item or item.get('type') == 'image':
                raise ApiError(400, 'System message cannot contain images.' if is_system
                               else 'image content is not supported: this engine is text only')
            if 'video' in item or item.get('type') == 'video':
                raise ApiError(400, 'System message cannot contain videos.' if is_system
                               else 'video content is not supported: this engine is text only')
            if 'text' in item:
                parts.append('' if item['text'] is None else str(item['text']))
            else:
                raise ApiError(400, 'Unexpected item type in content.')
        return ''.join(parts)
    raise ApiError(400, 'Unexpected content type.')


def normalize_effort(value):
    """OpenAI reasoning_effort -> the template's vocabulary (minimal->low, high->xhigh)."""
    return EFFORT_ALIASES.get(value, value) if isinstance(value, str) else value


def _tool_json(obj):
    """transformers' ``tojson`` filter: json.dumps(ensure_ascii=False), keys in the
    order the client sent them, no HTML escaping (plain jinja2 would sort keys and
    escape <>&', which is not what vLLM/transformers feed the model)."""
    return json.dumps(obj, ensure_ascii=False)


def _tool_call_view(tc):
    """One assistant tool_call as the template sees it: (name, arguments dict or None).
    OpenAI clients send ``function.arguments`` as a JSON string; the serving layer
    parses it into a mapping for the template's ``arguments|items`` (vLLM does the
    same), so a dict is accepted as well."""
    if not isinstance(tc, dict):
        raise ApiError(400, 'tool_calls entries must be objects', param='messages')
    fn = tc['function'] if 'function' in tc else tc
    if not isinstance(fn, dict) or not isinstance(fn.get('name'), str) or not fn['name']:
        raise ApiError(400, 'tool_calls entries need a function name', param='messages')
    args = fn.get('arguments')
    if args is None or args == '':
        return fn['name'], None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            raise ApiError(400, f'tool call {fn["name"]}: arguments is not valid JSON', param='messages')
    if not isinstance(args, dict):
        raise ApiError(400, f'tool call {fn["name"]}: arguments must be a JSON object', param='messages')
    return fn['name'], args


def render_chat(messages, enable_thinking=None, reasoning_effort=None, preserve_thinking=None, tools=None):
    """Render an OpenAI messages list exactly as chat_template.jinja would with
    add_generation_prompt=true.  ``enable_thinking``/``preserve_thinking``:
    None means "undefined" in the template (both default to on).  ``tools`` is
    the OpenAI list of {type: function, function: {...}} objects, rendered
    verbatim (tojson) into the system turn; an empty list renders no tools."""
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, 'No messages provided.', param='messages')
    for m in messages:
        if not isinstance(m, dict):
            raise ApiError(400, 'messages must be objects with role and content', param='messages')
    instructions = ''
    if enable_thinking is None or enable_thinking is True:
        effort = 'xhigh' if reasoning_effort is None else reasoning_effort
        if effort not in EFFORT_TEXT:
            raise ApiError(400, f'Unexpected reasoning effort {effort}. Supported types are xhigh (default), '
                                'medium, and low.', param='reasoning_effort')
        instructions = EFFORT_TEXT[effort]
    out = []
    first = messages[0]
    if tools and isinstance(tools, (list, tuple)):
        # tools branch: one system turn holding the effort line, the tool JSON and the
        # call-format instructions, then the client's system content (if any)
        out.append('<|im_start|>system\n')
        if instructions:
            out.append(instructions + '\n\n')
        out.append(TOOLS_HEAD)
        for t in tools:
            out.append('\n' + _tool_json(t))
        out.append(TOOLS_TAIL)
        if first.get('role') == 'system':
            content = _content_text(first.get('content'), True).strip()
            if content:
                out.append('\n\n' + content)
        out.append('<|im_end|>\n')
    elif first.get('role') == 'system':
        content = _content_text(first.get('content'), True).strip()
        if content:
            out.append('<|im_start|>system\n' + (instructions + '\n\n' if instructions else '') + content + '<|im_end|>\n')
        elif instructions:
            out.append('<|im_start|>system\n' + instructions + '<|im_end|>\n')
    elif instructions:
        out.append('<|im_start|>system\n' + instructions + '<|im_end|>\n')
    # index of the last real user query (tool_response-wrapped user turns do not count)
    last_query = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get('role') == 'user':
            c = _content_text(m.get('content')).strip()
            if not (c.startswith('<tool_response>') and c.endswith('</tool_response>')):
                last_query = i
                break
    if last_query is None:
        raise ApiError(400, 'No user query found in messages.', param='messages')
    keep_think = preserve_thinking is None or preserve_thinking is True
    n = len(messages)
    for i, m in enumerate(messages):
        role = m.get('role')
        content = _content_text(m.get('content'), role == 'system').strip()
        if role == 'system':
            if i != 0:
                raise ApiError(400, 'System message must be at the beginning.', param='messages')
        elif role == 'user':
            out.append('<|im_start|>user\n' + content + '<|im_end|>\n')
        elif role == 'assistant':
            reasoning = m.get('reasoning_content')
            reasoning = reasoning.strip() if isinstance(reasoning, str) else ''
            if keep_think or i > last_query:
                out.append('<|im_start|>assistant\n<think>\n' + reasoning + '\n</think>\n\n' + content)
            else:
                out.append('<|im_start|>assistant\n' + content)
            calls = m.get('tool_calls')
            if calls and isinstance(calls, (list, tuple)):
                for k, tc in enumerate(calls):
                    name, args = _tool_call_view(tc)
                    if k == 0:
                        out.append(('\n\n' if content else '') + '<tool_call>\n<function=' + name + '>\n')
                    else:
                        out.append('\n<tool_call>\n<function=' + name + '>\n')
                    if args:
                        for key, val in args.items():
                            # strings verbatim, everything else as JSON (args_value|tojson|safe)
                            out.append('<parameter=' + str(key) + '>\n' + (val if isinstance(val, str) else _tool_json(val))
                                       + '\n</parameter>\n')
                    out.append('</function>\n</tool_call>')
            out.append('<|im_end|>\n')
        elif role == 'tool':
            # consecutive tool results share one user turn of <tool_response> blocks
            if i > 0 and messages[i - 1].get('role') != 'tool':
                out.append('<|im_start|>user')
            out.append('\n<tool_response>\n' + content + '\n</tool_response>')
            if i == n - 1 or messages[i + 1].get('role') != 'tool':
                out.append('<|im_end|>\n')
        else:
            raise ApiError(400, 'Unexpected message role.', param='messages')
    out.append('<|im_start|>assistant\n')
    out.append('<think>\n\n</think>\n\n' if enable_thinking is False else '<think>\n')
    return ''.join(out)


# ---------------------------------------------------------------------------
# Output pipeline: ids -> text (incremental, UTF-8 safe) -> think split -> stop
# ---------------------------------------------------------------------------
def _new_suffix(old, new):
    if new.startswith(old):
        return new[len(old):]
    n = 0
    for a, b in zip(old, new):
        if a != b:
            break
        n += 1
    return new[n:]


class Detokenizer:
    """Decode the growing id list and emit only the new, stable suffix.  A
    trailing U+FFFD means an incomplete UTF-8 sequence: it is held back until
    the next token completes (or invalidates) it.  Once a decode ends cleanly
    the ids are dropped, so each push decodes a few tokens, not the whole run."""

    def __init__(self, tok):
        self.tok = tok
        self.ids = []
        self.emitted = ''

    def push(self, tid):
        self.ids.append(tid)
        s = self.tok.decode(self.ids)
        if s.endswith('�'):
            stable = s[:-1]
            new = _new_suffix(self.emitted, stable)
            self.emitted = stable
            return new
        new = _new_suffix(self.emitted, s)
        self.ids, self.emitted = [], ''
        return new

    def flush(self):
        if not self.ids:
            return ''
        new = _new_suffix(self.emitted, self.tok.decode(self.ids))
        self.ids, self.emitted = [], ''
        return new


class ThinkSplitter:
    """Routes text to ('reasoning', ...) until '</think>' then ('content', ...).
    A partial '</think>' at the tail is held back; the newlines the template
    puts after '</think>' are dropped from the content."""

    def __init__(self, thinking):
        self.in_reasoning = bool(thinking)
        self.buf = ''
        self.skip_nl = False

    def feed(self, text):
        out = []
        if not self.in_reasoning:
            return self._content(text, out)
        self.buf += text
        i = self.buf.find(THINK_END)
        if i >= 0:
            if i:
                out.append(('reasoning', self.buf[:i]))
            rest = self.buf[i + len(THINK_END):]
            self.buf, self.in_reasoning, self.skip_nl = '', False, True
            return self._content(rest, out)
        keep = 0
        for k in range(min(len(self.buf), len(THINK_END) - 1), 0, -1):
            if THINK_END.startswith(self.buf[-k:]):
                keep = k
                break
        emit = self.buf[:len(self.buf) - keep]
        self.buf = self.buf[len(emit):]
        if emit:
            out.append(('reasoning', emit))
        return out

    def _content(self, text, out):
        if self.skip_nl:
            text = text.lstrip('\n')
            if text:
                self.skip_nl = False
        if text:
            out.append(('content', text))
        return out

    def flush(self):
        out = []
        if self.buf:
            out.append(('reasoning' if self.in_reasoning else 'content', self.buf))
            self.buf = ''
        return out


class StopFilter:
    """OpenAI ``stop`` strings on the content stream.  The engine has no
    cancel, so a stop only truncates the text; the engine still runs to
    max_tokens (prefer max_tokens for bounded runs)."""

    def __init__(self, stops):
        self.stops = [s for s in stops if s]
        self.buf = ''
        self.stopped = False

    def feed(self, text):
        if not self.stops:
            return text
        if self.stopped:
            return ''
        self.buf += text
        best = None
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0 and (best is None or i < best):
                best = i
        if best is not None:
            emit, self.buf, self.stopped = self.buf[:best], '', True
            return emit
        keep = 0
        for s in self.stops:
            for k in range(min(len(self.buf), len(s) - 1), 0, -1):
                if s.startswith(self.buf[-k:]):
                    keep = max(keep, k)
                    break
        emit = self.buf[:len(self.buf) - keep]
        self.buf = self.buf[len(emit):]
        return emit

    def flush(self):
        if self.stopped:
            return ''
        emit, self.buf = self.buf, ''
        return emit


# ---------------------------------------------------------------------------
# Tool calls in the output.  The template teaches the model
#     <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n...</function>\n</tool_call>
# (several blocks separated by a newline, optional text BEFORE them); the JSON
# form <tool_call>{"name": .., "arguments": {..}}</tool_call> is accepted too.
# ---------------------------------------------------------------------------
_FUNC_OPEN_RE = re.compile(r'\s*<function=([^>]+)>')
_PARAM_RE = re.compile(r'<parameter=([^>]*)>(.*?)(?:</parameter>|\Z)', re.S)


def _coerce_param(value, schema):
    """An XML parameter value is text; the function's JSON schema says whether it
    is an integer, number, boolean, object or array (the rule vLLM's qwen3_coder
    parser applies).  A value that does not parse as its type stays a string."""
    types = schema.get('type') if isinstance(schema, dict) else None
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, list) or not types or 'string' in types:
        return value
    v = value.strip()
    if v == 'null':
        return None
    for t in types:
        try:
            if t == 'integer':
                return int(v)
            if t == 'number':
                f = float(v)
                if f != f or f in (float('inf'), float('-inf')):
                    continue
                return int(f) if f.is_integer() else f
            if t == 'boolean' and v.lower() in ('true', 'false'):
                return v.lower() == 'true'
            if t in ('object', 'array'):
                obj = json.loads(v)
                if isinstance(obj, dict if t == 'object' else list):
                    return obj
        except ValueError:
            pass
    return value


def parse_tool_call_block(block, schemas=None):
    """Text between <tool_call> and </tool_call> -> {'name': .., 'arguments': <JSON string>},
    or None when it is neither the template's <function=..> form nor the JSON form.
    ``schemas`` maps function name -> its parameters.properties (for value types)."""
    schemas = schemas or {}
    m = _FUNC_OPEN_RE.match(block)
    if m:
        name = m.group(1).strip()
        body = block[m.end():]
        end = body.rfind('</function>')
        if end >= 0:
            body = body[:end]
        props = schemas.get(name) or {}
        args = {}
        for pm in _PARAM_RE.finditer(body):
            key, val = pm.group(1).strip(), pm.group(2)
            if val.startswith('\n'):
                val = val[1:]
            if val.endswith('\n'):
                val = val[:-1]
            args[key] = _coerce_param(val, props.get(key))
        return {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}
    text = block.strip()
    if text.startswith('{'):
        try:
            obj = json.loads(text)
        except ValueError:
            return None
        if not isinstance(obj, dict) or not isinstance(obj.get('name'), str) or not obj['name']:
            return None
        args = obj.get('arguments', obj.get('parameters', {}))
        if isinstance(args, str):
            try:
                json.loads(args)
            except ValueError:
                return None
            return {'name': obj['name'], 'arguments': args}
        if isinstance(args, dict):
            return {'name': obj['name'], 'arguments': json.dumps(args, ensure_ascii=False)}
    return None


def tool_schemas(tools):
    """function name -> parameters.properties, from the OpenAI tools list."""
    out = {}
    for t in tools or []:
        fn = t.get('function', t) if isinstance(t, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get('name'), str):
            params = fn.get('parameters')
            props = params.get('properties') if isinstance(params, dict) else None
            out[fn['name']] = props if isinstance(props, dict) else {}
    return out


class ToolCallSplitter:
    """Routes the content stream to ('content', text) and ('tool_call', {name, arguments}).
    A <tool_call>...</tool_call> block is buffered until complete and then parsed;
    a block that does not parse (or is never closed) comes out as plain content
    with its tags.  A partial '<tool_call>' at the tail is held back, and so is
    trailing whitespace: it is dropped before a block and after the last one (the
    template's own trim) and emitted otherwise, so text without tool calls is
    unchanged."""

    def __init__(self, tools):
        self.schemas = tool_schemas(tools)
        self.buf = ''          # text not yet routed (the open block while in_call)
        self.ws = ''           # trailing whitespace held back
        self.in_call = False
        self.after_call = False

    def feed(self, text):
        out = []
        self.buf += text
        while True:
            if self.in_call:
                j = self.buf.find(TOOL_CALL_CLOSE)
                if j < 0:
                    if len(self.buf) > MAX_TOOL_CALL_CHARS:
                        self.in_call = False
                        self._text(TOOL_CALL_OPEN + self.buf, out)
                        self.buf = ''
                    return out
                block, self.buf = self.buf[:j], self.buf[j + len(TOOL_CALL_CLOSE):]
                self.in_call = False
                call = parse_tool_call_block(block, self.schemas)
                if call is None:
                    self._text(TOOL_CALL_OPEN + block + TOOL_CALL_CLOSE, out)
                else:
                    self.ws, self.after_call = '', True
                    out.append(('tool_call', call))
                continue
            i = self.buf.find(TOOL_CALL_OPEN)
            if i >= 0:
                self._text(self.buf[:i], out)   # trailing whitespace stays held: dropped only once the block parses
                self.buf = self.buf[i + len(TOOL_CALL_OPEN):]
                self.in_call = True
                continue
            keep = 0
            for k in range(min(len(self.buf), len(TOOL_CALL_OPEN) - 1), 0, -1):
                if TOOL_CALL_OPEN.startswith(self.buf[-k:]):
                    keep = k
                    break
            emit, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
            self._text(emit, out)
            return out

    def _text(self, text, out):
        text = self.ws + text
        stripped = text.rstrip()
        self.ws = text[len(stripped):]
        if not stripped:
            return
        if self.after_call:
            stripped, self.after_call = stripped.lstrip(), False
        if stripped:
            out.append(('content', stripped))

    def flush(self):
        out = []
        if self.in_call:
            self.in_call = False
            self._text(TOOL_CALL_OPEN + self.buf, out)
        elif self.buf:
            self._text(self.buf, out)
        self.buf = ''
        if self.ws and not self.after_call:
            out.append(('content', self.ws))
        self.ws = ''
        return out


# ---------------------------------------------------------------------------
# Engine client (one connection per request; the caller holds the engine lock)
# ---------------------------------------------------------------------------
class EngineError(Exception):
    pass


class DLine:
    """D <req> <reason> [<n_prompt> <n_gen> <prefill_ms> <decode_ms> [<drafter> <rounds> <commit> [n_cached]]]
    A short line is the serial format: drafter 0, no speculative stats."""

    def __init__(self, f):
        self.reason = f[2] if len(f) > 2 else 'error'
        self.n_prompt = int(f[3]) if len(f) > 3 else 0
        self.n_gen = int(f[4]) if len(f) > 4 else None
        self.prefill_ms = float(f[5]) if len(f) > 5 else 0.0
        self.decode_ms = float(f[6]) if len(f) > 6 else 0.0
        self.drafter = int(f[7]) if len(f) > 7 else 0
        self.rounds = int(f[8]) if len(f) > 8 else 0
        self.committed = int(f[9]) if len(f) > 9 else 0
        self.n_cached = int(f[10]) if len(f) > 10 else 0
        self.raw = ' '.join(f)


class Engine:
    def __init__(self, host, port, connect_timeout=10.0, first_token_timeout=1800.0, token_timeout=300.0):
        self.host, self.port = host, port
        self.connect_timeout = connect_timeout
        self.first_token_timeout = first_token_timeout
        self.token_timeout = token_timeout

    def _connect(self, timeout):
        s = socket.create_connection((self.host, self.port), timeout=timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    @staticmethod
    def _readline(s, buf):
        while b'\n' not in buf:
            d = s.recv(1 << 16)
            if not d:
                raise EngineError('engine closed the connection')
            buf += d
        ln, buf = buf.split(b'\n', 1)
        return ln.decode('ascii', 'replace'), buf

    def info(self, timeout=2.0):
        """INFO -> dict (fields absent from an older engine take conservative defaults)."""
        with self._connect(timeout) as s:
            s.settimeout(timeout)
            s.sendall(b'INFO\n')
            line, _ = self._readline(s, b'')
        f = line.split()
        if not f or f[0] != 'I':
            raise EngineError(f'unexpected INFO reply: {line!r}')
        names = ['mtp', 'draft_head', 'ctx', 'spec_rows', 'default_drafter', 'drafter_weights', 'dflash2',
                 'cache_mb', 'cache_align', 'kv_slots', 'slot_ctx']
        defaults = [0, 0, 262144, 8, 0, 0, 0, 0, 2048, 1, 262144]
        info = {}
        for i, (n, dflt) in enumerate(zip(names, defaults)):
            try:
                info[n] = int(f[i + 1])
            except (IndexError, ValueError):
                info[n] = dflt
        info['raw'] = line
        return info

    def generate(self, req, gen_line, on_token):
        """Send one GEN line; call on_token(id, logprob|None) per T line; return the D line."""
        s = self._connect(self.connect_timeout)
        try:
            s.sendall(gen_line.encode('ascii'))
            buf = b''
            timeout = self.first_token_timeout
            sreq = str(req)
            while True:
                s.settimeout(timeout)
                try:
                    line, buf = self._readline(s, buf)
                except socket.timeout:
                    raise EngineError(f'engine silent for {timeout:.0f} s (req {req})')
                f = line.split()
                if len(f) < 3 or f[1] != sreq:
                    continue
                if f[0] == 'T':
                    on_token(int(f[2]), float(f[3]) if len(f) > 3 else None)
                    timeout = self.token_timeout
                elif f[0] == 'D':
                    return DLine(f)
        finally:
            s.close()


class Sampling:
    def __init__(self, temperature, top_k, top_p, min_p, seed, presence, frequency, bias, logprobs):
        self.temperature, self.top_k, self.top_p, self.min_p, self.seed = temperature, top_k, top_p, min_p, seed
        self.presence, self.frequency, self.bias, self.logprobs = presence, frequency, bias, logprobs

    def as_json(self):
        return {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p, 'min_p': self.min_p,
                'seed': self.seed, 'presence_penalty': self.presence, 'frequency_penalty': self.frequency,
                'logit_bias': len(self.bias), 'logprobs': self.logprobs}


def build_gen_line(req, max_tokens, eos, ids, drafter, sp):
    """GEN <req> <max_tokens> <n_eos> <eos...> <n_ids> <ids...> <drafter>
       [SAMPLE temp top_k top_p min_p seed] [PENALTY p f] [BIAS n (tid val)*] [LOGPROBS]"""
    parts = ['GEN', str(req), str(max_tokens), str(len(eos))] + [str(e) for e in eos]
    parts += [str(len(ids))] + [str(i) for i in ids] + [str(drafter)]
    if sp is not None:
        parts += ['SAMPLE', g9(sp.temperature), str(sp.top_k), g9(sp.top_p), g9(sp.min_p), str(sp.seed)]
        if sp.presence or sp.frequency:
            parts += ['PENALTY', g9(sp.presence), g9(sp.frequency)]
        if sp.bias:
            parts += ['BIAS', str(len(sp.bias))]
            for tid, val in sp.bias:
                parts += [str(tid), g9(val)]
        if sp.logprobs:
            parts.append('LOGPROBS')
    return ' '.join(parts) + '\n'


# ---------------------------------------------------------------------------
# Request parsing helpers
# ---------------------------------------------------------------------------
def _num(body, key, default, lo=None, hi=None):
    v = body.get(key, default)
    if v is None:
        v = default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ApiError(400, f'{key} must be a number', param=key)
    v = float(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ApiError(400, f'{key} must be in [{lo}, {hi}]', param=key)
    return v


def parse_sampling(body):
    """The sampling clause, or None for greedy (temperature <= 0, the default)."""
    temperature = _num(body, 'temperature', 0.0, 0.0, None)
    top_p = _num(body, 'top_p', 1.0, 0.0, 1.0)
    min_p = _num(body, 'min_p', 0.0, 0.0, 1.0)
    top_k = body.get('top_k', 0)
    if top_k is None:
        top_k = 0
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ApiError(400, 'top_k must be a non-negative integer', param='top_k')
    presence = _num(body, 'presence_penalty', 0.0, -2.0, 2.0)
    frequency = _num(body, 'frequency_penalty', 0.0, -2.0, 2.0)
    logprobs = body.get('logprobs', False)
    logprobs = bool(logprobs) if not isinstance(logprobs, bool) else logprobs
    bias = []
    lb = body.get('logit_bias')
    if lb:
        if not isinstance(lb, dict):
            raise ApiError(400, 'logit_bias must be an object of token id -> bias', param='logit_bias')
        if len(lb) > MAX_BIAS:
            raise ApiError(400, f'logit_bias has more than {MAX_BIAS} entries', param='logit_bias')
        for k, v in lb.items():
            try:
                tid = int(k)
                val = float(v)
            except (TypeError, ValueError):
                raise ApiError(400, 'logit_bias entries must be token id -> number', param='logit_bias')
            if not 0 <= tid < VOCAB_LIMIT:
                raise ApiError(400, f'logit_bias token id {tid} out of range', param='logit_bias')
            bias.append((tid, val))
    if temperature <= 0:
        if presence or frequency or bias or logprobs:
            raise ApiError(400, 'presence_penalty, frequency_penalty, logit_bias and logprobs need '
                                'temperature > 0: greedy decoding has no sampler to apply them to')
        return None
    seed = body.get('seed')
    if seed is None:
        seed = random.getrandbits(64)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 64):
        raise ApiError(400, 'seed must be an integer in [0, 2^64)', param='seed')
    return Sampling(temperature, top_k, top_p, min_p, seed, presence, frequency, bias, logprobs)


def parse_drafter(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        raise ApiError(400, 'drafter must be 0/1/2 or serial/mtp/dflash2', param='drafter')
    if isinstance(value, int):
        d = value
    else:
        s = str(value).strip().lower()
        if s in DRAFTER_IDS:
            d = DRAFTER_IDS[s]
        else:
            try:
                d = int(s)
            except ValueError:
                raise ApiError(400, 'drafter must be 0/1/2 or serial/mtp/dflash2', param='drafter')
    if d not in DRAFTER_NAMES:
        raise ApiError(400, 'drafter must be 0/1/2 or serial/mtp/dflash2', param='drafter')
    return d


def parse_stops(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(s, str) for s in value):
        if len(value) > 4:
            raise ApiError(400, 'at most 4 stop strings', param='stop')
        return value
    raise ApiError(400, 'stop must be a string or a list of strings', param='stop')


def parse_tools(body):
    """OpenAI ``tools`` + ``tool_choice`` -> the function list to render, or None when
    nothing is rendered (no tools, an empty list, or tool_choice "none").  "auto",
    "required" and a named function all render the full list: the template has no
    way to force a call, so those two are best effort."""
    tools = body.get('tools')
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        raise ApiError(400, 'tools must be a list of {type: "function", function: {name, description, parameters}}',
                       param='tools')
    names = []
    for t in tools:
        fn = t.get('function') if isinstance(t, dict) else None
        if (not isinstance(t, dict) or t.get('type', 'function') != 'function' or not isinstance(fn, dict)
                or not isinstance(fn.get('name'), str) or not fn['name']):
            raise ApiError(400, 'tools entries must be {type: "function", function: {name, description, parameters}}',
                           param='tools')
        names.append(fn['name'])
    choice = body.get('tool_choice')
    if choice == 'none':
        return None
    if isinstance(choice, dict):
        fn = choice.get('function') if choice.get('type', 'function') == 'function' else None
        if not isinstance(fn, dict) or fn.get('name') not in names:
            raise ApiError(400, 'tool_choice names a function that is not in tools', param='tool_choice')
    elif choice not in (None, 'auto', 'required'):
        raise ApiError(400, 'tool_choice must be "auto", "none", "required" or {type: "function", function: {name}}',
                       param='tool_choice')
    return tools or None


def _tri(v):
    """tri-state bool: None (undefined), True, False."""
    return None if v is None else bool(v)


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------
class Service:
    def __init__(self, args):
        self.args = args
        self.tok = None
        self.engine = Engine(args.engine_host, args.engine_port, first_token_timeout=args.first_token_timeout,
                             token_timeout=args.token_timeout)
        self.engine_lock = threading.Lock()
        self.req_lock = threading.Lock()
        self.next_req = 1
        self.queued = 0
        self.info = None
        self.info_state = 'unknown'
        self.started = time.time()

    def load_tokenizer(self):
        t0 = time.perf_counter()
        path = Path(self.args.tokenizer)
        if path.is_dir():
            path = path / 'tokenizer.json'
        self.tok = Tokenizer(path)
        log(f'api: tokenizer {path} ({len(self.tok.vocab)} vocab, {len(self.tok.special)} added tokens) '
            f'loaded in {time.perf_counter() - t0:.2f} s')

    def new_req(self):
        with self.req_lock:
            r = self.next_req
            self.next_req += 1
            return r

    def probe(self, wait=1.0, timeout=2.0):
        """INFO under the engine lock; 'busy' when a generation holds it."""
        if not self.engine_lock.acquire(timeout=wait):
            self.info_state = 'busy'
            return self.info
        try:
            self.info = self.engine.info(timeout)
            self.info_state = 'ok'
        except (OSError, EngineError) as e:
            self.info_state = f'down ({e})'
        finally:
            self.engine_lock.release()
        return self.info

    @property
    def ctx(self):
        return self.info['ctx'] if self.info else None

    def health(self):
        self.probe()
        return {'status': 'ok', 'model': MODEL_ID, 'context': self.ctx or 262144,
                'drafter_default': DRAFTER_NAMES[self.args.default_drafter],
                'engine': {'host': self.args.engine_host, 'port': self.args.engine_port,
                           'state': self.info_state, 'info': self.info},
                'queue': self.queued, 'busy': self.engine_lock.locked(), 'uptime_s': round(time.time() - self.started, 1)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    timeout = 30            # idle keep-alive connections release their worker after 30 s
    svc = None              # set by main()
    server_version = 'chlorine-api/0.1'
    sys_version = ''

    # --- plumbing -----------------------------------------------------------
    def log_message(self, fmt, *args):
        if self.svc.args.verbose:
            log('http ' + fmt % args)

    def log_error(self, fmt, *args):
        log('http ' + fmt % args)

    def send_json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, err):
        try:
            self.send_json(err.status, err.body())
        except OSError:
            pass

    def read_body(self):
        if 'chunked' in self.headers.get('Transfer-Encoding', '').lower():
            raise ApiError(400, 'chunked request bodies are not supported; send Content-Length')
        try:
            n = int(self.headers.get('Content-Length', 0))
        except ValueError:
            raise ApiError(400, 'bad Content-Length')
        if n > self.svc.args.max_body_bytes:
            raise ApiError(413, 'request body too large')
        raw = self.rfile.read(n) if n else b''
        try:
            body = json.loads(raw.decode('utf-8')) if raw else {}
        except (UnicodeDecodeError, ValueError) as e:
            raise ApiError(400, f'request body is not valid JSON: {e}')
        if not isinstance(body, dict):
            raise ApiError(400, 'request body must be a JSON object')
        return body

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        try:
            if path in ('/v1/models', '/models'):
                self.send_json(200, {'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model',
                                                                 'created': int(self.svc.started), 'owned_by': 'chlorine'}]})
            elif path.startswith('/v1/models/'):
                if path[len('/v1/models/'):] != MODEL_ID:
                    raise ApiError(404, f'model not found: {path[len("/v1/models/"):]}', param='model', code='model_not_found')
                self.send_json(200, {'id': MODEL_ID, 'object': 'model', 'created': int(self.svc.started), 'owned_by': 'chlorine'})
            elif path in ('/health', '/v1/health', '/'):
                self.send_json(200, self.svc.health())
            else:
                raise ApiError(404, f'no route for GET {path}')
        except ApiError as e:
            self.send_error_json(e)

    def do_POST(self):
        t_recv = time.perf_counter()
        path = self.path.split('?', 1)[0]
        try:
            body = self.read_body()
            if path in ('/v1/chat/completions', '/chat/completions'):
                self.generate(body, 'chat', t_recv)
            elif path in ('/v1/completions', '/completions'):
                self.generate(body, 'completions', t_recv)
            else:
                raise ApiError(404, f'no route for POST {path}')
        except ApiError as e:
            self.send_error_json(e)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # --- generation ---------------------------------------------------------
    def generate(self, body, kind, t_recv):
        svc, args, tok = self.svc, self.svc.args, self.svc.tok
        chat = kind == 'chat'
        stream = bool(body.get('stream', False))
        if body.get('n', 1) not in (1, None):
            raise ApiError(400, 'n > 1 is not supported', param='n')
        model = body.get('model')
        if model not in (None, MODEL_ID) and not str(model).startswith(MODEL_ID):
            log(f'api: request for model {model!r} served as {MODEL_ID}')

        # prompt
        thinking = False
        tmpl = body.get('chat_template_kwargs') or {}
        if not isinstance(tmpl, dict):
            raise ApiError(400, 'chat_template_kwargs must be an object', param='chat_template_kwargs')
        t0 = time.perf_counter()
        tools = None
        if chat:
            enable_thinking = _tri(tmpl.get('enable_thinking', body.get('enable_thinking')))
            effort = normalize_effort(tmpl.get('reasoning_effort', body.get('reasoning_effort')))
            preserve = _tri(tmpl.get('preserve_thinking', body.get('preserve_thinking')))
            tools = parse_tools(body)
            text = render_chat(body.get('messages'), enable_thinking, effort, preserve, tools)
            thinking = enable_thinking is not False
            ids = tok.encode(text)
            eos = [IM_END, ENDOFTEXT]
        else:
            prompt = body.get('prompt')
            if isinstance(prompt, list) and prompt and all(isinstance(i, int) and not isinstance(i, bool) for i in prompt):
                ids = list(prompt)
                if any(not 0 <= i < VOCAB_LIMIT for i in ids):
                    raise ApiError(400, 'prompt token id out of range', param='prompt')
            elif isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
                ids = tok.encode(prompt[0])
            elif isinstance(prompt, str):
                ids = tok.encode(prompt)
            else:
                raise ApiError(400, 'prompt must be a string or a list of token ids (one prompt per request)', param='prompt')
            eos = [ENDOFTEXT]
        tokenize_ms = (time.perf_counter() - t0) * 1000.0
        if not ids:
            raise ApiError(400, 'empty prompt', param='messages' if chat else 'prompt')
        ctx = svc.ctx or 262144
        if len(ids) >= MAX_PROMPT_IDS or len(ids) >= ctx:
            raise ApiError(400, f'prompt too long: {len(ids)} tokens, context is {min(ctx, MAX_PROMPT_IDS)}',
                           code='context_length_exceeded')

        # knobs
        max_tokens = body.get('max_completion_tokens', body.get('max_tokens'))
        if max_tokens is None:
            max_tokens = args.default_max_tokens
        if isinstance(max_tokens, float) and max_tokens.is_integer():
            max_tokens = int(max_tokens)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ApiError(400, 'max_tokens must be a positive integer', param='max_tokens')
        max_tokens = min(max_tokens, args.max_tokens_cap)
        drafter = parse_drafter(body.get('drafter', body.get('x-drafter', self.headers.get('x-drafter'))),
                                args.default_drafter)
        sp = parse_sampling(body)
        stops = parse_stops(body.get('stop'))
        eos_set = set(eos)

        req = svc.new_req()
        gen_line = build_gen_line(req, max_tokens, eos, ids, drafter, sp)
        log(f'gen req={req} kind={kind} ids={len(ids)} max_tokens={max_tokens} eos={",".join(map(str, eos))} '
            f'drafter={drafter} ({DRAFTER_NAMES[drafter]}) '
            + (f'sample=temp:{g9(sp.temperature)}/top_k:{sp.top_k}/top_p:{g9(sp.top_p)}/min_p:{g9(sp.min_p)}'
               f'/seed:{sp.seed}' + ('/penalty' if sp.presence or sp.frequency else '')
               + (f'/bias:{len(sp.bias)}' if sp.bias else '') + ('/logprobs' if sp.logprobs else '')
               if sp else 'sample=greedy') + (f' stream=1' if stream else '')
            + (f' tools={len(tools)}' if tools else ''))

        # queue for the engine
        st = GenState(self, kind, req, stream, thinking, stops, tok, eos_set, tools)
        t_q = time.perf_counter()
        svc.queued += 1
        try:
            if not svc.engine_lock.acquire(timeout=args.queue_timeout):
                raise ApiError(503, f'engine busy: queue timeout after {args.queue_timeout:.0f} s', etype='server_error')
        finally:
            svc.queued -= 1
        queue_ms = (time.perf_counter() - t_q) * 1000.0
        t_e = time.perf_counter()
        try:
            try:
                d = svc.engine.generate(req, gen_line, st.on_token)
            except (OSError, EngineError) as e:
                log(f'api: req={req} engine error: {e}')
                err = ApiError(502, f'engine error: {e}', etype='server_error')
                st.fail(err)
                return
        finally:
            svc.engine_lock.release()
        engine_ms_total = (time.perf_counter() - t_e) * 1000.0

        if d.reason == 'error':
            log(f'api: req={req} engine rejected the request ({d.raw})')
            st.fail(ApiError(400, 'engine rejected the request (drafter not servable, prompt over capacity, '
                                  'or sampler clause not applicable)', code='engine_rejected'))
            return
        if d.reason == 'cancel':
            st.fail(ApiError(500, 'engine cancelled the request', etype='server_error'))
            return
        st.finish()
        finish = 'stop' if (d.reason == 'stop' or st.eos_hit or st.stops.stopped) else 'length'
        n_gen = d.n_gen if d.n_gen is not None else st.n_tokens
        usage = {'prompt_tokens': len(ids), 'completion_tokens': n_gen, 'total_tokens': len(ids) + n_gen}
        tok_s = n_gen / (d.decode_ms / 1000.0) if d.decode_ms > 0 else 0.0
        timings = {
            'engine_prefill_ms': d.prefill_ms, 'engine_decode_ms': d.decode_ms,
            'engine_ms_total': round(engine_ms_total, 3),
            'tokenize_ms': round(tokenize_ms, 3), 'detokenize_ms': round(st.detok_ms, 3),
            'queue_ms': round(queue_ms, 3),
            'first_token_ms': round((st.t_first - t_recv) * 1000.0, 3) if st.t_first else None,
            'http_ms': None,  # filled just before the final write
            'drafter': d.drafter, 'drafter_requested': drafter, 'spec_rounds': d.rounds,
            'spec_committed': d.committed, 'n_cached': d.n_cached,
            'decode_tok_s': round(tok_s, 3), 'engine_reason': d.reason,
        }
        log(f'ledger req={req} prompt={d.n_prompt or len(ids)} gen={n_gen} prefill_ms={d.prefill_ms:.1f} '
            f'decode_ms={d.decode_ms:.1f} drafter={d.drafter} rounds={d.rounds} commit={d.committed} tok_s={tok_s:.2f}')
        # halogen bench-serving.py ledger shape (its LEDGER regex); the drafter
        # NAME is the one the request asked for, which is how that script aligns
        # ledger lines with requests.
        cpr = d.committed / d.rounds if d.rounds else 0.0
        cached = f' ({d.n_cached} cached)' if d.n_cached else ''
        log(f'serve_api: {DRAFTER_NAMES[drafter]} {n_gen} tok in {d.decode_ms / 1000.0:.3f}s = {tok_s:.2f} t/s | '
            f'{d.rounds} rounds, commit {cpr:.2f}/round | prompt {d.n_prompt or len(ids)}{cached}, '
            f'prefill {d.prefill_ms / 1000.0:.3f}s')
        extra = {'sampling': sp.as_json()} if sp else {}
        st.respond(finish, usage, timings, t_recv, extra)


class GenState:
    """Per-request output state: detokenize, split, stop, and emit (SSE or buffer)."""

    def __init__(self, handler, kind, req, stream, thinking, stops, tok, eos_set, tools=None):
        self.h, self.kind, self.req, self.stream = handler, kind, req, stream
        self.chat = kind == 'chat'
        self.detok = Detokenizer(tok)
        self.tok = tok
        self.thinking = bool(thinking)
        self.split = ThinkSplitter(thinking) if self.chat else None
        self.tool_split = ToolCallSplitter(tools) if (self.chat and tools) else None
        self.calls = []
        self.stops = StopFilter(stops)
        self.eos_set = eos_set
        self.pieces = {'reasoning': [], 'content': []}
        self.logprobs = []
        self.n_tokens = 0
        self.eos_hit = False
        self.detok_ms = 0.0
        self.t_first = None
        self.headers_sent = False
        self.client_gone = False
        self.sent_role = False
        self.created = int(time.time())
        self.id = ('chatcmpl-' if self.chat else 'cmpl-') + f'{req}-{random.getrandbits(40):010x}'

    # engine callback
    def on_token(self, tid, logprob):
        self.n_tokens += 1
        if self.t_first is None:
            self.t_first = time.perf_counter()
        if self.eos_hit or self.stops.stopped:
            return  # draining: the engine cannot be cancelled
        if tid in self.eos_set:
            self.eos_hit = True
            return
        t0 = time.perf_counter()
        if logprob is not None:
            b = self.tok.decode_bytes([tid])
            self.logprobs.append({'token': b.decode('utf-8', 'replace'), 'logprob': logprob,
                                  'bytes': list(b), 'top_logprobs': []})
        self._route(self.detok.push(tid))
        self.detok_ms += (time.perf_counter() - t0) * 1000.0

    def _route(self, text):
        if not text:
            return
        self._emit(self.split.feed(text) if self.split else [('content', text)])

    def _emit(self, parts):
        """('reasoning'|'content', text) pieces: content goes through the tool-call
        splitter (when tools are rendered) and then the stop filter."""
        for kind, piece in parts:
            if self.stops.stopped:
                return
            if kind == 'content' and self.tool_split:
                for k, p in self.tool_split.feed(piece):
                    if self.stops.stopped:
                        return
                    self._piece(k, p)
            else:
                self._piece(kind, piece)

    def _piece(self, kind, piece):
        if kind == 'tool_call':
            piece['id'] = f'call_{next(_call_ids)}'
            self.calls.append(piece)
            if self.stream:
                self._delta_tool_call(len(self.calls) - 1, piece)
            return
        if kind == 'content':
            piece = self.stops.feed(piece)
        if piece:
            self.pieces[kind].append(piece)
            if self.stream:
                self._delta(kind, piece)

    def finish(self):
        t0 = time.perf_counter()
        self._route(self.detok.flush())
        if self.split:
            self._emit(self.split.flush())
        if self.tool_split:
            for kind, piece in self.tool_split.flush():
                if self.stops.stopped:
                    break
                self._piece(kind, piece)
        tail = self.stops.flush()
        if tail:
            self.pieces['content'].append(tail)
            if self.stream:
                self._delta('content', tail)
        self.detok_ms += (time.perf_counter() - t0) * 1000.0

    # --- output ---------------------------------------------------------------
    def _write(self, data):
        if self.client_gone:
            return
        try:
            self.h.wfile.write(data)
            self.h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.client_gone = True
            log(f'api: req={self.req} client disconnected; draining the engine')

    def _send_headers(self):
        if self.headers_sent:
            return
        self.headers_sent = True
        h = self.h
        h.send_response(200)
        h.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        h.send_header('Cache-Control', 'no-cache')
        h.send_header('X-Accel-Buffering', 'no')
        h.send_header('Transfer-Encoding', 'chunked')
        h.end_headers()

    def _chunk(self, obj):
        self._send_headers()
        payload = ('data: ' + json.dumps(obj, ensure_ascii=False) + '\n\n').encode('utf-8')
        self._write(f'{len(payload):x}\r\n'.encode('ascii') + payload + b'\r\n')

    def _done(self):
        payload = b'data: [DONE]\n\n'
        self._write(f'{len(payload):x}\r\n'.encode('ascii') + payload + b'\r\n')
        self._write(b'0\r\n\r\n')

    def _envelope(self, choice, **top):
        obj = {'id': self.id, 'object': 'chat.completion.chunk' if self.chat else 'text_completion',
               'created': self.created, 'model': MODEL_ID, 'choices': [choice]}
        obj.update(top)
        return obj

    def _delta(self, kind, piece):
        if self.chat:
            delta = {'role': 'assistant'} if not self.sent_role else {}
            delta['reasoning_content' if kind == 'reasoning' else 'content'] = piece
            self.sent_role = True
            choice = {'index': 0, 'delta': delta, 'finish_reason': None, 'logprobs': None}
        else:
            choice = {'index': 0, 'text': piece, 'finish_reason': None, 'logprobs': None}
        self._chunk(self._envelope(choice))

    @staticmethod
    def _tool_call_json(call, index=None):
        obj = {'id': call['id'], 'type': 'function', 'function': {'name': call['name'], 'arguments': call['arguments']}}
        if index is not None:
            obj = {'index': index, **obj}
        return obj

    def _delta_tool_call(self, index, call):
        """One complete tool call as an incremental delta.tool_calls chunk."""
        delta = {'role': 'assistant'} if not self.sent_role else {}
        self.sent_role = True
        delta['tool_calls'] = [self._tool_call_json(call, index)]
        self._chunk(self._envelope({'index': 0, 'delta': delta, 'finish_reason': None, 'logprobs': None}))

    def fail(self, err):
        """Error before any byte was sent -> proper status; mid-stream -> SSE error + [DONE]."""
        if not self.stream or not self.headers_sent:
            self.h.send_error_json(err)
            return
        self._chunk({'error': err.body()['error']})
        self._done()

    def respond(self, finish, usage, timings, t_recv, extra):
        reasoning = ''.join(self.pieces['reasoning'])
        content = ''.join(self.pieces['content'])
        lp = {'content': self.logprobs} if self.logprobs else None
        calls = [self._tool_call_json(c) for c in self.calls]
        if calls:
            finish = 'tool_calls'
        if self.stream:
            timings['http_ms'] = round((time.perf_counter() - t_recv) * 1000.0, 3)
            if self.chat:
                delta = {} if self.sent_role else {'role': 'assistant', 'content': ''}
                choice = {'index': 0, 'delta': delta, 'finish_reason': finish, 'logprobs': lp}
            else:
                choice = {'index': 0, 'text': '', 'finish_reason': finish, 'logprobs': lp}
            self._chunk(self._envelope(choice, usage=usage, timings=timings, **extra))
            self._done()
            return
        if self.chat:
            message = {'role': 'assistant', 'content': content if (content or not calls) else None}
            if self.thinking or reasoning:
                message['reasoning_content'] = reasoning
            if calls:
                message['tool_calls'] = calls
            choice = {'index': 0, 'message': message, 'finish_reason': finish, 'logprobs': lp}
            obj = {'id': self.id, 'object': 'chat.completion'}
        else:
            choice = {'index': 0, 'text': content, 'finish_reason': finish, 'logprobs': lp}
            obj = {'id': self.id, 'object': 'text_completion'}
        obj.update({'created': self.created, 'model': MODEL_ID, 'choices': [choice], 'usage': usage})
        obj.update(extra)
        timings['http_ms'] = round((time.perf_counter() - t_recv) * 1000.0, 3)
        obj['timings'] = timings
        self.h.send_json(200, obj)


class ApiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, addr, handler, threads):
        super().__init__(addr, handler)
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix='http')

    def process_request(self, request, client_address):
        self.pool.submit(self.process_request_thread, request, client_address)


def cpu_affinity_text():
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return 'unknown'
    ranges, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        ranges.append(f'{start}-{prev}' if start != prev else f'{start}')
        if c is not None:
            start = prev = c
    return f'{len(cpus)} cpus [{",".join(ranges)}]'


def main(argv=None):
    ap = argparse.ArgumentParser(description='OpenAI-compatible front end for the chlorine engine')
    ap.add_argument('--bind', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8731)
    ap.add_argument('--engine-host', default='127.0.0.1')
    ap.add_argument('--engine-port', type=int, default=8740)
    ap.add_argument('--tokenizer', default=DEFAULT_TOKENIZER, help='tokenizer dir (tokenizer.json) or file')
    ap.add_argument('--default-drafter', type=int, default=1, choices=[0, 1, 2], help='0 serial, 1 MTP, 2 DFlash2')
    ap.add_argument('--threads', type=int, default=8, help='HTTP worker threads (engine requests are serialized)')
    ap.add_argument('--default-max-tokens', type=int, default=512)
    ap.add_argument('--max-tokens-cap', type=int, default=65536)
    ap.add_argument('--max-body-bytes', type=int, default=64 << 20)
    ap.add_argument('--queue-timeout', type=float, default=7200.0, help='seconds a request may wait for the engine (503 after)')
    ap.add_argument('--first-token-timeout', type=float, default=1800.0, help='engine silence budget before the first token')
    ap.add_argument('--token-timeout', type=float, default=300.0, help='engine silence budget between tokens')
    ap.add_argument('--verbose', action='store_true', help='log every HTTP request')
    args = ap.parse_args(argv)

    svc = Service(args)
    log(f'api: pid {os.getpid()} cpu affinity {cpu_affinity_text()}')
    svc.load_tokenizer()
    info = svc.probe(wait=1.0, timeout=3.0)
    if info:
        log(f'api: engine {args.engine_host}:{args.engine_port} INFO {info["raw"]} '
            f'(ctx {info["ctx"]}, default drafter {info["default_drafter"]}, kv_slots {info["kv_slots"]})')
    else:
        log(f'api: engine {args.engine_host}:{args.engine_port} not probed ({svc.info_state}); requests will try anyway')
    Handler.svc = svc
    server = ApiServer((args.bind, args.port), Handler, args.threads)
    log(f'api: listening on http://{args.bind}:{args.port} (model {MODEL_ID}, default drafter '
        f'{args.default_drafter} {DRAFTER_NAMES[args.default_drafter]}, {args.threads} http threads, engine serialized)')

    def stop(signum, _frame):
        log(f'api: signal {signum}, stopping')
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        server.pool.shutdown(wait=False, cancel_futures=True)
    log('api: stopped')
    os._exit(0)


if __name__ == '__main__':
    main()
