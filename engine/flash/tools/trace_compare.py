#!/usr/bin/env python3
"""Compare bounded diagnostic traces; this never promotes a benchmark/model gate."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import struct

ABS, REL, FLOOR = 1e-5, 1e-4, 1e-12
MAX_BYTES, MAX_RECORDS = 64 << 20, 2048

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def load(directory, ignore_phase=False):
    directory = pathlib.Path(directory).resolve()
    index = directory / 'index.jsonl'
    if index.stat().st_size > 2 << 20:
        raise ValueError('trace index too large')
    lines = [json.loads(line) for line in index.read_text().splitlines()]
    if not lines:
        raise ValueError('empty trace')
    header = lines[0]
    if (header.get('schema_version') != 1 or header.get('kind') != 'chlorine-flash-diagnostic-trace'
            or header.get('dtype') != 'f32le' or header.get('timing_eligible') is not False
            or header.get('graph_replay') is not False):
        raise ValueError('unsupported or timing-eligible diagnostic trace')
    identity = header.get('identity', {})
    if not re.fullmatch('[0-9a-f]{64}', identity.get('manifest_sha256', '')):
        raise ValueError('trace requires checkpoint manifest identity')
    rows, files, total, request = {}, set(), 0, 0
    for row in lines[1:]:
        if row.get('event') == 'request':
            if type(row.get('request')) is not int or row['request'] != request + 1:
                raise ValueError('invalid request sequence')
            request += 1
            continue
        if row.get('event') != 'tensor' or row.get('dtype') != 'f32le':
            raise ValueError('invalid trace event/dtype')
        if any(type(row.get(k)) is not int for k in ('request', 'layer', 'position', 'bytes')):
            raise ValueError('invalid trace coordinates')
        if row['request'] != request or not -1 <= row['layer'] < 48 or not 0 <= row['position'] < 262144:
            raise ValueError('invalid trace coordinates')
        if row['phase'] not in ('prefill', 'decode') or not isinstance(row['stage'], str):
            raise ValueError('invalid trace stage/phase')
        if not re.fullmatch(r'tensor-\d+\.f32', row['file']) or row['file'] in files:
            raise ValueError('invalid or reused tensor file')
        shape = row['shape']
        if len(shape) != 1 or type(shape[0]) is not int or not 0 < shape[0] <= 248320 or row['bytes'] != shape[0] * 4:
            raise ValueError('invalid trace shape/size')
        total += row['bytes']
        if total > MAX_BYTES or len(rows) >= MAX_RECORDS:
            raise ValueError('trace exceeds bounded capture limits')
        file = directory / row['file']
        if file.is_symlink() or file.resolve().parent != directory or file.stat().st_size != row['bytes']:
            raise ValueError('invalid tensor path/size')
        raw = file.read_bytes()
        fnv = 14695981039346656037
        for byte in raw:
            fnv = ((fnv ^ byte) * 1099511628211) & ((1 << 64) - 1)
        if str(fnv) != row['fnv1a64']:
            raise ValueError('trace tensor checksum mismatch')
        values = struct.unpack('<' + str(shape[0]) + 'f', raw)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite trace tensor')
        key = (row['request'], '' if ignore_phase else row['phase'], row['stage'], row['layer'], row['position'])
        if key in rows:
            raise ValueError('duplicate trace coordinate')
        rows[key] = (values, raw, hashlib.sha256(raw).hexdigest())
        files.add(row['file'])
    if not rows:
        raise ValueError('trace contains no tensors')
    return header, rows, digest(index)

def nll(logits, target):
    peak = max(logits)
    return peak + math.log(math.fsum(math.exp(v - peak) for v in logits)) - logits[target]

def compare(reference, candidate, contract='fp32', ignore_phase=False, labels=None, reference_id=None):
    rh, rr, rs = load(reference, ignore_phase)
    ch, cr, cs = load(candidate, ignore_phase)
    if rh['identity']['manifest_sha256'] != ch['identity']['manifest_sha256']:
        raise ValueError('checkpoint manifest mismatch')
    if rr.keys() != cr.keys():
        raise ValueError('trace coordinates differ; recapture the same rows/layers/stages')
    if contract not in ('exact', 'fp32', 'model'):
        raise ValueError('unknown comparison contract')
    if contract == 'model' and (ignore_phase or not labels or not reference_id):
        raise ValueError('model comparison requires named reference and explicit target labels; phases must match')
    rows, logits, matches, deltas = [], 0, 0, []
    for key in sorted(rr):
        rv, rb, rhash = rr[key];cv, cb, chash = cr[key]
        if len(rv) != len(cv):
            raise ValueError('tensor shape mismatch')
        errors = [abs(a - b) for a, b in zip(cv, rv)]
        bitdiffs = sum(rb[i:i+4] != cb[i:i+4] for i in range(0, len(rb), 4))
        row = {'coordinate': key, 'values': len(rv), 'maxabs': max(errors),
               'maxrel': max(e / max(FLOOR, abs(v)) for e, v in zip(errors, rv)),
               'rms': math.sqrt(math.fsum(e*e for e in errors) / len(errors)),
               'bit_differences': bitdiffs, 'fp32_budget_passed': all(e <= ABS + REL * abs(v) for e, v in zip(errors, rv)),
               'reference_tensor_sha256': rhash, 'candidate_tensor_sha256': chash}
        if key[2] == 'logits':
            logits += 1
            rt = max(range(len(rv)), key=rv.__getitem__);ct = max(range(len(cv)), key=cv.__getitem__)
            matches += rt == ct;row.update(reference_top1=rt, candidate_top1=ct)
            if contract == 'model':
                lk = (key[0], key[1], key[4])
                if lk not in labels or type(labels[lk]) is not int or not 0 <= labels[lk] < len(rv):
                    raise ValueError('missing or invalid target label')
                delta = nll(cv, labels[lk]) - nll(rv, labels[lk]);deltas.append(delta)
                row.update(target_id=labels[lk], nll_increase_nats=delta)
        rows.append(row)
    if contract == 'model':
        expected = {(k[0], k[1], k[4]) for k in rr if k[2] == 'logits'}
        if not logits or set(labels) != expected:
            raise ValueError('target labels must cover exactly every captured logit row')
        passed = matches/logits >= .99 and math.fsum(deltas)/logits <= .01
    else:
        passed = all(r['bit_differences'] == 0 if contract == 'exact' else r['fp32_budget_passed'] for r in rows)
    return {'schema_version': 1, 'kind': 'diagnostic-trace-comparison', 'status': 'passed' if passed else 'failed',
            'qualification': 'diagnostic-only; no benchmark or complete-model gate promotion',
            'contract': contract, 'reference_id': reference_id, 'phase_matching': not ignore_phase,
            'reference': {'identity': rh['identity'], 'index_sha256': rs},
            'candidate': {'identity': ch['identity'], 'index_sha256': cs},
            'budget': {'absolute': ABS, 'relative': REL, 'relative_reporting_floor': FLOOR,
                       'minimum_top1_agreement': .99, 'maximum_mean_nll_increase_nats': .01},
            'top1_agreement': matches/logits if logits else None,
            'mean_nll_increase_nats': math.fsum(deltas)/logits if deltas else None, 'tensors': rows}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=pathlib.Path, required=True)
    ap.add_argument('--candidate', type=pathlib.Path, required=True)
    ap.add_argument('--contract', choices=('exact', 'fp32', 'model'), default='fp32')
    ap.add_argument('--ignore-phase', action='store_true', help='explicit prefill/decode operator comparison')
    ap.add_argument('--reference-id')
    ap.add_argument('--labels', type=pathlib.Path, help='JSON list of request/phase/position/target_id')
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    labels = None
    if a.labels:
        entries = json.loads(a.labels.read_text())
        labels = {(r['request'], r['phase'], r['position']): r['target_id'] for r in entries}
        if len(labels) != len(entries):
            ap.error('duplicate label coordinate')
    result = compare(a.reference, a.candidate, a.contract, a.ignore_phase, labels, a.reference_id)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2);f.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'contract', 'top1_agreement', 'mean_nll_increase_nats')}))
    return int(result['status'] != 'passed')

if __name__ == '__main__':
    raise SystemExit(main())
