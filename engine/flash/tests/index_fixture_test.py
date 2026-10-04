#!/usr/bin/env python3
"""Replay pinned HF index fixtures with no torch/model/GPU dependency."""
import json
import math
import pathlib
import subprocess
import sys

fixture = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert fixture['status'] == 'passed'
requests = [case['request'] for case in fixture['fixtures']]
proc = subprocess.run([sys.argv[1], '--json'], text=True, capture_output=True,
                      input=''.join(json.dumps(r) + '\n' for r in requests), check=True)
actual = [json.loads(line) for line in proc.stdout.splitlines()]
assert len(actual) == len(requests)
queries = 0
for got, case in zip(actual, fixture['fixtures']):
    assert len(got['rows']) == len(case['rows'])
    for row, ref in zip(got['rows'], case['rows']):
        assert row['position'] == ref['position']
        assert row['blocks'] == ref['blocks']
        p = ref['position']
        tokens = [b * 4 + t for b in ref['blocks'] for t in range(4)]
        tokens += list(range((p + 1) // 4 * 4, p + 1))
        assert row['tokens'] == tokens
        for field in ('query0', 'last_completed_key'):
            assert len(row[field]) == len(ref[field])
            assert all(math.isfinite(a) and math.isfinite(b) and
                       abs(a - b) <= 1e-5 + 1e-4 * abs(b)
                       for a, b in zip(row[field], ref[field]))
        queries += 1
print(f'PASS {queries} frozen independent HF index queries; exact block/token masks, bounded FP32 norm/RoPE error')
