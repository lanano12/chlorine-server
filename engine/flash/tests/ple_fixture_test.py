#!/usr/bin/env python3
"""Replay frozen independent CPU oracle without a PyTorch dependency."""
import json
import math
import pathlib
import subprocess
import sys

fixture=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert fixture['status']=='passed'
fixtures=[]
for case in fixture['cases']:
    if 'conv_fixture' in case:
        f=fixture['conv_fixtures'][case['conv_fixture']]
        fixtures.append({'request':dict(f['request'],chunks=case['chunks']),'reference':f['reference']})
    else:fixtures.append(case)
requests=[f['request'] for f in fixtures]
p=subprocess.run([sys.argv[1]],input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,check=True)
rows=[json.loads(line) for line in p.stdout.splitlines()]
assert len(rows)==len(requests)
for row,f in zip(rows,fixtures):
    assert row['mode']=='causal-dilation3-v1'
    ref=f['reference']
    if 'shifted' in ref:assert row['shifted']==ref['shifted']
    else:
        assert len(row['output'])==len(ref['output'])
        for value,expected in zip(row['output'],ref['output']):
            assert math.isfinite(value) and abs(value-expected)<=1e-5+1e-4*abs(expected)
assert all(r['output']==rows[0]['output'] and r['ring']==rows[0]['ring'] for r in rows[:8])
print('PASS 11 independent PLE fixtures, eight chunk schedules have identical outputs and final rings')
