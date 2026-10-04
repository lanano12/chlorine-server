#!/usr/bin/env python3
"""Replay frozen HF eager-attention samples without PyTorch or a GPU."""
import json
import math
import pathlib
import subprocess
import sys
fixture=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert fixture['status']=='passed'
requests=[f['request'] for f in fixture['fixtures']]
p=subprocess.run([sys.argv[1],'--json'],input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,check=True)
rows=[json.loads(r) for r in p.stdout.splitlines()]
assert len(rows)==len(requests)
samples=0
for row,f in zip(rows,fixture['fixtures']):
    assert row['sources']==f['sources']
    assert len(row['output'])==f['request']['rows']*24*256
    got=[row['output'][(r*24+h)*256+d] for r in range(f['request']['rows']) for h in range(24) for d in fixture['channels']]
    assert len(got)==len(f['sampled_output'])
    assert all(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-5+1e-4*abs(b) for a,b in zip(got,f['sampled_output']))
    samples+=len(got)
print(f'PASS {len(rows)} independent HF attention fixtures; {samples} sampled values, exact selected causal sources')
