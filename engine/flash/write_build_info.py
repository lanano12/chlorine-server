#!/usr/bin/env python3
"""Record compiler command and hashes immediately after successful HIP build."""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
ap=argparse.ArgumentParser()
ap.add_argument('--binary',type=pathlib.Path,required=True);ap.add_argument('--root',type=pathlib.Path,required=True)
ap.add_argument('--hipcc',required=True);ap.add_argument('--ple-causal',choices=('0','1'),required=True)
a=ap.parse_args();root=a.root.resolve();binary=a.binary.resolve()
command=[a.hipcc,'--offload-arch=gfx1151','-O3','-Werror','-DFLASH_PLE_CAUSAL='+a.ple_causal,
         str(root/'upstream/src/gpu/gdec.cpp'),'-lrocblas','-lhipblaslt','-o',str(binary)]
files=[p for folder in ('upstream/src','quant') for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.hpp','.inc')]
files += [root/'ple_contract.hpp',root/'model_config.hpp']
record={'schema_version':1,'binary_sha256':sha(binary),'hipcc_argv':command,'hipcc_command':shlex.join(command),
        'hipcc_version':subprocess.check_output([a.hipcc,'--version'],text=True).strip(),
        'ple_semantics':'causal-dilation3-v1' if a.ple_causal=='1' else 'legacy-nine-slot-alias-v1',
        'source_files':{str(p.relative_to(root)):sha(p) for p in sorted(files)}}
binary.with_suffix('.build.json').write_text(json.dumps(record,indent=2)+'\n')
