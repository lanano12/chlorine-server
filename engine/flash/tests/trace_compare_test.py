#!/usr/bin/env python3
"""Exercise strict trace admission and independent error/quality calculations."""
import json
import math
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
import trace_compare as tc
BINARY = sys.argv.pop(1)

class TraceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.root = pathlib.Path(self.temp.name)
        self.ref, self.got = self.root/'ref', self.root/'got'
        subprocess.run([BINARY, '--emit', str(self.ref)], check=True)
        shutil.copytree(self.ref, self.got)
    def tearDown(self):
        self.temp.cleanup()
    def edit(self, change):
        rows = [json.loads(x) for x in (self.got/'index.jsonl').read_text().splitlines()]
        change(rows)
        (self.got/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def tensor(self, values):
        raw = struct.pack('<'+str(len(values))+'f', *values)
        (self.got/'tensor-1.f32').write_bytes(raw)
        fnv = 14695981039346656037
        for b in raw:fnv = ((fnv^b)*1099511628211)&((1<<64)-1)
        self.edit(lambda r: r[-1].update(fnv1a64=str(fnv)))
    def test_exact_and_model_metrics(self):
        self.assertEqual(tc.compare(self.ref,self.got,'exact')['status'],'passed')
        r = tc.compare(self.ref,self.got,'model',labels={(1,'prefill',3):2},reference_id='host-test')
        self.assertEqual(r['top1_agreement'],1);self.assertEqual(r['mean_nll_increase_nats'],0)
    def test_precision_budget_and_bit_identity(self):
        self.tensor([0,1,2.00001,-1])
        self.assertEqual(tc.compare(self.ref,self.got,'fp32')['status'],'passed')
        self.assertEqual(tc.compare(self.ref,self.got,'exact')['status'],'failed')
        self.tensor([0,1,3,-1])
        self.assertEqual(tc.compare(self.ref,self.got,'fp32')['status'],'failed')
    def test_quality_independent_nll_and_top1(self):
        self.tensor([0,1,0,-1])
        r=tc.compare(self.ref,self.got,'model',labels={(1,'prefill',3):2},reference_id='host-test')
        self.assertEqual(r['status'],'failed');self.assertEqual(r['top1_agreement'],0)
        expected=math.log(2+math.exp(1)+math.exp(-1))-math.log(1+math.exp(1)+math.exp(2)+math.exp(-1))+2
        self.assertAlmostEqual(r['mean_nll_increase_nats'],expected,places=12)
    def test_quality_requires_targets_and_named_reference(self):
        with self.assertRaises(ValueError):tc.compare(self.ref,self.got,'model')
        with self.assertRaises(ValueError):tc.compare(self.ref,self.got,'model',labels={(1,'prefill',2):0},reference_id='test')
    def test_corruption_and_nonfinite(self):
        (self.got/'tensor-1.f32').write_bytes(b'bad')
        with self.assertRaises(ValueError):tc.load(self.got)
        self.tensor([0,1,float('nan'),-1])
        with self.assertRaises(ValueError):tc.load(self.got)
    def test_metadata_and_coordinate_mismatch(self):
        self.edit(lambda r: r[0]['identity'].update(manifest_sha256='c'*64))
        with self.assertRaises(ValueError):tc.compare(self.ref,self.got)
        self.edit(lambda r: r[0]['identity'].update(manifest_sha256='a'*64))
        self.edit(lambda r: r[-1].update(position=2))
        with self.assertRaises(ValueError):tc.compare(self.ref,self.got)
    def test_timing_and_filename_admission(self):
        self.edit(lambda r:r[0].update(timing_eligible=True))
        with self.assertRaises(ValueError):tc.load(self.got)
        self.edit(lambda r:r[0].update(timing_eligible=False))
        self.edit(lambda r:r[-1].update(file='../escape'))
        with self.assertRaises(ValueError):tc.load(self.got)
    def test_explicit_cross_phase_and_duplicate(self):
        self.edit(lambda r:r[-1].update(phase='decode'))
        with self.assertRaises(ValueError):tc.compare(self.ref,self.got)
        self.assertEqual(tc.compare(self.ref,self.got,ignore_phase=True)['status'],'passed')
        self.edit(lambda r:r.append(dict(r[-1])))
        with self.assertRaises(ValueError):tc.load(self.got)

if __name__=='__main__':unittest.main()
