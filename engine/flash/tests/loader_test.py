"""Adversarial GGUF fixtures exercised by the production C++ reader."""
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

INSPECT = sys.argv.pop(1)

def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b

def gguf(tensors=(), kv=(), version=3):
    h = b'GGUF' + struct.pack('<IQQ', version, len(tensors), len(kv))
    for name, value in kv:
        h += string(name) + struct.pack('<Iq', 11, value)
    for name, dims, typ, offset in tensors:
        h += string(name) + struct.pack('<I',len(dims))
        h += b''.join(struct.pack('<Q',d) for d in dims)
        h += struct.pack('<IQ',typ,offset)
    return h + bytes((-len(h)) % 32)

class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def run_file(self, data, message=None, name='test.gguf'):
        p=self.root/name;p.write_bytes(data)
        r=subprocess.run([INSPECT,str(p)],text=True,capture_output=True)
        if message is None:
            self.assertEqual(r.returncode,0,r.stderr)
            return json.loads(r.stdout)
        self.assertNotEqual(r.returncode,0)
        self.assertIn(message,r.stderr)
    def test_valid_block(self):
        j=self.run_file(gguf([('expert',[256],21,0)])+bytes(110))
        self.assertEqual(j['payload_bytes'],110)
    def test_truncated_header(self): self.run_file(b'GG','truncated')
    def test_magic(self): self.run_file(b'NOPE'+bytes(20),'magic')
    def test_version(self): self.run_file(gguf(version=4),'version')
    def test_zero_rank(self): self.run_file(gguf([('bad',[],0,0)]),'rank')
    def test_zero_dim(self): self.run_file(gguf([('bad',[0],0,0)]),'zero dimension')
    def test_overflow(self): self.run_file(gguf([('bad',[2**63,2],0,0)]),'overflow')
    def test_offset_overflow(self): self.run_file(gguf([('bad',[16],30,2**64-32)]),'overflow')
    def test_eof(self): self.run_file(gguf([('bad',[256],21,0)]),'EOF')
    def test_incomplete_block(self): self.run_file(gguf([('bad',[255],21,0)])+bytes(110),'block-aligned')
    def test_unknown_type(self): self.run_file(gguf([('bad',[256],99,0)])+bytes(110),'99 for bad')
    def test_unaligned(self): self.run_file(gguf([('bad',[16],30,1)])+bytes(64),'unaligned')
    def test_duplicate_tensor(self): self.run_file(gguf([('a',[16],30,0),('a',[16],30,32)])+bytes(64),'duplicate')
    def test_duplicate_kv(self): self.run_file(gguf(kv=[('x',1),('x',2)]),'duplicate metadata')
    def test_invalid_alignment(self):
        for v in (0,3,-1,2**30):
            with self.subTest(v=v):self.run_file(gguf(kv=[('general.alignment',v)]),'alignment')
    def test_truncated_string(self):
        b=b'GGUF'+struct.pack('<IQQ',3,1,0)+struct.pack('<Q',2**64-1)
        self.run_file(b,'truncated string')
    def test_shard_metadata(self):
        kv=[('split.no',1),('split.count',1)]
        self.run_file(gguf(kv=kv),'split.no')
    def test_overlap(self):
        self.run_file(gguf([('a',[32],30,0),('b',[16],30,32)])+bytes(64),'overlap')
    def test_empty_first_shard(self):
        for idx,ts in enumerate(([],[('x',[16],30,0)],[('y',[16],30,0)])):
            kv=[('split.no',idx),('split.count',3),('split.tensors.count',2)]
            (self.root/f'model-{idx+1:05}-of-00003.gguf').write_bytes(gguf(ts,kv)+bytes(32*len(ts)))
        p=self.root/'model-00001-of-00003.gguf'
        r=subprocess.run([INSPECT,str(p)],text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(json.loads(r.stdout)['tensor_count'],2)
        (self.root/'model-00003-of-00003.gguf').unlink()
        r=subprocess.run([INSPECT,str(p)],text=True,capture_output=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('00003',r.stderr)

if __name__=='__main__': unittest.main()
