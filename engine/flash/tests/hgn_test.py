#!/usr/bin/env python3
"""Host-only adversarial HGN directory validation."""
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
INSPECT=str(pathlib.Path(sys.argv.pop()).resolve())
HEADER=struct.Struct('<4sIIIQQQ64s')
RECORD=struct.Struct('<96sII4Q3Q')

def fixture(**kw):
    offset=320; data=bytes(64)
    record=[b'test',0,1,32,0,0,0,offset,len(data),0]
    fields={'dtype':1,'rank':2,'dim':3,'offset':7,'size':8,'name':0}
    for key,idx in fields.items():
        if key in kw:record[idx]=kw[key]
    h=[b'HGN1',kw.get('version',2),1,0,104,320,384,b'fixture']
    for key,idx in {'magic':0,'records_offset':4,'data_offset':5,'file_size':6}.items():
        if key in kw:h[idx]=kw[key]
    return HEADER.pack(*h)+RECORD.pack(*record)+bytes(56)+data

class HGNTests(unittest.TestCase):
    def check(self,data,good=False):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'test.hgn';p.write_bytes(data)
            r=subprocess.run([INSPECT,str(p),'--hgn'],capture_output=True,text=True)
            self.assertEqual(r.returncode==0,good,r.stdout+r.stderr)
            self.assertGreaterEqual(r.returncode,0,'loader crashed')
    def test_valid_v1_v2(self):
        for v in (1,2):self.check(fixture(version=v),True)
    def test_truncations(self):
        for n in (0,3,103,105,263,383):self.check(fixture()[:n])
    def test_header(self):
        for f in ({'magic':b'BAD!'},{'version':3},{'records_offset':2**64-1},{'data_offset':104},{'file_size':10}):self.check(fixture(**f))
    def test_records(self):
        for f in ({'rank':0},{'rank':5},{'dim':0},{'dim':2**63},{'dtype':21},{'offset':321},{'offset':2**64-1},{'size':63},{'name':b'x'*96}):self.check(fixture(**f))
    def test_quant_alignment(self):
        self.check(fixture(dtype=7,dim=32));self.check(fixture(dtype=5,dim=33))
    def test_duplicate_and_overlap(self):
        for name,offset in ((b'test',512),(b'other',448)):
            h=HEADER.pack(b'HGN1',2,2,0,104,448,576,b'fixture')
            r1=RECORD.pack(b'test',0,1,32,0,0,0,448,64,0)
            r2=RECORD.pack(name,0,1,32,0,0,0,offset,64,0)
            self.check(h+r1+r2+bytes(24)+bytes(128))
if __name__=='__main__':unittest.main()
