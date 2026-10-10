import json,tempfile,unittest
import hashlib
from pathlib import Path
import numpy as np
from infer import positions,mask_at,render
from selfcheck import verify_identity


class TestEdge(unittest.TestCase):
    def test_coverage(self):
        for H,W in [(512,512),(513,517),(1024,1024),(3072,4096)]:
            a=np.zeros((H,W,1),np.float32)
            for y in positions(H):
                for x in positions(W):a[y:y+512,x:x+512]+=mask_at(y,x,H,W)
            self.assertGreater(float(a.min()),0)
    def test_identity_all_levels(self):
        class Identity:
            def infer(self,x,prompt=''):return x
        img=(np.arange(513*517*3)%256).astype(np.uint8).reshape(513,517,3)
        self.assertTrue(np.array_equal(render(Identity(),img),img))
    def test_identity_white(self):
        class Identity:
            def infer(self,x,prompt=''):return x
        a=np.full((513,517,3),255,np.uint8)
        self.assertTrue(np.array_equal(render(Identity(),a),a))
    def test_nonfinite_rejected(self):
        class Bad:
            def infer(self,x,prompt=''):return x*float('nan')
        with self.assertRaises(RuntimeError):render(Bad(),np.zeros((512,512,3),np.uint8))
    def test_small_input_rejected(self):
        with self.assertRaises(ValueError):positions(511)
    def test_missing_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):verify_identity(d)
    def test_missing_entries_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'manifest.json').write_text(json.dumps({'files':{}}))
            with self.assertRaises(ValueError):verify_identity(d)
    def fixture(self,d):
        root=Path(d);names=['infer.py','selfcheck.py','requirements.txt','model_dir/your_model.pt','model_dir/runner.py']
        for rel in names:
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
        m={'version':'V4t1_EdgeFix_20260927','files':{rel:{'bytes':7,'sha256':hashlib.sha256(b'fixture').hexdigest()} for rel in names}}
        (root/'manifest.json').write_text(json.dumps(m));return root,m
    def test_content_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,_=self.fixture(d);(root/'infer.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed file'):verify_identity(root)
    def test_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,_=self.fixture(d);(root/'extra.py').write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'File set'):verify_identity(root)
    def test_nonparent_model_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,_=self.fixture(d)
            with self.assertRaisesRegex(ValueError,'Not V4t1 parent'):verify_identity(root)


if __name__=='__main__':unittest.main(verbosity=2)
