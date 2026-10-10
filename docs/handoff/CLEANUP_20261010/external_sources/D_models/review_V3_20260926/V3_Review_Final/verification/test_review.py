import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

package = Path(__file__).parent/"package"
if not package.exists():
    package = Path(__file__).parent.parent  # Delivered under verification/.
sys.path.insert(0, str(package))
import selfcheck as S
import infer as I
import numpy as np
from PIL import Image


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        files = {}
        for rel in ["model_dir/your_model.pt", "model_dir/runner.py", "infer.py", "selfcheck.py",
                    "samples/case3_in.jpg", "samples/case3_out.jpg", "samples/case17_in.jpg", "samples/case17_out.jpg"]:
            p = self.root/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(rel.encode())
            files[rel] = {"bytes": p.stat().st_size, "sha256": S.sha(p)}
        self.m = {"source_zip": {"sha256": S.EXPECTED_SOURCE}, "cases": ["case3", "case17"], "files": files}
        self.save()
        self.model_patch = patch.object(S, "EXPECTED_MODEL", files["model_dir/your_model.pt"]["sha256"])
        self.runner_patch = patch.object(S, "EXPECTED_RUNNER", files["model_dir/runner.py"]["sha256"])
        self.model_patch.start()
        self.runner_patch.start()

    def tearDown(self):
        self.model_patch.stop()
        self.runner_patch.stop()
        self.tmp.cleanup()

    def save(self):
        (self.root/"manifest.json").write_text(json.dumps(self.m), encoding="utf-8")

    def test_identity_complete_pass(self):
        self.assertEqual(S.verify_identity(self.root)["cases"], ["case3", "case17"])

    def test_missing_reference_fails(self):
        (self.root/"samples/case3_out.jpg").unlink()
        with self.assertRaises(ValueError): S.verify_identity(self.root)

    def test_missing_samples_fails(self):
        for p in (self.root/"samples").iterdir(): p.unlink()
        with self.assertRaises(ValueError): S.verify_identity(self.root)

    def test_missing_manifest_entry_fails(self):
        del self.m["files"]["samples/case17_out.jpg"]
        self.save()
        with self.assertRaises(ValueError): S.verify_identity(self.root)

    def test_changed_weight_fails(self):
        (self.root/"model_dir/your_model.pt").write_bytes(b"OTHER")
        with self.assertRaises(ValueError): S.verify_identity(self.root)

    def test_changed_runner_even_updated_manifest_fails(self):
        p = self.root/"model_dir/runner.py"
        p.write_bytes(b"OTHER")
        self.m["files"]["model_dir/runner.py"] = {"bytes":5,"sha256":S.sha(p)}
        self.save()
        with self.assertRaises(ValueError): S.verify_identity(self.root)

    def test_source_identity_fails(self):
        self.m["source_zip"]["sha256"] = "wrong"
        self.save()
        with self.assertRaises(ValueError): S.verify_identity(self.root)

    def test_path_escape_fails(self):
        with self.assertRaises(ValueError): S.safe_path(self.root, "../outside")

    def test_case_omission_fails(self):
        self.m["cases"] = ["case3"]
        self.save()
        with self.assertRaises(ValueError): S.verify_identity(self.root)

    def test_jpeg_byte_and_pixel_different_levels(self):
        b = io.BytesIO()
        Image.new("RGB", (32,32), (100,100,100)).save(b, format="JPEG", quality=98, subsampling=0)
        original = b.getvalue()
        self.assertTrue(S.compare_jpeg(original, original)["jpeg_byte_equal"])
        # Trailing bytes do not change decoded RGB, but strict byte criterion fails.
        r = S.compare_jpeg(original+b"metadata", original)
        self.assertFalse(r["jpeg_byte_equal"])
        self.assertTrue(r["decoded_rgb_equal"])

    def test_small_pixel_difference_is_not_approximate_pass(self):
        data = []
        for val in [100,101]:
            b = io.BytesIO()
            Image.new("RGB", (32,32), (val,val,val)).save(b, format="JPEG", quality=98, subsampling=0)
            data.append(b.getvalue())
        r = S.compare_jpeg(*data)
        self.assertFalse(r["jpeg_byte_equal"])
        self.assertFalse(r["decoded_rgb_equal"])

    def test_shared_budget_reserve_before_call_and_no_refund(self):
        p = self.root/"ledger.json"
        p.write_text(json.dumps({"deadline_epoch":time.time()+60,"reserved_calls":0,"call_cap":4}))
        S.Ledger(p).reserve(4)
        with self.assertRaises(RuntimeError): S.Ledger(p).reserve(1)
        self.assertEqual(json.loads(p.read_text())["reserved_calls"],4)

    def test_expired_deadline_fails(self):
        p = self.root/"ledger.json"
        p.write_text(json.dumps({"deadline_epoch":time.time()-1,"reserved_calls":0,"call_cap":4}))
        with self.assertRaises(RuntimeError): S.Ledger(p).reserve(1)

    def test_4k_geometry_63(self):
        self.assertEqual(len(I.positions(4096))*len(I.positions(3072)),63)

    def test_undersized_input_rejected(self):
        with self.assertRaises(ValueError): I.positions(511)

    def test_historical_border_retained(self):
        class Identity:
            def infer(self, x, prompt=""): return x
        out = I.render_4k(Identity(), np.full((512,512,3),255,np.uint8))
        self.assertEqual(out[0].max(),0)
        self.assertEqual(out[-1].max(),0)
        self.assertEqual(out[:,0].max(),0)
        self.assertEqual(out[:,-1].max(),0)
        self.assertTrue(np.all(out[256,256] == 255))


if __name__ == "__main__":
    unittest.main(verbosity=2)
