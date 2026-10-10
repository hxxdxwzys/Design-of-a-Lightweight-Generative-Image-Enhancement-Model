import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import common


class Tests(unittest.TestCase):
    def test_initial_fp32_equal(self):
        torch.manual_seed(42);p=common.parent();p.ref.float();g=common.initial_ref();x=torch.randn(1,3,32,32)
        with torch.no_grad():self.assertTrue(torch.allclose(p.ref(x),g(x),atol=1e-6,rtol=0))
    def test_parameter_count(self):self.assertEqual(sum(p.numel() for p in common.Refiner().parameters()),5523)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):common.finite(torch.tensor([float('nan')]))
    def test_parent_identity_rejects_change(self):
        with patch.object(common,'sha',return_value='bad'):
            with self.assertRaises(ValueError):common.parent()
    def test_budget_rejects_before_work(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'ledger.json').write_text(json.dumps({'caps':{'cache':0},'used':{'cache':0}}))
            with patch.object(common,'ROOT',root),patch.object(common,'guard'):
                with self.assertRaises(RuntimeError):common.reserve('cache')


if __name__=='__main__':unittest.main(verbosity=2)
