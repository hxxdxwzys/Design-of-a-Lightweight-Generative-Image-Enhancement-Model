# -*- coding: utf-8 -*-
"""verify_c1 比较逻辑的 CPU 测试（不加载真实模型、不用 GPU）。

覆盖此前两次真实失败的根因：CPU/GPU 设备不匹配；并验证形状校验与阈值判定路径。
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EB = HERE.parent
sys.path.insert(0, str(EB))

import verify_c1  # noqa: E402


class TestCompareCase(unittest.TestCase):
    def _write(self, tmp, name, arr):
        p = Path(tmp) / name
        np.save(p, arr)
        return p

    def test_identical_cpu_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = (np.random.default_rng(0).random((1, 3, 16, 16)) * 2 - 1).astype(np.float32)
            p = self._write(tmp, 'a.npy', a)
            rec = verify_c1.compare_case('case', p, torch.from_numpy(a))
            self.assertEqual(rec['same_precision_max_abs'], 0.0)
            self.assertEqual(rec['same_precision_mean_abs'], 0.0)
            self.assertTrue(rec['png_pixel_identical'])
            self.assertEqual(rec['out_shape'], [1, 3, 16, 16])

    def test_shape_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, 'a.npy', np.zeros((1, 3, 8, 8), dtype=np.float32))
            with self.assertRaises(ValueError):
                verify_c1.compare_case('case', p, torch.zeros(1, 3, 9, 9))

    def test_fp32_reference_and_lsb_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = np.zeros((1, 3, 8, 8), dtype=np.float32)
            p = self._write(tmp, 'a.npy', a)
            ref = self._write(tmp, 'ref.npy', np.full((1, 3, 8, 8), 2.0 / 255.0,
                                                      dtype=np.float32))
            rec = verify_c1.compare_case('case', p, torch.from_numpy(a), ref)
            self.assertAlmostEqual(rec['fp32_ref']['max_abs'], 2.0 / 255.0, places=7)
            self.assertFalse(rec['fp32_ref']['png_pixel_identical'])
            self.assertIn('skipped', verify_c1.compare_case(
                'case', p, torch.from_numpy(a),
                self._write(tmp, 'ref2.npy', np.zeros((1, 3, 4, 4), dtype=np.float32)))
                ['fp32_ref'])

    def test_float16_output_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = np.zeros((1, 3, 8, 8), dtype=np.float32)
            rec = verify_c1.compare_case('case', self._write(tmp, 'a.npy', a),
                                         torch.zeros(1, 3, 8, 8, dtype=torch.float16))
            self.assertEqual(rec['same_precision_max_abs'], 0.0)
            self.assertEqual(rec['out_dtype'], 'torch.float16')


if __name__ == '__main__':
    unittest.main(verbosity=2)
