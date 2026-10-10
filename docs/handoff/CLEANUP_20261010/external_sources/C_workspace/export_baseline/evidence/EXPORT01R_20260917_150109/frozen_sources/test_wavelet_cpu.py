# -*- coding: utf-8 -*-
"""EXPORT-01 C0/C1 CPU 测试：wavelet 忠实张量化（EXP-05 前置）。"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / 'pipeline_baseline'))

import core as PBcore  # noqa: E402
from chain import WaveletFix, gaussian_kernel_1d  # noqa: E402


class TestKernel(unittest.TestCase):
    def test_ksize_from_sigma(self):
        k = gaussian_kernel_1d(5.0)
        self.assertEqual(int(k.numel()), 41)
        self.assertAlmostEqual(float(k.sum()), 1.0, places=6)

    def test_matches_cv2_kernel(self):
        import cv2
        k_cv = cv2.getGaussianKernel(41, 5.0).ravel()
        k = gaussian_kernel_1d(5.0).numpy().astype(np.float64)
        # 核以 float32 存储（供 fp16 链使用），残差 ~1e-9 即 float32 表示误差；
        # float64 形式与 cv2 的差为 2.8e-17（见 evidence 记录）
        self.assertLess(float(np.abs(k - k_cv).max()), 1e-7)


class TestQuantization(unittest.TestCase):
    def test_matches_to_pil_image_truncation(self):
        from torchvision.transforms.functional import to_pil_image
        torch.manual_seed(0)
        t = torch.rand(3, 8, 8)
        got = WaveletFix._quantize_u8(t)
        ref = np.array(to_pil_image(t.clamp(0, 1))).transpose(2, 0, 1)
        self.assertTrue(np.array_equal(got.numpy(), ref))

    def test_half_boundary_is_truncated_not_rounded(self):
        # t*255 == 127.5 精确可表示：截断得 127（舍入会得 128）
        t = torch.full((1, 1, 1, 1), 0.5)
        self.assertEqual(int(WaveletFix._quantize_u8(t).item()), 127)


class TestBlur(unittest.TestCase):
    def test_matches_cv2(self):
        import cv2
        torch.manual_seed(1)
        w = WaveletFix(5.0)
        for shape in [(1, 3, 96, 80), (1, 3, 64, 64), (1, 3, 200, 151)]:
            x = torch.rand(*shape)
            got = w._blur(x).numpy()
            ref = np.stack([cv2.GaussianBlur(x[0, c].numpy(), (0, 0), 5.0) for c in range(3)])
            self.assertLess(float(np.abs(got[0] - ref).max()), 1e-6)

    def test_reflect_padding_used(self):
        w = WaveletFix(5.0)
        x = torch.rand(1, 3, 8, 8)
        with self.assertRaises(Exception):
            # 尺寸小于核半径时 reflect pad 应报错（而不是静默改成别的边界）
            w._blur(torch.rand(1, 3, 4, 4))


class TestEndToEndEquivalence(unittest.TestCase):
    def _cases(self):
        torch.manual_seed(2)
        return {
            'random_odd': (torch.rand(1, 3, 97, 79) * 2 - 1, torch.rand(1, 3, 97, 79) * 2 - 1),
            'gray': (torch.full((1, 3, 64, 64), -0.2), torch.full((1, 3, 64, 64), 0.3)),
            'extreme': (torch.full((1, 3, 33, 41), 1.0), torch.full((1, 3, 33, 41), -1.0)),
            'gradient': (torch.linspace(-1, 1, 64 * 64).reshape(1, 1, 64, 64).repeat(1, 3, 1, 1),
                         torch.linspace(1, -1, 64 * 64).reshape(1, 1, 64, 64).repeat(1, 3, 1, 1)),
        }

    # 判据（C0 冻结，理由见 evidence/WAVELET_TENSORIZATION.json）：
    #   张量化与 cv2 参考实现共用同一核公式与 reflect101 边界；残差来自卷积累加顺序，
    #   表现为整数个 uint8 LSB 翻转（Δout = 2/255 × k）。判定为 ≤1 LSB 且翻转像素占比 ≤1e-3；
    #   要求逐位相等在更大尺寸上不成立，故不写成硬判据（不得事后放宽，此处是事前登记值）。
    MAX_ABS = 2.0 / 255.0 * 1.0000001
    MAX_FRAC = 1e-3

    def test_matches_reference_within_1lsb(self):
        w = WaveletFix(5.0)
        for name, (tgt, src) in self._cases().items():
            with self.subTest(case=name):
                ref = PBcore.wavelet_color_fix_tensor(tgt, src, sigma=5)
                got = w(tgt, src)
                self.assertEqual(tuple(got.shape), tuple(ref.shape))
                d = (ref - got).abs()
                frac = float((d > 0).float().mean())
                self.assertLessEqual(float(d.max()), self.MAX_ABS,
                                     'wavelet 张量化残差超过 1 LSB（%s）' % name)
                self.assertLessEqual(frac, self.MAX_FRAC,
                                     'wavelet 张量化翻转像素占比过高（%s: %.3e）' % (name, frac))

    def test_shape_mismatch_rejected(self):
        w = WaveletFix(5.0)
        with self.assertRaises(ValueError):
            w(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 9, 8))

    def test_output_range(self):
        w = WaveletFix(5.0)
        got = w(torch.rand(1, 3, 40, 40) * 2 - 1, torch.rand(1, 3, 40, 40) * 2 - 1)
        self.assertGreaterEqual(float(got.min()), -1.0)
        self.assertLessEqual(float(got.max()), 1.0)


class TestScriptAndTrace(unittest.TestCase):
    def test_scriptable(self):
        m = torch.jit.script(WaveletFix(5.0))
        a = torch.rand(1, 3, 48, 48) * 2 - 1
        b = torch.rand(1, 3, 48, 48) * 2 - 1
        self.assertTrue(torch.equal(m(a, b), WaveletFix(5.0)(a, b)))

    def test_traced_graph_is_shape_generic(self):
        w = WaveletFix(5.0)
        tr = torch.jit.trace(w, (torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)))
        for shape in [(1, 3, 64, 64), (1, 3, 200, 150)]:
            a = torch.rand(*shape) * 2 - 1
            b = torch.rand(*shape) * 2 - 1
            self.assertTrue(torch.equal(tr(a, b), w(a, b)),
                            'trace 后的 wavelet 必须对其它尺寸同样生效 %s' % (shape,))


if __name__ == '__main__':
    unittest.main(verbosity=2)
