# -*- coding: utf-8 -*-
"""EXPORT-01R 行为测试：EXP-R01（分块融合语义）与 EXP-R03（量化前 NaN/Inf 拦截）。

要求（current_task §0A）：
  - 用**真实 DitStep** + 真实 runner 分块代码，对照**独立**算式 z − dt·加权速度场；
  - 覆盖单块 / 多块（含重叠）/ 64×65 奇数潜码，零/常数/空间变化速度场；
  - 原始解码输出 clamp 前、wavelet 量化前的 NaN/Inf 必须抛错，且证明“仅检查最终输出”不够。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EB = HERE.parent
sys.path.insert(0, str(EB))
sys.path.insert(0, str(EB.parent / 'pipeline_baseline'))

import core as PBcore  # noqa: E402
import delivery_runner as DR  # noqa: E402
from chain import DitStep, WaveletFix  # noqa: E402

_VOSR = str(PBcore.load_config()['vosr_dir'])
if _VOSR not in sys.path:
    sys.path.insert(0, _VOSR)
import tiled_vae as TV  # noqa: E402


class StubDit(torch.nn.Module):
    """速度场可解析控制的 DiT 替身（保持 forward_flexible 契约）。"""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward_flexible(self, x, t, r, z):
        lq, zz = x[:, :4], x[:, 4:]
        if self.mode == 'zero':
            return torch.zeros_like(zz)
        if self.mode == 'const':
            return torch.full_like(zz, 0.25)
        if self.mode == 'varying':
            return lq * 0.5 - zz * 0.1
        raise AssertionError(self.mode)


class BundleStub(torch.nn.Module):
    def __init__(self, dit_step):
        super().__init__()
        self.dit_step = dit_step

    def step(self, lq, z, tc, tn, feats):
        return self.dit_step(lq, z, tc, tn, feats)


class ChainStub(torch.nn.Module):
    """可注入 NaN/Inf 的整链替身（用于 EXP-R03 边界测试）。"""

    def __init__(self, bad=None):
        super().__init__()
        self.bad = bad or ''

    def _poison(self, t, tag):
        if self.bad == tag:
            t = t.clone()
            t.view(-1)[0] = float('inf') if tag == 'enc' else float('nan')
        return t

    def encode(self, x, tile_px):
        y = F.avg_pool2d(x, 8)
        z = torch.cat([y, y], 1)[:, :4]
        return self._poison(z, 'enc')

    def features(self, x):
        return self._poison(torch.zeros(1, 1024, 768), 'feat')

    def step(self, lq, z, tc, tn, feats):
        return self._poison(z, 'step')

    def decode(self, z, tile_latent):
        pix = F.interpolate(z[:, :3], scale_factor=8, mode='nearest')
        return self._poison(pix, 'dec')

    def post(self, sr, src):
        return sr


def make_runner(tmp, dit_step=None, chain=None, dtype='float32'):
    d = Path(tmp)
    recipe = {'model_file': 'your_model.pt', 'weight_dtype': dtype, 'seed': 42,
              't_start': 1.0, 't_end': 0.0, 'dit_tile_size': 512, 'dit_tile_overlap_cli': 32,
              'vae_tile_size': 1024, 'vae_tile_overlap_cli': 32, 'patch_size': 2,
              'vae_enc_trace_px': [512, 1024], 'dec_trace_latent': [64, 128]}
    (d / 'models_export.json').write_text(json.dumps(recipe), encoding='utf-8')

    class Id(torch.nn.Module):
        def forward(self, x):
            return x

    torch.jit.save(torch.jit.script(Id()), str(d / 'your_model.pt'))
    r = DR.Runner(str(d), device='cpu')
    if dit_step is not None:
        r.model = BundleStub(dit_step)
    elif chain is not None:
        r.model = chain
    return r


class TestExpR01Fusion(unittest.TestCase):
    """分块融合必须等于 z − dt·(Σg·u/Σg)；不得把已是 z1 的返回值再减一次。"""

    def _one_case(self, mode, lh, lw):
        with tempfile.TemporaryDirectory() as tmp:
            dit = StubDit(mode)
            step = DitStep(dit).eval()
            r = make_runner(tmp, dit_step=step)
            r.recipe['patch_size'] = 2
            r.dit_tile, r.dit_overlap_cli = 512, 32
            lt, lov = r._dit_params(lh, lw)
            h_pos, w_pos = DR._make_tile_grid(lh, lt, lov), DR._make_tile_grid(lw, lt, lov)
            lq = torch.randn(1, 4, lh, lw)
            feats = {(hi, wi): torch.randn(1, 1024, 768) for hi in h_pos for wi in w_pos}
            gen = torch.Generator(device='cpu')
            gen.manual_seed(42)
            got = r._one_step(lq, gen, feats, lh, lw, lt, h_pos, w_pos)

            # 独立参考：同样种子重放噪声，直接调用替身速度场并自行融合
            gen2 = torch.Generator(device='cpu')
            gen2.manual_seed(42)
            z = torch.randn(lq.shape, generator=gen2, device='cpu', dtype=torch.float32)
            dt = r.t_cur_val - r.t_nxt_val
            if lh <= lt and lw <= lt:
                u = dit.forward_flexible(torch.cat([lq, z], 1), torch.full((1,), 1.0),
                                         torch.full((1,), 0.0), [feats[(0, 0)]])
                expect = z - dt * u
            else:
                g = TV._gaussian_weights(lt, lt, int(lq.shape[1]), torch.device('cpu'))
                u_acc = torch.zeros_like(z)
                w_acc = torch.zeros_like(z)
                for hi in h_pos:
                    for wi in w_pos:
                        he, we = hi + lt, wi + lt
                        u = dit.forward_flexible(
                            torch.cat([lq[:, :, hi:he, wi:we], z[:, :, hi:he, wi:we]], 1),
                            torch.full((1,), 1.0), torch.full((1,), 0.0), [feats[(hi, wi)]])
                        u_acc[:, :, hi:he, wi:we] += u * g
                        w_acc[:, :, hi:he, wi:we] += g
                expect = z - dt * (u_acc / w_acc)
            return got, expect, z

    def test_zero_velocity_preserves_noise(self):
        # EXP-R01 核心反例：零速度场时 z1 必须等于 z（旧实现会再减一次得到 ~0）
        for lh, lw in ((64, 64), (64, 65), (96, 128)):
            with self.subTest(shape=(lh, lw)):
                got, expect, z = self._one_case('zero', lh, lw)
                self.assertTrue(torch.allclose(got, expect, atol=1e-6),
                                'zero 速度场必须保持 z：max diff %.3e'
                                % float((got - expect).abs().max()))
                if lh <= lw and (lh, lw) == (64, 64):
                    self.assertTrue(torch.allclose(got, z, atol=1e-6))

    def test_const_and_varying_velocity_match_independent_formula(self):
        for mode in ('const', 'varying'):
            for lh, lw in ((64, 64), (64, 65), (96, 128)):
                with self.subTest(mode=mode, shape=(lh, lw)):
                    got, expect, _ = self._one_case(mode, lh, lw)
                    d = float((got - expect).abs().max())
                    self.assertLess(d, 1e-6, '分块融合与独立算式不一致：%.3e' % d)

    def test_old_buggy_formula_differs(self):
        # 证明该测试能抓住旧缺陷（z - dt·blend(z1) ≠ z - dt·blend(u)）
        got, expect, z = self._one_case('const', 64, 65)
        lq = torch.zeros(1, 4, 64, 65)
        lt, lov = 64, 8
        h_pos, w_pos = DR._make_tile_grid(64, lt, lov), DR._make_tile_grid(65, lt, lov)
        g = TV._gaussian_weights(lt, lt, 4, torch.device('cpu'))
        z1_acc = torch.zeros_like(z)
        w_acc = torch.zeros_like(z)
        u = torch.full_like(z, 0.25)
        for hi in h_pos:
            for wi in w_pos:
                he, we = hi + lt, wi + lt
                z1_acc[:, :, hi:he, wi:we] += (z - 1.0 * u)[:, :, hi:he, wi:we] * g
                w_acc[:, :, hi:he, wi:we] += g
        buggy = z - 1.0 * (z1_acc / w_acc)
        self.assertGreater(float((buggy - expect).abs().max()), 1e-3)


class TestExpR03NonFinite(unittest.TestCase):
    def test_nan_inf_rejected_before_clamp_and_quantization(self):
        expect_where = {'enc': 'VAE 编码潜码', 'feat': 'DINO 条件特征',
                        'step': 'DiT 一步更新后的潜码', 'dec': '解码原始输出(clamp 前'}
        x = torch.rand(1, 3, 512, 512) * 2 - 1
        for tag, where in expect_where.items():
            with self.subTest(stage=tag):
                with tempfile.TemporaryDirectory() as tmp:
                    r = make_runner(tmp, chain=ChainStub(bad=tag))
                    with self.assertRaises(DR.RunnerError) as cm:
                        r.infer(x, prompt='')
                    self.assertIn(where, str(cm.exception))

    def test_wavelet_would_mask_nonfinite(self):
        # 证明“仅检查最终输出”不充分：NaN 经 wavelet 的 clamp/uint8 量化后变成有限像素
        nan = torch.full((1, 3, 64, 64), float('nan'))
        x = torch.rand(1, 3, 64, 64) * 2 - 1
        out = WaveletFix(5.0)(nan, x)
        self.assertTrue(bool(torch.isfinite(out).all()))
        inf = torch.full((1, 3, 64, 64), float('inf'))
        self.assertTrue(bool(torch.isfinite(WaveletFix(5.0)(inf, x)).all()))

    def test_clean_chain_still_passes(self):
        x = torch.rand(1, 3, 512, 512) * 2 - 1
        with tempfile.TemporaryDirectory() as tmp:
            r = make_runner(tmp, chain=ChainStub(bad=None))
            out = r.infer(x, prompt='')
            self.assertEqual(tuple(out.shape), (1, 3, 512, 512))
            self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
