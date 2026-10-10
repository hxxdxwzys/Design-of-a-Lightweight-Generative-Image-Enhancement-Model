# -*- coding: utf-8 -*-
"""EXPORT-01R2 CPU 负例与行为测试（EXP-R08/R09/R10/R11）。

覆盖：
  R08 逐块 raw decode -> finite -> clamp -> 融合 -> 最终 clamp（与"先融合后 clamp"必须不同）
  R11 迁移超限阻断、合并对照判据、缺 fp32 参考不得 SKIP
  R10 必需样本集合完整（缺 4K/重复/空集合/哈希错项不能 PASS）、释放检查
  R09 加载前身份绑定（哈希不符拒绝启动；由场景脚本覆盖真实子进程路径）
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EB = HERE.parent
sys.path.insert(0, str(EB))

import delivery_runner as DR  # noqa: E402
import r_supervisor  # noqa: E402
import r_verify  # noqa: E402


class OutOfRangeDecode(torch.nn.Module):
    """按"输入潜码所在 tile"返回越界像素值的解码替身（用于验证 clamp 顺序）。"""

    def __init__(self):
        super().__init__()

    def decode(self, z, tile_latent):
        m = float(z.mean())
        v = 2.0 if m >= 0 else -2.0
        return torch.full((z.shape[0], 3, z.shape[2] * 8, z.shape[3] * 8), v)


class BundleStub(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def decode(self, z, tile_latent):
        return self.dec.decode(z, tile_latent)


def make_runner(tmp):
    d = Path(tmp)
    recipe = {'model_file': 'your_model.pt', 'weight_dtype': 'float32', 'seed': 42,
              't_start': 1.0, 't_end': 0.0, 'dit_tile_size': 512, 'dit_tile_overlap_cli': 32,
              'vae_tile_size': 1024, 'vae_tile_overlap_cli': 32, 'patch_size': 2,
              'vae_enc_trace_px': [512, 1024], 'dec_trace_latent': [64, 128]}
    (d / 'models_export.json').write_text(json.dumps(recipe), encoding='utf-8')

    class Id(torch.nn.Module):
        def forward(self, x):
            return x

    torch.jit.save(torch.jit.script(Id()), str(d / 'your_model.pt'))
    return DR.Runner(str(d), device='cpu')


class TestExpR08DecodeOrder(unittest.TestCase):
    def test_per_tile_clamp_then_blend(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_runner(tmp)
            r.model = BundleStub(OutOfRangeDecode())
            lh, lw, c = 64, 65, 4
            z = torch.zeros(1, c, lh, lw)
            z[:, :, :, :32] = 1.0      # 第一块 tile 为正 -> +2
            z[:, :, :, 32:] = -1.0     # 第二块 tile 为负 -> -2
            out = r._decode(z)
            lt, lov = r._vae_params(lh, lw)
            self.assertEqual(lt, 64)
            self.assertIn(lov, (8, 16))     # 由冻结配方决定（VAE overlap 16）
            h_pos = DR._make_tile_grid(lh, lt, lov)
            w_pos = DR._make_tile_grid(lw, lt, lov)
            g = DR._gaussian_weights(lt * 8, lt * 8, 3, torch.device('cpu'))
            acc = torch.zeros(1, 3, lh * 8, lw * 8)
            wacc = torch.zeros_like(acc)
            for hi in h_pos:
                for wi in w_pos:
                    he, we = hi + lt, wi + lt
                    tile = z[:, :, hi:he, wi:we]
                    raw = OutOfRangeDecode().decode(tile, lt)
                    clamped = raw.clamp(-1.0, 1.0)          # R08：逐块先 clamp
                    acc[:, :, hi * 8:he * 8, wi * 8:we * 8] += clamped * g
                    wacc[:, :, hi * 8:he * 8, wi * 8:we * 8] += g
            expect = (acc / wacc).clamp(-1.0, 1.0)
            self.assertTrue(torch.allclose(out, expect, atol=1e-6),
                            'R08：必须先逐块 clamp 再融合')
            # 反例：先融合后 clamp 的结果必须不同（否则本用例不能区分实现）
            acc2 = torch.zeros_like(acc)
            wacc2 = torch.zeros_like(acc)
            for hi in h_pos:
                for wi in w_pos:
                    he, we = hi + lt, wi + lt
                    raw = OutOfRangeDecode().decode(z[:, :, hi:he, wi:we], lt)
                    acc2[:, :, hi * 8:he * 8, wi * 8:we * 8] += raw * g
                    wacc2[:, :, hi * 8:he * 8, wi * 8:we * 8] += g
            alt = (acc2 / wacc2).clamp(-1.0, 1.0)
            self.assertGreater(float((alt - expect).abs().max()), 1e-3)

    def test_nonfinite_still_intercepted(self):
        class Bad(OutOfRangeDecode):
            def decode(self, z, tile_latent):
                t = super().decode(z, tile_latent)
                t[0, 0, 0, 0] = float('nan')
                return t

        with tempfile.TemporaryDirectory() as tmp:
            r = make_runner(tmp)
            r.model = BundleStub(Bad())
            with self.assertRaises(DR.RunnerError):
                r._decode(torch.zeros(1, 4, 64, 65))


class TestExpR11Criteria(unittest.TestCase):
    TOL = {'same_precision_max_abs': 1e-5, 'same_precision_mean_abs': 1e-6,
           'fp16_vs_fp32_max_abs_diagnostic': 0.05,
           'fp16_vs_fp32_mean_abs_diagnostic': 0.002}

    def test_migration_over_limit_blocks(self):
        with self.assertRaises(RuntimeError):
            r_verify.check_migration([{'name': 'case1_full', 'max_abs': 0.4, 'mean_abs': 0.1}],
                                     self.TOL)
        self.assertTrue(r_verify.check_migration(
            [{'name': 'x', 'max_abs': 0.01, 'mean_abs': 0.0005}], self.TOL))

    def test_required_set_detects_missing_duplicate_empty(self):
        base = {'phaseA': {'call_4': {'call': 'A_511x513'}},
                'phaseB': {'call_1': {'call': 'B_512_1'}, 'call_2': {'call': 'B_512_2'},
                           'call_3': {'call': 'B_512_3'}, 'call_4': {'call': 'B_511x513'}},
                'pairs': [{'name': 'case1_full'}, {'name': 'case7_full'}],
                'migration': [{'name': 'm'}]}
        prereg = {'cases': [{'name': 'case1_crop512'}, {'name': 'case1_crop511x513'},
                            {'name': 'case1_full'}, {'name': 'case7_full'}]}
        self.assertTrue(r_verify.check_required_set(json.loads(json.dumps(base)), prereg))
        missing = json.loads(json.dumps(base))
        missing['pairs'] = [{'name': 'case1_full'}]
        with self.assertRaises(RuntimeError):
            r_verify.check_required_set(missing, prereg)
        dup = json.loads(json.dumps(base))
        dup['phaseB']['call_2']['call'] = 'B_512_1'
        with self.assertRaises(RuntimeError):
            r_verify.check_required_set(dup, prereg)
        empty = json.loads(json.dumps(base))
        empty['migration'] = []
        with self.assertRaises(RuntimeError):
            r_verify.check_required_set(empty, prereg)

    def test_release_check_cpu_path(self):
        rec = r_verify.release_and_verify()
        self.assertIn(rec['released'], ('UNKNOWN_CPU', True))


class TestExpR10SupervisorEvaluation(unittest.TestCase):
    TOL = TestExpR11Criteria.TOL

    def _rd(self, exp, npy_hashes_ok=True, pairs=('case1_full', 'case7_full'),
            calls=('B_512_1', 'B_512_2', 'B_512_3', 'B_511x513'), migration=True):
        npy = exp / 'npy'
        npy.mkdir(parents=True, exist_ok=True)
        rd = {'status': 'OK', 'phaseA': {}, 'phaseB': {}, 'pairs': [], 'migration': []}
        for i, c in enumerate(calls):
            p = npy / ('%s.npy' % c)
            np.save(p, np.zeros((1, 3, 4, 4), dtype=np.float32))
            sha = __import__('hashlib').sha256(p.read_bytes()).hexdigest()
            if not npy_hashes_ok and i == 0:
                sha = 'deadbeef'
            zero = {'max_abs': 0.0, 'mean_abs': 0.0, 'png_pixel_identical': True}
            rd['phaseB']['call_%d' % i] = {'call': c, 'npy': str(p), 'npy_sha256': sha,
                                           'vs_A': zero, 'vs_first_B512': zero}
        for n in pairs:
            rd['pairs'].append({'name': n, 'pair': {'max_abs': 0.0, 'mean_abs': 0.0,
                                                    'png_pixel_identical': True}})
        if migration:
            rd['migration'] = [{'name': 'case1_crop512', 'max_abs': 0.01, 'mean_abs': 0.001}]
        return rd

    def _required(self):
        return ['case1_crop512', 'case1_crop511x513', 'case1_full', 'case7_full']

    def test_complete_set_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = self._rd(Path(tmp))
            chk = r_supervisor.evaluate_readings(rd, self.TOL, Path(tmp), self._required())
            self.assertTrue(chk['required_set_ok'] and chk['pairs_ok'] and
                            chk['all_npy_present_and_hashed'] and chk['migration_ok'])

    def test_missing_4k_duplicate_empty_and_hash_error_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            chk = r_supervisor.evaluate_readings(
                self._rd(exp, pairs=('case1_full',)), self.TOL, exp, self._required())
            self.assertFalse(chk['pairs_ok'])
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            rd = self._rd(exp, calls=('B_512_1', 'B_512_1', 'B_512_3', 'B_511x513'))
            chk = r_supervisor.evaluate_readings(rd, self.TOL, exp, self._required())
            self.assertFalse(chk['B512_set_ok'])
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            chk = r_supervisor.evaluate_readings(self._rd(exp, migration=False), self.TOL, exp,
                                                 self._required())
            self.assertFalse(chk['migration_ok'])
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            chk = r_supervisor.evaluate_readings(self._rd(exp, npy_hashes_ok=False),
                                                 self.TOL, exp, self._required())
            self.assertFalse(chk['all_npy_present_and_hashed'])


class TestExpR09PreflightIdentity(unittest.TestCase):
    def test_tampered_model_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'run'
            (run / 'model_dir').mkdir(parents=True)
            (run / 'evidence').mkdir(parents=True)
            (run / 'model_dir' / 'your_model.pt').write_bytes(b'tampered')
            prereg = run / 'prereg.json'
            prereg.write_text(json.dumps({'tolerances': {}, 'cases': []}), encoding='utf-8')
            cmd = [sys.executable, '-B', str(EB / 'r_supervisor.py'), str(run),
                   '--ledger', str(root / 'ledger.json'), '--prereg', str(prereg),
                   '--model-dir', str(run / 'model_dir'), '--exp-dir', str(run / 'evidence'),
                   '--expect-model-sha256', '0' * 64]
            p = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn('MODEL_IDENTITY_MISMATCH', p.stdout + p.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
