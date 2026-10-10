# -*- coding: utf-8 -*-
"""EXPORT-01 CPU 测试：交付 runner 控制面、分块参数等价、磁盘/打包守卫（不需 GPU、不加载真实模型）。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EB = HERE.parent
sys.path.insert(0, str(EB))
sys.path.insert(0, str(EB.parent / 'pipeline_baseline'))

import core as PBcore  # noqa: E402
import delivery_runner as DR  # noqa: E402

_VOSR = str(PBcore.load_config()['vosr_dir'])   # 参考实现源码目录（只读）
if _VOSR not in sys.path:
    sys.path.insert(0, _VOSR)
import tiled_vae as TV  # noqa: E402


def make_stub_model_dir(tmp, recipe_overrides=None, with_model=True):
    d = Path(tmp)
    recipe = {
        'model_file': 'your_model.pt', 'weight_dtype': 'float16', 'seed': 42,
        't_start': 1.0, 't_end': 0.0, 'dit_tile_size': 512, 'dit_tile_overlap_cli': 32,
        'vae_tile_size': 1024, 'vae_tile_overlap_cli': 32, 'patch_size': 2,
        'vae_enc_trace_px': [512, 1024], 'dec_trace_latent': [64, 128],
    }
    recipe.update(recipe_overrides or {})
    (d / 'models_export.json').write_text(json.dumps(recipe), encoding='utf-8')
    if with_model:
        class Id(torch.nn.Module):
            def forward(self, x):
                return x
        torch.jit.save(torch.jit.script(Id()), str(d / 'your_model.pt'))
    return str(d)


class TestRunnerControlPlane(unittest.TestCase):
    def test_missing_recipe_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DR.RunnerError):
                DR.Runner(tmp, device='cpu')

    def test_missing_model_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_stub_model_dir(tmp, with_model=False)
            with self.assertRaises(DR.RunnerError):
                DR.Runner(d, device='cpu')

    def test_bad_input_shape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_stub_model_dir(tmp)
            r = DR.Runner(d, device='cpu')
            for bad in (torch.zeros(3, 8, 8), torch.zeros(2, 3, 8, 8), torch.zeros(1, 1, 8, 8),
                        [[1.0]]):
                with self.assertRaises(DR.RunnerError):
                    r.infer(bad)

    def test_recipe_missing_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_stub_model_dir(tmp)
            p = Path(d) / 'models_export.json'
            rec = json.loads(p.read_text(encoding='utf-8'))
            rec.pop('vae_enc_trace_px')
            p.write_text(json.dumps(rec), encoding='utf-8')
            with self.assertRaises(Exception):
                DR.Runner(d, device='cpu')


class TestTilingMathEquivalence(unittest.TestCase):
    """交付 runner 的分块参数必须与参考实现逐值一致（纯数学，无 GPU）。"""

    def setUp(self):
        self.cfg = PBcore.load_config()
        self.r = DR.Runner.__new__(DR.Runner)      # 只测纯函数，不构造模型
        self.r.recipe = {'patch_size': int(self.cfg['expected_patch_size'])}
        self.r.dit_tile = int(self.cfg['dit_tile_size'])
        self.r.dit_overlap_cli = int(self.cfg['dit_tile_overlap_cli'])
        self.r.vae_tile = int(self.cfg['vae_tile_size'])
        self.r.vae_overlap_cli = int(self.cfg['vae_tile_overlap_cli'])

    def test_dit_params_match_core(self):
        for lh, lw in ((64, 64), (64, 65), (512, 384), (384, 512), (8, 8), (1, 1)):
            with self.subTest(shape=(lh, lw)):
                self.assertEqual(self.r._dit_params(lh, lw),
                                 PBcore.dit_tile_params(self.cfg, 2, lh, lw))

    def test_vae_params_match_core(self):
        for lh, lw in ((64, 64), (64, 65), (512, 384), (384, 512), (8, 8), (1, 1)):
            with self.subTest(shape=(lh, lw)):
                self.assertEqual(self.r._vae_params(lh, lw),
                                 PBcore.vae_tile_params(self.cfg, lh, lw))

    def test_gaussian_weights_and_grid_match_tiled_vae(self):
        for tile, length, overlap in ((64, 512, 8), (128, 384, 16), (64, 65, 8)):
            with self.subTest(case=(tile, length, overlap)):
                self.assertEqual(DR._make_tile_grid(length, tile, overlap),
                                 TV._make_tile_grid(length, tile, overlap))
        g1 = DR._gaussian_weights(16, 24, 4, torch.device('cpu'))
        g2 = TV._gaussian_weights(16, 24, 4, torch.device('cpu'))
        self.assertTrue(torch.equal(g1, g2))


class TestDiskGuard(unittest.TestCase):
    def test_ok_and_fail(self):
        import disk_guard
        self.assertGreater(disk_guard.require_free(EB, 0.001, 'smoke'), 0.001)
        with self.assertRaises(SystemExit):
            disk_guard.require_free(EB, 10 ** 6, 'impossible')

    def test_cli_exit_codes(self):
        good = subprocess.run([sys.executable, '-B', str(EB / 'disk_guard.py'), str(EB), '0.001'],
                              capture_output=True, text=True)
        bad = subprocess.run([sys.executable, '-B', str(EB / 'disk_guard.py'), str(EB), '1e6'],
                             capture_output=True, text=True)
        self.assertEqual(good.returncode, 0)
        self.assertNotEqual(bad.returncode, 0)


class TestZipGuards(unittest.TestCase):
    def test_packaging_rejects_non_ascii_name_and_wrong_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'model_dir').mkdir()
            (run / 'model_dir' / 'your_model.pt').write_bytes(b'x')
            (run / 'output_dir').mkdir()
            for i in range(1, 4):
                (run / 'output_dir' / ('case%d.jpg' % i)).write_bytes(b'y')
            (run / 'evidence').mkdir()
            (run / 'evidence' / 'zip_record.json').write_text('{}', encoding='utf-8')
            bad_name = subprocess.run([sys.executable, '-B', str(EB / 'make_zip.py'), str(run),
                                       '--name', '中文包'], capture_output=True, text=True)
            self.assertNotEqual(bad_name.returncode, 0)
            few = subprocess.run([sys.executable, '-B', str(EB / 'make_zip.py'), str(run),
                                  '--name', 'Track2_TEST'], capture_output=True, text=True)
            self.assertNotEqual(few.returncode, 0)
            self.assertIn('100', few.stderr + few.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
