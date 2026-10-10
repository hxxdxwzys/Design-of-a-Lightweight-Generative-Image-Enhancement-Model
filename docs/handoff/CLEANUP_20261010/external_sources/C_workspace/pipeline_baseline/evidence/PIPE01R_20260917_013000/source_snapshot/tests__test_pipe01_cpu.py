# -*- coding: utf-8 -*-
"""PIPE-01 阶段A CPU 测试（mock / 临时小输入；不加载真实模型、不启动 CUDA、不联网）。

覆盖任务单 §5 要求：
  两入口调用同一核心 / 权重与 LoRA 加载失败传播 / 配置缺失与冲突 /
  输出尺寸与范围校验 / 局部 seed 不随图片顺序漂移 / 缺资源不联网 /
  任何失败不写“成功 manifest” / CLI --help 不加载模型或启动 CUDA /
  非整倍数形状（511x513）的补齐与裁回逻辑 / 冻结分块数值与任务单声明一致。
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

import core  # noqa: E402


def _cfg():
    return core.load_config()


class _UnsafePayload:
    """可被 pickle 但 weights_only=True 必须拒绝的对象（用于验证失败传播）。"""
    pass


class TestConfigFrozen(unittest.TestCase):
    def test_frozen_config_loads_and_locks(self):
        cfg = _cfg()
        self.assertEqual(cfg['dtype'], 'float32')
        self.assertFalse(cfg['autocast'])
        self.assertFalse(cfg['torch_compile'])
        self.assertTrue(cfg['vae_posterior_mode'])
        self.assertEqual(int(cfg['infer_steps']), 1)
        self.assertEqual(float(cfg['t_start']), 1.0)
        self.assertEqual(float(cfg['t_end']), 0.0)

    def _tamper(self, **kw):
        raw = json.loads((PKG / 'config.json').read_text(encoding='utf-8'))
        raw.update(kw)
        p = Path(tempfile.mkdtemp()) / 'cfg.json'
        p.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
        return p

    def test_rejects_tampered_switches(self):
        for kw in ({'dtype': 'float16'}, {'autocast': True}, {'torch_compile': True},
                   {'vae_posterior_mode': False}, {'infer_steps': 2},
                   {'t_start': 0.9}, {'align_method': 'adain'}, {'upscale': 2},
                   {'pad_mode': 'zeros'}, {'crop_back_to_input': False}):
            with self.assertRaises(core.ConfigError, msg=kw):
                core.load_config(self._tamper(**kw))

    def test_rejects_unknown_and_missing_keys(self):
        with self.assertRaises(core.ConfigError):
            core.load_config(self._tamper(new_key=1))
        raw = json.loads((PKG / 'config.json').read_text(encoding='utf-8'))
        raw.pop('lora_scale')
        p = Path(tempfile.mkdtemp()) / 'cfg2.json'
        p.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
        with self.assertRaises(core.ConfigError):
            core.load_config(p)

    def test_tile_params_match_task_sheet(self):
        cfg = _cfg()
        rep = core.tile_config_report(cfg, 2)
        self.assertEqual(rep['dit_tile_latent'], 64)
        self.assertEqual(rep['dit_overlap_latent'], 8)
        self.assertEqual(rep['dit_effective_overlap_px'], 64)
        self.assertEqual(rep['dit_stride_px'], 448)
        self.assertEqual(rep['vae_tile_latent'], 128)
        self.assertEqual(rep['vae_overlap_latent'], 16)
        self.assertEqual(rep['vae_effective_overlap_px'], 128)
        self.assertEqual(rep['vae_stride_px'], 896)

    def test_tile_report_rejects_recipe_change(self):
        p = self._tamper(dit_tile_size=640)
        with self.assertRaises(core.ConfigError):
            core.tile_config_report(core.load_config(p), 2)

    def test_args_ns_conflict_is_rejected(self):
        p = self._tamper(expected_ae_type='qwen')
        with self.assertRaises(core.ConfigError):
            core.build_args_ns(core.load_config(p))


class TestSharedCoreAndEntries(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='pipe01_cpu_'))
        self.calls = []
        self._orig_infer = core.infer_core
        self._orig_assets = core.Assets

        def fake_infer(assets, tensor, seed=None):
            import torch
            self.calls.append(('infer_core', tuple(tensor.shape)))
            out = torch.zeros_like(tensor)
            core.validate_output(out, int(tensor.shape[2]), int(tensor.shape[3]))
            return out

        class FakeAssets:
            def __init__(self, cfg, device):
                self.cfg, self.device = cfg, device
                self.asset_fp = {'fake': 1}
                self.source_fp = {'fake': 1}
                self.effective_fp = {'model': {'digest_sha256': 'x'}}
                self.load_report = {'fake': True}
                self.tile_report = {}
                self.args = None

        core.infer_core = fake_infer
        core.Assets = FakeAssets

    def tearDown(self):
        core.infer_core = self._orig_infer
        core.Assets = self._orig_assets

    def test_both_entries_call_same_core(self):
        import torch
        from PIL import Image
        import offline
        from runner import Runner
        cfg = _cfg()
        img = self.tmp / 'a.png'
        Image.fromarray((torch.zeros(3, 8, 8, dtype=torch.uint8)).permute(1, 2, 0).numpy()).save(img)
        r = Runner(cfg['base_dir'])
        r._ensure = lambda: FakeAssetsProxy(cfg)
        out_a = r.infer(torch.zeros(1, 3, 8, 8), prompt='')
        rec = offline.process_image(FakeAssetsProxy(cfg), img, self.tmp / 'out')
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0][1], self.calls[1][1])
        self.assertEqual(tuple(out_a.shape), (1, 3, 8, 8))
        self.assertTrue(Path(rec['png_path']).is_file())

    def test_prompt_must_be_empty(self):
        import torch
        from runner import Runner
        r = Runner(_cfg()['base_dir'])
        r._ensure = lambda: FakeAssetsProxy(_cfg())
        with self.assertRaises(core.PipelineError):
            r.infer(torch.zeros(1, 3, 8, 8), prompt='x')

    def test_model_dir_must_exist(self):
        from runner import Runner
        with self.assertRaises(core.AssetError):
            Runner(str(self.tmp / 'nope'))

    def test_png_encoder_is_shared_and_pixel_identical(self):
        import numpy as np
        import torch
        t = torch.rand(1, 3, 16, 16) * 2 - 1
        p1, p2 = self.tmp / 'x1.png', self.tmp / 'x2.png'
        core.tensor_to_pil(t).save(p1)
        core.tensor_to_pil(t).save(p2)
        from PIL import Image
        a = np.array(Image.open(p1).convert('RGB'))
        b = np.array(Image.open(p2).convert('RGB'))
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(core.sha256_file(p1), core.sha256_file(p2))


class FakeAssetsProxy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = 'cpu'


class TestFailurePropagation(unittest.TestCase):
    def test_lora_unsafe_payload_fails_closed(self):
        import torch
        p = Path(tempfile.mkdtemp()) / 'bad.pt'
        torch.save({'state_dict': {'x': torch.zeros(1)}, 'weird': _UnsafePayload()}, p)
        with self.assertRaises(core.AssetError):
            core.load_lora_payload(p)

    def test_lora_missing_file_fails(self):
        with self.assertRaises(core.AssetError):
            core.load_lora_payload(Path(tempfile.mkdtemp()) / 'nope.pt')

    def test_missing_dino_assets_do_not_go_online(self):
        import torch
        called = []
        orig = torch.hub.load
        torch.hub.load = lambda *a, **k: called.append(a) or (_ for _ in ()).throw(
            AssertionError('不得调用 torch.hub.load'))
        try:
            cfg = core.load_config()
            bad = dict(cfg)
            bad['dinov2_ckpt'] = str(Path(tempfile.mkdtemp()) / 'missing.pth')
            with self.assertRaises(core.OfflineError):
                core.load_dinov2_local(bad, 'cpu')
            bad2 = dict(cfg)
            bad2['dinov2_repo'] = str(Path(tempfile.mkdtemp()) / 'missing_repo')
            with self.assertRaises(core.OfflineError):
                core.load_dinov2_local(bad2, 'cpu')
            self.assertEqual(called, [])
        finally:
            torch.hub.load = orig

    def test_output_validation(self):
        import torch
        good = torch.zeros(1, 3, 8, 8)
        self.assertTrue(core.validate_output(good, 8, 8))
        with self.assertRaises(core.OutputValidationError):
            core.validate_output(torch.zeros(1, 3, 8, 9), 8, 8)
        with self.assertRaises(core.OutputValidationError):
            core.validate_output(torch.zeros(1, 3, 8, 8, dtype=torch.float16), 8, 8)
        with self.assertRaises(core.OutputValidationError):
            core.validate_output(torch.full((1, 3, 8, 8), float('nan')), 8, 8)
        with self.assertRaises(core.OutputValidationError):
            core.validate_output(torch.full((1, 3, 8, 8), 1.5), 8, 8)

    def test_failure_does_not_write_success_manifest(self):
        out = Path(tempfile.mkdtemp(prefix='pipe01_neg_')) / 'run'
        p = subprocess.run(
            [sys.executable, '-B', str(PKG / 'pair_verify.py'), '--run-dir', str(out),
             '--prereg', str(out / 'prereg.json'), '--case1', 'x.jpg', '--case7', 'y.jpg'],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('预登记', (p.stdout or '') + (p.stderr or ''))
        self.assertFalse(out.exists(), '拒绝执行时不得创建 run 目录')

    def test_make_evidence_refuses_existing_run_dir(self):
        d = Path(tempfile.mkdtemp(prefix='pipe01_exists_'))
        p = subprocess.run(
            [sys.executable, '-B', str(PKG / 'make_evidence.py'), '--run-dir', str(d)],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            cwd=str(PKG), env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('拒绝覆盖', (p.stdout or '') + (p.stderr or ''))


class TestSeedAndShapes(unittest.TestCase):
    def test_local_seed_is_order_independent(self):
        import torch
        order1 = ['a', 'b']
        order2 = ['b', 'a']
        vals1 = {}
        for name in order1:
            g = core.make_generator(42, 'cpu')
            vals1[name] = torch.randn(4, generator=g)
        vals2 = {}
        for name in order2:
            g = core.make_generator(42, 'cpu')
            vals2[name] = torch.randn(4, generator=g)
        for name in ('a', 'b'):
            self.assertTrue(torch.equal(vals1[name], vals2[name]))
        g1, g2 = core.make_generator(42, 'cpu'), core.make_generator(42, 'cpu')
        self.assertTrue(torch.equal(torch.randn(8, generator=g1), torch.randn(8, generator=g2)))
        g3 = core.make_generator(43, 'cpu')
        self.assertFalse(torch.equal(torch.randn(8, generator=g1), torch.randn(8, generator=g3)))

    def test_pad_and_crop_for_non_multiple_shapes(self):
        import torch
        core.ensure_vosr_import_path(_cfg())
        import tiled_vae as TV
        x = torch.zeros(1, 3, 511, 513)
        padded, h, w = TV._pad_to_multiple(x, 8)
        self.assertEqual((h, w), (511, 513))
        self.assertEqual(tuple(padded.shape[2:]), (512, 520))
        cropped = padded[:, :, :h, :w]
        self.assertEqual(tuple(cropped.shape[2:]), (511, 513))
        ok = torch.zeros(1, 3, 512, 512)
        padded2, h2, w2 = TV._pad_to_multiple(ok, 8)
        self.assertEqual(tuple(padded2.shape[2:]), (512, 512))


class TestDriverEndToEndCPU(unittest.TestCase):
    """用假核心把 pair_verify 的**全部驱动路径**在 CPU 上跑一遍（不是模型前向）。

    覆盖：记录/预算计数/成对对比/重复512/PNG 同一性/判据失败时不得报 OK。
    这样在花 GPU 预算前就能发现驱动缺陷。"""

    def setUp(self):
        import numpy as np
        from PIL import Image
        self.tmp = Path(tempfile.mkdtemp(prefix='pipe01_drv_'))
        rng = np.random.RandomState(0)
        for name in ('case1.jpg', 'case7.jpg'):
            Image.fromarray(rng.randint(0, 255, (640, 640, 3), dtype=np.uint8)).save(
                self.tmp / name, quality=95)
        self.run = self.tmp / 'run'
        self.run.mkdir()
        (self.run / 'prereg.json').write_text(json.dumps({
            'criteria': {'max_abs_diff': 1e-5, 'mean_abs_diff': 1e-6},
            'budget': {'max_forwards': 10, 'max_minutes': 30, 'max_output_gib': 4},
        }), encoding='utf-8')
        self._orig = (core.Assets, core.infer_core)

    def tearDown(self):
        core.Assets, core.infer_core = self._orig

    def _install_fake(self, delta_b=0.0):
        import torch
        made = {'n': 0}

        class FakeAssets:
            def __init__(self, cfg, device):
                made['n'] += 1
                self.cfg, self.device = cfg, device
                self.tag = 'A' if made['n'] == 1 else 'B'
                self.asset_fp = {'fake': 1}
                self.source_fp = {'fake': 1}
                self.effective_fp = {'model': {'digest_sha256': 'fake'}}
                self.load_report = {'fake': True}
                self.tile_report = {}

        class FakeRunner:  # Runner 内部用 core.Assets，这里只需 identity/release
            pass

        def fake_infer(assets, tensor, seed=None):
            y = torch.tanh(tensor.float() * 0.5)
            if delta_b and getattr(assets, 'tag', '') == 'B':
                y = y + float(delta_b)
            return y

        core.Assets = FakeAssets
        core.infer_core = fake_infer

    def _drive(self, delta_b=0.0):
        self._install_fake(delta_b)
        import pair_verify
        rc = pair_verify.main(['--run-dir', str(self.run), '--prereg',
                               str(self.run / 'prereg.json'), '--case1',
                               str(self.tmp / 'case1.jpg'), '--case7',
                               str(self.tmp / 'case7.jpg'), '--crop', '512',
                               '--device', 'cpu'])
        return rc, json.loads((self.run / 'paired_readings.json').read_text(encoding='utf-8'))

    def test_driver_full_flow_passes_with_identical_entries(self):
        rc, res = self._drive(0.0)
        self.assertEqual(rc, 0, res.get('instances', {}).get('error'))
        self.assertEqual(res['status'], 'OK')
        self.assertEqual(res['forwards'], 8)
        self.assertEqual(len(res['pairs']), 3)
        for tag, v in res['pairs'].items():
            self.assertTrue(v['pass_max'] and v['pass_mean'], tag)
            self.assertTrue(v['png_identical'], tag)
        self.assertEqual(len(res['repeat_512']), 2)
        for k, v in res['repeat_512'].items():
            self.assertTrue(v['pass_max'] and v['pass_mean'], k)
        self.assertTrue(res['criteria_all_pass'])
        self.assertEqual(res['instances']['device'], 'cpu')
        for tag in ('case1_full', 'case7_full'):
            self.assertEqual(res['inputs'][tag]['src_wh'], [640, 640])
        self.assertEqual(res['inputs']['case1_crop512']['crop_box'], [64, 64, 576, 576])

    def test_driver_marks_criteria_fail_instead_of_ok(self):
        rc, res = self._drive(1e-3)
        self.assertEqual(rc, 1)
        self.assertEqual(res['status'], 'CRITERIA_FAIL')
        self.assertFalse(res['criteria_all_pass'])
        self.assertFalse(res['pairs']['case1_crop512']['pass_max'])

    def test_driver_refuses_without_prereg(self):
        (self.run / 'prereg.json').unlink()
        with self.assertRaises(SystemExit):
            import pair_verify
            pair_verify.main(['--run-dir', str(self.run),
                              '--prereg', str(self.run / 'prereg.json'),
                              '--case1', str(self.tmp / 'case1.jpg'),
                              '--case7', str(self.tmp / 'case7.jpg')])


class TestImportSafety(unittest.TestCase):
    def test_imports_do_not_pull_torch(self):
        code = ("import sys, io, contextlib\n"
                "buf = io.StringIO()\n"
                "with contextlib.redirect_stdout(buf):\n"
                "    import core, runner, offline\n"
                "print('TORCH' if 'torch' in sys.modules else 'NO_TORCH')\n"
                "print('CUDA' if getattr(sys.modules.get('torch'), 'cuda', None) else 'NO_CUDA')\n")
        p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                           cwd=str(PKG), encoding='utf-8', errors='replace')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('NO_TORCH', p.stdout)

    def test_cli_help_does_not_load_model_or_cuda(self):
        code = ("import runpy, sys, io, contextlib\n"
                "sys.argv = ['offline.py', '--help']\n"
                "buf = io.StringIO()\n"
                "try:\n"
                "    with contextlib.redirect_stdout(buf):\n"
                "        runpy.run_path('offline.py', run_name='__main__')\n"
                "except SystemExit as e:\n"
                "    rc = e.code or 0\n"
                "print('RC', rc)\n"
                "print('TORCH' if 'torch' in sys.modules else 'NO_TORCH')\n"
                "print('HELP' if 'PIPE-01' in buf.getvalue() else 'NO_HELP')\n")
        p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                           cwd=str(PKG), encoding='utf-8', errors='replace')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('RC 0', p.stdout)
        self.assertIn('NO_TORCH', p.stdout)
        self.assertIn('HELP', p.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
