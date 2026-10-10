# -*- coding: utf-8 -*-
"""PIPE-01R 阶段A CPU 测试（mock / 临时小输入；不加载真实模型、不启动 CUDA、不联网、不导出JIT）。

保留 PIPE-01 首版 23 项的原始意图，并新增 review 要求（PIPE-R01–R06）的负测试与数值测试：
  R01 DINO 预处理与独立参考实现逐点比对（含 /255），黑/灰/白/梯度、双尺寸、边界
  R02 只允许安全加载（weights_only=True）且完全不使用 torch.hub；strict 键集；缺缓存不回退
  R03 A/B 完整身份必须相等；预登记与当前状态不符必须在模型加载前阻断
  R04 持久化预算总账：0 余额、调用前扣减、失败不退回、跨 attempt 累计、并发/复用拒绝
  R05 NaN/Inf 必须在量化前拦截（不得被 uint8 洗白），失败不产出成功图/manifest
  R06 每次调用唯一产物、不覆盖；A/B 使用同一 infer 计时边界；工具写临时目录

注意（诚实标注）：本文件只验证实现与判定逻辑，**不是**模型前向、质量或合规验收。
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
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


class FakeIdentityAssets:
    """只提供 identity/cfg 的假资产，用于驱动测试。"""

    def __init__(self, cfg, device='cpu', tag='A'):
        self.cfg, self.device, self.tag = cfg, device, tag
        self.seen = []

    def identity(self, entry=None, model_dir=None, device=None):
        return {'entry': entry, 'model_dir': model_dir, 'device': device or self.device,
                'config_sha256': 'FAKE', 'config': {'k': 1},
                'asset_fingerprint': {'fake': 1}, 'source_fingerprint': {'fake': 1},
                'implementation_fingerprint': {'fake': 1},
                'effective_fingerprint': {'model': {'digest_sha256': 'FAKE'},
                                          'dino': {'digest_sha256': 'FAKE'},
                                          'vae': {'digest_sha256': 'FAKE'},
                                          'light_decoder': {'digest_sha256': 'FAKE'}},
                'module_files': {'core': __file__},
                'load_report': {'light_decoder_weights_only': True,
                                'dino_loader': 'local_source+local_ckpt(weights_only=True,strict=True), 无 torch.hub'},
                'tile_report': {}, 'env': {'python': 'x'}}


# ───────────────────────── 冻结配置（原测试保留） ─────────────────────────

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
                   {'vae_posterior_mode': False}, {'infer_steps': 2}, {'t_start': 0.9},
                   {'align_method': 'adain'}, {'upscale': 2}, {'pad_mode': 'zeros'},
                   {'crop_back_to_input': False}):
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
        rep = core.tile_config_report(_cfg(), 2)
        self.assertEqual((rep['dit_tile_latent'], rep['dit_overlap_latent'],
                          rep['dit_effective_overlap_px'], rep['dit_stride_px']),
                         (64, 8, 64, 448))
        self.assertEqual((rep['vae_tile_latent'], rep['vae_overlap_latent'],
                          rep['vae_effective_overlap_px'], rep['vae_stride_px']),
                         (128, 16, 128, 896))

    def test_tile_report_rejects_recipe_change(self):
        with self.assertRaises(core.ConfigError):
            core.tile_config_report(core.load_config(self._tamper(dit_tile_size=640)), 2)

    def test_args_ns_conflict_is_rejected(self):
        with self.assertRaises(core.ConfigError):
            core.build_args_ns(core.load_config(self._tamper(expected_ae_type='qwen')))


# ───────────────────────── PIPE-R01：DINO 预处理数值 ─────────────────────────

def _reference_preprocess(lq, dinov2_size):
    """独立参考实现：逐字照抄 REF-A:131-135 的顺序（含 /255），不使用被测代码。"""
    import torch
    x = (0.5 * lq + 0.5) * 255.0
    x = x / 255.
    x = torch.nn.functional.interpolate(x, size=int(dinov2_size), mode='bicubic').clip(0., 1.)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (x - mean) / std


class TestDinoPreprocess(unittest.TestCase):
    def _cases(self):
        import torch
        g = torch.linspace(-1, 1, 64).view(1, 1, 8, 8).expand(1, 3, 8, 8)
        return {
            'black': torch.full((1, 3, 16, 16), -1.0),
            'gray': torch.zeros(1, 3, 16, 16),
            'white': torch.full((1, 3, 16, 16), 1.0),
            'gradient': g,
            'odd_13x17': torch.full((1, 3, 13, 17), 0.25),
        }

    def test_matches_reference_for_all_cases(self):
        import torch
        for size in (448, 512):
            for name, lq in self._cases().items():
                got = core.preprocess_dinov2_input(lq, size)
                ref = _reference_preprocess(lq, size)
                self.assertEqual(tuple(got.shape), tuple(ref.shape), (name, size))
                d = float((got - ref).abs().max())
                self.assertLessEqual(d, 1e-6, '%s@%d max_abs=%.3e' % (name, size, d))

    def test_gray_input_is_not_wrongly_saturated(self):
        import torch
        got = core.preprocess_dinov2_input(torch.zeros(1, 3, 64, 64), 448)
        # lq=0 对应像素 0.5（灰色），归一化后应为 (0.5-mean)/std
        expect = torch.tensor([(0.5 - 0.485) / 0.229, (0.5 - 0.456) / 0.224,
                               (0.5 - 0.406) / 0.225])
        for c in range(3):
            self.assertAlmostEqual(float(got[0, c].mean()), float(expect[c]), places=5)
        self.assertLess(float(got.max()), 0.5, '灰色输入不得被压成近白（旧缺陷会饱和到 1）')

    def test_old_wrong_order_would_differ(self):
        import torch
        lq = torch.zeros(1, 3, 32, 32)
        got = core.preprocess_dinov2_input(lq, 448)
        old = (0.5 * lq + 0.5) * 255.0
        old = torch.nn.functional.interpolate(old, size=448, mode='bicubic').clip(0., 1.)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        old = (old - mean) / std
        self.assertGreater(float((got - old).abs().max()), 1.0,
                           '必须与漏 /255 的旧实现显著不同（回归护栏）')

    def test_white_and_gradient_finite(self):
        import torch
        for name, lq in self._cases().items():
            out = core.preprocess_dinov2_input(lq, 448)
            self.assertTrue(bool(torch.isfinite(out).all()), name)


# ───────────────────────── PIPE-R02：安全加载与离线 ─────────────────────────

class TestSafeLoading(unittest.TestCase):
    def test_no_weights_only_false_in_sources(self):
        for f in ('core.py', 'runner.py', 'offline.py', 'pair_verify.py', 'make_evidence.py'):
            txt = (PKG / f).read_text(encoding='utf-8')
            self.assertNotIn('weights_only=False', txt, f)

    def test_no_torch_hub_calls_in_sources(self):
        """只禁止真实调用（torch.hub.xxx(）；说明性文字里提到 hub 不算违规。"""
        import re
        for f in ('core.py', 'runner.py', 'offline.py', 'pair_verify.py', 'make_evidence.py'):
            txt = (PKG / f).read_text(encoding='utf-8')
            m = re.search(r'torch\.hub\.\w+\s*\(', txt)
            self.assertIsNone(m, '%s 仍调用 torch.hub: %s' % (f, m.group(0) if m else ''))

    def test_lora_unsafe_payload_fails_closed(self):
        import torch
        p = Path(tempfile.mkdtemp()) / 'bad.pt'
        torch.save({'state_dict': {'x': torch.zeros(1)}, 'weird': _UnsafePayload()}, p)
        with self.assertRaises(core.AssetError):
            core.load_lora_payload(p)

    def test_lora_missing_file_fails(self):
        with self.assertRaises(core.AssetError):
            core.load_lora_payload(Path(tempfile.mkdtemp()) / 'nope.pt')

    def test_missing_or_wrong_dino_never_goes_online(self):
        import torch
        import urllib.request
        called = []
        orig = (torch.hub.load, torch.hub.load_state_dict_from_url,
                torch.hub.download_url_to_file, urllib.request.urlopen)

        def boom(*a, **k):
            called.append(a)
            raise AssertionError('不得触发任何网络/下载入口')
        torch.hub.load = boom
        torch.hub.load_state_dict_from_url = boom
        torch.hub.download_url_to_file = boom
        urllib.request.urlopen = boom
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
            (torch.hub.load, torch.hub.load_state_dict_from_url,
             torch.hub.download_url_to_file, urllib.request.urlopen) = orig

    def test_wrong_checkpoint_fails_strict(self):
        """指向一个能安全加载但不是 DINO 的文件 -> strict 键集必须报错（不静默）。"""
        import torch
        cfg = core.load_config()
        bad = dict(cfg)
        bad['dinov2_ckpt'] = cfg['light_decoder']
        with self.assertRaises((core.AssetError, core.OfflineError)):
            core.load_dinov2_local(bad, 'cpu')


# ───────────────────────── PIPE-R03：身份与预登记 ─────────────────────────

class TestIdentityAndPrereg(unittest.TestCase):
    def test_implementation_fingerprint_real(self):
        fp = core.implementation_fingerprint()
        for rel in ('core.py', 'runner.py', 'offline.py', 'pair_verify.py', 'make_evidence.py',
                    'config.json', 'tests/test_pipe01_cpu.py'):
            self.assertIn(rel, fp)
            self.assertEqual(len(fp[rel]['sha256']), 64)
            self.assertTrue(Path(fp[rel]['path']).is_file())

    def test_asset_fingerprint_covers_vae_and_dino_source(self):
        orig = core.sha256_file
        core.sha256_file = lambda p, chunk=None: 'FAKEHASH'
        try:
            fp = core.asset_fingerprint(_cfg())
        finally:
            core.sha256_file = orig
        self.assertIn('ae_vae_dir', fp)
        self.assertGreaterEqual(fp['ae_vae_dir']['n_files'], 1)
        self.assertIn('dinov2_vision_transformer', fp)
        self.assertIn('ae_vae_dir', fp)

    def test_identity_comparable_strips_entry_only(self):
        a = {'entry': 'A', 'x': 1}
        b = {'entry': 'B', 'x': 1}
        self.assertEqual(core.identity_comparable(a), core.identity_comparable(b))
        self.assertNotEqual(core.identity_comparable(a), core.identity_comparable({'entry': 'A', 'x': 2}))


# ───────────────────────── PIPE-R05：量化前数值边界 ─────────────────────────

class TestNaNBeforeQuantization(unittest.TestCase):
    def test_require_finite_blocks(self):
        import torch
        t = torch.zeros(1, 3, 8, 8)
        core.require_finite(t, 'ok')
        t[0, 0, 0, 0] = float('nan')
        with self.assertRaises(core.OutputValidationError):
            core.require_finite(t, 'test')
        t[0, 0, 0, 0] = float('inf')
        with self.assertRaises(core.OutputValidationError):
            core.require_finite(t, 'test')

    def test_uint8_quantization_would_hide_nan_but_precheck_blocks(self):
        import torch
        t = torch.zeros(1, 3, 8, 8)
        t[0, 0, 0, 0] = float('nan')
        pil = core.tensor_to_pil(t)            # 量化会把 NaN 变成合法像素
        import numpy as np
        arr = np.asarray(pil)
        self.assertTrue(np.isfinite(arr).all(), '量化确实会“洗白”NaN —— 所以必须前置拦截')
        with self.assertRaises(core.OutputValidationError):
            core.require_finite(t, '量化前')

    def test_infer_core_blocks_nan_latent(self):
        import torch
        core.ensure_vosr_import_path(_cfg())

        class FakeAssets:
            device = 'cpu'
            cfg = _cfg()
            def prepare_features(self, x):
                return torch.full((1, 4, 2, 2), float('nan')), None, None
        with self.assertRaises(core.OutputValidationError):
            core.infer_core(FakeAssets(), torch.zeros(1, 3, 16, 16))


# ───────────────────────── PIPE-R04：持久化预算总账 ─────────────────────────

class TestBudgetLedger(unittest.TestCase):
    def setUp(self):
        import pair_verify
        self.pv = pair_verify
        self.tmp = Path(tempfile.mkdtemp(prefix='pipe01_budget_'))
        self.led = self.tmp / 'ledger.json'

    def _mk(self, **kw):
        args = dict(max_forwards=4, max_minutes=30, max_gib=4, init=True)
        args.update(kw)
        return self.pv.BudgetLedger(self.led, args.pop('max_forwards'), args.pop('max_minutes'),
                                    args.pop('max_gib'), **args)

    def test_zero_quota_refuses(self):
        led = self._mk(max_forwards=0)
        with self.assertRaises(self.pv.BudgetStop):
            led.reserve('x')

    def test_reserve_is_persisted_before_call_and_not_refunded(self):
        led = self._mk(max_forwards=2)
        n = led.reserve('a')
        self.assertEqual(n, 1)
        on_disk = json.loads(self.led.read_text(encoding='utf-8'))
        self.assertEqual(on_disk['forwards_reserved'], 1, '必须在调用前落盘')
        with self.assertRaises(self.pv.BudgetStop):     # 失败不退回：第 3 次被拒
            led.reserve('b')
            led.reserve('c')
        self.assertEqual(json.loads(self.led.read_text(encoding='utf-8'))['forwards_reserved'], 2)

    def test_boundary_last_unit_allowed_then_refused(self):
        led = self._mk(max_forwards=1)
        led.reserve('a')
        with self.assertRaises(self.pv.BudgetStop):
            led.reserve('b')

    def test_cross_attempt_accumulates_and_interruption_counts(self):
        led = self._mk(max_forwards=10)
        led.begin_attempt('t1', self.tmp)
        led.data['open_attempt']['started_monotonic'] -= 5.0
        led._write()
        led.begin_attempt('t2', self.tmp)      # 上一次未结束 -> 记 INTERRUPTED 且时间计入
        self.assertEqual(len(led.data['attempts']), 1)
        self.assertEqual(led.data['attempts'][0]['status'], 'INTERRUPTED')
        self.assertGreaterEqual(led.data['wall_seconds_total'], 4.0)
        led.end_attempt('OK')
        self.assertIsNone(led.data['open_attempt'])

    def test_new_process_cannot_reset_quota(self):
        led = self._mk(max_forwards=4)
        led.reserve('a')
        with self.assertRaises(self.pv.BudgetStop):
            self.pv.BudgetLedger(self.led, 10, 30, 4, init=False)   # 试图改写配额

    def test_missing_ledger_without_init_refuses(self):
        with self.assertRaises(self.pv.BudgetStop):
            self.pv.BudgetLedger(self.tmp / 'nope.json', 4, 30, 4, init=False)

    def test_wall_stop(self):
        led = self._mk(max_forwards=4, max_minutes=0.001)
        led.begin_attempt('t1', self.tmp)
        time.sleep(0.2)
        with self.assertRaises(self.pv.BudgetStop):
            led.check_wall()

    def test_concurrent_lock_refused(self):
        led = self._mk(max_forwards=4)
        led.acquire_lock()
        try:
            other = self.pv.BudgetLedger(self.led, 4, 30, 4, init=False)
            with self.assertRaises(self.pv.BudgetStop):
                other.acquire_lock()
        finally:
            led.release_lock()

    def test_disk_budget(self):
        led = self._mk(max_forwards=4, max_gib=0.0000001)
        big = self.tmp / 'big.bin'
        big.write_bytes(b'0' * 4096)
        with self.assertRaises(self.pv.BudgetStop):
            led.check_disk(self.tmp)


# ───────────────────────── 原驱动/入口测试（保留意图，适配新接口） ─────────────────────────

class TestSharedCoreAndEntries(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='pipe01_cpu_'))
        self.calls = []
        self._orig_infer = core.infer_core
        self._orig_assets = core.Assets

        def fake_infer(assets, tensor, seed=None, timing=None):
            import torch
            if timing is not None:
                timing['infer_s'] = 0.001
            self.calls.append(('infer_core', tuple(tensor.shape)))
            out = torch.zeros_like(tensor)
            core.validate_output(out, int(tensor.shape[2]), int(tensor.shape[3]))
            return out

        core.infer_core = fake_infer
        core.Assets = FakeIdentityAssets

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
        Image.fromarray(torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()).save(img)
        r = Runner(cfg['base_dir'])
        r._ensure = lambda: FakeIdentityAssets(cfg)
        out_a = r.infer(torch.zeros(1, 3, 8, 8), prompt='')
        rec = offline.process_image(FakeIdentityAssets(cfg), img, self.tmp / 'out')
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0][1], self.calls[1][1])
        self.assertEqual(tuple(out_a.shape), (1, 3, 8, 8))
        self.assertTrue(Path(rec['png_path']).is_file())

    def test_prompt_must_be_empty(self):
        import torch
        from runner import Runner
        r = Runner(_cfg()['base_dir'])
        r._ensure = lambda: FakeIdentityAssets(_cfg())
        with self.assertRaises(core.PipelineError):
            r.infer(torch.zeros(1, 3, 8, 8), prompt='x')

    def test_model_dir_must_exist(self):
        from runner import Runner
        with self.assertRaises(core.AssetError):
            Runner(str(self.tmp / 'nope'))

    def test_png_encoder_is_shared_and_pixel_identical(self):
        import numpy as np
        import torch
        from PIL import Image
        t = torch.rand(1, 3, 16, 16) * 2 - 1
        p1, p2 = self.tmp / 'x1.png', self.tmp / 'x2.png'
        core.tensor_to_pil(t).save(p1)
        core.tensor_to_pil(t).save(p2)
        self.assertTrue(np.array_equal(np.array(Image.open(p1).convert('RGB')),
                                       np.array(Image.open(p2).convert('RGB'))))
        self.assertEqual(core.sha256_file(p1), core.sha256_file(p2))

    def test_offline_records_separate_io_timing(self):
        from PIL import Image
        import torch
        import offline
        img = self.tmp / 'b.png'
        Image.fromarray(torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()).save(img)
        rec = offline.process_image(FakeIdentityAssets(_cfg()), img, self.tmp / 'out2',
                                    save_tensor_dir=self.tmp / 'out2_npy', call_id='UNIQUE1')
        self.assertIn('input_decode_s', rec['timing'])
        self.assertIn('infer_s', rec['timing'])
        self.assertIn('output_encode_s', rec['timing'])
        self.assertTrue(rec['png_path'].endswith('UNIQUE1.png'))
        self.assertTrue(rec['tensor_npy'].endswith('UNIQUE1.npy'))


class TestFailurePropagation(unittest.TestCase):
    def test_output_validation(self):
        import torch
        self.assertTrue(core.validate_output(torch.zeros(1, 3, 8, 8), 8, 8))
        for bad in (torch.zeros(1, 3, 8, 9), torch.zeros(1, 3, 8, 8, dtype=torch.float16),
                    torch.full((1, 3, 8, 8), float('nan')), torch.full((1, 3, 8, 8), 1.5)):
            with self.assertRaises(core.OutputValidationError):
                core.validate_output(bad, 8, 8)

    def test_driver_refuses_without_prereg(self):
        out = Path(tempfile.mkdtemp(prefix='pipe01_neg_')) / 'run'
        p = subprocess.run(
            [sys.executable, '-B', str(PKG / 'pair_verify.py'), '--run-dir', str(out),
             '--prereg', str(out / 'prereg.json'), '--case1', 'x.jpg', '--case7', 'y.jpg'],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('预登记', (p.stdout or '') + (p.stderr or ''))
        self.assertFalse(out.exists(), '拒绝执行时不得创建 run 目录')

    def test_make_evidence_refuses_existing_run_dir_without_writing(self):
        """首版会先改写 evidence 再检查 run 目录；此测试固定在临时目录并核对无写入。"""
        d = Path(tempfile.mkdtemp(prefix='pipe01_exists_'))
        (d / 'leftover.txt').write_text('x', encoding='utf-8')   # 非空目录才应被拒绝
        ev = Path(tempfile.mkdtemp(prefix='pipe01_ev_')) / 'ev'
        probe = ev / 'hashes.json'
        p = subprocess.run(
            [sys.executable, '-B', str(PKG / 'make_evidence.py'), '--run-dir', str(d),
             '--evidence-dir', str(ev), '--ledger', str(Path(tempfile.mkdtemp()) / 'l.json')],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            cwd=str(PKG), env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('拒绝复用', (p.stdout or '') + (p.stderr or ''))
        self.assertFalse(probe.exists(), '被拒绝时不得写入任何 evidence 文件')


class TestSeedAndShapes(unittest.TestCase):
    def test_local_seed_is_order_independent(self):
        import torch
        vals = {}
        for order in (('a', 'b'), ('b', 'a')):
            for name in order:
                g = core.make_generator(42, 'cpu')
                vals.setdefault(name, []).append(torch.randn(4, generator=g))
        for name in ('a', 'b'):
            self.assertTrue(torch.equal(vals[name][0], vals[name][1]))
        g1, g2, g3 = (core.make_generator(s, 'cpu') for s in (42, 42, 43))
        self.assertTrue(torch.equal(torch.randn(8, generator=g1), torch.randn(8, generator=g2)))
        self.assertFalse(torch.equal(torch.randn(8, generator=g1), torch.randn(8, generator=g3)))

    def test_pad_and_crop_for_non_multiple_shapes(self):
        import torch
        core.ensure_vosr_import_path(_cfg())
        import tiled_vae as TV
        padded, h, w = TV._pad_to_multiple(torch.zeros(1, 3, 511, 513), 8)
        self.assertEqual((h, w), (511, 513))
        self.assertEqual(tuple(padded.shape[2:]), (512, 520))
        self.assertEqual(tuple(padded[:, :, :h, :w].shape[2:]), (511, 513))
        padded2, h2, w2 = TV._pad_to_multiple(torch.zeros(1, 3, 512, 512), 8)
        self.assertEqual(tuple(padded2.shape[2:]), (512, 512))


class TestImportSafety(unittest.TestCase):
    def test_imports_do_not_pull_torch(self):
        code = ("import sys, io, contextlib\n"
                "buf = io.StringIO()\n"
                "with contextlib.redirect_stdout(buf):\n"
                "    import core, runner, offline\n"
                "print('TORCH' if 'torch' in sys.modules else 'NO_TORCH')\n")
        p = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
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
        p = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
                           cwd=str(PKG), encoding='utf-8', errors='replace')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('RC 0', p.stdout)
        self.assertIn('NO_TORCH', p.stdout)
        self.assertIn('HELP', p.stdout)


# ───────────────────────── 驱动器端到端起搏（含身份/预算/唯一产物） ─────────────────────────

class TestDriverEndToEndCPU(unittest.TestCase):
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
        self.ledger = self.tmp / 'budget_ledger.json'
        self._orig = (core.Assets, core.infer_core, core.asset_fingerprint,
                      core.source_fingerprint, core.implementation_fingerprint)
        self._write_prereg()

    def tearDown(self):
        (core.Assets, core.infer_core, core.asset_fingerprint, core.source_fingerprint,
         core.implementation_fingerprint) = self._orig

    def _write_prereg(self, max_forwards=10):
        (self.run / 'prereg.json').write_text(json.dumps({
            'criteria': {'max_abs_diff': 1e-5, 'mean_abs_diff': 1e-6},
            'budget': {'max_forwards': max_forwards, 'max_minutes': 30,
                       'max_output_gib': 4},
            'config_sha256': core.load_config()['_config_sha256'],
            'asset_fingerprint': {'fake_assets': 1},
            'source_fingerprint': {'fake_src': 1},
            'implementation_fingerprint': {'fake_impl': 1},
        }), encoding='utf-8')

    def _install_fake(self, delta_b=0.0, identity_mismatch=False):
        import torch
        made = {'n': 0}

        class FakeAssets(FakeIdentityAssets):
            def __init__(self, cfg, device='cpu', tag=None):
                made['n'] += 1
                super().__init__(cfg, device, tag or ('A' if made['n'] == 1 else 'B'))
                self.tag = tag or ('A' if made['n'] == 1 else 'B')

            def identity(self, entry=None, model_dir=None, device=None):
                idy = super().identity(entry, model_dir, device)
                if identity_mismatch and self.tag == 'B':
                    idy['effective_fingerprint']['model']['digest_sha256'] = 'DIFFERENT'
                return idy

        def fake_infer(assets, tensor, seed=None, timing=None):
            y = torch.tanh(tensor.float() * 0.5)
            if delta_b and getattr(assets, 'tag', '') == 'B':
                y = y + float(delta_b)
            if timing is not None:
                timing['infer_s'] = 0.001
            return y

        core.Assets = FakeAssets
        core.infer_core = fake_infer
        core.asset_fingerprint = lambda cfg: {'fake_assets': 1}
        core.source_fingerprint = lambda cfg: {'fake_src': 1}
        core.implementation_fingerprint = lambda: {'fake_impl': 1}

    def _drive(self, delta_b=0.0, identity_mismatch=False, extra=None):
        self._install_fake(delta_b, identity_mismatch)
        import pair_verify
        argv = ['--run-dir', str(self.run), '--prereg', str(self.run / 'prereg.json'),
                '--case1', str(self.tmp / 'case1.jpg'), '--case7', str(self.tmp / 'case7.jpg'),
                '--crop', '512', '--device', 'cpu', '--budget-ledger', str(self.ledger),
                '--init-ledger']
        rc = pair_verify.main(argv + (extra or []))
        return rc, json.loads((self.run / 'paired_readings.json').read_text(encoding='utf-8'))

    def test_driver_full_flow_passes_with_identical_entries(self):
        rc, res = self._drive(0.0)
        self.assertEqual(rc, 0, '%s | %s | %s' % (
            res.get('criteria_eval_error'), res.get('note'),
            json.dumps(res.get('criteria_detail'), ensure_ascii=False)))
        self.assertEqual(res['status'], 'OK')
        self.assertTrue(res['criteria_all_pass'])
        self.assertEqual(res['forwards_reserved_total'], 8)
        self.assertTrue(res['criteria_detail']['identity_equal'])
        self.assertEqual(len(res['pairs']), 3)
        for tag, v in res['pairs'].items():
            self.assertTrue(v['pass_max'] and v['pass_mean'] and v['png_identical'], tag)
        self.assertEqual(len(res['repeat_512']), 2)
        for k, v in res['repeat_512'].items():
            self.assertTrue(v['pass_max'] and v['pass_mean'], k)
        man = json.loads((self.run / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(man['calls']), 8)
        ids = [c['call_id'] for c in man['calls']]
        self.assertEqual(len(ids), len(set(ids)), 'call_id 必须唯一')
        npy = [c['tensor_npy'] for c in man['calls']]
        self.assertEqual(len(npy), len(set(npy)), 'NPY 不得同名覆盖')
        png = [c['png_path'] for c in man['calls']]
        self.assertEqual(len(png), len(set(png)), 'PNG 不得同名覆盖')
        for c in man['calls']:
            self.assertTrue(Path(c['tensor_npy']).is_file())
            self.assertTrue(Path(c['png_path']).is_file())
        self.assertIn('identity', man['instances']['A'])
        self.assertIn('identity', man['instances']['B'])
        self.assertIn('dino', man['instances']['A']['identity']['effective_fingerprint'])
        led = json.loads(self.ledger.read_text(encoding='utf-8'))
        self.assertEqual(led['forwards_reserved'], 8)
        self.assertEqual(led['attempts'][-1]['status'], 'OK')

    def test_identity_mismatch_fails_even_with_zero_diff(self):
        rc, res = self._drive(0.0, identity_mismatch=True)
        self.assertEqual(rc, 1)
        self.assertEqual(res['status'], 'CRITERIA_FAIL')
        self.assertFalse(res['criteria_detail']['identity_equal'])
        for tag, v in res['pairs'].items():
            self.assertEqual(v['max_abs_diff'], 0.0, tag)     # 输出仍逐位相同
        self.assertFalse(res['criteria_all_pass'])

    def test_criteria_fail_is_reported(self):
        rc, res = self._drive(1e-3)
        self.assertEqual(rc, 1)
        self.assertEqual(res['status'], 'CRITERIA_FAIL')
        self.assertFalse(res['pairs']['case1_crop512']['pass_max'])

    def test_prereg_mismatch_blocks_before_any_load(self):
        pq = json.loads((self.run / 'prereg.json').read_text(encoding='utf-8'))
        pq['config_sha256'] = 'DEADBEEF'
        (self.run / 'prereg.json').write_text(json.dumps(pq), encoding='utf-8')
        self._install_fake()
        import pair_verify
        with self.assertRaises(SystemExit) as cm:
            pair_verify.main(['--run-dir', str(self.run), '--prereg',
                              str(self.run / 'prereg.json'), '--case1',
                              str(self.tmp / 'case1.jpg'), '--case7',
                              str(self.tmp / 'case7.jpg'), '--crop', '512',
                              '--device', 'cpu', '--budget-ledger', str(self.ledger),
                              '--init-ledger'])
        self.assertIn('PREREG_MISMATCH', str(cm.exception))

    def test_zero_quota_blocks_before_any_forward(self):
        self._install_fake()
        self._write_prereg(max_forwards=0)          # 预登记与总账一致：额度为 0
        self.ledger.write_text(json.dumps({
            'task': 'PIPE-01', 'created_at': 'x', 'max_forwards': 0, 'max_minutes': 30,
            'max_gib': 4, 'forwards_reserved': 0, 'wall_seconds_total': 0.0,
            'attempts': [], 'open_attempt': None, 'reservations': [], 'stale_locks': []}),
            encoding='utf-8')
        import pair_verify
        rc = pair_verify.main(['--run-dir', str(self.run), '--prereg',
                               str(self.run / 'prereg.json'), '--case1',
                               str(self.tmp / 'case1.jpg'), '--case7',
                               str(self.tmp / 'case7.jpg'), '--crop', '512',
                               '--device', 'cpu', '--budget-ledger', str(self.ledger)])
        res = json.loads((self.run / 'paired_readings.json').read_text(encoding='utf-8'))
        self.assertEqual(rc, 1)
        self.assertEqual(res['status'], 'BUDGET_STOP')
        self.assertEqual(res['forwards_reserved_total'], 0)

    def test_nonempty_run_dir_is_refused(self):
        (self.run / 'leftover.txt').write_text('x', encoding='utf-8')
        self._install_fake()
        import pair_verify
        with self.assertRaises(SystemExit) as cm:
            pair_verify.main(['--run-dir', str(self.run), '--prereg',
                              str(self.run / 'prereg.json'), '--case1',
                              str(self.tmp / 'case1.jpg'), '--case7',
                              str(self.tmp / 'case7.jpg'), '--device', 'cpu',
                              '--budget-ledger', str(self.ledger), '--init-ledger'])
        self.assertIn('拒绝复用', str(cm.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
