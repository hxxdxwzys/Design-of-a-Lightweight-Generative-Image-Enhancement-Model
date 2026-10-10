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

    def release(self):
        pass

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

class DriverHarness(unittest.TestCase):
    """驱动器端到端起搏的可复用夹具（假核心；不加载真实模型）。"""

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
                      core.source_fingerprint, core.implementation_fingerprint,
                      core.env_fingerprint)
        self._write_prereg()

    def tearDown(self):
        (core.Assets, core.infer_core, core.asset_fingerprint, core.source_fingerprint,
         core.implementation_fingerprint, core.env_fingerprint) = self._orig

    def _write_prereg(self, max_forwards=8, protocol=True, crop=512, case1=None, case7=None,
                      deltas=None):
        import hashlib
        from PIL import Image
        c1 = Path(case1 or (self.tmp / 'case1.jpg'))
        c7 = Path(case7 or (self.tmp / 'case7.jpg'))
        im1, im7 = Image.open(c1), Image.open(c7)
        w, h = im1.size
        box = [(w - crop) // 2, (h - crop) // 2, (w - crop) // 2 + crop,
               (h - crop) // 2 + crop]
        proto = None
        if protocol:
            proto = {
                'crop': crop, 'warmups': 0,
                'order_per_instance': ['case1_crop512', 'case1_full', 'case7_full',
                                       'case1_crop512'],
                'instances': ['A=tensor_runner', 'B=offline_file_entry'],
                'forwards_total': 8,
                'samples': {
                    'case1_full': {'input_file_sha256': hashlib.sha256(
                        c1.read_bytes()).hexdigest(), 'input_wh': [w, h]},
                    'case7_full': {'input_file_sha256': hashlib.sha256(
                        c7.read_bytes()).hexdigest(), 'input_wh': list(im7.size)},
                    'case1_crop512': {'crop_box': box, 'input_wh': [crop, crop],
                                      'input_file_sha256': None,
                                      'source': 'case1_full',
                                      'source_sha256': hashlib.sha256(c1.read_bytes()).hexdigest()},
                },
            }
        pq = {
            'criteria': {'max_abs_diff': 1e-5, 'mean_abs_diff': 1e-6},
            'budget': {'max_forwards': max_forwards, 'max_minutes': 30,
                       'max_output_gib': 4, 'warmups': 0},
            'protocol': proto,
            'config_sha256': core.load_config()['_config_sha256'],
            'asset_fingerprint': {'fake_assets': 1},
            'source_fingerprint': {'fake_src': 1},
            'implementation_fingerprint': {'fake_impl': 1},
            'env': {'python': 'x'},          # 与 _install_fake 的假环境一致
        }
        if deltas:
            pq.update(deltas)
        (self.run / 'prereg.json').write_text(json.dumps(pq), encoding='utf-8')
        return pq

    def _install_fake(self, delta_b=0.0, identity_mismatch=False, fail_on_call=None,
                      lifecycle=None, weakrefs=None):
        import torch
        import weakref as _weakref
        made = {'n': 0, 'calls': 0}

        class FakeAssets(FakeIdentityAssets):
            def __init__(self, cfg, device='cpu', tag=None):
                made['n'] += 1
                super().__init__(cfg, device, tag or ('A' if made['n'] == 1 else 'B'))
                self.tag = tag or ('A' if made['n'] == 1 else 'B')
                if lifecycle is not None:
                    lifecycle.append('construct_%s' % self.tag)
                if weakrefs is not None:
                    weakrefs[self.tag] = _weakref.ref(self)

            def release(self):
                if lifecycle is not None:
                    lifecycle.append('release_%s' % self.tag)

            def identity(self, entry=None, model_dir=None, device=None):
                idy = super().identity(entry, model_dir, device)
                if identity_mismatch and self.tag == 'B':
                    idy['effective_fingerprint']['model']['digest_sha256'] = 'DIFFERENT'
                return idy

        def fake_infer(assets, tensor, seed=None, timing=None):
            made['calls'] += 1
            if lifecycle is not None:
                lifecycle.append('infer_%s_%d' % (getattr(assets, 'tag', '?'), made['calls']))
            if fail_on_call is not None and made['calls'] == fail_on_call:
                raise RuntimeError('SYNTHETIC_FORWARD_FAILURE')
            y = torch.tanh(tensor.float() * 0.5)
            if delta_b and getattr(assets, 'tag', '') == 'B':
                y = y + float(delta_b)
            if timing is not None:
                timing['events'] = ['enter', 'to_device', 'prepare_features', 'one_step',
                                    'decode', 'postprocess', 'exit']
                timing['infer_s'] = 0.001
            return y

        core.Assets = FakeAssets
        core.infer_core = fake_infer
        core.asset_fingerprint = lambda cfg: {'fake_assets': 1}
        core.source_fingerprint = lambda cfg: {'fake_src': 1}
        core.implementation_fingerprint = lambda: {'fake_impl': 1}
        core.env_fingerprint = lambda: {'python': 'x'}

    def _drive(self, delta_b=0.0, identity_mismatch=False, extra=None, **fake_kw):
        self._install_fake(delta_b, identity_mismatch, **fake_kw)
        import pair_verify
        argv = ['--run-dir', str(self.run), '--prereg', str(self.run / 'prereg.json'),
                '--case1', str(self.tmp / 'case1.jpg'), '--case7', str(self.tmp / 'case7.jpg'),
                '--crop', '512', '--device', 'cpu', '--budget-ledger', str(self.ledger),
                '--init-ledger', '--no-supervise']
        rc = pair_verify.main(argv + (extra or []))
        return rc, json.loads((self.run / 'paired_readings.json').read_text(encoding='utf-8'))

class TestDriverEndToEndCPU(DriverHarness):

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

    def test_identity_mismatch_never_passes(self):
        """身份不符时不得 PASS，且在 B 首个前向之前就阻断（不依赖输出是否相同）。"""
        rc, res = self._drive(0.0, identity_mismatch=True)
        self.assertEqual(rc, 1)
        self.assertEqual(res['status'], 'STOPPED')
        self.assertIn('IDENTITY_MISMATCH', res['note'])
        self.assertFalse(res['criteria_all_pass'])

    def test_criteria_exceeded_stops_immediately(self):
        rc, res = self._drive(1e-3)
        self.assertEqual(rc, 1)
        self.assertEqual(res['status'], 'STOPPED')
        self.assertIn('CRITERIA_EXCEEDED', res['note'])
        self.assertFalse(res['criteria_all_pass'])

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
                              '--init-ledger', '--no-supervise'])
        self.assertIn('PREREG_MISMATCH', str(cm.exception))

    def test_zero_quota_blocks_before_any_load(self):
        self._install_fake()
        self._write_prereg(max_forwards=0)          # 预登记与总账一致：额度为 0
        self.ledger.write_text(json.dumps({
            'task': 'PIPE-01', 'created_at': 'x', 'max_forwards': 0, 'max_minutes': 30,
            'max_gib': 4, 'forwards_reserved': 0, 'wall_seconds_total': 0.0,
            'disk_bytes_total': 0,
            'attempts': [], 'open_attempt': None, 'reservations': [], 'stale_locks': [],
            'prereg_sha256': core.sha256_file(self.run / 'prereg.json')}),
            encoding='utf-8')
        import pair_verify
        with self.assertRaises(SystemExit) as cm:
            pair_verify.main(['--run-dir', str(self.run), '--prereg',
                              str(self.run / 'prereg.json'), '--case1',
                              str(self.tmp / 'case1.jpg'), '--case7',
                              str(self.tmp / 'case7.jpg'), '--crop', '512',
                              '--device', 'cpu', '--budget-ledger', str(self.ledger),
                              '--no-supervise'])
        self.assertIn('额度', str(cm.exception))
        self.assertFalse((self.run / 'manifest.json').exists(), '无额度时不得产生任何调用记录')
        self.assertEqual(json.loads(self.ledger.read_text(encoding='utf-8'))['forwards_reserved'], 0)

    def test_nonempty_run_dir_is_refused(self):
        (self.run / 'leftover.txt').write_text('x', encoding='utf-8')
        self._install_fake()
        import pair_verify
        with self.assertRaises(SystemExit) as cm:
            pair_verify.main(['--run-dir', str(self.run), '--prereg',
                              str(self.run / 'prereg.json'), '--case1',
                              str(self.tmp / 'case1.jpg'), '--case7',
                              str(self.tmp / 'case7.jpg'), '--device', 'cpu',
                              '--budget-ledger', str(self.ledger), '--init-ledger',
                              '--no-supervise'])
        self.assertIn('拒绝复用', str(cm.exception))



# ───────────────────────── PIPE-RA01：真实 Assets 组装控制流 ─────────────────────────

class TestRealAssetsAssembly(unittest.TestCase):
    """用轻量替身执行**真实 Assets.__init__ 控制流**（不替换整个类）。

    这样能捕获“只有真实组装路径才会触发”的错误（例如引用未定义名字），
    而此前的端到端测试把整个 core.Assets 换掉，恰好漏掉这段。
    """

    def setUp(self):
        import torch
        import diffusers
        import safetensors.torch as st
        core.ensure_vosr_import_path(_cfg())
        import models.light_decoder as ldec
        self.tmp = Path(tempfile.mkdtemp(prefix='pipe01_ra01_'))
        self.undo = []

        class FakeModel:
            forward_flexible = None

            def to(self, *a, **k):
                return self

            def eval(self):
                return self

            def parameters(self):
                return iter(())

            def state_dict(self):
                return {'w': torch.zeros(1)}

            def load_state_dict(self, sd, strict=False):
                return type('R', (), {'missing_keys': [], 'unexpected_keys': []})()

        class FakeDec(FakeModel):
            def __init__(self, **kw):
                self.kw = kw

        def patch(obj, name, val):
            self.undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)

        patch(core, 'asset_fingerprint', lambda cfg: {'fake': 1})
        patch(core, 'source_fingerprint', lambda cfg: {'fake': 1})
        patch(core, 'implementation_fingerprint', lambda: {'fake': 1})
        patch(core, 'env_fingerprint', lambda: {'python': 'x'})
        patch(core, 'build_dit', lambda cfg, ns, base_state=None: (
            FakeModel(), {'missing': [], 'unexpected': []}))
        patch(core, 'load_lora_payload', lambda p: {'state_dict': {'lora_A': 1},
                                                    'lora_config': '{}'})
        patch(core, 'apply_lora', lambda m, p, s: (m, {'lora_keys_loaded': 1,
                                                       'lora_scale': float(s)}))
        patch(core, 'load_dinov2_local', lambda cfg, dev: (FakeModel(), {
            'loader': 'FAKE_LOCAL_SOURCE', 'n_keys': 7, 'weights_only': True,
            'strict': True, 'uses_torch_hub': False}))
        patch(core, 'fingerprint_state_dict', lambda sd, with_bytes=True: {
            'digest_sha256': 'FAKE', 'n_keys': 1, 'n_params': 1, 'dtypes': {},
            'shape_histogram': {}, 'first_keys': [], 'last_keys': [], 'byte_protocol': 'x'})
        patch(st, 'load_file', lambda p: {'w': torch.zeros(1)})
        patch(diffusers.AutoencoderKL, 'from_pretrained',
              classmethod(lambda cls, *a, **k: FakeModel()))
        patch(ldec, 'LightDecoder', FakeDec)
        patch(torch, 'load', lambda *a, **k: {
            'config': {'in_channels': 4, 'out_channels': 3,
                       'block_out_channels': [1], 'layers_per_block': 1},
            'model_state_dict': {}})

    def tearDown(self):
        for obj, name, old in reversed(self.undo):
            setattr(obj, name, old)

    def test_real_assets_init_control_flow(self):
        a = core.Assets(_cfg(), 'cpu')
        self.assertEqual(a.load_report['dino']['n_keys'], 7)
        self.assertTrue(a.load_report['light_decoder_weights_only'])
        self.assertEqual(a.load_report['lora']['lora_keys_loaded'], 1)
        for key in ('model', 'vae', 'dino', 'light_decoder'):
            self.assertIn(key, a.effective_fp)
        idy = a.identity(entry='A')
        for key in ('config_sha256', 'asset_fingerprint', 'source_fingerprint',
                    'implementation_fingerprint', 'effective_fingerprint', 'module_files',
                    'load_report', 'tile_report', 'env'):
            self.assertIn(key, idy)

    def test_real_assets_release_clears_modules(self):
        a = core.Assets(_cfg(), 'cpu')
        a.release()
        for attr in ('model', 'vae', 'venc', 'dec'):
            self.assertIsNone(getattr(a, attr), attr)


# ───────────────────────── PIPE-RA02：失败/不符立即停止 ─────────────────────────

class TestImmediateStop(DriverHarness):

    def _manifest(self):
        return json.loads((self.run / 'manifest.json').read_text(encoding='utf-8'))

    def test_stops_after_first_call_failure_and_keeps_charge(self):
        rc, res = self._drive(0.0, fail_on_call=2)
        man = self._manifest()
        self.assertEqual(res['status'], 'STOPPED')
        self.assertEqual([c['status'] for c in man['calls']], ['OK', 'FAILED'])
        self.assertNotIn('B', man['instances'], 'A 失败后不得构造 B')
        self.assertEqual(man['calls'][1]['artifacts']['status'], 'NOT_PRODUCED')
        led = json.loads(self.ledger.read_text(encoding='utf-8'))
        self.assertEqual(led['forwards_reserved'], 2, '失败调用必须保留扣款')
        self.assertEqual(led['attempts'][-1]['status'], 'STOPPED')
        self.assertFalse(self.ledger.with_suffix('.lock').exists(), '终态必须释放锁')

    def test_identity_mismatch_blocks_before_b_first_forward(self):
        rc, res = self._drive(0.0, identity_mismatch=True)
        man = self._manifest()
        self.assertEqual(res['status'], 'STOPPED')
        self.assertIn('IDENTITY_MISMATCH', res['note'])
        self.assertEqual(len([c for c in man['calls'] if c['entry'] == 'A']), 4)
        self.assertEqual(len([c for c in man['calls'] if c['entry'] == 'B']), 0,
                         'B 不得在身份不符后执行任何前向')
        self.assertIn('B', man['instances'], 'B 必须已构造以取得身份')

    def test_stops_at_first_threshold_exceeded(self):
        rc, res = self._drive(1e-3)
        man = self._manifest()
        self.assertEqual(res['status'], 'STOPPED')
        self.assertIn('CRITERIA_EXCEEDED', res['note'])
        self.assertEqual(len([c for c in man['calls'] if c['entry'] == 'B']), 1,
                         '首个超阈值样本后不得继续')
        self.assertFalse(res['criteria_all_pass'])


# ───────────────────────── PIPE-RA03：实例释放顺序 ─────────────────────────

class TestInstanceLifecycle(DriverHarness):

    def test_a_released_before_b_constructed(self):
        life, refs = [], {}
        rc, res = self._drive(0.0, lifecycle=life, weakrefs=refs)
        self.assertEqual(res['status'], 'OK')
        self.assertIn('release_A', life)
        self.assertIn('construct_B', life)
        self.assertLess(life.index('release_A'), life.index('construct_B'),
                        '必须先释放 A 再加载 B')
        import gc
        gc.collect()
        self.assertIsNone(refs['A'](), 'A 的资产对象必须已无强引用（弱引用失效）')

    def test_a_released_even_when_a_fails(self):
        life, refs = [], {}
        rc, res = self._drive(0.0, fail_on_call=2, lifecycle=life, weakrefs=refs)
        self.assertIn('release_A', life, 'A 失败也必须释放')
        self.assertNotIn('construct_B', life)


# ───────────────────────── PIPE-RA04：锁/超时/收尾 ─────────────────────────

class TestLedgerHardening(TestBudgetLedger):

    def test_exclusive_lock_and_stale_takeover(self):
        led = self._mk(max_forwards=4)
        led.acquire_lock()
        try:
            other = self.pv.BudgetLedger(self.led, 4, 30, 4, init=False)
            with self.assertRaises(self.pv.BudgetStop):
                other.acquire_lock()
        finally:
            led.release_lock()
        self.led.with_suffix('.lock').write_text(json.dumps({'pid': 999999}),
                                                 encoding='utf-8')
        led2 = self.pv.BudgetLedger(self.led, 4, 30, 4, init=False)
        led2.acquire_lock()
        try:
            self.assertTrue(led2.data['stale_locks'], '陈旧锁接管必须留痕')
        finally:
            led2.release_lock()

    def test_pid_state_tristate_and_unknown_fails_closed(self):
        self.assertTrue(self.pv._pid_state(os.getpid()))
        self.assertFalse(self.pv._pid_state(999999))
        self._mk(max_forwards=4)                    # 总账必须先存在
        self.led.with_suffix('.lock').write_text(json.dumps({'pid': 12345}),
                                                 encoding='utf-8')
        orig = self.pv._pid_state
        self.pv._pid_state = lambda pid: None
        try:
            led = self.pv.BudgetLedger(self.led, 4, 30, 4, init=False)
            with self.assertRaises(self.pv.BudgetStop):
                led.acquire_lock()
        finally:
            self.pv._pid_state = orig
            try:
                self.led.with_suffix('.lock').unlink()
            except Exception:
                pass

    def test_check_startable_refuses_zero_quota(self):
        led = self._mk(max_forwards=0)
        with self.assertRaises(self.pv.BudgetStop):
            led.check_startable()

    def test_check_startable_refuses_when_time_exhausted(self):
        p = self.tmp / 'led2.json'
        led = self.pv.BudgetLedger(p, 4, 0.001, 4, init=True)
        led.begin_attempt('t', self.tmp)
        time.sleep(0.2)
        with self.assertRaises(self.pv.BudgetStop):
            led.check_startable()

    def test_disk_is_cumulative_across_attempts(self):
        led = self._mk(max_forwards=4, max_gib=0.000002)
        rd = self.tmp / 'run'
        rd.mkdir(exist_ok=True)
        (rd / 'a.bin').write_bytes(b'0' * 1500)
        led.begin_attempt('t1', rd)
        led.end_attempt('OK', '', run_dir=rd)
        (rd / 'b.bin').write_bytes(b'0' * 1500)
        with self.assertRaises(self.pv.BudgetStop):
            led.check_disk(rd)

    def test_end_attempt_records_disk_and_wall(self):
        led = self._mk(max_forwards=4)
        rd = self.tmp / 'run2'
        rd.mkdir(exist_ok=True)
        (rd / 'x.bin').write_bytes(b'0' * 100)
        led.begin_attempt('t1', rd)
        led.end_attempt('OK', '', run_dir=rd)
        self.assertEqual(led.data['attempts'][-1]['disk_bytes'], 100)
        self.assertEqual(led.data['disk_bytes_total'], 100)
        self.assertGreaterEqual(led.data['wall_seconds_total'], 0.0)


class TestSupervisorHardWall(unittest.TestCase):
    """硬墙钟：只监督本任务自建的工作进程（不触碰用户其他进程）。"""

    def test_timeout_terminates_child(self):
        import pair_verify
        cmd = [sys.executable, '-B', '-c', 'import time; time.sleep(30)']
        rc, timed_out, wall = pair_verify.run_supervised(cmd, deadline_s=1.5, poll=0.2)
        self.assertTrue(timed_out)
        self.assertLess(wall, 15.0)

    def test_child_within_deadline_completes(self):
        import pair_verify
        cmd = [sys.executable, '-B', '-c', 'print(1)']
        rc, timed_out, wall = pair_verify.run_supervised(cmd, deadline_s=30, poll=0.1)
        self.assertFalse(timed_out)
        self.assertEqual(rc, 0)


# ───────────────────────── PIPE-RA05 / RA07：协议绑定与 manifest 语义 ─────────────────────────

class TestProtocolBinding(DriverHarness):

    def _inputs(self, crop=512):
        import pair_verify
        return pair_verify.build_inputs(self.tmp / 'case1.jpg', self.tmp / 'case7.jpg',
                                        crop, self.tmp / ('in%d' % crop))

    def test_accepts_matching_protocol(self):
        import pair_verify
        pq = self._write_prereg()
        inputs = self._inputs()
        self.assertTrue(pair_verify.verify_protocol(
            pq, inputs, 512, pair_verify.ORDER,
            ['A=tensor_runner', 'B=offline_file_entry']))

    def test_rejects_crop_change_and_wrong_samples(self):
        import pair_verify
        pq = self._write_prereg()
        inputs = self._inputs()
        with self.assertRaises(pair_verify.BudgetStop):
            pair_verify.verify_protocol(pq, inputs, 256, pair_verify.ORDER,
                                        ['A=tensor_runner', 'B=offline_file_entry'])
        bad = json.loads(json.dumps(pq))
        bad['protocol']['samples']['case1_full']['input_file_sha256'] = 'DEAD'
        with self.assertRaises(pair_verify.BudgetStop):
            pair_verify.verify_protocol(bad, inputs, 512, pair_verify.ORDER,
                                        ['A=tensor_runner', 'B=offline_file_entry'])
        bad2 = json.loads(json.dumps(pq))
        bad2['protocol']['samples']['case1_crop512']['crop_box'] = [0, 0, 512, 512]
        with self.assertRaises(pair_verify.BudgetStop):
            pair_verify.verify_protocol(bad2, inputs, 512, pair_verify.ORDER,
                                        ['A=tensor_runner', 'B=offline_file_entry'])
        bad3 = json.loads(json.dumps(pq))
        bad3['protocol']['warmups'] = 2
        with self.assertRaises(pair_verify.BudgetStop):
            pair_verify.verify_protocol(bad3, inputs, 512, pair_verify.ORDER,
                                        ['A=tensor_runner', 'B=offline_file_entry'])

    def test_generator_freezes_eight_forwards_and_zero_warmups(self):
        import make_evidence
        self.assertEqual(make_evidence.BUDGET['max_forwards'], 8)
        self.assertEqual(make_evidence.BUDGET['warmups'], 0)
        proto = make_evidence.build_protocol(self.tmp / 'case1.jpg', self.tmp / 'case7.jpg', 512)
        self.assertEqual(proto['forwards_total'], 8)
        self.assertEqual(proto['warmups'], 0)
        self.assertEqual(len(proto['samples']), 3)

    def test_driver_blocks_on_protocol_mismatch_before_load(self):
        import pair_verify
        pq = self._write_prereg()
        pq['protocol']['samples']['case7_full']['input_file_sha256'] = 'DEAD'
        (self.run / 'prereg.json').write_text(json.dumps(pq), encoding='utf-8')
        self._install_fake()
        with self.assertRaises(SystemExit) as cm:
            pair_verify.main(['--run-dir', str(self.run), '--prereg',
                              str(self.run / 'prereg.json'), '--case1',
                              str(self.tmp / 'case1.jpg'), '--case7',
                              str(self.tmp / 'case7.jpg'), '--crop', '512',
                              '--device', 'cpu', '--budget-ledger', str(self.ledger),
                              '--init-ledger', '--no-supervise'])
        self.assertIn('PROTOCOL_MISMATCH', str(cm.exception))

    def test_ledger_binding_mismatch_blocks(self):
        import pair_verify
        led = pair_verify.BudgetLedger(self.ledger, 8, 30, 4, init=True,
                                       prereg_sha256=core.sha256_file(self.run / 'prereg.json'))
        with self.assertRaises(pair_verify.BudgetStop):
            pair_verify.verify_ledger_binding(led, 'DEADBEEF')


class TestManifestSemantics(DriverHarness):

    def test_tensor_and_npy_hashes_are_separately_recomputable(self):
        import hashlib
        import numpy as np
        rc, res = self._drive(0.0)
        man = json.loads((self.run / 'manifest.json').read_text(encoding='utf-8'))
        seen = 0
        for c in man['calls']:
            self.assertEqual(c['artifacts']['status'], 'PRODUCED')
            self.assertEqual(c['tensor_byte_protocol'],
                             'float32-little-endian-raw-C-contiguous')
            arr = np.load(c['tensor_npy'], allow_pickle=False)
            self.assertEqual(hashlib.sha256(arr.tobytes()).hexdigest(),
                             c['tensor_content_sha256'])
            self.assertEqual(core.sha256_file(c['tensor_npy']), c['tensor_npy_sha256'])
            self.assertNotEqual(c['tensor_content_sha256'], c['tensor_npy_sha256'])
            seen += 1
        self.assertEqual(seen, 8)

    def test_failed_call_marks_artifacts_not_produced(self):
        rc, res = self._drive(0.0, fail_on_call=3)
        man = json.loads((self.run / 'manifest.json').read_text(encoding='utf-8'))
        failed = [c for c in man['calls'] if c['status'] == 'FAILED']
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['artifacts']['status'], 'NOT_PRODUCED')
        self.assertIsNone(failed[0].get('tensor_content_sha256'))
        self.assertIsNone(failed[0].get('tensor_npy_sha256'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
