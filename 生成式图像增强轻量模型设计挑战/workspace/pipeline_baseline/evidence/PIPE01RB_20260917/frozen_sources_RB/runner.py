# -*- coding: utf-8 -*-
"""PIPE-01 入口之一：公开 tensor runner —— Runner(model_dir).infer(image_tensor, prompt="").

薄入口：只承担接口契约、设备选择与身份上报；全部数值逻辑在 core 内，与离线入口共享同一实现。
本阶段是本地 eager 研究基线，NOT_JIT_DELIVERABLE：不代表官方交付格式通过。

PIPE-01R 改动：infer 增加可选 timing 出口（与离线入口使用同一计时口径），identity() 上报完整身份。
公开接口签名 Runner(model_dir).infer(image_tensor, prompt="") 保持不变（timing 为可选关键字）。
"""
from __future__ import annotations

import time
from pathlib import Path

import core


class Runner:
    def __init__(self, model_dir, config_path=None, device=None):
        self.model_dir = str(model_dir)
        if not Path(self.model_dir).is_dir():
            raise core.AssetError('model_dir 不存在: %s' % self.model_dir)
        self.config_path = config_path
        self.cfg = core.load_config(config_path)
        self.assets = None
        self.device = None
        self._device_hint = device

    def _ensure(self):
        if self.assets is None:
            import torch
            self.device = self._device_hint or (
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.assets = core.Assets(self.cfg, self.device)
        return self.assets

    def infer(self, image_tensor, prompt='', timing=None):
        if prompt not in ('', None):
            raise core.PipelineError('本阶段 prompt 必须为空字符串，收到 %r' % (prompt,))
        t0 = time.perf_counter()
        assets = self._ensure()
        if timing is not None:
            timing['load_s'] = timing.get('load_s', 0.0) + (time.perf_counter() - t0)
        return core.infer_core(assets, image_tensor, timing=timing)

    def identity(self, entry='tensor_runner'):
        a = self._ensure()
        return a.identity(entry=entry, model_dir=self.model_dir, device=self.device)

    def release(self):
        """释放实例（成对协议要求：先后加载，不同时占显存）。"""
        import gc
        self.assets = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
