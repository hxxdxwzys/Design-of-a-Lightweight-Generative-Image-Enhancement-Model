# -*- coding: utf-8 -*-
"""PIPE-01 入口之一：公开 tensor runner —— Runner(model_dir).infer(image_tensor, prompt="")。

薄入口：只承担接口契约、设备选择与身份上报；全部数值逻辑在 core 内，与离线入口共享同一实现。
本阶段是本地 eager 研究基线，NOT_JIT_DELIVERABLE：不代表官方交付格式通过。
"""
from __future__ import annotations

from pathlib import Path

import core


class Runner:
    """官方形态的 tensor 入口。model_dir 仅按接口保留并被校验/记录，资产来自冻结配置。"""

    def __init__(self, model_dir, config_path=None):
        self.model_dir = str(model_dir)
        if not Path(self.model_dir).is_dir():
            raise core.AssetError('model_dir 不存在: %s' % self.model_dir)
        self.config_path = config_path
        self.cfg = core.load_config(config_path)
        self.assets = None
        self.device = None

    def _ensure(self):
        if self.assets is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.assets = core.Assets(self.cfg, self.device)
        return self.assets

    def infer(self, image_tensor, prompt=""):
        if prompt not in ('', None):
            raise core.PipelineError('本阶段 prompt 必须为空字符串，收到 %r' % (prompt,))
        assets = self._ensure()
        return core.infer_core(assets, image_tensor)

    def identity(self):
        a = self._ensure()
        return {
            'entry': 'tensor_runner',
            'model_dir': self.model_dir,
            'device': self.device,
            'config_sha256': self.cfg['_config_sha256'],
            'asset_fingerprint': a.asset_fp,
            'source_fingerprint': a.source_fp,
            'effective_fingerprint': a.effective_fp,
            'load_report': a.load_report,
            'tile_report': a.tile_report,
        }

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
