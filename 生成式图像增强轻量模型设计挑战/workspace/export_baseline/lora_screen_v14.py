# -*- coding: utf-8 -*-
"""LORA-SCREEN（v14 兼容版）：v14 检查点含 optimizer 状态，core.load_lora_payload 的
weights_only=True 会失败 → 运行时回退到 weights_only=False（仅本地可信文件，不改冻结源码）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))
import torch  # noqa: E402
import core as PBcore  # noqa: E402

_orig = PBcore.load_lora_payload


def patched(path):
    try:
        return _orig(path)
    except Exception:
        payload = torch.load(str(path), map_location='cpu', weights_only=False)
        return payload


PBcore.load_lora_payload = patched
import lora_screen  # noqa: E402

sys.argv = [sys.argv[0]] + sys.argv[1:]
lora_screen.main()
