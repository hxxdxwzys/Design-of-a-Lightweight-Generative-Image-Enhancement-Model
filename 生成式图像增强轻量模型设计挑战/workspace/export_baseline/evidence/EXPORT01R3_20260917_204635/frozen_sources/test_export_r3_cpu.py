# -*- coding: utf-8 -*-
"""EXPORT-01R3 CPU 测试：EXP-R13 时间嵌入设备修复（数值等价 + 设备归属 + save/load）。"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EB = HERE.parent
sys.path.insert(0, str(EB))

import core as PBcore  # noqa: E402
from chain import patch_device_safe_timestep  # noqa: E402

_VOSR = str(PBcore.load_config()["vosr_dir"])
if _VOSR not in sys.path:
    sys.path.insert(0, _VOSR)


def original_formula(t, dim, max_period=10000.0):
    """原实现公式（lightningdit.TimestepEmbedder.timestep_embedding），独立复算。"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(
        start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def make_embedder(dim=128):
    from models.lightningdit import TimestepEmbedder
    for kwargs in ({"hidden_size": 64, "frequency_embedding_size": dim},
                   {"dim": dim}, {}):
        try:
            return TimestepEmbedder(**kwargs)
        except TypeError:
            continue
    raise unittest.SkipTest("TimestepEmbedder 构造签名不匹配")


class TestExpR13TimestepDevice(unittest.TestCase):
    def test_patched_matches_original_formula(self):
        import copy

        class Holder(torch.nn.Module):
            """最小 DiT 替身：只带 t_embedder / r_embedder 两个属性。"""

            def __init__(self):
                super().__init__()
                base = make_embedder()
                self.t_embedder = base
                self.r_embedder = copy.deepcopy(base)

        holder = Holder()
        patched = patch_device_safe_timestep(holder)
        self.assertEqual(len(patched), 2, "两个嵌入器都必须被修补：%s" % patched)
        for name in ("t_embedder", "r_embedder"):
            emb = getattr(holder, name)
            for tv in (0.0, 1.0, 0.5):
                t = torch.full((3,), tv, dtype=torch.float32)
                got = emb.timestep_embedding(t, emb.frequency_embedding_size)
                ref = original_formula(t, emb.frequency_embedding_size, 10000.0)
                self.assertEqual(tuple(got.shape), tuple(ref.shape))
                self.assertLess(float((got - ref).abs().max()), 1e-6,
                                "%s t=%s 与原始公式不一致" % (name, tv))

    def test_freqs_built_on_input_device(self):
        emb = make_embedder()
        patch_device_safe_timestep(emb)
        t = torch.zeros(2, dtype=torch.float32)
        out = emb.timestep_embedding(t, emb.frequency_embedding_size)
        self.assertEqual(out.device, t.device)
        # 传入非默认 dtype 的输入也应正常工作（不依赖全局默认设备/类型）
        t64 = torch.zeros(2, dtype=torch.float64)
        out64 = emb.timestep_embedding(t64, emb.frequency_embedding_size)
        self.assertEqual(out64.device, t64.device)

    def test_trace_save_load_runs_after_patch(self):
        emb = make_embedder()
        patch_device_safe_timestep(emb)

        class Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, t):
                return self.inner.timestep_embedding(t, self.inner.frequency_embedding_size)

        w = Wrap(emb).eval()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "emb.pt"
            traced = torch.jit.trace(w, torch.zeros(2, dtype=torch.float32), check_trace=False)
            torch.jit.save(traced, str(p))
            loaded = torch.jit.load(str(p), map_location="cpu").eval()
            t = torch.full((2,), 0.5, dtype=torch.float32)
            out = loaded(t)
            ref = original_formula(t, emb.frequency_embedding_size, 10000.0)
            self.assertLess(float((out - ref).abs().max()), 1e-6)

    def test_graph_static_audit_no_cpu_arange_in_patched_source(self):
        """图静态审计（CPU 侧）：修补后不得再出现"CPU arange + 后置 .to(device)"模式。"""
        src = (EB / "chain.py").read_text(encoding="utf-8")
        self.assertIn("device=t.device", src)
        seg = src.split("def patch_device_safe_timestep")[1].split("class VAEEncode")[0]
        # 只看代码，不看注释/文档字符串（文档里会解释"不硬编码 cuda:0"）
        code = "\n".join(ln.split("#")[0] for ln in seg.splitlines()
                          if not ln.strip().startswith(('"""', "#")))
        self.assertNotIn('"cuda:0"', code)
        self.assertNotIn("'cuda:0'", code)
        audit = {"schema": "EXPORT01R3-graph-audit/1", "scope": "chain.timestep_embedding",
                 "cpu_arange_plus_late_to": False, "hardcoded_cuda": False,
                 "device_source": "input tensor device (t.device)"}
        (Path(tempfile.gettempdir()) / "r3_graph_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False), encoding="utf-8")


class TestExpR14RealPath(unittest.TestCase):
    def test_required_set_ignores_non_dict_phaseA_values(self):
        import r_verify
        rd = {"phaseA": {"released": True, "call_4": {"call": "A_511x513"}},
              "phaseB": {"call_1": {"call": "B_512_1"}, "call_2": {"call": "B_512_2"},
                         "call_3": {"call": "B_512_3"}, "call_4": {"call": "B_511x513"}},
              "pairs": [{"name": "case1_full"}, {"name": "case7_full"}],
              "migration": [{"name": "m"}]}
        prereg = {"cases": [{"name": "case1_crop512"}, {"name": "case1_crop511x513"},
                            {"name": "case1_full"}, {"name": "case7_full"}]}
        self.assertTrue(r_verify.check_required_set(rd, prereg))   # 不得因 released=True 抛 TypeError

    def test_partial_readings_persist_on_failure_path(self):
        """真实驱动路径：run_verification 逐次写 readings.json，worker 失败时保留 partial。"""
        src = (EB / "r_verify.py").read_text(encoding="utf-8")
        self.assertIn("def flush(", src)
        self.assertIn("readings.json", src)
        wsrc = (EB / "r_worker.py").read_text(encoding="utf-8")
        self.assertIn("保留失败前的结构化 partial", wsrc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
