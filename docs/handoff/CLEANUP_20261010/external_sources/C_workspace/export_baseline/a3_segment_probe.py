# -*- coding: utf-8 -*-
"""DELIV-01R A3 分段诊断（≤20 次组件调用）：编码器 / 解码器各自按段对照 eager 与 traced。

做法：以本包已有的 enc_512 输入与 dec_64 输入为基准；每段用**上一段 eager 输出**作为两侧共同输入，
从而把"本段自身的 traced 误差"与"上游传播"分开。所有段边界两侧原始输出都落盘。

用法: python a3_segment_probe.py <exp_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_PB = HERE.parent / "pipeline_baseline"
if str(_PB) not in sys.path:
    sys.path.insert(0, str(_PB))


class EncChunk(torch.nn.Module):
    def __init__(self, vae, part):
        super().__init__()
        self.vae = vae
        self.part = part

    def forward(self, x):
        enc = self.vae.encoder
        h = x
        if self.part == "front":
            h = enc.conv_in(h)
            for i in (0, 1):
                o = enc.down_blocks[i](h)
                h = o[0] if isinstance(o, (tuple, list)) else o
        elif self.part == "mid":
            for i in (2, 3):
                o = enc.down_blocks[i](h)
                h = o[0] if isinstance(o, (tuple, list)) else o
            h = enc.mid_block(h)
        elif self.part == "tail":
            h = enc.conv_norm_out(h)
            h = enc.conv_act(h)
            h = enc.conv_out(h)
            h = self.vae.quant_conv(h)
        return h


class DecChunk(torch.nn.Module):
    def __init__(self, dec, part):
        super().__init__()
        self.dec = dec
        self.part = part

    def forward(self, z):
        d = self.dec
        h = z
        if self.part == "front":
            h = d.conv_in(h)
            h = d.mid_block(h)
        elif self.part == "mid":
            for i in (0, 1):
                o = d.up_blocks[i](h)
                h = o[0] if isinstance(o, (tuple, list)) else o
        elif self.part == "tail":
            for i in (2, 3):
                o = d.up_blocks[i](h)
                h = o[0] if isinstance(o, (tuple, list)) else o
            h = d.conv_norm_out(h)
            h = d.conv_act(h)
            h = d.conv_out(h)
        return h


def main():
    exp_dir = Path(sys.argv[1]).resolve()
    import frozen_mode
    flags = frozen_mode.apply_frozen_mode()
    import core as PBcore
    out_dir = exp_dir / "a3_segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cfg = PBcore.load_config()
    assets = PBcore.Assets(cfg, dev)
    vae = assets.vae.to(torch.float16).eval()
    dec = assets.dec.to(torch.float16).eval()
    ref_dir = exp_dir / "component_refs"
    x_enc = torch.from_numpy(np.load(ref_dir / "in_enc_512.npy", allow_pickle=False)) \
        .to(dev).to(torch.float16)
    z_dec = torch.from_numpy(np.load(ref_dir / "in_dec_64.npy", allow_pickle=False)) \
        .to(dev).to(torch.float16)
    res = {"schema": "DELIV01R-a3-segments/1", "flags": flags, "segments": {},
           "note": "每段输入 = 上一段 eager 输出（隔离本段误差与上游传播）；两侧原始输出已落盘"}
    calls = {"n": 0}

    def probe(group, name, mod, inp):
        """1 次 trace + 1 次 traced 运行 + 1 次 eager 运行 = 3 次真实组件前向。"""
        tr = torch.jit.trace(mod, inp, strict=False, check_trace=False)
        calls["n"] += 1
        with torch.no_grad():
            got = tr(inp)
            calls["n"] += 1
            exp = mod(inp)
            calls["n"] += 1
        g = got.detach().float().cpu().numpy()
        e = exp.detach().float().cpu().numpy()
        d = np.abs(g - e)
        np.save(out_dir / ("%s_%s_eager.npy" % (group, name)), e)
        np.save(out_dir / ("%s_%s_traced.npy" % (group, name)), g)
        rec = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
               "finite": bool(np.isfinite(g).all()), "shape": list(g.shape),
               "dtype": str(got.dtype),
               "exact_equal": bool(torch.equal(got, exp))}
        res["segments"]["%s.%s" % (group, name)] = rec
        return exp.detach()      # 下一段的共同输入 = eager 输出

    h = x_enc
    for part in ("front", "mid", "tail"):
        h = probe("enc", part, EncChunk(vae, part).to(dev).eval(), h)
    d_in = z_dec
    for part in ("front", "mid", "tail"):
        d_in = probe("dec", part, DecChunk(dec, part).to(dev).eval(), d_in)
    res["n_component_forwards"] = calls["n"]
    first = None
    for k, v in res["segments"].items():
        if v["max_abs"] > 1e-5 and first is None:
            first = k
    res["first_divergent_segment"] = first
    res["interpretation"] = ("首个分歧段：%s（若为 None 表示各段均在 1e-5 内）" % first)
    (exp_dir / "a3_segment_probe.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n_forwards": calls["n"], "first": first,
                      "segments": {k: round(v["max_abs"], 8) for k, v in
                                   res["segments"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
