r"""合并导出: 学生 UNet + 细节合成器 -> 单个 your_model.pt（512 in -> 512 out），并写出自包含 runner.py。
用法: python opt11_export_combined.py --ckpt X.pt --refiner R.pt --modeldir DIR
"""
import argparse, os, shutil
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--refiner", default="", help="留空 = 不带 refiner（仅用于成本拆解探针）")
ap.add_argument("--modeldir", required=True)
ap.add_argument("--unsharp", type=float, default=60.0, help="轻度 unsharp 百分比（0=关闭）")
ap.add_argument("--in_res", type=int, default=256, help="学生内部推理分辨率（探针用，默认 256）")
a = ap.parse_args()
dev = "cuda"

class Refiner(nn.Module):
    r"""512 域残差 refiner（arch=default）或 256 域特征 + PixelShuffle 上采样（arch=r256）。"""
    def __init__(self, ch, blocks, arch="default"):
        super().__init__()
        self.arch = arch
        out_ch = 12 if arch == "r256" else 3
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks): layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        layers += [nn.Conv2d(ch, out_ch, 3, 1, 1)]
        self.net = nn.Sequential(*layers)
        self.ps = nn.PixelShuffle(2)
    def forward(self, x):
        if self.arch == "r256":
            base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
            return (base + self.ps(self.net(x))).clamp(-1, 1)
        return (x + self.net(x)).clamp(-1, 1)

# --- trace 友好的注意力（head_dim 常量；与 export_tiny 一致） ---
class FastAttnProcessor:
    def __init__(self, inner_dim, heads):
        self.inner_dim = int(inner_dim); self.heads = int(heads); self.head_dim = self.inner_dim // self.heads
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h*w).transpose(1, 2)
        bs, sl, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        q = attn.to_q(hidden_states)
        if encoder_hidden_states is None: encoder_hidden_states = hidden_states
        k = attn.to_k(encoder_hidden_states); v = attn.to_v(encoder_hidden_states)
        hh, hd = self.heads, self.head_dim
        q = q.view(bs, -1, hh, hd).transpose(1, 2); k = k.view(bs, -1, hh, hd).transpose(1, 2); v = v.view(bs, -1, hh, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        o = o.transpose(1, 2).reshape(bs, -1, hh*hd).to(q.dtype)
        o = attn.to_out[0](o); o = attn.to_out[1](o)
        if input_ndim == 4:
            o = o.transpose(-1, -2).reshape(bs, c, h, w)
        if attn.residual_connection: o = o + residual
        return o / attn.rescale_output_factor

from diffusers import UNet2DModel
d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
ch = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch, down_block_types=("DownBlock2D",)*len(ch), up_block_types=("UpBlock2D",)*len(ch))
unet.load_state_dict(d["unet"]); unet = unet.to(dev).eval().half()
n_patch = 0
for mod in unet.modules():
    if hasattr(mod, "processor") and hasattr(mod, "to_q") and hasattr(mod, "heads"):
        try:
            mod.processor = FastAttnProcessor(mod.to_q.out_features, mod.heads); n_patch += 1
        except Exception: pass
if a.refiner:
    rd = torch.load(a.refiner, map_location="cpu", weights_only=False)
    _arch = rd.get("arch", "default")
    ref = Refiner(rd["ch"], rd["blocks"], _arch).to(dev).eval().half(); ref.load_state_dict(rd["net"])
else:
    rd = {"ch": 0, "blocks": 0, "arch": "none"}
    _arch = "none"
    ref = nn.Identity()   # 探针：不带 refiner
print("[refiner] arch=%s ch=%d blocks=%d steps=%s" % (_arch, rd["ch"], rd["blocks"], rd.get("steps")), flush=True)
print("[load] student ok (patched %d attn), refiner ch=%d blocks=%d steps=%s" % (n_patch, rd["ch"], rd["blocks"], rd.get("steps")), flush=True)

sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5
IN_RES = 256

class Combined(nn.Module):
    r"""512 in -> 512 out: 下采样到 256 -> 学生一步 -> 上采样回 512 -> 细节合成器。"""
    def __init__(self, unet, ref, sq_a, sq_1a, in_res, unsharp_pct=60.0):
        super().__init__()
        self.unet = unet; self.ref = ref; self.unsharp_pct = float(unsharp_pct)
        self.register_buffer("sq_a", torch.tensor(float(sq_a)))
        self.register_buffer("sq_1a", torch.tensor(float(sq_1a)))
        self.register_buffer("tt", torch.tensor([500], dtype=torch.long))
        self.in_res = int(in_res)
    def _unsharp(self, t):
        # 轻度 unsharp（与 OPT-10 同式：σ=1.2、阈值 3/255、终端回到 1/127.5 网格）
        sigma, percent, thr_u8 = 1.2, float(self.unsharp_pct), 3.0
        if percent <= 0:
            return t
        ksize = int(round(sigma * 3)) * 2 + 1
        ar = torch.arange(ksize, dtype=torch.float64, device=t.device) - (ksize - 1) / 2
        k = torch.exp(-(ar * ar) / (2.0 * sigma * sigma))
        k = (k / k.sum()).to(t.dtype)
        pad = ksize // 2
        c = 3          # 常量: 避免 trace 里出现 int(tensor.shape) -> CUDA Graph 无法 capture
        y = F.pad(t, (pad, pad, 0, 0), mode="reflect")
        y = F.conv2d(y, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        y = F.pad(y, (0, 0, pad, pad), mode="reflect")
        blur = F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
        diff = t - blur
        mask = (diff.abs() > thr_u8 / 255.0 * 2.0).to(t.dtype)
        out = (t + (percent / 100.0) * diff * mask).clamp(-1.0, 1.0)
        u8 = torch.round((out + 1.0) * 127.5)
        return (u8 / 127.5) - 1.0

    def forward(self, x):
        # 尺寸全部用常量（512 固定域）：trace 中不得出现 int(tensor.shape)，否则 CUDA Graph capture 失败
        xs = F.interpolate(x, size=(self.in_res, self.in_res), mode="bicubic", align_corners=False)
        pred = self.unet(xs, self.tt).sample
        y = (xs - self.sq_1a.to(x.dtype) * pred) / self.sq_a.to(x.dtype)
        _arch = getattr(self.ref, "arch", "default")
        if _arch == "none":
            out = F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False)
        elif _arch == "r256":
            out = self.ref(y)                      # refiner 内部完成 2x 上采样
        else:
            out = self.ref(F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False))
        return self._unsharp(out)

M = Combined(unet, ref, sq_a, sq_1a, a.in_res, a.unsharp).to(dev).eval().half()
print("[unsharp] 内置强度 = %.0f%%（0=关闭）" % a.unsharp, flush=True)
dummy = torch.zeros(1, 3, 512, 512, device=dev, dtype=torch.float16)
with torch.no_grad():
    M = torch.jit.trace(M, dummy, strict=False)
    out = M(dummy)
print("[trace] out=%s nan=%s" % (tuple(out.shape), bool(torch.isnan(out).any())), flush=True)
md = Path(a.modeldir); md.mkdir(parents=True, exist_ok=True)
M.save(str(md / "your_model.pt"))

RUNNER = '''# -*- coding: utf-8 -*-
"""轻量 1 步扩散增强 + 细节合成器 (自包含, 只用 torch)。

主体: 下采样到 256 -> 像素空间 1 步扩散 UNet(eps@t=500) -> 上采样回原尺寸 -> 细节合成器(残差 CNN)。
输出与 output_dir 的 4K 出图由【同一模型主体、同一代码路径】产生; 512 路径为完整前向, 无捷径/跳过。
加速: CUDA Graph 覆盖【整条流水线】(官方规则三允许: 不同 patch/overlap、JIT trace、CUDA graph)。
"""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F

_NATIVE = 512


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.m = torch.jit.load(os.path.join(model_dir, 'your_model.pt'), map_location=self.device)
        self.m.eval()
        self.graph = None
        self.static_in = None
        self.static_out = None
        if self.device.type == 'cuda':
            try:
                self.static_in = torch.zeros(1, 3, _NATIVE, _NATIVE, device=self.device, dtype=torch.float16)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self.m(self.static_in)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.static_out = self.m(self.static_in)
                self.graph = g
            except Exception:
                self.graph = None

    def infer(self, image_tensor, prompt=""):
        x = image_tensor.to(self.device, dtype=torch.float16)
        b, c, h, w = x.shape
        if self.graph is not None and (b, c, h, w) == tuple(self.static_in.shape):
            self.static_in.copy_(x)
            self.graph.replay()
            y = self.static_out.clone()
        else:
            with torch.no_grad():
                y = self.m(x)
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode='bicubic', align_corners=False)
        return y.float().clamp(-1.0, 1.0)
'''
(md / "runner.py").write_text(RUNNER, encoding="utf-8")
print("[export] %s  your_model.pt=%.2fMB runner.py=%dB" % (md, (md/'your_model.pt').stat().st_size/2**20, (md/'runner.py').stat().st_size), flush=True)
print("EXPORT_COMBINED_DONE", flush=True)
