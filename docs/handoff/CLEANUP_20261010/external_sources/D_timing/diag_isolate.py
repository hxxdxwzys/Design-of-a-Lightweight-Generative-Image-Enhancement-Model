# -*- coding: utf-8 -*-
"""隔离：512 patch 上 packaged runner 与 student-direct 管线的差异来源。"""
import os, sys, importlib.util
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

P = r"D:\models\finals_20260927\Track2_light_diffusion_inference"
MD = os.path.join(P, "model_dir")
TEST = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集"
dev = "cuda"

img = np.asarray(Image.open(os.path.join(TEST, "case3.jpg")).convert("RGB"), dtype=np.uint8)
patch = img[0:512, 0:512].copy()
print("patch mean %.2f" % patch.mean(), flush=True)


def prep_fp32(p):
    return torch.from_numpy(np.ascontiguousarray(p)).permute(2, 0, 1)[None].to(dev).float().div_(127.5).sub_(1.0)


def to_u8(y):
    o = y[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip((o + 1) * 127.5, 0, 255)


def cmp(n1, x, n2, y):
    dd = np.abs(x.astype(np.int16) - y.astype(np.int16))
    print("  %-30s vs %-30s max=%3d mean=%.4f pct!=0=%.2f%%" % (
        n1, n2, dd.max(), dd.mean(), (dd != 0).mean() * 100), flush=True)


m = torch.jit.load(os.path.join(MD, "your_model.pt"), map_location=dev)
m.eval()
with torch.no_grad():
    a = to_u8(m(prep_fp32(patch).half()))       # packaged, fp16 input, no graph
    b = to_u8(m(prep_fp32(patch)))              # packaged, fp32 input, no graph

spec = importlib.util.spec_from_file_location("rr", os.path.join(MD, "runner.py"))
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)
R = mm.Runner(MD)
with torch.no_grad():
    c = to_u8(R.infer(prep_fp32(patch), prompt=""))
R2 = mm.Runner(MD)
with torch.no_grad():
    d = to_u8(R2.infer(prep_fp32(patch), prompt=""))

print("== 512 patch ==", flush=True)
cmp("jit fp16-in", a, "jit fp32-in", b)
cmp("jit fp32-in", b, "graph fp32-in", c)
cmp("graph run1", c, "graph run2 (new Runner)", d)

# ---- student-direct pipeline (old render script math) ----
sys.path.insert(0, r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
from diffusers import UNet2DModel


class FastAttnProcessor:
    def __init__(self, inner_dim, heads):
        self.inner_dim = int(inner_dim); self.heads = int(heads); self.head_dim = self.inner_dim // self.heads

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b_, c_, h_, w_ = hidden_states.shape
            hidden_states = hidden_states.view(b_, c_, h_ * w_).transpose(1, 2)
        bs, sl, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        q = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        k = attn.to_k(encoder_hidden_states); v = attn.to_v(encoder_hidden_states)
        hh, hd = self.heads, self.head_dim
        q = q.view(bs, -1, hh, hd).transpose(1, 2)
        k = k.view(bs, -1, hh, hd).transpose(1, 2)
        v = v.view(bs, -1, hh, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        o = o.transpose(1, 2).reshape(bs, -1, hh * hd).to(q.dtype)
        o = attn.to_out[0](o); o = attn.to_out[1](o)
        if input_ndim == 4:
            o = o.transpose(-1, -2).reshape(bs, c_, h_, w_)
        if attn.residual_connection:
            o = o + residual
        return o / attn.rescale_output_factor


class Refiner(nn.Module):
    def __init__(self, ch, blocks, arch="default"):
        super().__init__()
        self.arch = arch
        out_ch = 12 if arch == "r256" else 3
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks):
            layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        layers += [nn.Conv2d(ch, out_ch, 3, 1, 1)]
        self.net = nn.Sequential(*layers); self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        if self.arch == "r256":
            base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
            return (base + self.ps(self.net(x))).clamp(-1, 1)
        return (x + self.net(x)).clamp(-1, 1)


ck = r"D:\models\tiny_v1\v21_s21000.pt"
rf = r"D:\models\refine_v1\refiner_thin2_s6000.pt"
print("ckpt exists:", os.path.exists(ck), "refiner exists:", os.path.exists(rf), flush=True)
if os.path.exists(ck) and os.path.exists(rf):
    d0 = torch.load(ck, map_location="cpu", weights_only=False)
    ch = tuple(d0["ch"]); res = int(d0["res"]); alpha = float(d0["alpha"])
    print("  ckpt=%s ch=%s res=%d alpha=%.6f steps=%s" % (os.path.basename(ck), ch, res, alpha, d0.get("steps")), flush=True)
    unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                       block_out_channels=ch, down_block_types=("DownBlock2D",) * len(ch),
                       up_block_types=("UpBlock2D",) * len(ch))
    unet.load_state_dict(d0["unet"]); unet = unet.to(dev).eval().half()
    n_patch = 0
    for mod in unet.modules():
        if hasattr(mod, "processor") and hasattr(mod, "to_q") and hasattr(mod, "heads"):
            try:
                mod.processor = FastAttnProcessor(mod.to_q.out_features, mod.heads); n_patch += 1
            except Exception:
                pass
    rd = torch.load(rf, map_location="cpu", weights_only=False)
    print("  refiner ch=%d blocks=%d arch=%s steps=%s (patched attn=%d)" % (
        rd["ch"], rd["blocks"], rd.get("arch", "default"), rd.get("steps"), n_patch), flush=True)
    ref = Refiner(rd["ch"], rd["blocks"], rd.get("arch", "default")).to(dev).eval().half()
    ref.load_state_dict(rd["net"])
    sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5
    th = prep_fp32(patch).half()
    xs = F.interpolate(th, size=(256, 256), mode="bicubic", align_corners=False)
    with torch.no_grad():
        pred = unet(xs, torch.full((1,), 500, dtype=torch.long, device=dev)).sample
        y = (xs - sq_1a * pred) / sq_a
        y2 = F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False)
        e_raw = to_u8(y2)
        e_ref = to_u8(ref(y2.half()))
    print("== student-direct vs packaged ==", flush=True)
    cmp("student raw (no refiner)", e_raw, "packaged graph", c)
    cmp("student+refiner (no unsharp)", e_ref, "packaged graph", c)
