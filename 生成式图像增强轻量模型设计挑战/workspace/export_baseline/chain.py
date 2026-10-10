# -*- coding: utf-8 -*-
"""EXPORT-01 数值链组件：可导出（torch.jit trace/script 友好）的完整推理链。

设计要点（与 EXPORT-01 任务单一致）：
  - 不含 PIL / cv2 / numpy：wavelet 颜色处理已忠实张量化（CPU 预登记验证 max_abs=0）。
  - 噪声不是模块内部随机：z 由 runner 用局部 Generator 生成后**显式传入**（review §160）。
  - 分块/拼接等编排留在薄 runner；本模块只承载数值主体，权重与参考实现同源。
  - 后处理（wavelet）固定 fp32：与参考实现同精度，避免 fp16 舍入翻转 uint8 边界。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def gaussian_kernel_1d(sigma: float, ksize: int = 0) -> torch.Tensor:
    """与 cv2.getGaussianKernel(sigma, ksize) 一致的 1-D 核（float32）。

    ksize=0 时按 OpenCV 对 CV_32F 的规则推导：cvRound(sigma*4*2+1)|1。
    """
    if ksize <= 0:
        ksize = int(round(float(sigma) * 4 * 2 + 1)) | 1
    if ksize % 2 == 0:
        ksize += 1
    # 逐字对齐 OpenCV getGaussianKernel 的运算顺序（exp(-0.5/sigma^2 * x^2)），
    # CPU 实测与 cv2 核最大差 2.8e-17（另一种等价写法差 3.6e-9，会放大边界翻转概率）
    scale2 = -0.5 / (float(sigma) ** 2)
    x = torch.arange(ksize, dtype=torch.float64) - (ksize - 1) * 0.5
    k = torch.exp(scale2 * x * x)
    k = k / k.sum()
    return k.to(torch.float32)


class WaveletFix(nn.Module):
    """忠实张量化的 wavelet 颜色处理（等价于参考 numpy/cv2/PIL 实现）。

    参考语义：
      tgt = uint8(target); src = uint8(source)            # 截断，等价 to_pil_image
      src.resize(tgt.size) -> 同尺寸时恒等（CPU 已验证）
      source_low = blur(src/255); target_low = blur(tgt/255)
      result = clip(source_low + (tgt/255 - target_low), 0, 1) * 255 -> uint8（截断）
    全部在 fp32 下执行，最后返回 fp32 [-1,1]。
    """

    def __init__(self, sigma: float = 5.0):
        super().__init__()
        self.sigma = float(sigma)
        k1 = gaussian_kernel_1d(self.sigma)
        self.ksize = int(k1.numel())
        w1 = k1.view(1, 1, 1, self.ksize).expand(3, 1, 1, self.ksize).contiguous()
        w2 = k1.view(1, 1, self.ksize, 1).expand(3, 1, self.ksize, 1).contiguous()
        self.register_buffer('w1', w1)
        self.register_buffer('w2', w2)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        p = self.ksize // 2
        y = F.pad(x, (p, p, 0, 0), mode='reflect')
        y = F.conv2d(y, self.w1, groups=3)
        y = F.pad(y, (0, 0, p, p), mode='reflect')
        return F.conv2d(y, self.w2, groups=3)

    @staticmethod
    def _quantize_u8(t: torch.Tensor) -> torch.Tensor:
        # 与 torchvision.to_pil_image 一致：clamp(0,1) -> *255 -> 截断
        return torch.clamp(t * 255.0, 0.0, 255.0).to(torch.uint8)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        target = target.float()
        source = source.float()
        if target.shape[2] != source.shape[2] or target.shape[3] != source.shape[3]:
            raise ValueError('wavelet: target/source 尺寸必须相同（同尺寸 resize 恒等）')
        tgt_u8 = self._quantize_u8(torch.clamp(target, -1.0, 1.0) * 0.5 + 0.5)
        src_u8 = self._quantize_u8(torch.clamp(source, -1.0, 1.0) * 0.5 + 0.5)
        tn = tgt_u8.float() / 255.0
        sn = src_u8.float() / 255.0
        t_low = self._blur(tn)
        s_low = self._blur(sn)
        res = torch.clamp(s_low + (tn - t_low), 0.0, 1.0) * 255.0
        res_u8 = torch.clamp(torch.floor(res), 0.0, 255.0)
        return (res_u8 / 255.0) * 2.0 - 1.0


class CastIn(nn.Module):
    """把输入转成指定 dtype 后再进内部模块。

    用途：LightningDiT 的 TimestepEmbedder.forward 内部先用 float32 算 cos/sin 嵌入
    （lightningdit.py:198-207），随后直接 self.mlp(t_freq)。fp16 权重下必须在进入 mlp
    之前转换，否则 Linear 出现 Float×Half 不匹配。对 fp32 原实现该转换是恒等的。
    """

    def __init__(self, inner: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.inner = inner
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x.to(self.dtype))


class CastOut(nn.Module):
    """把内部模块的输出转成指定 dtype。

    用途（转换记录的一部分）：LightningDiT 的 TimestepEmbedder 内部固定用 float32
    计算 cos/sin（lightningdit.py:198-207），fp16 权重会导致 t_block/final_layer 的
    Linear 出现 Float×Half 不匹配。对 fp32 原实现该转换是恒等的（no-op）。
    """

    def __init__(self, inner: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.inner = inner
        self.dtype = dtype

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.inner(t).to(self.dtype)


def patch_fp16_embedders(dit: nn.Module, dtype: torch.dtype = torch.float16) -> list:
    """把 DiT 的时间嵌入对齐到工作 dtype；返回被改动的属性路径（写进转换记录）。

    优先只包住嵌入器内部的 mlp（入口 CastIn），使 cos/sin 仍在 float32 计算、
    进入 Linear 前转 fp16；无 mlp 时退化为整体 CastOut。
    """
    patched = []
    for name in ('t_embedder', 'r_embedder'):
        emb = getattr(dit, name, None)
        if emb is None:
            continue
        if hasattr(emb, 'mlp'):
            emb.mlp = CastIn(emb.mlp, dtype)
            patched.append(name + '.mlp')
        else:
            setattr(dit, name, CastOut(emb, dtype))
            patched.append(name)
    return patched


def patch_device_safe_timestep(dit: nn.Module, max_period: float = 10000.0) -> list:
    """EXP-R13：消除 t/r 时间嵌入里"CPU arange × 异设备常量"的路径。

    原实现（lightningdit.TimestepEmbedder.timestep_embedding）在 CPU 上构造 arange，
    再用 Python 常量做乘法，最后才 .to(device=t.device)。trace 保存/重载后，常量可能
    落在执行设备上，简单执行器下就出现 cuda:0 与 cpu 混用。这里把 arange 直接建在
    **输入张量的设备**上（不硬编码 cuda:0），频率公式、运算顺序与 fp32 精度完全不变。
    """
    import math
    import types
    patched = []
    for name in ('t_embedder', 'r_embedder'):
        emb = getattr(dit, name, None)
        if emb is None or not hasattr(emb, 'timestep_embedding'):
            continue

        def _device_safe(self, t, dim, _max_period=max_period):
            half = dim // 2
            freqs = torch.exp(-math.log(_max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half)
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

        emb.timestep_embedding = types.MethodType(_device_safe, emb)
        patched.append(name + '.timestep_embedding(device=t.device)')
    return patched


class VAEEncode(nn.Module):
    """SD2 VAE 编码 -> 归一化潜码（posterior mode，确定性）。

    DELIV-01R A2：允许编码器用 FP32 计算/权重，并在**边界显式转换** dtype
    （输入 -> 编码器 dtype；潜码 -> 下游 DiT 的 fp16）。转换记录进 recipe。
    """

    def __init__(self, vae: nn.Module, compute_dtype: torch.dtype = torch.float16,
                 out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.vae = vae
        self.scale = float(vae.config.scaling_factor)
        self.compute_dtype = compute_dtype
        self.out_dtype = out_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(x.to(self.compute_dtype)).latent_dist.mode() * self.scale
        return z.to(self.out_dtype)


class Decode(nn.Module):
    """潜码 -> **原始解码输出**（不做 clamp）。

    EXP-R03：clamp 会把 Inf/NaN 变成有限的 [-1,1]，掩盖数值异常。因此 clamp 移出模块，
    由 runner 在"clamp 前"先做 finite 检查（原始解码输出），检查通过后再 clamp。
    """

    def __init__(self, dec: nn.Module, scale: float,
                 compute_dtype: torch.dtype = torch.float16,
                 out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.dec = dec
        self.scale = float(scale)
        self.compute_dtype = compute_dtype
        self.out_dtype = out_dtype

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z.to(self.compute_dtype) / self.scale).to(self.out_dtype)


class DinoFeat(nn.Module):
    """DINOv2 条件特征（preprocess -> layer_8 特征），输入 [-1,1] 像素块。"""

    def __init__(self, dino: nn.Module, size: int = 448, layer_index: int = 8,
                 n_blocks: int = 12):
        super().__init__()
        self.dino = dino
        self.size = int(size)
        self.layer_index = int(layer_index)
        self.n_blocks = int(n_blocks)
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 逐字对齐 core.preprocess_dinov2_input：x(0..1)*255 -> /255 -> bicubic -> clip -> ImageNet
        y = (0.5 * x + 0.5) * 255.0
        y = y / 255.0
        y = F.interpolate(y, size=self.size, mode='bicubic')
        y = y.clip(0.0, 1.0)
        y = (y - self.mean.to(y.dtype)) / self.std.to(y.dtype)
        feats, x_norm = self.dino.forward_with_features(y)
        z = [feats['layer_%d' % i] for i in range(self.n_blocks)]
        z[-1] = x_norm
        return z[self.layer_index]


class DitStep(nn.Module):
    """DiT 一步流匹配更新：输入 (lq_latent, z, t_cur, t_nxt, feats) -> z1。"""

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.dit = dit

    def forward(self, lq_latent: torch.Tensor, z: torch.Tensor, t_cur: torch.Tensor,
                t_nxt: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        u = self.dit.forward_flexible(torch.cat([lq_latent, z], dim=1), t_cur, t_nxt, [feats])
        dt = t_cur[0] - t_nxt[0]
        return z - dt * u


class ExportBundle(nn.Module):
    """单一可保存/重载的数值链容器（子模块各自为已 trace 的 ScriptModule）。

    编排（分块/拼接/RNG）由薄 runner 负责；本容器保证同一权重主体与同一数值链。
    """

    def __init__(self, vae_enc_512, vae_enc_1024, dino_feat, dit_step, dec_64, dec_128,
                 wavelet):
        super().__init__()
        self.vae_enc_512 = vae_enc_512
        self.vae_enc_1024 = vae_enc_1024
        self.dino_feat = dino_feat
        self.dit_step = dit_step
        self.dec_64 = dec_64
        self.dec_128 = dec_128
        self.wavelet = wavelet

    def forward(self, x: torch.Tensor, tile_px: int, z: torch.Tensor, t_cur: torch.Tensor,
                t_nxt: torch.Tensor) -> torch.Tensor:
        """单块便利路径（输入需能放进一个 tile，例如 512x512 自检/隔离预检）。

        4K 的分块编排由薄 runner 负责；这里只串起同一条数值链。
        """
        lq = self.encode(x, tile_px)
        feats = self.features(x)
        z1 = self.step(lq, z, t_cur, t_nxt, feats)
        sr = self.decode(z1, 128 if tile_px == 1024 else 64)
        return self.post(sr, x)

    @torch.jit.export
    def encode(self, x: torch.Tensor, tile_px: int) -> torch.Tensor:
        if tile_px == 1024:
            return self.vae_enc_1024(x)
        return self.vae_enc_512(x)

    @torch.jit.export
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.dino_feat(x)

    @torch.jit.export
    def step(self, lq_latent: torch.Tensor, z: torch.Tensor, t_cur: torch.Tensor,
             t_nxt: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        return self.dit_step(lq_latent, z, t_cur, t_nxt, feats)

    @torch.jit.export
    def decode(self, z: torch.Tensor, tile_latent: int) -> torch.Tensor:
        if tile_latent == 128:
            return self.dec_128(z)
        return self.dec_64(z)

    @torch.jit.export
    def post(self, sr: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        return self.wavelet(sr, src)
