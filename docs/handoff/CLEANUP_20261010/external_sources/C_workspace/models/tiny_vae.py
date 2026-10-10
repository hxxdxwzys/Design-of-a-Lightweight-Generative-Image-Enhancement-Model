"""TAESD 轻量 VAE 适配层: 把 AutoencoderTiny 封装成与 LightLatentDiff.encode/decode 兼容的接口。

目的: 用 ~27ms 的 TAESD 替换 ~285ms 的 SD2.1 VAE, 大幅降低推理时延 (加速比评分项)。
兼容性: TAESD latent 与 SD2.1 latent 分布相似 (归一化后 mean/max 接近),
        v3 学生 UNet 无需重训即可直接适配 (需实测验证)。
"""
import torch
import torch.nn as nn
from diffusers import AutoencoderTiny


class TinyVAE(nn.Module):
    """包装 AutoencoderTiny: encode(img)->latent, decode(latent)->img, 均 [-1,1]。

    接口与 LightLatentDiff 的 vae 子模块兼容:
      - .encode(img): [-1,1] -> latent (已乘 scaling)
      - .decode(latent): latent(含scaling) -> [-1,1]
    """

    def __init__(self, model_id='madebyollin/taesd', device='cuda', fp16=True):
        super().__init__()
        self.vae = AutoencoderTiny.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.vae.eval()
        self.scaling = 1.0  # TAESD latent 已是最终值域, 无需额外 scaling
        self.config = type('C', (), {'scaling_factor': self.scaling})()
        self.dtype = torch.float16 if fp16 else torch.float32
        # requires_grad=False 使 trainable_params() 只返回 UNet
        for p in self.parameters():
            p.requires_grad_(False)

    class _LatentDist:
        """伪装 latent_dist 对象: LightLatentDiff.encode 调 .latent_dist.sample()。"""

        def __init__(self, latents):
            self._latents = latents

        def sample(self):
            return self._latents

    class _Encoded:
        """伪装 encode 输出: LightLatentDiff.encode 调 vae.encode(img).latent_dist.sample()。"""

        def __init__(self, latents):
            self.latent_dist = TinyVAE._LatentDist(latents)

    @torch.no_grad()
    def encode(self, img):
        """返回 .latent_dist.sample() 兼容对象 (LightLatentDiff.encode 会再乘 scaling)。
        注意: 只返回原始 latents, 不乘 scaling (与 AutoencoderKL.encode 契约一致)。"""
        x = img.to(self.dtype)
        latents = self.vae.encode(x).latents  # AutoencoderTiny 直接给 latents
        return self._Encoded(latents.float())

    class _Decoded:
        """伪装 decode 输出对象: LightLatentDiff.decode 调 .sample。"""

        def __init__(self, out):
            self.sample = out

    @torch.no_grad()
    def decode(self, latent):
        """latent (含scaling) -> .sample 兼容对象 ([-1,1])"""
        z = latent.float() / self.scaling
        out = self.vae.decode(z.half()).sample
        return self._Decoded(out.float().clamp(-1, 1))
