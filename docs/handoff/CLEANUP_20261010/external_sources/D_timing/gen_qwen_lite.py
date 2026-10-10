# -*- coding: utf-8 -*-
"""生成 qwen VAE 自包含裁剪版 (2026-09-08): 去 diffusers 依赖
输出: D:/models/final_compliant_14b/model_dir/models/qwen_vae_lite.py
"""
import re
from pathlib import Path

SRC = Path(r'D:/models/finals/VOSR/models/qwenimage_vae2d.py')
OUT = Path(r'D:/models/final_compliant_14b/model_dir/models/qwen_vae_lite.py')
OUT.parent.mkdir(parents=True, exist_ok=True)

src = SRC.read_text(encoding='utf-8')

# 1) 删 diffusers import
for pat in [
    'from diffusers.configuration_utils import ConfigMixin, register_to_config\n',
    'from diffusers.loaders import FromOriginalModelMixin\n',
    'from diffusers.models.activations import get_activation\n',
    'from diffusers.models.modeling_outputs import AutoencoderKLOutput\n',
    'from diffusers.models.modeling_utils import ModelMixin\n',
    'from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution\n',
]:
    src = src.replace(pat, '')

# 2) 类头 + 装饰器
src = src.replace('class AutoencoderKLQwenImage2D(ModelMixin, ConfigMixin, FromOriginalModelMixin):',
                  'class AutoencoderKLQwenImage2D(nn.Module):')
src = src.replace('    @register_to_config\n', '')

# 3) encode/decode 简化 (无 return_dict 包装)
src = src.replace('''    def encode(self, x: torch.Tensor, return_dict: bool = True):
        h = self.encoder(x)
        h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)
''', '''    def encode(self, x: torch.Tensor, return_dict: bool = True):
        h = self.encoder(x)
        h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior
''')
src = src.replace('''    def decode(self, z: torch.Tensor, return_dict: bool = True):
        x = self.post_quant_conv(z)
        dec = self.decoder(x)
        dec = torch.clamp(dec, -1.0, 1.0)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
''', '''    def decode(self, z: torch.Tensor, return_dict: bool = True):
        x = self.post_quant_conv(z)
        dec = self.decoder(x)
        dec = torch.clamp(dec, -1.0, 1.0)
        return dec
''')
src = src.replace('''        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec''', '''        posterior = self.encode(sample)
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return dec''')

# 4) 追加自包含尾部
src += '''

# ---- 自包含补充 (去 diffusers) ----
import torch.nn.functional as _F  # noqa: E402


def get_activation(name: str):
    return {'silu': nn.SiLU(), 'gelu': nn.GELU(), 'relu': nn.ReLU()}.get(name, nn.SiLU())


class DiagonalGaussianDistribution:
    """简化版: 前 half 通道为 mean, 后 half 为 logvar"""

    def __init__(self, parameters: torch.Tensor):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)

    def mode(self):
        return self.mean

    def sample(self, generator=None):
        eps = torch.randn_like(self.mean)
        return self.mean + eps * torch.exp(0.5 * self.logvar)
'''
OUT.write_text(src, encoding='utf-8')
print(f'written: {OUT} ({len(src.splitlines())} lines)')
