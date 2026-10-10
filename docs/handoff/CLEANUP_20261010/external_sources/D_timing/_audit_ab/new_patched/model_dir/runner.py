from __future__ import annotations
"""合规 model_dir runner (2026-09-03 自包含版, 规则三合规)


模型主体: VOSR 0.5B os + v7_s5000 LoRA(sc0.8, 已合并进权重) 完整推理链


  VAE encode -> dinov2 vitb14 特征 -> LightningDiT 一步 -> LightDecoder decode


完全自包含: 仅依赖 torch/torchvision/numpy (官方环境标配); 无 diffusers/timm/fairscale/peft/hub/网络


接口: Runner(model_dir).infer(image_tensor, prompt="") -> [1,3,H,W] in [-1,1]


说明: 512x512 完整前向 (fp16), 无任何捷径/跳过; 与出图(output_dir)同一模型主体同一套权重。


"""


import os


import sys





# 无 triton 环境 (Windows/部分 Linux) 下 dynamo 自动编译会崩: 静默回退 eager


os.environ.setdefault('TORCHDYNAMO_SUPPRESS_ERRORS', '1')





import numpy as np  # noqa: F401  (vision_transformer 顶层需要)


import torch


import torch.nn as nn


import torch.nn.functional as F





_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_PATH_BEFORE = list(sys.path)
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)
from models.lightningdit import LightningDiT  # noqa: E402


from models.light_decoder import LightDecoder  # noqa: E402


from dinov2.models import vision_transformer as _dvt  # noqa: E402


from vae_shim.autoencoder_kl import AutoencoderKL  # noqa: E402

# ── 交还 import 环境 (关键: 官方 IQA 评测器与我们在同一进程) ────────────────
# 本目录自带的是【裁剪桩】: timm 仅 3 个文件 (无 Block); einops 仅支持 3 种 pattern。
# 官方质量评测用 pyiqa (CLIPIQA/MUSIQ/MANIQA/NIQE), 它会 import timm/einops; 若命中桩包
# 就报 "Failed to load IQAEvaluator" (2026-09-11 实测 MANIQA:
# ImportError: cannot import name 'Block' from 'timm.models.vision_transformer')。
# 处置: 撤掉置顶路径, 并清掉 sys.modules 中来自本目录的同名包, 让官方环境里的真包生效;
#       本目录仅追加到 sys.path 末尾作为回落 (真包缺失时才用我们的桩)。
_MD_ABS = os.path.abspath(_MODEL_DIR)
# 关键: 只清 sys.modules 不够 —— 调用方 (或本文件) 把 model_dir 留在 sys.path 里,
# 下次 import timm/einops 仍命中桩包。必须同时把它挪到路径【末尾】。
sys.path[:] = [p for p in _PATH_BEFORE if os.path.abspath(p) != _MD_ABS]
for _n in [n for n in sys.modules if n.split('.')[0] in
           ('timm', 'einops', 'diffusers', 'fairscale')]:
    _f = getattr(sys.modules[_n], '__file__', None) or ''
    if _f and os.path.abspath(_f).startswith(_MD_ABS + os.sep):
        del sys.modules[_n]
if _MODEL_DIR not in sys.path:
    sys.path.append(_MODEL_DIR)






_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float16).view(1, 3, 1, 1)


_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float16).view(1, 3, 1, 1)








def _build_venc(pkg, device):


    """dinov2 vitb14 (与官方 hub 构造一致: init_values=1.0, block_chunks=0, img_size=518)"""


    venc = _dvt.vit_base(patch_size=14, img_size=518, init_values=1.0,


                         block_chunks=0, ffn_layer='mlp')


    venc.load_state_dict(pkg['venc'], strict=False)


    venc.eval()


    return venc








def _build_vae(pkg, device):


    cfg = pkg['vae_cfg']


    keys = ('in_channels', 'out_channels', 'down_block_types', 'up_block_types',


            'block_out_channels', 'layers_per_block', 'act_fn', 'latent_channels',


            'norm_num_groups', 'sample_size', 'scaling_factor', 'shift_factor',


            'force_upcast', 'use_quant_conv', 'use_post_quant_conv', 'mid_block_add_attention')


    kw = {k: cfg[k] for k in keys if k in cfg}


    kw.setdefault('use_quant_conv', True)


    kw.setdefault('use_post_quant_conv', True)


    kw.setdefault('mid_block_add_attention', True)


    kw.setdefault('force_upcast', False)


    kw.setdefault('shift_factor', 0.0)


    vae = AutoencoderKL(**kw)


    vae.load_state_dict(pkg['vae'], strict=True)


    vae.eval()


    return vae








class _FullChain(nn.Module):


    """完整推理链 (fp16 + autocast, 与训练/出图一致)"""





    def __init__(self, pkg):


        super().__init__()


        self.dinov2_size = int(pkg['dinov2_size'])          # 448


        self.layer_list = list(pkg['layer_dinov2b_list'])   # [8]


        self.scale = float(pkg['scale_factor'])


        self.venc = _build_venc(pkg, None)


        self.vae = _build_vae(pkg, None)


        self.diT = LightningDiT(**pkg['diT_cfg'])


        self.diT.load_state_dict(pkg['diT'], strict=True)


        self.diT.forward = self.diT.forward_flexible


        self.dec = LightDecoder(**pkg['dec_cfg'])


        self.dec.load_state_dict(pkg['dec'], strict=True)


        self.register_buffer('mean', _MEAN)


        self.register_buffer('std', _STD)





    def forward(self, x, z):


        with torch.autocast('cuda', dtype=torch.float16):


            # dinov2 特征: 只跑到所需层 (layer 8), 与全跑输出一致


            raw = F.interpolate((0.5 * x + 0.5) * 255.0, size=self.dinov2_size,


                                mode='bicubic').clamp(0., 255.)


            xn = (raw / 255.0 - self.mean) / self.std


            v = self.venc.prepare_tokens_with_masks(xn, None)


            zfeat = None


            for i, blk in enumerate(self.venc.blocks):


                v = blk(v)


                if i in self.layer_list:


                    zfeat = v[:, 1:].detach()


            # VAE encode


            lat = self.vae.encode(x).latent_dist.sample() * self.scale


            # 一步 denoise


            u = self.diT(torch.cat([lat, z], 1),


                         torch.ones(x.shape[0], device=x.device),


                         torch.zeros(x.shape[0], device=x.device),


                         [zfeat])


            z1 = z - u


            # decode


            sr = self.dec(z1 / self.scale).clamp(-1, 1)


        return sr








class Runner:


    """官方 harness 统一接口"""





    def __init__(self, model_dir: str):


        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        try:
            pkg = torch.load(os.path.join(model_dir, 'your_model.pt'),


                             map_location='cpu', weights_only=False)
        except TypeError:  # torch<2.0 无 weights_only 参数
            pkg = torch.load(os.path.join(model_dir, 'your_model.pt'), map_location='cpu')


        self.model = _FullChain(pkg).to(self.device).half().eval()


        for p in self.model.parameters():


            p.requires_grad_(False)





    def infer(self, image_tensor, prompt=""):


        """image_tensor: [1,3,H,W] float in [-1,1]; 返回同尺寸 [-1,1] (fp32)"""


        x = image_tensor.to(self.device, dtype=torch.float16)


        b, _, h, w = x.shape


        # 防御: pad 到 8 的倍数 (官方测速 512x512, 无需 pad; 其余尺寸安全处理)


        ph = (-h) % 8


        pw = (-w) % 8


        if ph or pw:


            x = F.pad(x, (0, pw, 0, ph))


        z = torch.randn(b, 4, (h + ph) // 8, (w + pw) // 8,


                        device=self.device, dtype=torch.float16)


        with torch.no_grad():


            sr = self.model(x, z)


        if ph or pw:


            sr = sr[:, :, :h, :w]


        return sr.float().clamp(-1.0, 1.0)


