# -*- coding: utf-8 -*-
"""合规自检: output_dir 的图(用 ckpt 渲染) 与 runner 的 your_model.pt 是不是同一主体/同一输出"""
import sys, torch, numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

sys.path.insert(0, r'D:/tmp_timing')
TILE = 512
ck_path = r'D:/models/tiny_v1/v1b_s40000.pt'
d = torch.load(ck_path, map_location='cpu', weights_only=False)
from diffusers import UNet2DModel
ch = tuple(d['ch']); res = int(d['res']); alpha = float(d['alpha'])
unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch, down_block_types=('DownBlock2D',)*len(ch),
                   up_block_types=('UpBlock2D',)*len(ch))
unet.load_state_dict(d['unet']); unet = unet.to('cuda').eval().half()
sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5

md = Path(r'D:/tmp_timing/_tinyv1_verify/model_dir')
sys.path.insert(0, str(md))
import runner as R
rr = R.Runner(str(md))

img = np.asarray(Image.open(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集/case35.jpg').convert('RGB'), dtype=np.uint8)[1000:1000+TILE, 1000:1000+TILE]
t = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].to('cuda').float().div_(127.5).sub_(1)
with torch.no_grad():
    th = t.half()
    pred = unet(th, torch.full((1,), 500, dtype=torch.long, device='cuda')).sample
    a = ((th - sq_1a * pred.float()) / sq_a).clamp(-1, 1)
    b = rr.infer(t, "")
diff = (a - b).abs()
print(f'ckpt 路径 vs packaged runner: max|Δ|={diff.max().item():.5f}  mean|Δ|={diff.mean().item():.6f}  (范围 [-1,1])')
print(f'  -> {"一致 (同一模型主体)" if diff.max().item() < 0.02 else "不一致, 需检查!"}')
print('SUBJECT_CHECK_DONE')
