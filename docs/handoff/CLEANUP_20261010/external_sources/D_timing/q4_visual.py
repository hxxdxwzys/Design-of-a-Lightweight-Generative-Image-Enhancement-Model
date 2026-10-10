# -*- coding: utf-8 -*-
"""生成对比图: LQ / v4g@128 / v11b@128 / v11b@160 / v11b@192 / v11b@256 / M24-4K-down512"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
M24D = Path(r'D:/models/final_compliant/output_dir')
dev='cuda'
CROP=512
nm = 'case12'
lq4 = Image.open(TEST/(nm+'.jpg')).convert('RGB')
m24 = Image.open(M24D/(nm+'.jpg')).convert('RGB')
W,H = lq4.size
# 中心裁剪 1024 再缩到 512 显示 (保留细节)
cx, cy = W//2, H//2
box = (cx-CROP//2, cy-CROP//2, cx+CROP//2, cy+CROP//2)
lqc = lq4.crop(box)
m24c = m24.crop(box)
def to_t(im):
    a = np.asarray(im).astype(np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0)
x = (to_t(lqc)*2-1).cuda().half()
panels = [('LQ', lqc), ('M24-4K', m24c)]
for tag, pkg, nats in (('v4g', r'D:/models/pkg_TinyV4g', [128]),
                       ('v11b', r'D:/models/pkg_TinyV11b', [128, 160, 192, 256]),
                       ('v7gan', r'D:/models/pkg_TinyV7gan', [160, 256])):
    m = torch.jit.load(str(Path(pkg)/'model_dir'/'your_model.pt'), map_location=dev).eval()
    for n in nats:
        with torch.no_grad():
            xs = F.interpolate(x, size=(n,n), mode='bicubic', align_corners=False)
            y = m(xs)
            if y.shape[-2:] != x.shape[-2:]:
                y = F.interpolate(y, size=x.shape[-2:], mode='bicubic', align_corners=False)
            y = y.clamp(-1,1)
        a = ((y[0].float().permute(1,2,0).cpu().numpy()+1)/2*255).clip(0,255).astype(np.uint8)
        panels.append(('%s@%d' % (tag, n), Image.fromarray(a)))
    del m; torch.cuda.empty_cache()
# 拼成 2 行 x 4 列
cols=4; rows=(len(panels)+cols-1)//cols
Wp=Hp=512
sheet = Image.new('RGB', (cols*Wp, rows*(Hp+22)), (255,255,255))
from PIL import ImageDraw
d = ImageDraw.Draw(sheet)
for i,(t,im) in enumerate(panels):
    r,c = divmod(i, cols)
    sheet.paste(im.resize((Wp,Hp)), (c*Wp, r*(Hp+22)+22))
    d.text((c*Wp+6, r*(Hp+22)+5), t, fill=(0,0,0))
out = Path(r'D:/tmp_timing/q4_visual.png')
sheet.save(out)
print('saved', out, sheet.size)
print('VISUAL_DONE')
