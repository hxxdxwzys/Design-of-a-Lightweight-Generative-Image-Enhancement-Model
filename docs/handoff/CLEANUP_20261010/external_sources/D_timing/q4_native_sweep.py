# -*- coding: utf-8 -*-
"""q4_native_sweep.py — 同一个 JIT 权重, 扫 _NATIVE(内部工作分辨率): 时延 vs 输出真实性.
产出: 每个 (pkg, NATIVE) 的 infer 时延 + 512 输出的 clipiqa/lapvar/与 LQ 的差异.
"""
import os, sys, time, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, pyiqa
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
PKGS = [r'D:/models/pkg_TinyV7gan', r'D:/models/pkg_TinyV5d', r'D:/models/pkg_TinyV11b']
NATIVES = [128, 160, 192, 224, 256]
NQ = 8

dev = 'cuda'
cq = pyiqa.create_metric('clipiqa', device=dev)
names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:NQ]]
lq = []
for nm in names:
    im = Image.open(TEST / (nm + '.jpg')).convert('RGB').resize((512, 512), Image.BICUBIC)
    a = np.asarray(im).astype(np.float32) / 255.0
    lq.append(torch.from_numpy(a).permute(2, 0, 1))
lq_t = torch.stack(lq)                      # [NQ,3,512,512] 0..1
lq_g = (lq_t * 2 - 1).cuda().half()         # [-1,1] fp16
def lapvar(x):                              # x: [N,3,H,W] in [-1,1]
    g = (x.float().mean(1) * 255.0)
    lap = (-4*g[:,1:-1,1:-1] + g[:,:-2,1:-1] + g[:,2:,1:-1] + g[:,1:-1,:-2] + g[:,1:-1,2:])
    return float(lap.var().item())
with torch.no_grad():
    lq_cq = float(cq(lq_t.cuda()).mean().item())
lq_lv = lapvar(lq_g)
print('LQ 参考: clipiqa512=%.4f lapvar=%.1f' % (lq_cq, lq_lv), flush=True)

out_all = {}
for pkg in PKGS:
    tag = Path(pkg).name
    try:
        m = torch.jit.load(str(Path(pkg) / 'model_dir' / 'your_model.pt'), map_location=dev).eval()
    except Exception as e:
        print('[load fail]', tag, repr(e)[:200]); continue
    out_all[tag] = {}
    for n in NATIVES:
        try:
            def fwd(x):
                xs = F.interpolate(x, size=(n, n), mode='bicubic', align_corners=False)
                y = m(xs)
                if y.shape[-2:] != x.shape[-2:]:
                    y = F.interpolate(y, size=x.shape[-2:], mode='bicubic', align_corners=False)
                return y.clamp(-1, 1)
            static = torch.zeros(1, 3, 512, 512, device=dev, dtype=torch.float16)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): fwd(static)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): out = fwd(static)
            def one():
                static.copy_(lq_g[:1]); g.replay()
            for _ in range(30): one()
            torch.cuda.synchronize()
            ts = []
            for _ in range(200):
                t0 = time.perf_counter(); one(); torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
            med = float(np.median(ts))
            with torch.no_grad():
                static.copy_(lq_g[0]); g.replay()
                o = out.clone()
            with torch.no_grad():
                cqv = float(cq(((o.float()+1)/2).clamp(0,1)).item())
            lv = lapvar(o)
            diff = float((o.float()-lq_g[0].float()).abs().mean().item()*255/2)
            out_all[tag][n] = dict(ms=round(med,3), clipiqa=round(cqv,4), lapvar=round(lv,1), diff255=round(diff,2))
            r = 285.6/(med+2.515)
            print('%-16s NATIVE=%3d  ms=%6.3f  r=%6.1f  clipiqa=%0.4f lapvar=%7.1f  |out-LQ|=%5.2f/255' % (
                tag, n, med, r, cqv, lv, diff), flush=True)
            del g, out
            torch.cuda.empty_cache()
        except Exception as e:
            print('%-16s NATIVE=%3d FAIL %r' % (tag, n, repr(e)[:160]), flush=True)
    del m; torch.cuda.empty_cache()
Path(r'D:/tmp_timing/q4_native_sweep.json').write_text(json.dumps(out_all, indent=1), encoding='utf-8')
print('Q4_SWEEP_DONE')
