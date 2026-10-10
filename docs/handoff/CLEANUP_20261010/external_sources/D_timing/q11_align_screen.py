# -*- coding: utf-8 -*-
"""q11_align_screen.py — 先识别"已提交的 M24 100 张"到底用的哪个颜色对齐, 再比指标.
对齐方式会整体改变色彩/对比统计(实测 nofix 与已提交件差 7.75/255), 是比 tiling 更一阶的旋钮.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch, pyiqa
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PY = r'D:\Python\python.exe'; CLI = r'D:\tmp_timing\vosr_lora_infer2.py'
CKPT = 'C:/models/finals/preset/ckpts/VOSR_0.5B_os'; LORA = 'D:/models/vosr_lora/v7/lora_s5000.pt'
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
M24 = Path(r'D:/models/final_compliant/output_dir'); ROOT = Path(r'D:/models/probe_render')
CASES = ['case1', 'case7', 'case12', 'case30', 'case54', 'case77']
iq = {k: pyiqa.create_metric(k, device='cuda') for k in ('maniqa', 'clipiqa', 'niqe')}
def lapvar(g):
    lap = (-4*g[1:-1,1:-1] + g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:])
    return float(lap.var())
def seam(g, step=512):
    d = np.abs(np.diff(g, axis=1)); med = float(np.median(d))
    cols = [x for x in range(step, g.shape[1]-1, step)]
    return float(np.mean([float(d[:, x].mean()) for x in cols if x < d.shape[1]]) / med) if cols and med > 0 else None
def metrics(p):
    im = Image.open(p).convert('RGB'); g = np.asarray(im.convert('L'), dtype=np.float64)
    t = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).unsqueeze(0).cuda().float().div_(255)
    o = {}
    for k, m in iq.items():
        with torch.no_grad(): o[k] = float(m(t).item())
    o['lapvar'] = lapvar(g); o['seam'] = seam(g); del t
    return o
res = {}
print('%-9s | %-28s | %8s %8s %8s %7s %6s' % ('align', '与已提交M24件的平均差/255', 'maniqa','clipiqa','niqe','lapvar','seam'), flush=True)
for align in ('nofix', 'wavelet', 'adain'):
    tag = 'align_' + align
    od = ROOT / tag; od.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TORCHDYNAMO_DISABLE='1', TORCHDYNAMO_SUPPRESS_ERRORS='1',
               TORCH_HOME=r'C:/models/finals/preset/ckpts/torch_cache', HF_HUB_OFFLINE='1')
    diffs, acc = [], {}
    for c in CASES:
        png = od / (c + '.png')
        if not png.exists():
            cmd = [PY, CLI, '-c', CKPT, '--lora', LORA, '--lora-scale', '0.8', '-i', str(TEST/(c+'.jpg')),
                   '-o', str(od), '-u', '1', '--align_method', align, '--seed', '8']
            subprocess.run(cmd, cwd=r'C:\models\finals', env=env, capture_output=True, text=True, errors='replace')
        if not png.exists(): continue
        a = np.asarray(Image.open(png).convert('RGB')).astype(np.float32)
        b = np.asarray(Image.open(M24/(c+'.jpg')).convert('RGB')).astype(np.float32)
        if a.shape == b.shape: diffs.append(float(np.abs(a-b).mean()))
        for k, v in metrics(png).items(): acc.setdefault(k, []).append(v if v is not None else np.nan)
    row = {k: round(float(np.nanmean(v)), 4) for k, v in acc.items()}
    row['diff_vs_submitted'] = round(float(np.mean(diffs)), 3)
    res[tag] = row
    print('%-9s | %-28.2f | %8.4f %8.4f %8.4f %7.1f %6.3f' % (
        align, row['diff_vs_submitted'], row['maniqa'], row['clipiqa'], row['niqe'], row['lapvar'], row['seam']), flush=True)
Path(r'D:/tmp_timing/q11_align_screen.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
print('Q11_DONE')
