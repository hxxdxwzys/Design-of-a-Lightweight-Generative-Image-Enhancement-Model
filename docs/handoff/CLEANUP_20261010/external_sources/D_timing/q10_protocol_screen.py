# -*- coding: utf-8 -*-
"""q10_protocol_screen.py — 渲染协议筛选(新模型下的第一次): 同 seed 同图, 只改分块/步数.
对每个配置渲 6 张 4K 图, 测 native-4K 的 maniqa/clipiqa/niqe + lapvar + 竖缝比.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PY = r'D:\Python\python.exe'
CLI = r'D:\tmp_timing\vosr_lora_infer2.py'
CKPT = 'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
LORA = 'D:/models/vosr_lora/v7/lora_s5000.pt'
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
M24 = Path(r'D:/models/final_compliant/output_dir')
ROOT = Path(r'D:/models/probe_render')
CASES = ['case1', 'case7', 'case12', 'case30', 'case54', 'case77']

CONFIGS = [
    ('A_base',      []),
    ('B_vaefull',   ['--vae_tile_size', '4096', '--vae_tile_overlap', '0']),
    ('C_vae2048',   ['--vae_tile_size', '2048', '--vae_tile_overlap', '128']),
    ('D_dit768',    ['--tile_size', '768', '--tile_overlap', '96']),
    ('E_dit1024',   ['--tile_size', '1024', '--tile_overlap', '128']),
    ('F_step2',     ['--infer_steps', '2']),
]

OUTJ = Path(r'D:/tmp_timing/q10_protocol_screen.json')
res = json.loads(OUTJ.read_text(encoding='utf-8')) if OUTJ.exists() else {}

def render(tag, extra, case):
    od = ROOT / tag
    od.mkdir(parents=True, exist_ok=True)
    png = od / (case + '.png')
    if png.exists():
        return True
    env = dict(os.environ, TORCHDYNAMO_DISABLE='1', TORCHDYNAMO_SUPPRESS_ERRORS='1',
               TORCH_HOME=r'C:/models/finals/preset/ckpts/torch_cache', HF_HUB_OFFLINE='1')
    cmd = [PY, CLI, '-c', CKPT, '--lora', LORA, '--lora-scale', '0.8',
           '-i', str(TEST / (case + '.jpg')), '-o', str(od), '-u', '1',
           '--align_method', 'nofix', '--seed', '8'] + extra
    p = subprocess.run(cmd, cwd=r'C:\models\finals', env=env, capture_output=True, text=True, errors='replace')
    return png.exists()

def lapvar(g):
    lap = (-4*g[1:-1,1:-1] + g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:])
    return float(lap.var())

def seam(g, step=512):
    """竖缝比: 分块边界处 |dx| 的均值 / 全图 |dx| 中位数."""
    d = np.abs(np.diff(g, axis=1))
    med = float(np.median(d))
    cols = [x for x in range(step, g.shape[1] - 1, step)]
    if not cols or med <= 0:
        return None
    vals = [float(d[:, x].mean()) for x in cols if x < d.shape[1]]
    return float(np.mean(vals) / med) if vals else None

# 先测参照: 现有 M24 出图
import torch, pyiqa
iq = {k: pyiqa.create_metric(k, device='cuda') for k in ('maniqa', 'clipiqa', 'niqe')}
def metrics(path):
    im = Image.open(path).convert('RGB')
    g = np.asarray(im.convert('L'), dtype=np.float64)
    t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).unsqueeze(0).cuda().float().div_(255)
    out = {}
    for k, m in iq.items():
        with torch.no_grad():
            out[k] = float(m(t).item())
    out['lapvar'] = lapvar(g); out['seam'] = seam(g)
    del t
    return out

for tag, extra in CONFIGS:
    if res.get(tag, {}).get('done'):
        print('[skip]', tag, flush=True); continue
    t0 = time.time()
    ok = all(render(tag, extra, c) for c in CASES)
    if not ok:
        print('[fail]', tag, flush=True); res[tag] = {'done': False}; OUTJ.write_text(json.dumps(res, indent=1), encoding='utf-8'); continue
    acc = {}
    for c in CASES:
        m = metrics(ROOT / tag / (c + '.png'))
        for k, v in m.items():
            acc.setdefault(k, []).append(v if v is not None else np.nan)
    row = {k: round(float(np.nanmean(v)), 4) for k, v in acc.items()}
    row['done'] = True
    res[tag] = row
    OUTJ.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('[cfg] %-11s maniqa=%.4f clipiqa=%.4f niqe=%.4f lapvar=%7.1f seam=%s  (%.0fs)' % (
        tag, row['maniqa'], row['clipiqa'], row['niqe'], row['lapvar'],
        ('%.3f' % row['seam']) if row.get('seam') else 'n/a', time.time() - t0), flush=True)

# 参照行: 现有 M24
if 'REF_M24' not in res:
    acc = {}
    for c in CASES:
        p = M24 / (c + '.jpg')
        if not p.exists(): continue
        m = metrics(p)
        for k, v in m.items():
            acc.setdefault(k, []).append(v if v is not None else np.nan)
    row = {k: round(float(np.nanmean(v)), 4) for k, v in acc.items()}
    res['REF_M24'] = row
    OUTJ.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('[REF] M24(jpg) maniqa=%.4f clipiqa=%.4f niqe=%.4f lapvar=%.1f seam=%.3f' % (
        row['maniqa'], row['clipiqa'], row['niqe'], row['lapvar'], row['seam']), flush=True)
print('Q10_DONE')
