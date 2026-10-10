# -*- coding: utf-8 -*-
"""q_fit_measure2.py — native 4K 指标(clipiqa/maniqa/niqe/lapvar) 统一测量, 供 Q 拟合.
musiq 单图 51s 太慢, 单独按需补测. 增量落盘.
"""
import json, os, sys, time
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
for _k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):
    os.environ.pop(_k, None)
import numpy as np, torch, pyiqa
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
VAL  = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
OUT  = Path(r'D:/tmp_timing/q_fit_metrics2.json')
SETS = [
    ('M24',        r'D:/models/final_compliant/output_dir'),
    ('probeA',     r'D:/models/probe_a/output_dir'),
    ('TinyV9b',    r'D:/models/pkg_TinyV9b/output_dir'),
    ('TinyV11b',   r'D:/models/pkg_TinyV11b/output_dir'),
    ('TinyV7gan',  r'D:/models/pkg_TinyV7gan/output_dir'),
    ('TinyV5d',    r'D:/models/pkg_TinyV5d/output_dir'),
    ('TinyV2g',    r'D:/models/pkg_TinyV2g/output_dir'),
    ('TinyV1',     r'D:/models/pkg_TinyV1/output_dir'),
    ('PixB',       r'D:/models/pkg_PixB/output_dir'),
    ('w2_m0s1_s090', r'D:/models/wave2_m0s1_s090/output_dir'),
    ('w2_m0s7_s097', r'D:/models/wave2_m0s7_s097/output_dir'),
    ('HYPIR',      r'D:/models/hypir/out_test100'),
]
N = 30
METRICS = ['clipiqa', 'maniqa', 'niqe']
dev = 'cuda'
iqa = {k: pyiqa.create_metric(k, device=dev) for k in METRICS}
print('[ok] metrics', flush=True)
res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:N]]
def lapvar(im):
    g = np.asarray(im.convert('L'), dtype=np.float64)
    lap = (-4*g[1:-1,1:-1] + g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:])
    return float(lap.var())
def run(tag, d, pat='case{n}.jpg'):
    vals = {k: [] for k in METRICS}; vals['lapvar'] = []
    t0 = time.time()
    for nm in names:
        idx = ''.join(c for c in nm if c.isdigit())
        p = Path(d) / pat.format(n=idx)
        if not p.exists(): continue
        im = Image.open(p).convert('RGB')
        t = torch.from_numpy(np.asarray(im)).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
        for k, m in iqa.items():
            with torch.no_grad(): vals[k].append(float(m(t).item()))
        vals['lapvar'].append(lapvar(im))
        del t
    out = {k: (round(float(np.mean(v)), 4) if v else None) for k, v in vals.items()}
    out['n'] = len(vals['lapvar'])
    res[tag] = out
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('[set] %-14s %s (%.0fs)' % (tag, json.dumps(out, ensure_ascii=False), time.time()-t0), flush=True)
for tag, d in SETS:
    if res.get(tag, {}).get('n', 0) >= N: print('[skip]', tag, flush=True); continue
    try: run(tag, d)
    except Exception as e: print('[err]', tag, repr(e)[:200], flush=True)
if 'LQ' not in res: run('LQ', str(TEST))
if 'GT_val' not in res: run('GT_val', str(VAL), '*_gt.*')
print('Q_FIT_MEASURE2_DONE', flush=True)
