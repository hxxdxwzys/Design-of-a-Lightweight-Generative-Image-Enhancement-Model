# -*- coding: utf-8 -*-
"""q_fit_measure.py — 为 Q=f(指标) 拟合统一测量: 各已知 Q 图像集的 native 4K 指标.
口径: 测试集 sorted()[:N] 前缀 (与 test30 历史口径一致). 增量落盘, 可中断续跑.
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
OUT  = Path(r'D:/tmp_timing/q_fit_metrics.json')

SETS = [
    ('LQ',        str(TEST),                              'case{n}.jpg'),
    ('M24',       r'D:/models/final_compliant/output_dir', 'case{n}.jpg'),
    ('probeA',    r'D:/models/probe_a/output_dir',         'case{n}.jpg'),
    ('TinyV9b',   r'D:/models/pkg_TinyV9b/output_dir',     'case{n}.jpg'),
    ('TinyV7gan', r'D:/models/pkg_TinyV7gan/output_dir',   'case{n}.jpg'),
    ('TinyV5d',   r'D:/models/pkg_TinyV5d/output_dir',     'case{n}.jpg'),
    ('TinyV2g',   r'D:/models/pkg_TinyV2g/output_dir',     'case{n}.jpg'),
    ('TinyV2',    r'D:/models/pkg_TinyV2/output_dir',      'case{n}.jpg'),
    ('TinyV1',    r'D:/models/pkg_TinyV1/output_dir',      'case{n}.jpg'),
    ('PixB',      r'D:/models/pkg_PixB/output_dir',        'case{n}.jpg'),
    ('HYPIR',     r'D:/models/hypir/out_test100',          'case{n}.jpg'),
    ('w2_m0s1_s090', r'D:/models/wave2_m0s1_s090/output_dir', 'case{n}.jpg'),
    ('w2_m0s7_s097', r'D:/models/wave2_m0s7_s097/output_dir', 'case{n}.jpg'),
    ('M24img_v9brun', r'D:/models/pkg_M24img_V9brun/output_dir', 'case{n}.jpg'),
]
N = 30
METRICS = ['clipiqa', 'musiq', 'maniqa', 'niqe', 'lapvar']

dev = 'cuda'
iqa = {}
for k in METRICS:
    if k == 'lapvar':
        continue
    try:
        iqa[k] = pyiqa.create_metric(k, device=dev)
        print('[ok] metric', k, flush=True)
    except Exception as e:
        print('[FAIL] metric', k, repr(e)[:200], flush=True)

res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:N]]
print('names:', len(names), flush=True)

def lapvar(im):
    g = np.asarray(im.convert('L'), dtype=np.float64)
    lap = (-4*g[1:-1,1:-1] + g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:])
    return float(lap.var())

for tag, d, pat in SETS:
    if tag in res and res[tag].get('n', 0) >= N:
        print('[skip]', tag, flush=True)
        continue
    vals = {k: [] for k in METRICS}
    t0 = time.time()
    got = 0
    for nm in names:
        idx = ''.join(ch for ch in nm if ch.isdigit())
        p = Path(d) / pat.format(n=idx)
        if not p.exists():
            continue
        try:
            im = Image.open(p).convert('RGB')
            t = torch.from_numpy(np.asarray(im)).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
            for k, m in iqa.items():
                with torch.no_grad():
                    vals[k].append(float(m(t).item()))
            vals['lapvar'].append(lapvar(im))
            got += 1
            del t
        except Exception as e:
            print('[err]', tag, nm, repr(e)[:120], flush=True)
    out = {k: (round(float(np.mean(v)), 4) if v else None) for k, v in vals.items()}
    out['n'] = got
    res[tag] = out
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('[set] %-16s %s  (%.1fs, n=%d)' % (tag, json.dumps(out, ensure_ascii=False), time.time()-t0, got), flush=True)

# GT (val, 3 张, native)
if 'GT_val' not in res:
    gvals = {k: [] for k in METRICS}
    for f in sorted(VAL.glob('*_gt.*')):
        im = Image.open(f).convert('RGB')
        t = torch.from_numpy(np.asarray(im)).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
        for k, m in iqa.items():
            with torch.no_grad():
                gvals[k].append(float(m(t).item()))
        gvals['lapvar'].append(lapvar(im))
        del t
    out = {k: (round(float(np.mean(v)), 4) if v else None) for k, v in gvals.items()}
    out['n'] = len(gvals['lapvar'])
    res['GT_val'] = out
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('[set] GT_val', json.dumps(out, ensure_ascii=False), flush=True)

print('Q_FIT_MEASURE_DONE', flush=True)
