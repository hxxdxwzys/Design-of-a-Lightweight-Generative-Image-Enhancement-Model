# -*- coding: utf-8 -*-
"""Q1 渲染配置筛选 (2026-09-15) —— 见 Q_CAMPAIGN_20260915.md §1
每个配置渲染 N 张测试图(字典序前 N, 与 test30 同口径) -> native 4K clipiqa/lapvar/接缝比 -> JSON
本地代理只用于【排序】; 最优配置才全量渲染 100 张并提交(读官方 Q)。
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
OUT = T / 'q1_screen.json'
LOG = T / 'q1_screen.log'
BASE = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
V7 = r'D:/models/vosr_lora/v7/lora_s5000.pt'
WORK = Path(r'D:/models/q1_screen')
TILE = 512


def log(m):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + chr(10))


def run(name, cmd, timeout=7200):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
        rc, out = p.returncode, (p.stdout or '') + (p.stderr or '')
    except subprocess.TimeoutExpired as e:
        rc, out = 124, 'TIMEOUT %s' % e
    except Exception as e:
        rc, out = 125, 'EXC %s: %s' % (type(e).__name__, e)
    with LOG.open('a', encoding='utf-8') as f:
        f.write('--- %s rc=%d (%.1f min)' % (name, rc, (time.time() - t0) / 60) + chr(10) + out[-3000:] + chr(10))
    return rc, out


def seam_stats(d, n=6):
    cs, rs = [], []
    for f in sorted(d.glob('*.png'))[:n]:
        g = np.asarray(Image.open(f).convert('L'), np.float32)
        dx = np.abs(np.diff(g, axis=1)).mean(axis=0)
        dy = np.abs(np.diff(g, axis=0)).mean(axis=1)
        for arr, acc in ((dx, cs), (dy, rs)):
            for p in range(TILE, len(arr) - 30, TILE):
                b = np.concatenate([arr[p - 24:p - 5], arr[p + 6:p + 25]])
                if len(b) and b.mean() > 0:
                    acc.append(float(arr[p] / b.mean()))
    return (float(np.median(cs)) if cs else float('nan'), float(np.median(rs)) if rs else float('nan'))


CONFIGS = [
    ('base_sc0.8_t512_wavelet', dict(lora=V7, scale=0.8, tile=512, ov=32, align='wavelet', seed=42, vae=1024, vaeov=64)),
    ('sc1.0_t512_wavelet',      dict(lora=V7, scale=1.0, tile=512, ov=32, align='wavelet', seed=42, vae=1024, vaeov=64)),
    ('sc1.2_t512_wavelet',      dict(lora=V7, scale=1.2, tile=512, ov=32, align='wavelet', seed=42, vae=1024, vaeov=64)),
    ('sc0.8_t512_adain',        dict(lora=V7, scale=0.8, tile=512, ov=32, align='adain',   seed=42, vae=1024, vaeov=64)),
    ('sc0.8_t768_ov64',         dict(lora=V7, scale=0.8, tile=768, ov=64, align='wavelet', seed=42, vae=1024, vaeov=64)),
    ('sc0.8_t512_ov128',        dict(lora=V7, scale=0.8, tile=512, ov=128, align='wavelet', seed=42, vae=1024, vaeov=64)),
    ('sc0.8_t512_vae2048',      dict(lora=V7, scale=0.8, tile=512, ov=32, align='wavelet', seed=42, vae=2048, vaeov=128)),
    ('sc0.8_t512_vae4096',      dict(lora=V7, scale=0.8, tile=512, ov=32, align='wavelet', seed=42, vae=4096, vaeov=256)),
]

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=6)
ap.add_argument('--only', default='')
a = ap.parse_args()
only = set(x.strip() for x in a.only.split(',') if x.strip())

res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
WORK.mkdir(parents=True, exist_ok=True)
names = sorted(TEST.glob('*.jpg'))[:a.n]
inp = WORK / ('in%d' % a.n)
inp.mkdir(parents=True, exist_ok=True)
for f in names:
    dst = inp / f.name
    if not dst.exists():
        try:
            os.link(f, dst)
        except OSError:
            import shutil
            shutil.copy2(f, dst)
log('input dir %s (%d imgs)' % (inp, len(names)))

for tag, c in CONFIGS:
    if only and tag not in only:
        continue
    if tag in res and res[tag].get('clipiqa') is not None:
        log('[skip] %s' % tag)
        continue
    od = WORK / tag
    od.mkdir(parents=True, exist_ok=True)
    cmd = [PY, '-u', str(T / 'vosr_lora_infer.py'), '-c', BASE, '--lora', c['lora'],
           '--lora-scale', str(c['scale']), '-i', str(inp), '-o', str(od),
           '--seed', str(c['seed']), '--tile_size', str(c['tile']), '--tile_overlap', str(c['ov']),
           '--align_method', c['align'], '--infer_steps', '1', '--upscale', '1',
           '--vae_tile_size', str(c.get('vae', 1024)), '--vae_tile_overlap', str(c.get('vaeov', 64))]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    rc = -1
    n_out = len(list(od.glob('*.png'))) + len(list(od.glob('*.jpg')))
    for attempt in range(3):   # dinov2 hub 加载会间歇性退到联网 => 重试
        if n_out >= a.n:
            break
        _p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                            timeout=7200, env=env)
        rc = _p.returncode
        with LOG.open('a', encoding='utf-8') as f:
            f.write('--- render %s attempt%d rc=%d' % (tag, attempt + 1, rc) + chr(10) + ((_p.stdout or '') + (_p.stderr or ''))[-1500:] + chr(10))
        n_out = len(list(od.glob('*.png'))) + len(list(od.glob('*.jpg')))
        if n_out >= a.n:
            break
        log('  retry %s (%d/%d)' % (tag, n_out, a.n))
    if n_out == 0:
        log('!! %s 无出图 rc=%d' % (tag, rc))
        res[tag] = {'cfg': c, 'rc': rc, 'n': 0}
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        continue
    mtag = 'q1_' + tag.replace('.', 'p')
    run('measure ' + tag, [PY, '-u', str(T / 'measure_dir.py'), '--dir', str(od), '--n', str(a.n),
                           '--pattern', 'case{n}.png', '--metrics', 'clipiqa,lapvar', '--tag', mtag])
    mj = WORK / ('metrics_%s.json' % mtag)
    m = json.loads(mj.read_text(encoding='utf-8')) if mj.exists() else {}
    sc, sr = seam_stats(od, a.n)
    res[tag] = {'cfg': {k: v for k, v in c.items()}, 'n': n_out, 'rc': rc,
                'clipiqa': m.get('clipiqa'), 'lapvar': m.get('lapvar'),
                'seam_v': round(sc, 4), 'seam_h': round(sr, 4)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    log('%-26s clipiqa=%-8s lapvar=%-8s seam=%.3f/%.3f n=%d' % (tag, m.get('clipiqa'), m.get('lapvar'), sc, sr, n_out))

log('=== 汇总 ===')
for k, v in res.items():
    log('%-26s clipiqa=%-8s lapvar=%-8s seam=%.3f/%.3f' % (k, v.get('clipiqa'), v.get('lapvar'), v.get('seam_v', float('nan')), v.get('seam_h', float('nan'))))
log('Q1_SCREEN_DONE')