# -*- coding: utf-8 -*-
"""diag_panel.py v2 (2026-09-15) — 官方无参四项 × 多口径 面板

动机: ① 官方无参四项里 MUSIQ / MANIQA 我们从未测过;
     ② 时延实测显示这些指标对【输入口径】极度敏感 (M24 case1: clipiqa 4K 0.879 vs ms512 0.427;
        musiq 4K 42.7 vs ms512 61.3), 而官方 IQAEvaluator 的口径未知
        (MUSIQ 原生 4K 要 109 s/张 => 官方几乎不可能在原生 4K 上算 100 张)。
     ③ 分数模型无法解释 84~217; 唯一合规查法是测量。
做法: 对【已拿到官方分的 15 个图像集】+若干未评分集, 在同一进程内扫 (指标 × 口径),
      再与官方分/加速比做相关与二元回归 => 找出真正驱动官方分的那个组合。
纪律: 数值全部由本脚本从落盘图/zip 原图直算, 不改口径混用; 只测量不结论。
用法: python diag_panel.py --n 10 --modes ms512,ms1024,c1024,native [--only tag,tag]
输出: D:/tmp_timing/diag_panel.json (逐集增量, 可断点续跑)
"""
import argparse, io, json, os, sys, time, zipfile
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
for _k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(_k, None)
import numpy as np
import torch
import pyiqa
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path('C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
HIST = Path('C:/生成式图像增强轻量模型设计挑战/提交历史')
M = Path('D:/models')
OUTJ = Path('D:/tmp_timing/diag_panel.json')

SOURCES = [
    ('LQ',           'dir', str(TEST),                               None,     None,  '原图基线'),
    ('M24',          'dir', str(M / 'final_compliant/output_dir'),   43.6951,  0.78,  '我方最佳'),
    ('probeA_sc04',  'dir', str(M / 'probe_a/pred_sc0.4_jpg95'),     43.2837,  0.78,  'LoRA sc0.4'),
    ('TinyV7gan',    'dir', str(M / 'pkg_TinyV7gan/output_dir'),     41.7346,  61.9,  '256域+GAN 0.87M'),
    ('TinyV9b',      'dir', str(M / 'pkg_TinyV9b/output_dir'),       None,     12.4,  '512域 0.38M 待评测'),
    ('TinyV5d',      'dir', str(M / 'pkg_TinyV5d/output_dir'),       33.0382,  61.4,  '无GAN蒸馏'),
    ('TinyV2g',      'dir', str(M / 'pkg_TinyV2g/output_dir'),       33.2872,  53.9,  'GT目标 纯速探针'),
    ('TinyV2',       'dir', str(M / 'pkg_TinyV2/output_dir'),        29.4225,  19.5,  '同图慢版'),
    ('TinyV1',       'dir', str(M / 'pkg_TinyV1/output_dir'),        27.3521,  6.1,   '512域无GAN'),
    ('PixB',         'dir', str(M / 'pkg_PixB/output_dir'),          18.9710,  1.5,   '外来源'),
    ('TinyV4g',      'dir', str(M / 'pkg_TinyV4g/output_dir'),       None,     111.6, '128域 未交(图太弱)'),
    ('w2_m0s1_s090', 'dir', str(M / 'wave2_m0s1_s090/output_dir'),   34.0989,  None,  '1.4B sc0.90'),
    ('w2_m0s7_s097', 'dir', str(M / 'wave2_m0s7_s097/output_dir'),   31.3942,  None,  '1.4B m0s7 sc0.97'),
    ('w2_m0s1_s097', 'dir', str(M / 'wave2_m0s1_s097/output_dir'),   None,     None,  '1.4B sc0.97 未交'),
    ('w2_m0s1_s100', 'dir', str(M / 'wave2_m0s1_s100/output_dir'),   None,     None,  '1.4B sc1.00 未交'),
    ('ProbeDetail',  'zip', str(HIST / 'Track2_ProbeDetail——43.0699.zip'), 43.0699, 0.78, 'lapvar x2.27'),
    ('ProbeBlend',   'zip', str(HIST / 'Track2_ProbeBlend——40.5354.zip'),  40.5354, 0.78, '向 LQ 回退'),
    ('M24sc00',      'zip', str(HIST / 'Track2_M24sc00——37.6316.zip'),     37.6316, 0.78, '去 LoRA 发糊'),
    ('M24avg2',      'zip', str(HIST / 'Track2_M24avg2——36.6414.zip'),     36.6414, 0.382, '2 样本平均'),
    ('M24p4',        'zip', str(HIST / 'Track2_M24p4——7.1829.zip'),        7.1829,  0.975, '剪枝毁图'),
    ('ProbeSlow8',   'zip', str(HIST / 'Track2_ProbeSlow8——28.7538.zip'),  28.7538, 0.098, '图同M24 8x慢'),
    ('HYPIR',        'dir', str(M / 'hypir/out_test100'),            None,     None,  '账号B 阶梯 37.20~51.89 同组图不同速(旧世界标注入, 仅参考)'),
    ('OSEDiff',      'dir', 'D:/tmp/osediff_test100_out',            None,     None,  '账号A 10.1196 1x 基线(旧世界, 仅参考)'),
    ('v7g_traj8k',   'dir', 'D:/models/traj/v7g8k',                  None,     None,  'v7gan 轨迹 8k(未评分)'),
    ('v7g_traj16k',  'dir', 'D:/models/traj/v7g16k',                 None,     None,  'v7gan 轨迹 16k(未评分)'),
]


def resize_maxside(im, ms):
    w, h = im.size
    s = float(ms) / max(w, h)
    if s >= 1.0:
        return im
    return im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BICUBIC)


def center_crop(im, size):
    w, h = im.size
    if w <= size or h <= size:
        return im
    x, y = (w - size) // 2, (h - size) // 2
    return im.crop((x, y, x + size, y + size))


MODES = {
    'native': lambda im: im,
    'ms1024': lambda im: resize_maxside(im, 1024),
    'ms512':  lambda im: resize_maxside(im, 512),
    'c1024':  lambda im: center_crop(im, 1024),
}

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=10)
ap.add_argument('--modes', default='ms512,ms1024,c1024,native')
ap.add_argument('--metrics', default='clipiqa,musiq,maniqa,niqe')
ap.add_argument('--native-skip', default='musiq')
ap.add_argument('--only', default='')
a = ap.parse_args()
want = [x.strip() for x in a.metrics.split(',') if x.strip()]
modes = [x.strip() for x in a.modes.split(',') if x.strip()]
native_skip = set(x.strip() for x in a.native_skip.split(',') if x.strip())
only = set(x.strip() for x in a.only.split(',') if x.strip())

names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:a.n]]
idx = [''.join(c for c in s if c.isdigit()) for s in names]
dev = 'cuda'
iqa = {k: pyiqa.create_metric(k, device=dev) for k in want}
print('[panel] n=%d modes=%s metrics=%s (native 跳过 %s)' % (len(idx), modes, want, sorted(native_skip)), flush=True)

res = {}
if OUTJ.exists():
    try:
        res = json.loads(OUTJ.read_text(encoding='utf-8'))
    except Exception:
        res = {}
_MODES = ('native', 'ms512', 'ms1024', 'c1024')
if res and not all(isinstance(v, dict) and any(k in v for k in _MODES) for v in res.values()):
    print('[panel] !! 旧版(单口径)结果, 重置', flush=True)
    res = {}


def score_one(t, m):
    with torch.no_grad():
        return float(m(t).item())


def lapvar_of(im):
    g = np.asarray(im.convert('L'), dtype=np.float64)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


t_all = time.time()
for tag, kind, src, score, ratio, note in SOURCES:
    if only and tag not in only:
        continue
    ent = res.get(tag) or {}
    todo = [md for md in modes if md not in ent]
    if not todo:
        print('[skip] %s' % tag, flush=True)
        continue
    zf = zipfile.ZipFile(src) if kind == 'zip' else None
    acc = {md: {k: [] for k in want + ['lapvar']} for md in todo}
    t0 = time.time()
    try:
        for i in idx:
            im = None
            if kind == 'dir':
                for ext in ('.jpg', '.png'):
                    p = Path(src) / ('case%s%s' % (i, ext))
                    if p.exists():
                        im = Image.open(p).convert('RGB')
                        break
            else:
                for nm in ('output_dir/case%s.jpg', 'output_dir/case%s.png'):
                    try:
                        with zf.open(nm % i) as fh:
                            im = Image.open(io.BytesIO(fh.read())).convert('RGB')
                        break
                    except KeyError:
                        continue
            if im is None:
                continue
            for md in todo:
                v = MODES[md](im)
                arr = np.asarray(v).copy()
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
                for k, mm in iqa.items():
                    if md == 'native' and k in native_skip:
                        continue
                    try:
                        acc[md][k].append(score_one(t, mm))
                    except Exception as e:
                        print('  !! %s %s %s: %s' % (tag, md, k, str(e)[:120]), flush=True)
                acc[md]['lapvar'].append(lapvar_of(v))
                del t, arr
                if v is not im:
                    del v
            del im
            if 'native' in todo:
                torch.cuda.empty_cache()
    finally:
        if zf is not None:
            zf.close()
    for md in todo:
        d = {k: round(float(np.mean(v)), 4) for k, v in acc[md].items() if v}
        d['n'] = len(acc[md]['lapvar'])
        ent[md] = d
    ent['score'] = score
    ent['ratio'] = ratio
    ent['note'] = note
    res[tag] = ent
    OUTJ.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    msg = ' | '.join('%s: %s' % (md, ' '.join('%s=%.4f' % (k, ent[md][k]) for k in want if k in ent[md])) for md in todo)
    print('[%4.1f min] %-14s %s' % ((time.time() - t0) / 60, tag, msg), flush=True)

# ---------------- 汇总与相关分析 ----------------
print('\n=== 各口径 · 指标 vs 官方分 ===', flush=True)
rows = [(tg, v) for tg, v in res.items() if v.get('score') is not None]
rows.sort(key=lambda x: -x[1]['score'])
for md in modes:
    sub = [(tg, v) for tg, v in rows if md in v and v[md].get('n')]
    if len(sub) < 4:
        continue
    print('\n--- 口径 %s (n_sets=%d) ---' % (md, len(sub)), flush=True)
    y = np.array([v['score'] for _, v in sub])
    lg = np.array([np.log2(v['ratio']) if v.get('ratio') else np.nan for _, v in sub])
    print('%-16s %8s %7s | %s' % ('tag', 'score', 'ratio', ' '.join('%9s' % k for k in want + ['lapvar'])), flush=True)
    for tg, v in sub:
        print('%-16s %8.4f %7s | %s' % (tg, v['score'], v.get('ratio'),
              ' '.join('%9.4f' % v[md].get(k, float('nan')) for k in want + ['lapvar'])), flush=True)
    print('  单指标 Pearson r: ' + '  '.join('%s=%+.3f' % (k, float(np.corrcoef([v[md][k] for _, v in sub], y)[0, 1]))
                                            for k in want + ['lapvar'] if all(k in v[md] for _, v in sub)), flush=True)
    ok = [(tg, v) for tg, v in sub if np.isfinite(np.log2(v['ratio']) if v.get('ratio') else np.nan)]
    if len(ok) >= 5:
        yy = np.array([v['score'] for _, v in ok])
        lg2 = np.array([np.log2(v['ratio']) for _, v in ok])
        for k in want + ['lapvar']:
            if not all(k in v[md] for _, v in ok):
                continue
            x = np.array([v[md][k] for _, v in ok])
            A = np.stack([x, lg2, np.ones_like(x)], 1)
            c, *_ = np.linalg.lstsq(A, yy, rcond=None)
            pred = A @ c
            r2 = 1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            print('  二元回归 %-8s a=%+9.3f b=%+7.3f c=%+8.2f R2=%.3f rmse=%.2f' % (k, c[0], c[1], c[2], r2,
                  float(np.sqrt(((yy - pred) ** 2).mean()))), flush=True)

print('\n[panel] 总耗时 %.1f min' % ((time.time() - t_all) / 60), flush=True)
print('DIAG_PANEL_DONE', flush=True)
