# -*- coding: utf-8 -*-
"""波 2 渲染队列 v2 (2026-09-11 夜, 优先级排序 + 自愈重试)

排序原则: 优先保证"不可替代"的靶点先落地 ——
  1. m0s7_s097  T 高分支最优 (val lpips 0.3542, 全场最低)
  2. m0s1_s097  T 低分支唯一可产 (r16 锁在不可达的服务器上)
  3. m0s1_s090  T 高分支次选
  4. m0s1_s100  T 低分支新增点 (val clip 0.9394 为可产资产中最高, 本轮首次可产)

每个候选做四件事: 反推 pkg -> val 3 张门控(独立确认) -> 测试集 100 张 4K 渲染+jpg
                 -> 测试集 100 张 clipiqa 直证 (§16.82 指出 val->test 跌幅达 0.044-0.064,
                    必须用真实测试集数字做最终裁决)
失败自愈: 每候选最多 3 轮, 未满 100 张就重跑 (断点续跑, 已渲的图不重复)。

用法: D:/Python/python.exe wave2_queue.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

PY = r'D:/Python/python.exe'
TOOLS = Path(r'D:/tmp_timing')
BASE_C = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
VAL_LQ = r'D:/tmp_timing/p100_val/lq'
VAL_GT = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
ROOT_BASE = Path(r'D:/models')
OUT = Path(r'D:/tmp_timing/wave2_overnight.json')
LOG = Path(r'D:/tmp_timing/wave2_queue.log')

SRC_M0S1 = r'D:/models/final_compliant_14b/model_dir/your_model.pt'
SRC_M0S7 = r'D:/models/final_compliant_m0s7/model_dir/your_model.pt'

# (tag, 源 pkg, 目标 scale, 预登记 val (clip, lpips), 分支, 说明)
CANDIDATES = [
    ('m0s7_s097', SRC_M0S7, 0.97, (0.9203, 0.3542), 'T_high', 'T高最优 lpips'),
    ('m0s1_s097', SRC_M0S1, 0.97, (0.9311, 0.3595), 'T_low', 'T低唯一可产'),
    ('m0s1_s090', SRC_M0S1, 0.90, (0.9209, 0.3556), 'T_high', 'T高次选'),
    ('m0s1_s100', SRC_M0S1, 1.00, (0.9394, 0.3717), 'T_low', '最高 clip 可产档'),
]


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def sh(cmd, timeout):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, ((r.stdout or '') + (r.stderr or '')).strip()


def main():
    res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    log('=' * 72)
    log(f'波 2 队列 v2 启动: {len(CANDIDATES)} 个候选 (优先级: 不可替代者先渲)')

    for tag, src, scale, val_expect, branch, note in CANDIDATES:
        ROOT = ROOT_BASE / f'wave2_{tag}'
        PKG = ROOT / 'model_dir' / 'your_model.pt'
        OUT_DIR = ROOT / 'output_dir'
        log('=' * 72)
        log(f'候选 {tag} [{branch}] {note}  预登记 val={val_expect}')
        entry = {'branch': branch, 'note': note, 'scale': scale,
                 'src_pkg': Path(src).parent.parent.name, 'val_expect': list(val_expect)}
        t0 = time.time()

        # ① 反推 pkg
        if not PKG.exists():
            rc, o = sh([PY, TOOLS / 'remerge_scale_14b.py', '--src-pkg', src,
                        '--scale', scale, '--out', PKG], 1800)
            log(f'[1/4] remerge rc={rc} :: {o.splitlines()[-1] if o else ""}')
            if rc != 0:
                entry['error'] = 'remerge failed'
                res[tag] = entry
                OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
                continue
        else:
            log('[1/4] remerge 已有 pkg, 复用')

        # ② val 3 张门控 (独立确认, 尤其 m0s1_s100 是首次外推)
        val_dir = ROOT / 'val_render'
        gate = ROOT / 'val_gate.json'
        if not gate.exists():
            rc, o = sh([PY, TOOLS / 'vosr_pkg_infer_14b.py', '-c', BASE_C, '--pkg', PKG,
                        '--fp16', '-i', VAL_LQ, '-o', val_dir, '--align_method', 'wavelet'], 3600)
            if rc == 0:
                sh([PY, TOOLS / 'val_gate.py', '--src', val_dir, '--val-dir', VAL_GT,
                    '--clip-min', 0.919, '--lpips-max', 0.375, '--out', gate], 3600)
            log(f'[2/4] val 渲染 rc={rc}')
        if gate.exists():
            g = json.loads(gate.read_text(encoding='utf-8'))
            entry['val'] = {'clip': g['clip'], 'lpips': g['lpips'],
                            'd_clip': round(g['clip'] - val_expect[0], 4),
                            'd_lpips': round(g['lpips'] - val_expect[1], 4)}
            log(f'[2/4] val 实测={entry["val"]}')

        # ③ 测试集 100 张 (最多 3 轮自愈)
        for attempt in range(3):
            n = len(list(OUT_DIR.glob('*.jpg'))) if OUT_DIR.exists() else 0
            if n >= 100:
                break
            log(f'[3/4] 测试集渲染 attempt {attempt+1}, 现有 {n}/100')
            rc, o = sh([PY, TOOLS / 'render_pkg_test100.py', '--src-pkg', src,
                        '--scale', scale, '--tag', tag], 12 * 3600)
            log(f'[3/4] rc={rc} :: {o.splitlines()[-1] if o else ""}')
            n2 = len(list(OUT_DIR.glob('*.jpg'))) if OUT_DIR.exists() else 0
            if n2 >= 100:
                break
            log(f'[3/4] 仍缺 {100-n2} 张, 重试')
            time.sleep(30)
        n = len(list(OUT_DIR.glob('*.jpg'))) if OUT_DIR.exists() else 0
        entry['n_jpg'] = n
        log(f'[3/4] 测试集完成 {n}/100, 耗时 {(time.time()-t0)/60:.1f} min')

        # ④ 测试集 clipiqa 直证
        if n == 100:
            cj = Path(rf'D:/tmp_timing/wave2_clip_{tag}.json')
            rc, o = sh([PY, TOOLS / 'nr_clipiqa_dir.py', '--dir', OUT_DIR, '--out', cj], 6 * 3600)
            log(f'[4/4] clipiqa rc={rc} :: {o.splitlines()[-1] if o else ""}')
            if cj.exists():
                c = json.loads(cj.read_text(encoding='utf-8'))
                entry['test_clip'] = {'n': c.get('n'), 'mean': c.get('mean'),
                                      'std': c.get('std'), 'min': c.get('min'), 'max': c.get('max')}
        entry['elapsed_min'] = round((time.time() - t0) / 60, 1)
        res[tag] = entry
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        log(f'[记录] {json.dumps(entry, ensure_ascii=False)}')
        # 该候选已收官则删掉临时 pkg 省 3.5GB (需要时 7 秒可重生)
        if n == 100 and PKG.exists():
            PKG.unlink()
            log('[cleanup] 已删临时 pkg')

    log('=' * 72)
    log('最终汇总:')
    for tag, e in res.items():
        tc = e.get('test_clip')
        v = e.get('val')
        log(f'  {tag:12s} [{e.get("branch")}] n={e.get("n_jpg")} '
            f'val={None if not v else (v["clip"], v["lpips"])} '
            f'test_clip={None if not tc else tc["mean"]}')
    log('WAVE2_QUEUE_DONE')


if __name__ == '__main__':
    main()
