# -*- coding: utf-8 -*-
"""探针 A 一键构建 (2026-09-12): 渲染 100 张 4K -> 质检 -> 合并 -> 打包 -> 门禁 -> 归位 upload/
用法: python build_probe_a.py --scale 0.4 --tag probeA_sc04 [--skip-render]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'          # 代理已关: 强制用本地缓存(所有指标已在缓存)
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
MODELS = Path(r'D:/models')
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
LOG = T / 'build_probe_a.log'


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd, timeout=14400):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=timeout, cwd=r'C:/models/finals')
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--tag', required=True, help='如 probeA_sc04')
    ap.add_argument('--skip-render', action='store_true')
    a = ap.parse_args()

    outdir = MODELS / f'final_compliant_{a.tag}' / 'output_dir'
    # ① 渲染 100 张 (用上一轮的 P3 脚本, 断点续跑)
    if not a.skip_render:
        log(f'① 渲染 100 张 @ scale={a.scale} -> {outdir}')
        rc, out = run([PY, str(T / 'render_test100.py'), '--lora', LORA,
                       '--scale', str(a.scale), '--tag', a.tag])
        log(f'   渲染 rc={rc}, 出图 {len(list(outdir.glob("*.jpg")))} 张')
        if rc != 0:
            log('   !! 渲染失败: ' + out[-500:]); return
    n = len(list(outdir.glob('*.jpg')))
    log(f'① 完成: {n}/100 张')
    if n < 100:
        log('   !! 不足 100 张, 停止'); return

    # ② 质检
    log('② 质检 (qc_test100)')
    rc, out = run([PY, str(T / 'qc_test100.py'), str(outdir)])
    log('   ' + (out.strip().splitlines()[-1] if out.strip() else f'rc={rc}'))

    # ③ 组装 model_dir: 用 remerge 从 M24(sc0.8) 反推本探针 scale 的权重 (与出图严格等价)
    log(f'③ 组装 model_dir (remerge sc0.8 -> sc{a.scale})')
    md = MODELS / f'final_compliant_{a.tag}' / 'model_dir'
    src_md = MODELS / 'final_compliant' / 'model_dir'
    md.mkdir(parents=True, exist_ok=True)
    import shutil
    for p in src_md.rglob('*'):          # 拷支撑文件(含加固 runner / _vendor), your_model.pt 单独处理
        if p.is_file() and p.name != 'your_model.pt':
            rel = p.relative_to(src_md)
            d = md / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d)
    rc, out = run([PY, str(T / 'remerge_scale_05b.py'),
                   '--src-pkg', str(src_md / 'your_model.pt'),
                   '--scale', str(a.scale),
                   '--out', str(md / 'your_model.pt')])
    ok_rm = 'REMERGE_DONE' in out
    log(f'   remerge {"OK" if ok_rm else "失败"}: ' +
        (out.strip().splitlines()[-2] if out.strip() else f'rc={rc}'))
    if not ok_rm:
        log('   !! remerge 失败, 停止'); return
    log(f'   model_dir 文件数 {sum(1 for _ in md.rglob("*") if _.is_file())}')

    # ④ 打包
    zname = f'Track2_{a.tag}'
    log(f'④ 打包 {zname}')
    rc, out = run([PY, str(T / 'package_generic.py'), '--name', zname,
                   '--outdir', str(outdir), '--modeldir', str(md)])
    log('   ' + (out.strip().splitlines()[-2] if out.strip() else f'rc={rc}'))

    # ⑤ 门禁 + 归位
    log('⑤ 10 项门禁 + 归位 upload/')
    rc, out = run([PY, str(T / 'preupload_gate.py'), '--stage-upload',
                   '--upload-name', f'{zname}.zip', str(MODELS / f'{zname}.zip')])
    ok = 'PREUPLOAD_ALL_PASS' in out
    for ln in out.strip().splitlines():
        if any(k in ln for k in ('✅', '❌', '==>')):
            log('   ' + ln.strip())
    log(f'⑤ {"✅ 可以上传" if ok else "❌ 门禁未过"}')

    (T / f'build_{a.tag}.json').write_text(json.dumps(
        {'tag': a.tag, 'scale': a.scale, 'zip': str(MODELS / f'{zname}.zip'),
         'gate_pass': ok}, indent=1), encoding='utf-8')
    log('[done]')


if __name__ == '__main__':
    main()
