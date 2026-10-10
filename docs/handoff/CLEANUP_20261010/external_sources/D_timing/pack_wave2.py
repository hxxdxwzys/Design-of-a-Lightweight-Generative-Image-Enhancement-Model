# -*- coding: utf-8 -*-
"""波 2 一键打包+验证 (2026-09-11): tag -> 可直接上传的 zip

从已渲染的 wave2_<tag>/output_dir + 反推的 pkg 组装 model_dir, 打包, 并跑两道验证:
  ① harness_sim_zip (解压+纯净 import+512 infer)
  ② audit_py_compat  (官方 Py<3.10 的 |None 语法审计)
model_dir 的 runner.py 与依赖目录沿用已提交的合规包 (final_compliant_14b/model_dir),
只替换 your_model.pt -> 保证与已验证过的合规结构完全一致。

用法:
  D:/Python/python.exe pack_wave2.py --tag m0s7_s097 --src-pkg <源 pkg> --scale 0.97 \
      --name Track2_14B_M0S7_SC097
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PY = r'D:/Python/python.exe'
TOOLS = Path(r'D:/tmp_timing')
TEMPLATE_MD = Path(r'D:/models/final_compliant_14b/model_dir')   # runner.py + 依赖的合规模板
ROOT_BASE = Path(r'D:/models')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='wave2_<tag> 的 tag')
    ap.add_argument('--src-pkg', required=True, help='反推用的源 pkg')
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--name', required=True, help='zip 名, 纯 ASCII, 不含 .zip')
    args = ap.parse_args()

    ROOT = ROOT_BASE / f'wave2_{args.tag}'
    OUT_DIR = ROOT / 'output_dir'
    jpgs = sorted(OUT_DIR.glob('*.jpg'))
    assert len(jpgs) == 100, f'{OUT_DIR} 需 100 jpg, 实际 {len(jpgs)} -> 先跑 render_pkg_test100.py'
    print(f'[1] output_dir OK: {len(jpgs)} jpg')

    # 反推目标 pkg (若渲染时已删则重生, 7 秒)
    pkg = ROOT / 'model_dir' / 'your_model.pt'
    if not pkg.exists():
        r = subprocess.run([PY, str(TOOLS / 'remerge_scale_14b.py'), '--src-pkg', args.src_pkg,
                            '--scale', str(args.scale), '--out', str(pkg)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f'remerge 失败: {r.stdout[-400:]} {r.stderr[-400:]}'
        print(f'[2] remerge OK: {pkg.stat().st_size/1e9:.2f} GB')

    # 组装 model_dir: 模板 (runner.py + 依赖) + 新 pkg
    md = ROOT / 'pack_model_dir'
    if md.exists():
        shutil.rmtree(md)
    md.mkdir(parents=True)
    n = 0
    for p in sorted(TEMPLATE_MD.rglob('*')):
        if not p.is_file():
            continue
        if p.name == 'your_model.pt':
            continue
        if '__pycache__' in p.parts or p.suffix in ('.pyc', '.log'):
            continue
        hardlink_or_copy(p, md / p.relative_to(TEMPLATE_MD))
        n += 1
    hardlink_or_copy(pkg, md / 'your_model.pt')
    print(f'[3] model_dir 组装: {n} 个模板文件 + your_model.pt')

    # 打包
    r = subprocess.run([PY, str(TOOLS / 'package_generic.py'), '--name', args.name,
                        '--outdir', str(OUT_DIR), '--modeldir', str(md)],
                       capture_output=True, text=True)
    print('[4] 打包:', (r.stdout or '').strip()[-200:], (r.stderr or '').strip()[-200:])
    assert r.returncode == 0 and 'PACK_PASS' in (r.stdout or ''), '打包失败'

    zip_path = ROOT_BASE / f'{args.name}.zip'
    # 验证 ① harness sim
    r = subprocess.run([PY, str(TOOLS / 'harness_sim_zip.py'), str(zip_path)],
                       capture_output=True, text=True)
    print('[5] harness sim:', (r.stdout or '').strip()[-400:], (r.stderr or '').strip()[-200:])
    sim_ok = r.returncode == 0 and 'infer OK' in (r.stdout or '')
    # 验证 ② 语法审计 (官方 Py<3.10 的 |None)
    r = subprocess.run([PY, str(TOOLS / 'audit_py_compat.py'), str(zip_path)],
                       capture_output=True, text=True)
    aud_txt = (r.stdout or '').strip()
    aud_ok = '合计有风险文件数: 0' in aud_txt
    print(f'[6] 语法审计: {"PASS" if aud_ok else "FAIL"}')
    if not aud_ok:
        print(aud_txt[-800:])

    # 验证 ③ IQA 依赖遮蔽审计 (官方 "Failed to load IQAEvaluator" 的根因之一)
    sim_md = ROOT_BASE / '_harness_zip_sim' / 'model_dir'
    shadow_ok = False
    if sim_md.exists():
        r = subprocess.run([PY, str(TOOLS / 'audit_iqa_shadow.py'), str(sim_md)],
                           capture_output=True, text=True)
        shadow_ok = 'IQA_SHADOW_AUDIT_PASS' in ((r.stdout or '') + (r.stderr or ''))
        print(f'[7] IQA 遮蔽审计: {"PASS" if shadow_ok else "FAIL"}')
        if not shadow_ok:
            print(((r.stdout or '') + (r.stderr or ''))[-600:])
    else:
        print('[7] IQA 遮蔽审计: SKIP (harness sim 未产出解压目录)')

    print(f'[size] {zip_path.stat().st_size/1e9:.2f} GB  (>10GB 违规)')
    verdict = 'PACK_WAVE2_READY' if (sim_ok and aud_ok and shadow_ok) else 'PACK_WAVE2_PROBLEM'
    print(verdict)
    print(f'zip: {zip_path}')


if __name__ == '__main__':
    main()
