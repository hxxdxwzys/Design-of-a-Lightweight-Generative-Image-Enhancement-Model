# -*- coding: utf-8 -*-
"""冲刺七提交链一键封装 (分支 A): 备份旧选择 -> 重算 choice -> 装配 -> 打包
用法: D:/Python/python.exe make_submit_s7.py --margin 0.010 --out D:/models/cnn_out_mix_s7 --name <ASCII作品名>
步骤:
  1. 备份 choice_maps.json / hybrid_choice.json (加 .pre_s7 后缀, 保 M01 记录)
  2. compute_choice.py 重算 (自动纳入 sweep_results 新条目) + hybrid_rule.py 对照
  3. assemble_mix.py --strategy map:M<margin> --out <out>
  4. package_submit_v2.py --out-dir <out> --name <name>
"""
import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PY = r'D:/Python/python.exe'
TD = Path(r'D:/tmp_timing')


def run(args):
    print('[run]', ' '.join(args), flush=True)
    subprocess.run([PY, *args], cwd=str(TD), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--margin', default='0.010')
    ap.add_argument('--out', default=r'D:/models/cnn_out_mix_s7')
    ap.add_argument('--name', required=True, help='纯 ASCII 作品名')
    ap.add_argument('--skip-hybrid', action='store_true')
    args = ap.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    for f in ('choice_maps.json', 'hybrid_choice.json'):
        p = TD / f
        if p.exists():
            shutil.copy2(p, TD / f'{f}.pre_s7_{ts}')
            print(f'[backup] {p.name} -> {p.name}.pre_s7_{ts}', flush=True)

    run(['compute_choice.py'])
    if not args.skip_hybrid:
        run(['hybrid_rule.py'])
    run(['assemble_mix.py', '--strategy', f'map:M{args.margin}', '--out', args.out])
    run(['package_submit_v2.py', '--out-dir', args.out,
         '--model-dir', r'D:/models/final_graph7', '--name', args.name])
    print('[all ok]', flush=True)


if __name__ == '__main__':
    main()
