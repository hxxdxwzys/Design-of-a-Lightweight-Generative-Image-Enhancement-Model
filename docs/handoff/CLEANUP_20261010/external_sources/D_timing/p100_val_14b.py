# -*- coding: utf-8 -*-
"""1.4B 训练档验证 (2026-09-03): 下载 -> 本地 4060 fp16 渲染 val 3 张 -> 门控
用法: python p100_val_14b.py s500 [s1000 ...]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r'D:/tmp_timing')
from p100_upload import connect  # noqa: E402

CKPT_DIR = Path(r'D:/tmp_timing/p100_ckpts')
OUT_DIR = Path(r'D:/tmp_timing/p100_val')
VAL_LQ = r'D:/tmp_timing/p100_val/lq'
VAL_GT = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
BASE_CKPT = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
GATE_CLIP, GATE_LPIPS = 0.919, 0.360
VARIANTS = ['p512_14b_w20', 'p512_14b_w15']
INFER14 = r'D:/Python/python.exe D:/tmp_timing/vosr_lora_infer_14b.py'
GATE = r'D:/Python/python.exe D:/tmp_timing/val_gate.py'


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with (OUT_DIR / 'val14_history.txt').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    steps = sys.argv[1:] or ['s500']
    cli = connect()
    sftp = cli.open_sftp()
    for v in VARIANTS:
        for s in steps:
            remote = f'p100/runs/{v}/lora_{s}.pt'
            local = CKPT_DIR / f'{v}_lora_{s}.pt'
            tag = f'{v}_{s}'
            outdir = OUT_DIR / f'out_{tag}'
            if (OUT_DIR / f'gate_{tag}.json').exists():
                log(f'[skip] {tag} 已测')
                continue
            try:
                sftp.stat(remote)
            except Exception:
                log(f'[warn] {tag} 服务器无此存档')
                continue
            t0 = time.time()
            log(f'[下载] {tag}')
            sftp.get(remote, str(local))
            log(f'[渲染] {tag} (4060 fp16 4K, ~10min/张)')
            r = subprocess.run(
                f'{INFER14} -c {BASE_CKPT} --lora {local} --fp16 -i {VAL_LQ} -o {outdir} '
                f'--align_method wavelet',
                shell=True, capture_output=True, text=True, timeout=7200)
            if r.returncode != 0:
                log(f'[FAIL] {tag}: {r.stderr[-300:]}')
                continue
            subprocess.run(
                f'{GATE} --src {outdir} --clip-min {GATE_CLIP} --lpips-max {GATE_LPIPS} '
                f'--val-dir {VAL_GT} --out {OUT_DIR / f"gate_{tag}.json"}',
                shell=True, capture_output=True, text=True, timeout=1800)
            try:
                g = json.loads((OUT_DIR / f'gate_{tag}.json').read_text(encoding='utf-8'))
                log(f'[GATE] {tag}: clip={g["clip"]} lpips={g["lpips"]} pass={g["pass"]} '
                    f'({time.time() - t0:.0f}s)')
            except Exception as e:
                log(f'[GATE_FAIL] {tag}: {e}')
    sftp.close()
    cli.close()
    log('[DONE]')


if __name__ == '__main__':
    main()
