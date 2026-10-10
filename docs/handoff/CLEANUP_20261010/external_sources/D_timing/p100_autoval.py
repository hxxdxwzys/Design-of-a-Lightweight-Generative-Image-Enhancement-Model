# -*- coding: utf-8 -*-
"""P100 训练自动验证链 (2026-09-02): 轮询新 ckpt -> 下载 -> 本地 4060 渲染 val -> 门控
用法: D:/Python/python.exe p100_autoval.py   (Ctrl+C 或训练结束可停)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r'D:/tmp_timing')
from p100_upload import connect  # noqa: E402

CKPT_DIR = Path(r'D:/tmp_timing/p100_ckpts')
OUT_DIR = Path(r'D:/tmp_timing/p100_val')
VAL_LQ = r'D:/tmp_timing/p100_val/lq'
INFER = r'D:/Python/python.exe D:/tmp_timing/vosr_lora_infer.py'
GATE = r'D:/Python/python.exe D:/tmp_timing/val_gate.py'
BASE_CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
VALDIR_GT = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
VARIANTS = ['p768_w20', 'p768_w15']
GATE_CLIP, GATE_LPIPS = 0.919, 0.360
POLL = 600  # 秒
LOG = OUT_DIR / 'autoval_history.txt'


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def remote_ckpts(cli, variant):
    _in, _out, _err = cli.exec_command(f'ls ~/p100/runs/{variant}/lora_s*.pt 2>/dev/null', timeout=30)
    return sorted(p.strip().rsplit('/', 1)[-1] for p in _out.read().decode().splitlines())


def already_done(variant, name):
    """宽松判定: 任何 out_* 目录含该 variant 标识 + step 即视为已测 (兼容不同命名)"""
    step = ''.join(ch for ch in name if ch.isdigit())
    vkey = variant.split('_')[1]  # 如 w20
    for d in OUT_DIR.glob('out_*'):
        if d.is_dir() and vkey in d.name and f's{step}' in d.name:
            return True
    return False


def main():
    done = set()
    log(f'[boot] 自动验证链启动, 轮询 {POLL}s, 门控 ({GATE_CLIP},{GATE_LPIPS})')
    while True:
        try:
            cli = connect()
            sftp = cli.open_sftp()
            for variant in VARIANTS:
                for name in remote_ckpts(cli, variant):
                    if name in done or already_done(variant, name):
                        continue
                    local = CKPT_DIR / f'{variant}_{name}'
                    log(f'[新存档] {variant}/{name} -> 下载')
                    sftp.get(f'p100/runs/{variant}/{name}', str(local))
                    tag = f'{variant}_{name.replace(".pt", "")}'
                    outdir = OUT_DIR / f'out_{tag}'
                    r1 = subprocess.run(
                        f'{INFER} -c {BASE_CKPT} --lora {local} -i {VAL_LQ} -o {outdir} --align_method wavelet',
                        shell=True, capture_output=True, text=True, timeout=3600)
                    if r1.returncode != 0:
                        log(f'[渲染失败] {tag}: {r1.stderr[-500:]}')
                        done.add(name)
                        continue
                    gate = OUT_DIR / f'gate_{tag}.json'
                    subprocess.run(
                        f'{GATE} --src {outdir} --clip-min {GATE_CLIP} --lpips-max {GATE_LPIPS} '
                        f'--val-dir {VALDIR_GT} --out {gate}',
                        shell=True, capture_output=True, text=True, timeout=1800)
                    try:
                        import json
                        g = json.loads(gate.read_text(encoding='utf-8'))
                        log(f'[门控] {tag}: clip={g["clip"]} lpips={g["lpips"]} pass={g["pass"]}')
                    except Exception as e:
                        log(f'[门控异常] {tag}: {e}')
                    done.add(name)
            sftp.close()
            cli.close()
        except Exception as e:
            log(f'[warn] {type(e).__name__}: {str(e)[:150]}')
        time.sleep(POLL)


if __name__ == '__main__':
    main()
