# -*- coding: utf-8 -*-
"""GAN/容量实验链 (2026-09-14 晚): v7 收尾 -> v8(3.5M宽) -> v9(512域)
每个候选: 导出(graph) -> 出100张 -> 本地质量代理(clipiqa/lapvar) -> 打包 -> 门禁 -> 归位
日志: D:/tmp_timing/gan_chain.log
"""
from __future__ import annotations
import os, shutil, subprocess, sys, time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'gan_chain.log'
UPLOAD = Path(r'D:/models/upload')


def log(m):
    s = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def run(name, cmd, timeout=28800):
    log(f'>>> {name}')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env, timeout=timeout)
    out = (p.stdout or '') + (p.stderr or '')
    with LOG.open('a', encoding='utf-8') as f:
        f.write(out + '\n')
    log(f'<<< {name} rc={p.returncode} ({(time.time()-t0)/60:.1f} min)')
    return p.returncode, out


def wait_marker(f: Path, marker: str, timeout_s: int):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if marker in f.read_text(encoding='utf-8', errors='replace'):
                return True
        except Exception:
            pass
        time.sleep(60)
    return False


def productize(ckpt: Path, tag: str, down: int, note: str):
    log(f'=== productize {tag} (down={down}) ===')
    pkg = Path(r'D:/models') / f'pkg_{tag}'
    _, eout = run(f'{tag} export', [PY, '-u', str(T / 'export_tiny.py'), '--ckpt', str(ckpt),
                                    '--modeldir', str(pkg / 'model_dir'), '--down', str(down)])
    run(f'{tag} render100', [PY, '-u', str(T / 'render_tiny.py'), '--ckpt', str(ckpt),
                             '--out', str(pkg / 'output_dir'), '--down', str(down)])
    n = len(list((pkg / 'output_dir').glob('*.jpg')))
    _, mout = run(f'{tag} measure', [PY, '-u', str(T / 'measure_test30.py'), '--dir', str(pkg / 'output_dir')])
    metric = next((ln.strip() for ln in mout.splitlines() if '[measure]' in ln), '')
    _, lout = run(f'{tag} latency', [PY, '-u', str(T / 'measure_latency.py'), str(pkg / 'model_dir')])
    lat = next((ln.strip() for ln in lout.splitlines() if '[latency]' in ln), '')
    packed = False
    if n == 100:
        run(f'{tag} package', [PY, '-u', str(T / 'package_generic.py'), '--name', f'Track2_{tag}',
                               '--outdir', str(pkg / 'output_dir'), '--modeldir', str(pkg / 'model_dir')])
        zp = Path(r'D:/models') / f'Track2_{tag}.zip'
        _, gout = run(f'{tag} gate', [PY, '-u', str(T / 'preupload_gate.py'), str(zp)])
        if 'PREUPLOAD_ALL_PASS' in gout:
            UPLOAD.mkdir(parents=True, exist_ok=True)
            dst = UPLOAD / f'Track2_{tag}.zip'
            if dst.exists():
                dst.unlink()
            try:
                os.link(zp, dst)
            except OSError:
                shutil.copy2(zp, dst)
            packed = True
            log(f'[upload] {dst}')
    else:
        log(f'!! {tag} 只出了 {n}/100')
    return {'tag': tag, 'down': down, 'note': note, 'metric': metric, 'latency': lat, 'packed': packed}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    log('=' * 70)
    log('GAN/容量链启动: 等 v7 -> v7 productize -> 训 v8(3.5M宽) -> v8 productize -> 训 v9(512域) -> v9 productize')
    res = []
    wait_marker(Path(r'D:/models/tiny_v1/train_gan.log'), 'TRAIN_GAN_DONE', 3 * 3600)
    ck = Path(r'D:/models/tiny_v1/v7gan_s25000.pt')
    if ck.exists():
        res.append(productize(ck, 'TinyV7gan', 2, '0.87M GAN蒸馏 256域'))

    run('train v8 (3.5M 宽 + GAN)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '20000',
                                     '--bs', '4', '--ch', '64,64,64,128', '--tag', 'v8gan'])
    ck = Path(r'D:/models/tiny_v1/v8gan_s20000.pt')
    if ck.exists():
        res.append(productize(ck, 'TinyV8gan', 2, '3.5M GAN蒸馏 256域'))

    run('train v9 (512 域 + GAN)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '15000',
                                    '--bs', '4', '--res', '512', '--tag', 'v9gan'])
    ck = Path(r'D:/models/tiny_v1/v9gan_s15000.pt')
    if ck.exists():
        res.append(productize(ck, 'TinyV9gan', 1, '0.87M GAN蒸馏 512域'))

    with (T / 'GAN_CHAIN_REPORT.md').open('w', encoding='utf-8') as f:
        f.write('# GAN/容量链报告 (2026-09-14)\n\n')
        f.write('参照: M24 clipiqa 0.9023 / lapvar 508 -> 43.6951 | v5 clipiqa 0.3897 / lapvar 92 -> 33.0382\n\n')
        f.write('| 候选 | 域 | 质量代理 | 时延 | 门禁/归位 |\n|---|---|---|---|---|\n')
        for r in res:
            f.write(f"| {r['tag']} | {r['down']} | {r['metric']} | {r['latency']} | {r['packed']} |\n")
    log('GAN_CHAIN_DONE')


if __name__ == '__main__':
    main()
