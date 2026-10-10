# -*- coding: utf-8 -*-
"""今晚->明天 全自动流水线 (2026-09-13 夜, 用户睡觉)
 1) 等 v1 训练完成  2) v1: 评估 -> 导出 -> 出100张4K -> 打包 -> 门禁 -> 归位 upload/
 3) 训练 v2 (down=2, 256 域, ~4x 提速)  4) v2 同样处理
 5) 训练 v3 (更宽 ch=64,64,64,128, 512)  6) v3 同样处理
 7) 写 NIGHT_REPORT.md
日志: D:/tmp_timing/night_final.log   产物: D:/models/pkg_<tag>/, D:/models/Track2_<tag>.zip, D:/models/upload/
"""
from __future__ import annotations
import os, shutil, subprocess, sys, time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'night_final.log'
UPLOAD = Path(r'D:/models/upload')
REPORT = T / 'NIGHT_REPORT.md'


def log(m):
    s = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def run(name, cmd, timeout=14400):
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


def wait_marker(logfile: Path, marker: str, timeout_s: int):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if marker in logfile.read_text(encoding='utf-8', errors='replace'):
                return True
        except Exception:
            pass
        time.sleep(60)
    log(f'[wait] {marker} 超时 ({timeout_s/3600:.1f}h) -> 用当前最新 ckpt')
    return False


def latest_ckpt(tag: str):
    cks = sorted(Path(r'D:/models/tiny_v1').glob(f'{tag}_s*.pt'),
                 key=lambda p: int(p.stem.split('_s')[-1]))
    return cks[-1] if cks else None


def productize(ckpt: Path, tag: str, down: int, note: str):
    """评估 -> 导出 -> 出图 -> 打包 -> 门禁 -> 归位"""
    log(f'=== productize {tag} (ckpt={ckpt.name} down={down}) ===')
    code, out = run(f'{tag} eval', [PY, '-u', str(T / 'eval_tiny.py'), '--ckpt', str(ckpt),
                                    '--down', str(down), '--out', str(Path(r'D:/models') / f'pkg_{tag}' / 'eval')])
    pkg = Path(r'D:/models') / f'pkg_{tag}'
    run(f'{tag} export', [PY, '-u', str(T / 'export_tiny.py'), '--ckpt', str(ckpt),
                          '--modeldir', str(pkg / 'model_dir'), '--down', str(down)])
    run(f'{tag} render100', [PY, '-u', str(T / 'render_tiny.py'), '--ckpt', str(ckpt),
                             '--out', str(pkg / 'output_dir'), '--down', str(down)])
    n = len(list((pkg / 'output_dir').glob('*.jpg')))
    if n != 100:
        log(f'!! {tag} 出图只有 {n}/100, 跳过打包')
        return {'tag': tag, 'ok': False, 'reason': f'only {n} images'}
    run(f'{tag} package', [PY, '-u', str(T / 'package_generic.py'), '--name', f'Track2_{tag}',
                           '--outdir', str(pkg / 'output_dir'), '--modeldir', str(pkg / 'model_dir')])
    zip_p = Path(r'D:/models') / f'Track2_{tag}.zip'
    code, gout = run(f'{tag} gate', [PY, '-u', str(T / 'preupload_gate.py'), str(zip_p)])
    passed = 'PREUPLOAD_ALL_PASS' in gout
    if passed:
        UPLOAD.mkdir(parents=True, exist_ok=True)
        dst = UPLOAD / f'Track2_{tag}.zip'
        if dst.exists():
            dst.unlink()
        try:
            os.link(zip_p, dst)
        except OSError:
            shutil.copy2(zip_p, dst)
        log(f'[upload] {dst} 已归位 (门禁 PASS)')
    else:
        log(f'!! {tag} 门禁未过, 不归位')
    lat = ''
    for ln in (out or '').splitlines():
        if '[latency]' in ln:
            lat = ln.strip()
    return {'tag': tag, 'ok': passed, 'zip': str(zip_p), 'latency': lat, 'note': note}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    log('=' * 70)
    log('NIGHT FINAL 启动: v1 等训练 -> 三条候选逐个 productize')
    results = []

    # ---- v1 ----
    wait_marker(Path(r'D:/models/tiny_v1/train.log'), 'TRAIN_TINY_DONE', 4 * 3600)
    ck = latest_ckpt('v1b')
    if ck:
        results.append(productize(ck, 'TinyV1', 1, '512 域 0.87M, 目标 ~10x'))
    else:
        log('!! 找不到 v1 ckpt')

    # ---- v2: 256 域 (~4x 提速) ----
    run('train v2 (down=2)', [PY, '-u', str(T / 'train_tiny.py'), '--res', '512', '--down', '2',
                              '--bs', '8', '--steps', '40000', '--tag', 'v2'])
    ck = latest_ckpt('v2')
    if ck:
        results.append(productize(ck, 'TinyV2', 2, '256 域 0.87M, 目标 ~40x'))

    # ---- v3: 更宽 512 域 ----
    run('train v3 (wide)', [PY, '-u', str(T / 'train_tiny.py'), '--res', '512', '--down', '1',
                            '--bs', '2', '--steps', '30000', '--ch', '64,64,64,128', '--tag', 'v3'])
    ck = latest_ckpt('v3')
    if ck:
        results.append(productize(ck, 'TinyV3', 1, '512 域 3.5M 宽模型, 目标 ~3x, 质量更好'))

    # ---- 报告 ----
    with REPORT.open('w', encoding='utf-8') as f:
        f.write('# 夜间流水线报告 (2026-09-13 夜 -> 09-14)\n\n')
        f.write('速度轴已证实: 同图只改时延, 4.98~6.84 分/翻倍 (M24 43.6951 -> Slow8 28.7538)\n\n')
        f.write('| 候选 | 门禁 | 512 时延 | 说明 | 包 |\n|---|---|---|---|---|\n')
        for r in results:
            f.write(f"| {r['tag']} | {'✅' if r['ok'] else '❌'} | {r.get('latency','')} | {r['note']} | {r.get('zip','')} |\n")
        f.write('\n详细日志: D:/tmp_timing/night_final.log\n')
        f.write('每个候选的 val 指标在 D:/models/pkg_<tag>/eval/eval.json\n')
    log(f'[report] {REPORT}')
    log('NIGHT_FINAL_DONE')


if __name__ == '__main__':
    main()
