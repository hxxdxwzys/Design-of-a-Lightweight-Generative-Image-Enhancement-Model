# -*- coding: utf-8 -*-
"""蒸馏链 (2026-09-14): 等数据 -> 训 v5(256域) -> 全流程 -> 训 v6(512域) -> 全流程
每个候选都会做: 导出(graph) -> 出100张 -> 本地质量代理(测试集 clipiqa/niqe) -> 打包 -> 门禁 -> 归位
日志: D:/tmp_timing/distill_chain.log
"""
from __future__ import annotations
import os, shutil, subprocess, sys, time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'distill_chain.log'
UPLOAD = Path(r'D:/models/upload')
REPORT = T / 'DISTILL_REPORT.md'


def log(m):
    s = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def run(name, cmd, timeout=21600):
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
    log(f'[wait] {marker} 超时')
    return False


def productize(ckpt: Path, tag: str, down: int, note: str):
    log(f'=== productize {tag} (ckpt={ckpt.name} down={down}) ===')
    pkg = Path(r'D:/models') / f'pkg_{tag}'
    _, eout = run(f'{tag} export(graph)', [PY, '-u', str(T / 'export_tiny.py'), '--ckpt', str(ckpt),
                                            '--modeldir', str(pkg / 'model_dir'), '--down', str(down)])
    graph_ok = '[graph] 可 capture' in eout
    run(f'{tag} render100', [PY, '-u', str(T / 'render_tiny.py'), '--ckpt', str(ckpt),
                             '--out', str(pkg / 'output_dir'), '--down', str(down)])
    n = len(list((pkg / 'output_dir').glob('*.jpg')))
    _, mout = run(f'{tag} measure', [PY, '-u', str(T / 'measure_test30.py'), '--dir', str(pkg / 'output_dir')])
    metric = ''
    for ln in mout.splitlines():
        if '[measure]' in ln:
            metric = ln.strip()
    # 时延
    _, tout = run(f'{tag} latency', [PY, '-u', '-c', (
        "import sys,time,torch;sys.path.insert(0,r'%s');import runner as R;"
        "r=R.Runner(r'%s');x=(torch.rand(1,3,512,512)*2-1);"
        "y=r.infer(x);"
        "[r.infer(x) for _ in range(20)];torch.cuda.synchronize();t=time.perf_counter();"
        "[r.infer(x) for _ in range(200)];torch.cuda.synchronize();"
        "ms=(time.perf_counter()-t)/200*1000;"
        "print('[latency] %%%.2f ms -> %%%.1fx graph=%%s'%%(ms,285.6/ms,r.graph is not None))"
      ) % (str(pkg / 'model_dir'), str(pkg / 'model_dir'))])
    lat = ''
    for ln in tout.splitlines():
        if '[latency]' in ln:
            lat = ln.strip()
    ok_pack = False
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
            ok_pack = True
            log(f'[upload] {dst} 已归位')
    else:
        log(f'!! {tag} 只出了 {n}/100')
    return {'tag': tag, 'down': down, 'note': note, 'images': n, 'metric': metric,
            'latency': lat, 'graph': graph_ok, 'packed': ok_pack}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    log('=' * 70)
    log('DISTILL CHAIN 启动')
    # 等数据生成器 (轮询 lq 目录; 生成器不在本进程内, 用文件计数判断)
    t0 = time.time()
    while time.time() - t0 < 3 * 3600:
        nd = len(list((Path(r'D:/datasets/distill256/lq')).glob('*.jpg')))
        if nd >= 2800:
            break
        time.sleep(60)
    nd = len(list((Path(r'D:/datasets/distill256/lq')).glob('*.jpg')))
    log(f'蒸馏数据 {nd} 对 (等待 {(time.time()-t0)/60:.0f} min)')
    res = []
    # v5: 256 域 (快, graph 后 ~50x)
    run('train v5 (distill 256)', [PY, '-u', str(T / 'train_tiny.py'), '--distill_dir', r'D:/datasets/distill256',
                                   '--down', '2', '--bs', '8', '--steps', '40000', '--tag', 'v5'])
    ck = Path(r'D:/models/tiny_v1/v5_s40000.pt')
    if ck.exists():
        res.append(productize(ck, 'TinyV5d', 2, '蒸馏 M24 观感, 256 域 + graph'))
    # v6: 512 域 (质量上限更高)
    run('train v6 (distill 512)', [PY, '-u', str(T / 'train_tiny.py'), '--distill_dir', r'D:/datasets/distill256',
                                   '--down', '1', '--bs', '4', '--steps', '25000', '--tag', 'v6'])
    ck = Path(r'D:/models/tiny_v1/v6_s25000.pt')
    if ck.exists():
        res.append(productize(ck, 'TinyV6d', 1, '蒸馏 M24 观感, 512 域 + graph'))
    with REPORT.open('w', encoding='utf-8') as f:
        f.write('# 蒸馏链报告 (2026-09-14)\n\n')
        f.write('动机: 微模型 clipiqa 掉到 LQ 档(0.34) -> 官方质量项主要认"被增强过多少"\n')
        f.write('做法: 训练目标从 GT 换成 **M24 的输出** (蒸馏观感)\n\n')
        f.write('| 候选 | 域 | 加速比 | 本地质量代理 | 门禁 | 归位 |\n|---|---|---|---|---|---|\n')
        for r in res:
            f.write(f"| {r['tag']} | {r['down']} | {r['latency']} | {r['metric']} | {'OK' if r['packed'] else 'FAIL'} | {r['packed']} |\n")
        f.write('\n参照: M24 clipiqa 0.9023 / niqe 5.2559 | TinyV1 0.3367 | TinyV2 0.3532 | LQ 0.2752\n')
    log(f'[report] {REPORT}')
    log('DISTILL_CHAIN_DONE')


if __name__ == '__main__':
    main()
