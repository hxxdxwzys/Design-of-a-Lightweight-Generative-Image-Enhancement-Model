# -*- coding: utf-8 -*-
"""FiDeSR 可续管线 (2026-08-25): val 门控 -> 旋钮扫描 -> 测试集逐图渲染 -> 入池 -> choice -> 打包
可暂停: 检测 D:/tmp_timing/fidesr_pause.flag -> 单元边界干净停 (exit 0)
可续跑: 状态 D:/tmp_timing/b_fidesr_state.json + 所有产出幂等 (存在即跳过)
用法: cd FiDeSR-main && D:/Python/python.exe D:/tmp_timing/b_fidesr_pipe.py [--phase val|test|pool]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
TT = Path(r'D:/tmp_timing')
FD = r'D:/models/new_src/fidesr/code/FiDeSR-main'
SD = r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base'
PKL = r'D:/models/new_src/fidesr/fidesr.pkl'
VAL_IN = r'D:/tmp_timing/codiff_val_in'
TEST_DIR = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'
OUT_ROOT = Path(r'D:/models/new_src/fidesr')
STATE = TT / 'b_fidesr_state.json'
PAUSE_FLAG = TT / 'fidesr_pause.flag'
LOG = TT / 'b_fidesr_pipe.log'

# 门控: 新家族, clip>=0.90 且 lpips<=0.376
GATE = {'clip': 0.90, 'lpips': 0.376}

# val 配置队列: (upscale, lf, hf, tag); 第一个为默认, 不过门才跑后续
VAL_CONFIGS = [
    (1, 0.2, 0.3, 'u1_lf02_hf03'),
    (1, 0.1, 0.5, 'u1_lf01_hf05'),
    (1, 0.5, 0.3, 'u1_lf05_hf03'),
    (1, 0.5, 0.5, 'u1_lf05_hf05'),
    (4, 0.2, 0.3, 'u4_lf02_hf03'),
]


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def paused():
    return PAUSE_FLAG.exists()


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding='utf-8')


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    return {'phase': 'val', 'val_done': [], 'val_best': None, 'chosen': None,
            'test_done': [], 'test_total': 0}


def render_val(upscale, lf, hf, tag):
    """渲染 val 3 张 -> gate; 返回 (ok, gate) 或 (False, None)"""
    out = OUT_ROOT / f'val_out_{tag}'
    gj = OUT_ROOT / f'val_gate_{tag}.json'
    if not gj.exists():
        out.mkdir(parents=True, exist_ok=True)
        cmd = [PY, 'test_fidesr.py', '--pretrained_model_path', SD, '--pretrained_path', PKL,
               '--input_image', VAL_IN, '--output_dir', str(out),
               '--upscale', str(upscale), '--align_method', 'wavelet',
               '--lf_scale', str(lf), '--hf_scale', str(hf),
               '--mixed_precision', 'fp16', '--seed', '42']
        log(f'[val] render {tag} (u{upscale} lf{lf} hf{hf})...')
        t0 = time.time()
        r = subprocess.run(cmd, cwd=FD)
        log(f'[val] {tag} rc={r.returncode} ({time.time()-t0:.0f}s)')
        if r.returncode != 0:
            return False, None
        # 4x 退路: 下采样回 1x (LANCZOS, 逐图幂等)
        if upscale > 1:
            from PIL import Image
            for p in sorted(out.glob('*.png')):
                im = Image.open(p)
                if max(im.size) > 4200:
                    s = 1.0 / upscale
                    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
                    im.save(p)
        subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(out), '--out', str(gj)])
    g = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else None
    return g is not None, g


def pick_best(val_done):
    """val_done: [{tag, clip, lpips}] -> 最优 (过门优先, 其次 lpips 最低且 clip>=0.87)"""
    passed = [d for d in val_done if d.get('clip', 0) >= GATE['clip'] and d.get('lpips', 9) <= GATE['lpips']]
    if passed:
        return min(passed, key=lambda d: d['lpips'])
    viable = [d for d in val_done if d.get('clip', 0) >= 0.87]
    if viable:
        return min(viable, key=lambda d: d['lpips'])
    return min(val_done, key=lambda d: (d.get('lpips', 9), -d.get('clip', 0))) if val_done else None


def render_test(cfg, done, total):
    """逐图渲染测试集 100 (幂等 + 暂停检查)"""
    upscale, lf, hf, tag = cfg
    out = OUT_ROOT / f'test_out_{tag}'
    out.mkdir(parents=True, exist_ok=True)
    cases = [f'case{i}.jpg' for i in range(1, 101)]
    remaining = [c for c in cases if not (out / c).exists()]
    for i, c in enumerate(remaining):
        if paused():
            log('[pause] flag found, clean stop')
            return False
        cmd = [PY, 'test_fidesr.py', '--pretrained_model_path', SD, '--pretrained_path', PKL,
               '--input_image', str(Path(TEST_DIR) / c), '--output_dir', str(out),
               '--upscale', str(upscale), '--align_method', 'wavelet',
               '--lf_scale', str(lf), '--hf_scale', str(hf),
               '--mixed_precision', 'fp16', '--seed', '42']
        t0 = time.time()
        r = subprocess.run(cmd, cwd=FD)
        if r.returncode != 0:
            log(f'[test] {c} FAILED rc={r.returncode}')
            continue
        if upscale > 1:
            from PIL import Image
            p = out / c
            if p.exists():
                im = Image.open(p)
                if max(im.size) > 4200:
                    s = 1.0 / upscale
                    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
                    im.save(p)
        log(f'[test] {c} done ({time.time()-t0:.0f}s) [{len(remaining)-i-1} left]')
        done.append(c)
    return True


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='auto', choices=['auto', 'val', 'test', 'pool'])
    args = ap.parse_args()
    st = load_state()
    log(f'[boot] phase={args.phase} state={st["phase"]} val_done={len(st["val_done"])}')

    # ---- Phase 1: val 门控 + 旋钮扫描 ----
    if args.phase in ('auto', 'val') and st['phase'] == 'val':
        for cfg in VAL_CONFIGS:
            if paused():
                log('[pause] flag found, clean stop')
                save_state(st)
                sys.exit(0)
            upscale, lf, hf, tag = cfg
            if tag in [d['tag'] for d in st['val_done']]:
                continue
            ok, g = render_val(upscale, lf, hf, tag)
            if not ok:
                continue
            st['val_done'].append({'tag': tag, 'upscale': upscale, 'lf': lf, 'hf': hf,
                                   'clip': g.get('clip'), 'lpips': g.get('lpips')})
            log(f'[gate] {tag}: clip={g.get("clip")} lpips={g.get("lpips")}')
            save_state(st)
            if g.get('clip', 0) >= GATE['clip'] and g.get('lpips', 9) <= GATE['lpips']:
                log(f'[gate] {tag} PASSED, stop scanning')
                break
        st['val_best'] = pick_best(st['val_done'])
        st['chosen'] = st['val_best']
        if st['val_best']:
            log(f'[decide] best = {st["val_best"]["tag"]} clip={st["val_best"]["clip"]} lpips={st["val_best"]["lpips"]}')
        passed = any(d.get('clip', 0) >= GATE['clip'] and d.get('lpips', 9) <= GATE['lpips'] for d in st['val_done'])
        if not passed and (not st['val_best'] or st['val_best'].get('clip', 0) < 0.87):
            log('[valfail] 无配置过门且 clip <0.87, 停止并呈报 (不渲染测试集)')
            st['phase'] = 'valfail'
            save_state(st)
            sys.exit(0)
        st['phase'] = 'test'
        save_state(st)

    # ---- Phase 2: 测试集逐图渲染 ----
    if args.phase in ('auto', 'test') and st['phase'] == 'test':
        if not st['chosen']:
            log('[err] no chosen config')
            sys.exit(3)
        c = (st['chosen']['upscale'], st['chosen']['lf'], st['chosen']['hf'], st['chosen']['tag'])
        done = render_test(c, st['test_done'], st['test_total'])
        st['test_done'] = sorted(set(st['test_done']))
        save_state(st)
        if not done:
            log('[pause] stopped mid-test, resume later')
            sys.exit(0)
        log(f'[test] complete: {len(st["test_done"])} images')
        st['phase'] = 'pool'
        save_state(st)

    # ---- Phase 3: 入池 + choice + 打包 ----
    if args.phase in ('auto', 'pool') and st['phase'] == 'pool':
        tag = st['chosen']['tag']
        out = OUT_ROOT / f'test_out_{tag}'
        name = f'fidesr_{tag}'
        if paused():
            log('[pause] flag found, clean stop')
            save_state(st)
            sys.exit(0)
        subprocess.run([PY, str(TT / 'pool_add_source.py'), '--src', str(out), '--name', name],
                       cwd=FD)
        subprocess.run([PY, str(TT / 'compute_choice.py')])
        subprocess.run([PY, str(TT / 'package_submit_v2.py'),
                        '--out-dir', str(Path(r'D:/models/sweep_test_out') / name),
                        '--model-dir', r'D:/models/final_graph7/model_dir',
                        '--name', f'MetricMixS7Fidesr{tag}', '--account', 'Yuki'])
        st['phase'] = 'done'
        save_state(st)
        log('[done] pipeline complete')

    log(f'[exit] phase={st["phase"]}')


if __name__ == '__main__':
    main()
