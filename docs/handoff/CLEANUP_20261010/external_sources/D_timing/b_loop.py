# -*- coding: utf-8 -*-
"""B 全流程编排 v2 (2026-08-18): VOSR LoRA 3 变体训练(每500步存盘) -> val 门控全部 ckpt
-> 每变体择优渲染 1-2 个 ckpt 测试集 -> 入池 -> choice -> 唤醒
用法: D:/Python/python.exe b_loop.py [--resume]
状态: D:/tmp_timing/b_loop_state.json; 日志: D:/tmp_timing/b_loop.log
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
PY = r'D:/Python/python.exe'
TT = Path(r'D:/tmp_timing')
LOG = TT / 'b_loop.log'
STATE = TT / 'b_loop_state.json'
OUT = Path(r'D:/models/vosr_lora')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
TEST_DIR = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'
VAL_IN = TT / 'codiff_val_in'
RENDER_OUT = Path(r'D:/models/sweep_test_out')

VARIANTS = [('v1', 0.3), ('v2', 0.8), ('v3', 1.5)]
STEPS = 2500
SAVE_EVERY = 500
MAX_RENDER_CKPT = 2


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding='utf-8'))
        except Exception as e:
            log(f'[warn] state 读取失败: {e}')
    return {'stage': 0, 'variant_idx': 0, 'results': {}}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding='utf-8')


def run(cmd, cwd, logfile, tag):
    log(f'[run] {tag}')
    t0 = time.time()
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('ab') as f:
        r = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    log(f'[run] {tag}: exit={r.returncode} ({time.time() - t0:.0f}s)')
    if r.returncode != 0:
        tail = logfile.read_text(encoding='utf-8', errors='ignore').strip().splitlines()[-8:]
        for t in tail:
            log(f'  | {t}')
    return r.returncode == 0


def variant_ckpts(name):
    d = OUT / name
    if not d.is_dir():
        return []
    return sorted([p.name for p in d.glob('lora_s*.pt')],
                  key=lambda n: int(n.replace('lora_s', '').replace('.pt', '')))


def summary_write(st, extra=None):
    lines = ['=== B 全流程汇总 ===', f'时间: {time.strftime("%Y-%m-%d %H:%M:%S")}']
    for k, v in st['results'].items():
        lines.append(f'{k}: {json.dumps(v, ensure_ascii=False)}')
    if extra:
        lines.append('--- choice ---')
        lines.extend(extra)
    (TT / 'b_loop_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    st = load_state()
    stage, vi = st['stage'], st['variant_idx']
    log(f'[boot] stage={stage} variant={VARIANTS[vi][0] if vi < len(VARIANTS) else "END"}')

    while True:
        # ============ TRAIN ============
        if stage == 0 and vi < len(VARIANTS):
            name, wc = VARIANTS[vi]
            if st['results'].get(f'{name}_trained'):
                vi += 1
                st['variant_idx'] = vi
                save_state(st)
                continue
            ckpts = variant_ckpts(name)
            if ckpts and variant_ckpts(name)[-1] == f'lora_s{STEPS}.pt':
                log(f'[skip] {name} 最终 ckpt 已存在')
                st['results'][f'{name}_trained'] = True
                vi += 1
                st['variant_idx'] = vi
                save_state(st)
                continue
            log(f'[train] {name} (w_clip={wc}, {STEPS} 步, 每 {SAVE_EVERY} 步存盘)')
            ok = run([PY, str(TT / 'train_vosr_lora.py'), '--variant', name,
                      '--w-clip', str(wc), '--steps', str(STEPS), '--save-every', str(SAVE_EVERY)],
                     TT, OUT / name / 'train.log', f'train_{name}')
            if not ok:
                log(f'[warn] {name} 训练退出码非 0; 已存 ckpt: {variant_ckpts(name)}')
            ckpts = variant_ckpts(name)
            if not ckpts or variant_ckpts(name)[-1] != f'lora_s{STEPS}.pt':
                log(f'[err] {name} 无最终 ckpt -> 唤醒代理')
                summary_write(st)
                save_state(st)
                sys.exit(2)
            st['results'][f'{name}_trained'] = True
            vi += 1
            st['variant_idx'] = vi
            save_state(st)
            continue
        if stage == 0:
            stage = 1
            st['stage'] = 1
            st['variant_idx'] = 0   # 关键: 进入 VALGATE 前重置变体索引
            save_state(st)

        # ============ VALGATE (全部 ckpt) ============
        if stage == 1 and vi < len(VARIANTS):
            name, wc = VARIANTS[vi]
            gates = st['results'].get(f'{name}_gates', {})
            ckpts = variant_ckpts(name)
            pending = [c for c in ckpts if c not in gates]
            for ck in pending:
                val_out = OUT / name / f'val_out_{ck[:-3]}'
                gj = OUT / name / f'val_gate_{ck[:-3]}.json'
                log(f'[valgate] {name}/{ck}')
                ok = run([PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(OUT / name / ck),
                          '-i', str(VAL_IN), '-o', str(val_out), '--align_method', 'nofix', '--seed', '42'],
                         TT, OUT / name / 'val_render.log', f'val_render_{name}_{ck}')
                if ok:
                    run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
                        TT, OUT / name / 'val_gate_run.log', f'val_gate_{name}_{ck}')
                gates[ck] = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {'clip': 0, 'lpips': 9, 'pass': False}
                st['results'][f'{name}_gates'] = gates
                save_state(st)
                log(f'[valgate] {name}/{ck}: clip={gates[ck]["clip"]} lpips={gates[ck]["lpips"]} pass={gates[ck]["pass"]}')
            vi += 1
            st['variant_idx'] = vi
            save_state(st)
            continue
        if stage == 1:
            stage = 2
            st['stage'] = 2
            st['variant_idx'] = 0
            save_state(st)

        # ============ RENDER (每变体择优 1-2 个) ============
        if stage == 2 and vi < len(VARIANTS):
            name, wc = VARIANTS[vi]
            gates = st['results'].get(f'{name}_gates', {})
            ok_ck = sorted([c for c, g in gates.items() if g.get('pass')],
                           key=lambda c: -gates[c]['clip'])
            if not ok_ck:
                log(f'[skip] {name} 全部 ckpt 未过门控')
                vi += 1
                st['variant_idx'] = vi
                save_state(st)
                continue
            chosen = ok_ck[:MAX_RENDER_CKPT]
            st['results'][f'{name}_best'] = chosen
            save_state(st)
            rendered = st['results'].get(f'{name}_rendered', {})
            for ck in chosen:
                if rendered.get(ck):
                    continue
                test_out = RENDER_OUT / f'vosr_lora_{name}_{ck[:-3]}'
                if test_out.exists() and len(list(test_out.glob('*.png'))) >= 100:
                    log(f'[skip] {name}/{ck} 渲染已存在')
                    rendered[ck] = True
                    st['results'][f'{name}_rendered'] = rendered
                    save_state(st)
                    continue
                test_out.mkdir(parents=True, exist_ok=True)
                log(f'[render] {name}/{ck} 测试集 100 张 (~1.6h)')
                ok = run([PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(OUT / name / ck),
                          '-i', TEST_DIR, '-o', str(test_out), '--align_method', 'nofix', '--seed', '42'],
                         TT, OUT / name / 'test_render.log', f'test_render_{name}_{ck}')
                if not ok or len(list(test_out.glob('*.png'))) < 100:
                    log(f'[err] {name}/{ck} 渲染失败 -> 唤醒代理')
                    summary_write(st)
                    save_state(st)
                    sys.exit(3)
                rendered[ck] = True
                st['results'][f'{name}_rendered'] = rendered
                save_state(st)
            vi += 1
            st['variant_idx'] = vi
            save_state(st)
            continue
        if stage == 2:
            stage = 3
            st['stage'] = 3
            st['variant_idx'] = 0
            save_state(st)

        # ============ POOL ============
        if stage == 3 and vi < len(VARIANTS):
            name, wc = VARIANTS[vi]
            chosen = st['results'].get(f'{name}_best', [])
            pooled = st['results'].get(f'{name}_pooled', {})
            for ck in chosen:
                if pooled.get(ck):
                    continue
                src = RENDER_OUT / f'vosr_lora_{name}_{ck[:-3]}'
                pname = f'vosr_lora_{name}_{ck[:-3]}'
                log(f'[pool] {pname}')
                ok = run([PY, str(TT / 'pool_add_source.py'), '--src', str(src),
                          '--name', pname, '--val-src', str(OUT / name / f'val_out_{ck[:-3]}')],
                         TT, TT / f'b_pool_{pname}.log', f'pool_{pname}')
                if not ok:
                    log(f'[err] {pname} 入池失败 -> 唤醒代理')
                    summary_write(st)
                    save_state(st)
                    sys.exit(4)
                pooled[ck] = True
                st['results'][f'{name}_pooled'] = pooled
                save_state(st)
            vi += 1
            st['variant_idx'] = vi
            save_state(st)
            continue
        if stage == 3:
            stage = 4
            st['stage'] = 4
            save_state(st)

        # ============ DONE ============
        run([PY, str(TT / 'compute_choice.py')], TT, TT / 'b_choice.log', 'choice_final')
        extra = []
        try:
            sr = json.loads(Path(r'D:/models/sweep_test_out/sweep_results.json').read_text(encoding='utf-8'))
            for r in sr['results']:
                if r['name'].startswith('vosr_lora'):
                    extra.append(f"{r['name']}: test_clipiqa={r['test_clipiqa']}")
            cm = json.loads(Path(r'D:/tmp_timing/choice_maps.json').read_text(encoding='utf-8'))
            for m in ('M0.000', 'M0.005', 'M0.010'):
                mm = cm['maps'].get(m)
                if mm:
                    extra.append(f'{m}: mean={mm["mean"]} delta={mm["delta"]} swaps={mm["n_swap"]}')
        except Exception as e:
            extra.append(f'[warn] 汇总失败: {e}')
        summary_write(st, extra)
        log('[done] B 全流程完成, 唤醒代理')
        save_state(st)
        sys.exit(0)


if __name__ == '__main__':
    main()
