# -*- coding: utf-8 -*-
"""M21 自动链 (2026-08-21): 等 v6 训练完成 -> val wavelet 门控(v6 5 ckpt + sc0.9)
-> 线性模型打分 f = 938*clip_test_est - 1110*lpips_val -> 选点 -> 渲染测试集 -> 入池 -> 打包 M21
标定: M18(0.8719,0.405)->198.07 / M19(0.9263,0.415)->237.97 / M20(0.9053,0.3867)->249.70
    -> c=938 clip, w=-1110 lpips (test clip + val-wavelet lpips 口径)
规则: v6 需领先 sc0.9 >= 3 分才选 v6, 否则 sc0.9 (保底方向, 转移比已实证)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
PY = r'D:/Python/python.exe'
TT = Path(r'D:/tmp_timing')
OUT = Path(r'D:/models/vosr_lora')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
VAL_IN = TT / 'codiff_val_in'
TEST_DIR = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'
LOG = TT / 'b_m21_auto.log'
DECISION = TT / 'b_m21_decision.txt'

C_CLIP, W_LP = 938.0, -1110.0  # 线性模型系数
TRANSFER = 0.976  # val(nofix)->test 转移比 (sc0.8 实测 0.9764)
SC09_GATE_EST = (0.9365, 0.3954)  # sc0.9 val wavelet 预估 (nofix 0.9415/0.4004 - 0.005)


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(args, logf):
    with logf.open('ab') as f:
        return subprocess.run(args, stdout=f, stderr=subprocess.STDOUT).returncode


def render_val(ckpt, scale, out_dir, logf, tag):
    if out_dir.exists() and len(list(out_dir.glob('*.png'))) >= 3:
        log(f'[skip] {tag} val 渲染已存在')
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(ckpt), '-i', str(VAL_IN),
            '-o', str(out_dir), '--align_method', 'wavelet', '--seed', '42']
    if scale is not None:
        args += ['--lora-scale', str(scale)]
    r = run(args, logf)
    ok = r == 0 and len(list(out_dir.glob('*.png'))) >= 3
    log(f'[val] {tag} render exit={r} ok={ok}')
    return ok


def gate(out_dir, gj):
    r = subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(out_dir), '--out', str(gj)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if gj.exists():
        g = json.loads(gj.read_text(encoding='utf-8'))
        log(f'[gate] {gj.stem}: clip={g["clip"]} lpips={g["lpips"]}')
        return g
    log(f'[warn] gate 缺失: {gj}')
    return None


def score(test_clip, lpips):
    """预估官方分 (相对 M20=249.70 基准, 用差值): f = 938*clip - 1110*lpips"""
    return C_CLIP * test_clip + W_LP * lpips


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    log('[boot] M21 自动链启动')

    # 1) 等 v6 训练完成 (最多 9h)
    train_log = OUT / 'v6' / 'train.log'
    t0 = time.time()
    while True:
        if train_log.exists() and '[done]' in train_log.read_text(encoding='utf-8', errors='ignore'):
            log('[step1] v6 训练完成')
            break
        if time.time() - t0 > 9 * 3600:
            log('[warn] v6 训练 9h 超时, 用已存 ckpt 继续')
            break
        time.sleep(60)

    # 2) val wavelet 门控: v6 5 ckpt + sc0.9
    log('[step2] val wavelet 门控')
    cands = []  # (tag, ckpt_path, scale, gate)
    for s in (1000, 2000, 3000, 4000, 5000):
        ck = OUT / 'v6' / f'lora_s{s}.pt'
        if not ck.exists():
            log(f'[warn] {ck.name} 不存在, 跳过')
            continue
        vout = OUT / 'v6' / f'val_out_v6_s{s}_wavelet'
        gj = OUT / 'v6' / f'val_gate_v6_s{s}_wavelet.json'
        if render_val(ck, None, vout, TT / 'b_m21_val_render.log', f'v6_s{s}'):
            g = gate(vout, gj)
            if g:
                cands.append((f'v6_s{s}', ck, None, g))
    vout9 = OUT / 'v5' / 'val_out_v5_s4000_sc0.9_wavelet'
    gj9 = OUT / 'v5' / 'val_gate_v5_s4000_sc0.9_wavelet.json'
    if render_val(OUT / 'v5' / 'v5_s4000.pt', 0.9, vout9, TT / 'b_m21_val_render.log', 'v5_s4000_sc0.9'):
        g9 = gate(vout9, gj9)
        if g9:
            cands.append(('v5_s4000_sc0.9', OUT / 'v5' / 'v5_s4000.pt', 0.9, g9))

    # 3) 打分决策
    log('[step3] 打分决策')
    base = score(0.9053, 0.3867)  # M20 基准
    scored = []
    for tag, ck, scale, g in cands:
        test_est = g['clip'] * TRANSFER if tag != 'v5_s4000_sc0.9' else g['clip'] * TRANSFER
        s = score(test_est, g['lpips'])
        scored.append((s, tag, ck, scale, g, test_est))
        log(f'[score] {tag}: val_clip={g["clip"]} lpips={g["lpips"]} test_est={test_est:.4f} score={s:.1f} (M20基准 {base:.1f}, 相对 {s-base:+.1f})')
    if not scored:
        log('[err] 无候选, 退出')
        sys.exit(2)
    scored.sort(reverse=True)
    best_s, best_tag, best_ck, best_scale, _, _ = scored[0]
    sc09 = next((x for x in scored if x[1] == 'v5_s4000_sc0.9'), None)
    if sc09 and best_tag.startswith('v6') and (best_s - sc09[0]) < 3.0:
        log(f'[decision] v6 领先不足 3 分 ({best_s-sc09[0]:+.1f}), 保守选 sc0.9')
        best_s, best_tag, best_ck, best_scale = sc09[0], sc09[1], sc09[2], sc09[3]
    else:
        log(f'[decision] 选 {best_tag} (score={best_s:.1f})')
    DECISION.write_text(json.dumps({
        'cands': [{'tag': t, 'val_clip': g['clip'], 'lpips': g['lpips'], 'test_est': round(te, 4),
                   'score': round(s, 1)} for s, t, _, _, g, te in scored],
        'best': best_tag, 'best_score': round(best_s, 1),
        'M20_base': round(base, 1), 'rel': round(best_s - base, 1),
    }, indent=1, ensure_ascii=False), encoding='utf-8')

    # 4) 渲染胜者测试集
    log(f'[step4] 渲染 {best_tag} 测试集 100 张 (~1.6h)')
    test_out = OUT / (('v6' if best_tag.startswith('v6') else 'v5')) / f'test_out_{best_tag}'
    if not (test_out.exists() and len(list(test_out.glob('*.png'))) >= 100):
        test_out.mkdir(parents=True, exist_ok=True)
        args = [PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(best_ck),
                '-i', TEST_DIR, '-o', str(test_out), '--align_method', 'wavelet', '--seed', '42']
        if best_scale is not None:
            args += ['--lora-scale', str(best_scale)]
        r = run(args, TT / 'b_m21_test_render.log')
        if r != 0 or len(list(test_out.glob('*.png'))) < 100:
            log('[err] 测试集渲染失败')
            sys.exit(3)
    log(f'[render] {best_tag} 完成')

    # 5) 入池 + choice
    pool_name = f'vosr_lora_{best_tag}'
    run([PY, str(TT / 'pool_add_source.py'), '--src', str(test_out), '--name', pool_name],
        TT / 'b_m21_pool.log')
    run([PY, str(TT / 'compute_choice.py')], TT / 'b_m21_choice.log')
    log('[pool] 入池+choice 完成')

    # 6) 打包 M21
    run([PY, str(TT / 'package_submit_v2.py'), '--out-dir', str(Path(r'D:/models/sweep_test_out') / pool_name),
         '--model-dir', r'D:/models/final_graph7/model_dir', '--name', 'MetricMixS7M21', '--account', 'Remay'],
        TT / 'b_m21_package.log')
    log('[done] M21 自动链完成')
    sys.exit(0)


if __name__ == '__main__':
    main()
