# -*- coding: utf-8 -*-
"""长线自动管线 v1 (2026-08-17 深夜): 种子收尾 -> 入池 -> choice 重算 -> CODiff 评估 -> 入池 -> choice 重算 -> 唤醒
用法: D:/Python/python.exe auto_pipeline.py            (全新启动)
      D:/Python/python.exe auto_pipeline.py --resume   (断点续跑)
状态机: 每阶段完成写 auto_pipeline_state.json; 全程日志 auto_pipeline.log; 结束写 auto_pipeline_summary.txt
所有 GPU 阶段串行 (单卡 4060)。进程结束时 = 唤醒宿主代理接管分析。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
TT = Path(r'D:/tmp_timing')
LOG = TT / 'auto_pipeline.log'
STATE = TT / 'auto_pipeline_state.json'
SUMMARY = TT / 'auto_pipeline_summary.txt'

SEED_RUNNER_PID = 58316
SEED_LOG = TT / 'seed_runner3.log'
SEEDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
SEED_SRC = Path(r'C:/models/finals/preset')          # vosr_seed{seed}_test

VAL_IN = TT / 'codiff_val_in'
REPO = Path(r'D:/models/finals/codiff/CODiff-main/CODiff-main')
VAL_ADAIN = Path(r'D:/models/codiff_out_val_adain')
VAL_NOFIX = Path(r'D:/models/codiff_out_val_nofix')
FULL_OUT = Path(r'D:/models/codiff_out_test')

SWEEP_JSON = Path(r'D:/models/sweep_test_out/sweep_results.json')
CHOICE_JSON = TT / 'choice_maps.json'
CODIFF_CLOSED = TT / 'CODIFF_CLOSED'   # 存在则跳过阶段 3-6 (CODiff 评估, 2026-08-18 已关闭)

STAGES = ['WAIT_SEEDS', 'POOL_SEEDS', 'CHOICE1', 'CODIFF_VAL', 'GATE',
          'CODIFF_FULL', 'POOL_CODIFF', 'CHOICE2', 'DONE']


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
            log(f'[warn] state 读取失败: {e}, 从 0 开始')
    return {'stage': 0, 'results': {}}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding='utf-8')


def pid_alive(pid):
    """Windows 进程存活检测 (ctypes 正确句柄类型; os.kill 在 Windows 不可靠)"""
    import ctypes
    import ctypes.wintypes
    if pid <= 0:
        return False
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD]
    kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
    h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    kernel32.CloseHandle(h)
    return True


def seed_dir_ok(seed):
    d = SEED_SRC / f'vosr_seed{seed}_test'
    return d.is_dir() and len(list(d.glob('*.png'))) >= 100


def run_inline(cmd, cwd, logfile, tag):
    log(f'[run] {tag}: {" ".join(cmd[:4])}...')
    t0 = time.time()
    with logfile.open('ab') as f:
        r = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    log(f'[run] {tag}: exit={r.returncode} ({time.time() - t0:.0f}s)')
    if r.returncode != 0:
        tail = logfile.read_text(encoding='utf-8', errors='ignore').strip().splitlines()[-5:]
        for t in tail:
            log(f'  | {t}')
    return r.returncode == 0


def run_detached(cmd, cwd, logfile, tag):
    """脱离会话启动 (防托管 job 意外死亡), 返回 pid"""
    log(f'[detach] {tag}')
    with logfile.open('ab') as f:
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    log(f'[detach] {tag}: pid={p.pid}')
    return p.pid


def wait_pid(pid, tag, timeout_s, logfile, ok_marker=None):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not pid_alive(pid):
            txt = logfile.read_text(encoding='utf-8', errors='ignore')
            ok = (ok_marker is None) or (ok_marker in txt)
            log(f'[wait] {tag}: 进程结束 ok={ok}')
            return ok
        time.sleep(30)
    log(f'[wait] {tag}: 超时 {timeout_s}s')
    return False


def summary_write(st, extra=None):
    lines = ['=== auto_pipeline 汇总 ===', f'时间: {time.strftime("%Y-%m-%d %H:%M:%S")}',
             f'阶段: {STAGES[st["stage"]]}', '--- results ---']
    for k, v in st.get('results', {}).items():
        lines.append(f'{k}: {v}')
    if extra:
        lines.append('--- extra ---')
        lines.extend(extra)
    SUMMARY.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log('[summary] 已写入 auto_pipeline_summary.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    st = load_state()
    if not args.resume and st['stage'] > 0:
        log('[warn] 检测到已有状态, 自动进入 --resume 模式')
    stage = st['stage']
    if CODIFF_CLOSED.exists() and stage in (3, 4, 5, 6):
        log('[skip] CODiff 已关闭 (CODIFF_CLOSED), 跳过阶段 3-6, 跳至 CHOICE2')
        st['stage'] = 7
        stage = 7
        save_state(st)
    log(f'[boot] 从阶段 {stage} ({STAGES[stage]}) 继续')

    # 启动安全备份
    if stage <= 1:
        shutil.copy2(SWEEP_JSON, TT / 'sweep_results.pre_auto.json') if SWEEP_JSON.exists() else None
    if stage <= 2:
        if CHOICE_JSON.exists():
            shutil.copy2(CHOICE_JSON, TT / 'choice_maps.pre_auto.json')

    # ── 0 WAIT_SEEDS ──
    if stage == 0:
        log('[stage 0] 等待种子渲染完成 (含自动续跑)...')
        active_runner = SEED_RUNNER_PID
        heal_pid = st['results'].get('heal_pid', 0)
        if heal_pid and pid_alive(heal_pid):
            active_runner = heal_pid
            log(f'[stage 0] 采用续跑实例 pid={heal_pid}')
        heals = st['results'].get('heals', 0)
        t0 = time.time()
        while time.time() - t0 < 20 * 3600:
            if all(seed_dir_ok(s) for s in SEEDS):
                log('[stage 0] 全部 10 种子渲染完成')
                break
            if not pid_alive(active_runner):
                missing = [s for s in SEEDS if not seed_dir_ok(s)]
                log(f'[stage 0] runner(pid={active_runner}) 死亡, 缺失种子: {missing}')
                if heals >= 2:
                    log(f'[stage 0] [WARN] 续跑已重试 {heals} 次仍失败 -> 唤醒代理')
                    st['results']['seed_fail'] = str(missing)
                    summary_write(st)
                    save_state(st)
                    sys.exit(2)
                heals += 1
                st['results']['heals'] = heals
                st['results']['heal_missing'] = str(missing)
                save_state(st)
                # 幂等续跑: seed_runner4 会跳过已完成的种子
                logfile = TT / 'auto_heal_runner.log'
                with logfile.open('ab') as f:
                    p = subprocess.Popen(
                        [PY, str(TT / 'seed_runner4.py')], cwd=str(TT),
                        stdout=f, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                active_runner = p.pid
                st['results']['heal_pid'] = active_runner
                save_state(st)
                log(f'[stage 0] 已重启续跑: pid={active_runner} (第 {heals} 次)')
            time.sleep(60)
        else:
            log('[stage 0] [WARN] 等待超时 20h -> 唤醒代理')
            summary_write(st)
            save_state(st)
            sys.exit(3)
        st['stage'] = 1
        save_state(st)

    # ── 1 POOL_SEEDS ──
    if stage <= 1:
        log('[stage 1] 种子入池 (vosr_seeds_to_pool2)')
        ok = run_inline([PY, str(TT / 'vosr_seeds_to_pool2.py'), '--skip-existing'], TT,
                        TT / 'auto_pool_seeds.log', 'pool_seeds')
        st['results']['pool_seeds'] = 'ok' if ok else 'FAIL'
        if not ok:
            summary_write(st)
            save_state(st)
            sys.exit(4)
        st['stage'] = 2
        save_state(st)

    # ── 2 CHOICE1 ──
    if stage <= 2:
        log('[stage 2] choice 重算 (compute_choice)')
        ok = run_inline([PY, str(TT / 'compute_choice.py')], TT, TT / 'auto_choice1.log', 'choice1')
        if not ok:
            summary_write(st)
            save_state(st)
            sys.exit(5)
        st['stage'] = 3
        save_state(st)

    # ── 3 CODIFF_VAL ──
    if stage <= 3:
        log('[stage 3] CODiff val 3 张 (adain + nofix)')
        if not VAL_IN.is_dir() or len(list(VAL_IN.glob('*.jpg'))) < 3:
            log(f'[err] val 输入目录缺失: {VAL_IN}')
            summary_write(st)
            save_state(st)
            sys.exit(6)
        for mode, out in (('adain', VAL_ADAIN), ('nofix', VAL_NOFIX)):
            out.mkdir(parents=True, exist_ok=True)
            if len(list(out.glob('*.png'))) >= 3:
                log(f'[stage 3] {mode} 已有输出, 跳过')
                continue
            cmd = [PY, str(TT / 'codiff_infer_4k.py'), '-i', str(VAL_IN), '-o', str(out),
                   '--color-fix', mode, '--limit', '3']
            ok = run_inline(cmd, REPO, TT / f'auto_codiff_val_{mode}.log', f'codiff_val_{mode}')
            if not ok:
                log(f'[err] CODiff val {mode} 失败, 唤醒代理')
                st['results']['codiff_val'] = f'{mode} FAIL'
                summary_write(st)
                save_state(st)
                sys.exit(7)
        st['stage'] = 4
        save_state(st)

    # ── 4 GATE ──
    if stage <= 4:
        log('[stage 4] val 门控判定')
        gates = {}
        for mode, out in (('adain', VAL_ADAIN), ('nofix', VAL_NOFIX)):
            gj = TT / f'val_gate_{mode}.json'
            run_inline([PY, str(TT / 'val_gate.py'), '--src', str(out), '--out', str(gj)],
                       TT, TT / f'auto_val_gate_{mode}.log', f'val_gate_{mode}')
            if gj.exists():
                gates[mode] = json.loads(gj.read_text(encoding='utf-8'))
                log(f'[gate] {mode}: clip={gates[mode]["clip"]} lpips={gates[mode]["lpips"]} pass={gates[mode]["pass"]}')
        st['results']['gate'] = gates
        ok_modes = [m for m, g in gates.items() if g.get('pass')]
        if not ok_modes:
            log('[gate] [WARN] 两种模式均未过门, 唤醒代理人工决策')
            summary_write(st)
            save_state(st)
            sys.exit(8)
        chosen = max(ok_modes, key=lambda m: gates[m]['clip'])
        st['results']['codiff_color_fix'] = chosen
        log(f'[gate] 选定 color-fix={chosen}')
        st['stage'] = 5
        save_state(st)

    # ── 5 CODIFF_FULL ──
    if stage <= 5:
        chosen = st['results'].get('codiff_color_fix', 'nofix')
        log(f'[stage 5] CODiff 全量 100 张 (color-fix={chosen})')
        FULL_OUT.mkdir(parents=True, exist_ok=True)
        done_pngs = len(list(FULL_OUT.glob('case*.png')))
        if done_pngs < 100:
            env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            for attempt, dec_tile in enumerate((512, 224)):
                log(f'[stage 5] 尝试 {attempt + 1}: vae_decoder_tiled_size={dec_tile}')
                cmd = [PY, str(TT / 'codiff_infer_4k.py'), '-i',
                       r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集',
                       '-o', str(FULL_OUT), '--color-fix', chosen, '--vae_decoder_tiled_size', str(dec_tile)]
                logfile = TT / 'auto_codiff_full.log'
                pid = run_detached(cmd, REPO, logfile, 'codiff_full')
                ok = wait_pid(pid, 'codiff_full', 6 * 3600, logfile, ok_marker='[done]')
                if ok and len(list(FULL_OUT.glob('case*.png'))) >= 100:
                    log('[stage 5] CODiff 全量完成')
                    break
                log(f'[stage 5] 尝试 {attempt + 1} 失败 (ok={ok}, pngs={len(list(FULL_OUT.glob("case*.png")))})')
            else:
                log('[err] CODiff 全量两次尝试均失败, 唤醒代理')
                st['results']['codiff_full'] = 'FAIL'
                summary_write(st)
                save_state(st)
                sys.exit(9)
        st['stage'] = 6
        save_state(st)

    # ── 6 POOL_CODIFF ──
    if stage <= 6:
        chosen = st['results'].get('codiff_color_fix', 'nofix')
        val_src = VAL_ADAIN if chosen == 'adain' else VAL_NOFIX
        log('[stage 6] CODiff 入池 (pool_add_source)')
        ok = run_inline([PY, str(TT / 'pool_add_source.py'), '--src', str(FULL_OUT),
                         '--name', 'codiff', '--val-src', str(val_src)],
                        TT, TT / 'auto_pool_codiff.log', 'pool_codiff')
        st['results']['pool_codiff'] = 'ok' if ok else 'FAIL'
        if not ok:
            summary_write(st)
            save_state(st)
            sys.exit(10)
        st['stage'] = 7
        save_state(st)

    # ── 7 CHOICE2 ──
    if stage <= 7:
        log('[stage 7] 最终 choice 重算')
        ok = run_inline([PY, str(TT / 'compute_choice.py')], TT, TT / 'auto_choice2.log', 'choice2')
        if not ok:
            summary_write(st)
            save_state(st)
            sys.exit(11)
        st['stage'] = 8
        save_state(st)

    # ── 8 DONE ──
    extra = []
    try:
        sr = json.loads(SWEEP_JSON.read_text(encoding='utf-8'))
        entries = {r['name']: r for r in sr['results']}
        extra.append(f'池条目数: {len(sr["results"])}')
        if 'codiff' in entries:
            extra.append(f'CODiff test clipiqa: {entries["codiff"]["test_clipiqa"]}')
        for s in SEEDS:
            nm = f'vosr_seed{s}'
            if nm in entries:
                extra.append(f'{nm}: {entries[nm]["test_clipiqa"]}')
        cm = json.loads(CHOICE_JSON.read_text(encoding='utf-8'))
        for m in ('M0.000', 'M0.005', 'M0.010'):
            if m in cm['maps']:
                mm = cm['maps'][m]
                extra.append(f'{m}: mean={mm["mean"]} delta={mm["delta"]} swaps={mm["n_swap"]}')
    except Exception as e:
        extra.append(f'[warn] 汇总读取失败: {e}')
    summary_write(st, extra)
    log('[done] 管线完成, 唤醒代理接管')
    save_state(st)
    sys.exit(0)


if __name__ == '__main__':
    main()
