# -*- coding: utf-8 -*-
"""1.4B 前沿扫描驱动 (2026-09-01, Phase 1 自动化)

网格: w_clip {0.5,1.0,1.5,2.0} x steps {1000,2000,4000} (+ 预阶段 w_clip=0, steps=1000)
每点链路: train (train_vosr_lora_14b.py, patch1024) -> val 渲染 (vosr_lora_infer_14b.py --fp16)
          -> val_gate (clip>=0.919, lpips<=0.360) -> 状态落盘
决策门: 任一点 pass -> 打印 -> (可选 --stop-on-pass 停机) 进 Phase 2
任何时刻可杀进程; state json 每阶段后落盘, 重启后自动跳过已完成点。

用法 (服务器 /data/proj/tools):
  python b_14b_frontier.py --out /data/proj/frontier --data-dirs /data/datasets/div2k/pngtxt_ori /data/datasets/flickr2k \
      --ckpt /data/weights/VOSR_1.4B_os --qwen-ae /data/weights/Qwen-Image-vae-2d \
      --vosr-dir /data/proj/VOSR --tools-dir /data/proj/tools --val-lq /data/val/lq --python /usr/bin/python \
      --parallel 2 --stop-on-pass
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GRID = [(wc, st) for wc in (0.5, 1.0, 1.5, 2.0) for st in (1000, 2000, 4000)]
PRE_STAGE = (0.0, 1000)  # 纯保真预阶段 (LPIPS+L1, 无 clip 项)
GATE_CLIP = 0.919
GATE_LPIPS = 0.360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='状态/日志/val_out 根目录')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--data-dirs', nargs='+', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--qwen-ae', required=True)
    ap.add_argument('--vosr-dir', required=True, help='VOSR 仓库目录 (渲染 cwd)')
    ap.add_argument('--tools-dir', required=True)
    ap.add_argument('--val-lq', required=True, help='val LQ 目录')
    ap.add_argument('--val-gt', default=None, help='val GT 目录 (val_gate 用, 默认 val_lq 同目录)')
    ap.add_argument('--torch-cache', default=None)
    ap.add_argument('--deg-json', required=True)
    ap.add_argument('--jpeg-json', required=True)
    ap.add_argument('--pause-flag', required=True)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--patch', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--parallel', type=int, default=1, help='并发训练数 (2 卡设 2, 轮询 CUDA_VISIBLE_DEVICES)')
    ap.add_argument('--n-gpus', type=int, default=1)
    ap.add_argument('--include-prestage', action='store_true', help='先跑 w_clip=0 预阶段')
    ap.add_argument('--stop-on-pass', action='store_true', help='任一点过门即停止')
    ap.add_argument('--force-reset', action='store_true',
                    help='重启时把 status=running 的点重置为 pending (先确认无孤儿训练进程!)')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_p = out / 'frontier_state.json'
    state = json.loads(state_p.read_text(encoding='utf-8')) if state_p.exists() else {}

    def log(msg):
        line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        with (out / 'frontier.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    def save_state():
        state_p.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding='utf-8')

    def variant(wc, st):
        wc_s = f'w{int(wc * 10):02d}'.replace('w00', 'w0')
        return f'{wc_s}s{st}'

    def train_cmd(name, wc, st):
        c = [args.python, os.path.join(args.tools_dir, 'train_vosr_lora_14b.py'),
             '--variant', name, '--w-clip', str(wc), '--steps', str(st),
             '--patch', str(args.patch), '--batch', str(args.batch), '--accum', str(args.accum),
             '--save-every', '500', '--lr', str(args.lr),
             '--out', str(out), '--data-dirs'] + args.data_dirs + [
             '--ckpt', args.ckpt, '--qwen-ae', args.qwen_ae,
             '--deg-json', args.deg_json, '--jpeg-json', args.jpeg_json,
             '--pause-flag', args.pause_flag, '--vosr-dir', args.vosr_dir, '--tools-dir', args.tools_dir]
        if args.torch_cache:
            c += ['--torch-cache', args.torch_cache]
        return c

    def render_cmd(name):
        return [args.python, os.path.join(args.tools_dir, 'vosr_lora_infer_14b.py'),
                '-c', args.ckpt, '--lora', str(out / name / f'lora_s{st_of(name)}.pt'),
                '--fp16', '-i', args.val_lq, '-o', str(out / 'val_out' / name),
                '--align_method', 'wavelet']

    def gate_cmd(name):
        c = [args.python, os.path.join(args.tools_dir, 'val_gate.py'),
             '--src', str(out / 'val_out' / name),
             '--clip-min', str(GATE_CLIP), '--lpips-max', str(GATE_LPIPS),
             '--out', str(out / f'gate_{name}.json')]
        if args.val_gt:
            c += ['--val-dir', args.val_gt]
        return c

    def st_of(name):
        return int(name.split('s')[-1])

    grid = ([PRE_STAGE] if args.include_prestage else []) + GRID
    name2cfg = {variant(*g): g for g in grid}
    log(f'[boot] 网格 {len(grid)} 点: {[variant(*g) for g in grid]} parallel={args.parallel} '
        f'patch={args.patch} batch={args.batch} gate=({GATE_CLIP},{GATE_LPIPS})')

    # 断点续跑: fail/render_fail 自动重试; running 需 --force-reset (防孤儿训练双实例)
    for name, st_ in list(state.items()):
        if st_.get('status') in ('fail', 'render_fail'):
            state[name]['status'] = 'pending'
            log(f'[reset] {name} ({st_["status"]}) -> pending (自动重试)')
        elif st_.get('status') == 'running' and args.force_reset:
            state[name]['status'] = 'pending'
            log(f'[reset] {name} (running, --force-reset) -> pending')

    pending = [name for name in name2cfg if name not in state or state[name].get('status') == 'pending']
    for name in pending:
        state[name] = {'status': 'pending'}

    # GPU 轮询分配
    inflight = {}  # name -> (Popen, logfile, gpu)

    def launch(name):
        gpu = len(inflight) % args.parallel
        env = dict(os.environ)
        if args.parallel > 1:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu % args.n_gpus)
        lg = (out / f'train_{name}.log').open('w', encoding='utf-8')
        p = subprocess.Popen(train_cmd(name, *name2cfg[name]),
                             stdout=lg, stderr=subprocess.STDOUT, env=env)
        state[name]['status'] = 'running'
        state[name]['start'] = time.strftime('%m-%d %H:%M:%S')
        state[name]['pid'] = p.pid
        inflight[name] = (p, lg, gpu)
        save_state()
        log(f'[launch] {name} (gpu={gpu}, pid={p.pid})')

    # 启动首批
    for name in list(state):
        if state[name]['status'] == 'pending' and len(inflight) < args.parallel:
            launch(name)

    while inflight:
        done = []
        for name, (p, lg, gpu) in inflight.items():
            rc = p.poll()
            if rc is not None:
                lg.close()
                done.append(name)
        for name in done:
            p, lg, gpu = inflight.pop(name)
            if p.returncode != 0:
                state[name]['status'] = 'fail'
                state[name]['rc'] = p.returncode
                log(f'[fail] {name} train rc={p.returncode} (log: train_{name}.log)')
                save_state()
                continue
            log(f'[train-done] {name} -> val 渲染 (fp16)')
            # 渲染 + 门控 (串行, 用训练同卡)
            env = dict(os.environ)
            if args.parallel > 1:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu % args.n_gpus)
            r = subprocess.run(render_cmd(name), cwd=args.vosr_dir, env=env,
                               capture_output=True, text=True, timeout=7200)
            if r.returncode != 0:
                state[name]['status'] = 'render_fail'
                state[name]['render_err'] = r.stderr[-2000:]
                log(f'[render-fail] {name} rc={r.returncode}')
                save_state()
                continue
            g = subprocess.run(gate_cmd(name), capture_output=True, text=True, timeout=1800)
            try:
                gate = json.loads((out / f'gate_{name}.json').read_text(encoding='utf-8'))
            except Exception:
                gate = {'clip': -1, 'lpips': 99, 'pass': False, 'err': g.stderr[-2000:]}
            state[name]['status'] = 'done'
            state[name]['gate'] = {k: gate.get(k) for k in ('clip', 'lpips', 'pass', 'per')}
            log(f'[gate] {name}: clip={gate.get("clip")} lpips={gate.get("lpips")} pass={gate.get("pass")}')
            save_state()
            if gate.get('pass') and args.stop_on_pass:
                log(f'[DECISION] {name} 过门 (clip>={GATE_CLIP}, lpips<={GATE_LPIPS}) -> Phase 2')
                for n2, (p2, lg2, _) in inflight.items():
                    p2.kill()
                    lg2.close()
                return
            # 释放一个槽位, 补下一个 pending
            for nxt in list(state):
                if state[nxt]['status'] == 'pending' and len(inflight) < args.parallel:
                    launch(nxt)
        time.sleep(15)

    # 汇总
    log('=== 扫描汇总 ===')
    best = None
    for name in sorted(state):
        g = state[name].get('gate') or {}
        log(f'{name}: {state[name]["status"]} clip={g.get("clip")} lpips={g.get("lpips")} pass={g.get("pass")}')
        if g.get('clip') is not None and g.get('lpips') is not None:
            if best is None or g['lpips'] < best[2]:
                best = (name, g['clip'], g['lpips'])
    if best:
        log(f'[best] {best[0]} (clip={best[1]}, lpips={best[2]}) — 对照 v14(patch512): (0.9172, 0.3845)')
    log('[done]')


if __name__ == '__main__':
    main()
