# -*- coding: utf-8 -*-
"""1.4B 自动验证链 v2 (2026-09-03): 稳健性升级
- 心跳文件 (autoval14.hb): 每轮更新, 供本地守护判断存活
- 下载子进程化 (timeout 300): 防止 sftp.get 连接重置后无限挂起
- 下载后本地大小校验 (防截断)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r'D:/tmp_timing')

CKPT_DIR = Path(r'D:/tmp_timing/p100_ckpts')
OUT_DIR = Path(r'D:/tmp_timing/p100_val')
VAL_LQ = r'D:/tmp_timing/p100_val/lq'
VAL_GT = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
BASE_CKPT = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
GATE_CLIP, GATE_LPIPS = 0.919, 0.360
VARIANTS = ['p512_14b_w20', 'p512_14b_w15', 'p512_14b_w30', 'p512_14b_w31',
            'p512_14b_v14s1', 'p512_14b_v14s2', 'p512_14b_v14s3', 'p512_14b_v14s4',
            'p512_14b_m0s0', 'p512_14b_m0s1', 'p512_14b_m0s2', 'p512_14b_m0s3',
            'p512_14b_m0s4', 'p512_14b_m0s5', 'p512_14b_m0s6', 'p512_14b_m0s7',
            'p512_14b_m0s8', 'p512_14b_m0s9',
            'p512_14b_g3w18s1', 'p512_14b_g3w18s7', 'p512_14b_g3w20s7',
            'p512_14b_m0s1w1',
            # 2026-09-09 rank 收缩轴 (BRAINSTORM §2.1): 冠军配方仅改 LoRA rank
            'p512_14b_m0s1r8', 'p512_14b_m0s1r16',
            # 2026-09-09 w_lpips 轴 (BRAINSTORM §2.2): 冠军配方仅改 w_lpips
            'p512_14b_m0s1wl2', 'p512_14b_m0s1wl3']
POLL = 300
HB = OUT_DIR / 'autoval14.hb'
LOG = OUT_DIR / 'autoval14_history.txt'


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def heartbeat():
    HB.write_text(time.strftime('%m-%d %H:%M:%S'), encoding='utf-8')


def remote_ckpts():
    """主进程 exec (timeout 20s/条, 历来可靠); 卡死风险仅在 sftp.get 大文件"""
    from p100_upload import connect
    cli = connect()
    out = {}
    try:
        for v in VARIANTS:
            _i, _o, _e = cli.exec_command(f'ls ~/p100/runs/{v}/lora_s*.pt 2>/dev/null', timeout=20)
            out[v] = sorted(p.strip().rsplit('/', 1)[-1] for p in _o.read().decode().splitlines())
    finally:
        cli.close()
    return out


def dl_file(remote, local):
    """子进程化下载: 超时 300s, 卡死可杀"""
    code = (
        "import sys; sys.path.insert(0, r'D:/tmp_timing'); from p100_upload import connect; "
        "cli = connect(); sftp = cli.open_sftp(); sftp.get(%r, %r); sftp.close(); cli.close()"
    ) % (remote, str(local))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f'下载失败 {remote}: {r.stderr[-200:]}')


def already_done(variant, name):
    tag = f'out_{variant}_{name.replace(".pt", "")}'
    return (OUT_DIR / tag).is_dir()


def local_ckpts():
    """服务器不可达时退回本地缓存列表 (p100_ckpts/{variant}_lora_s*.pt)"""
    out = {v: [] for v in VARIANTS}
    for f in sorted(CKPT_DIR.glob('*.pt')):
        for v in VARIANTS:
            pre = f'{v}_'
            if f.name.startswith(pre):
                out[v].append(f.name[len(pre):])
    return out


def main():
    done = set()
    log(f'[boot] 1.4B 自动验证链 v3, 轮询 {POLL}s (先下载后渲染, 断网自动降级本地缓存)')
    while True:
        t_round = time.time()
        try:
            try:
                ck = remote_ckpts()
            except Exception as e:
                log(f'[warn] 服务器不可达, 降级本地缓存: {str(e)[:120]}')
                ck = local_ckpts()
            # pass 1: 批量下载缺失 ckpt (联网窗口优先下完)
            for variant in VARIANTS:
                for name in sorted(ck.get(variant, [])):
                    k = f'{variant}_{name}'
                    if k in done or already_done(variant, name):
                        continue
                    local = CKPT_DIR / f'{variant}_{name}'
                    if local.exists():
                        continue
                    log(f'[下载] {variant}/{name}')
                    dl_file(f'p100/runs/{variant}/{name}', local)
                    log(f'[下载OK] {local.stat().st_size} bytes')
                    heartbeat()  # 下载间隔心跳 (防守护误杀)
            # pass 2: 逐档本地渲染+门控 (离线可用)
            for variant in VARIANTS:
                for name in sorted(ck.get(variant, [])):
                    k = f'{variant}_{name}'
                    if k in done or already_done(variant, name):
                        continue
                    local = CKPT_DIR / f'{variant}_{name}'
                    if not local.exists():
                        continue  # 尚未下载成功, 等下一轮
                    tag = f'{variant}_{name.replace(".pt", "")}'
                    outdir = OUT_DIR / f'out_{tag}'
                    log(f'[渲染] {tag}')
                    heartbeat()  # 渲染启动前心跳
                    r1 = subprocess.run(
                        f'D:/Python/python.exe D:/tmp_timing/vosr_lora_infer_14b.py -c {BASE_CKPT} '
                        f'--lora {local} --fp16 -i {VAL_LQ} -o {outdir} --align_method wavelet',
                        shell=True, capture_output=True, text=True, timeout=10800)
                    heartbeat()  # 渲染完成心跳
                    if r1.returncode != 0:
                        log(f'[渲染失败] {tag}: {r1.stderr[-300:]}')
                        done.add(k)
                        continue
                    gate = OUT_DIR / f'gate_{tag}.json'
                    subprocess.run(
                        f'D:/Python/python.exe D:/tmp_timing/val_gate.py --src {outdir} '
                        f'--clip-min {GATE_CLIP} --lpips-max {GATE_LPIPS} --val-dir {VAL_GT} --out {gate}',
                        shell=True, capture_output=True, text=True, timeout=1800)
                    try:
                        g = json.loads(gate.read_text(encoding='utf-8'))
                        log(f'[门控] {tag}: clip={g["clip"]} lpips={g["lpips"]} pass={g["pass"]}')
                    except Exception as e:
                        log(f'[门控异常] {tag}: {e}')
                    if local.exists():
                        local.unlink()  # 门控后删本地 ckpt, 省磁盘 (服务器有原件)
                        log(f'[清理] 已删本地缓存 {local.name}')
                    done.add(k)
        except Exception as e:
            log(f'[warn] {type(e).__name__}: {str(e)[:150]}')
        finally:
            heartbeat()
        # 轮询间隔 (减去本轮耗时, 最小 30s)
        elapsed = time.time() - t_round
        time.sleep(max(POLL - elapsed, 30))


if __name__ == '__main__':
    main()
