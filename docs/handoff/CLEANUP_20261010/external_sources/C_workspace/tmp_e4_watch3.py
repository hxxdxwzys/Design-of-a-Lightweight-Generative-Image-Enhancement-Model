"""E4 门控守护 v3 (修复 v2 缺陷):
  1) 只测"最新完整 checkpoint" (不再补测旧 checkpoint -> 不再回退训练)
  2) 门控子进程 45 分钟超时 (防崩溃/卡死阻塞守护)
  3) 启动前检查重复训练进程
  4) 无训练进程时自愈: 从最新 checkpoint 重启
门槛: lpips < 0.3364 (主判据) 且 musiq > 35 且 clipiqa > 0.42 -> PASS 停训
用法: python tmp_e4_watch3.py (PowerShell 脱离会话启动)
"""
import subprocess
import sys
import time
from pathlib import Path

CKPT_DIR = Path(r"D:/models/hypir_ft_v3")
WS = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace")
HYPIR_DIR = Path(r"D:/models/hypir_code/HYPIR-main/HYPIR-main")
DONE = WS / "results" / "gate_done_v3.txt"
PASS = WS / "results" / "gate_PASS_v3.txt"
MIN_TAG = 250
GATE_TIMEOUT = 2700  # 45 分钟

done_tags = set()
if DONE.exists():
    done_tags = set(DONE.read_text(encoding="utf-8").split())

print("gate watcher v3 started; watching", CKPT_DIR, flush=True)


def complete_ckpts():
    out = []
    if not CKPT_DIR.exists():
        return out
    for ck in CKPT_DIR.glob("checkpoint-*"):
        if (ck / "ema_state_dict.pth").exists():
            try:
                out.append((int(ck.name.split("-")[1]), ck))
            except ValueError:
                pass
    return sorted(out)


def train_pids():
    r = subprocess.run(
        ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine", "/format:list"],
        capture_output=True, text=True)
    pids, cur = [], {}
    for line in r.stdout.splitlines():
        if line.startswith("CommandLine="):
            cur["cmd"] = line.split("=", 1)[1]
        elif line.startswith("ProcessId="):
            cur["pid"] = line.split("=", 1)[1]
            if "train.py" in cur.get("cmd", "") and "watch" not in cur.get("cmd", ""):
                pids.append(cur["pid"])
            cur = {}
    return pids


def kill_training():
    for pid in train_pids():
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        print(f"killed training PID {pid}", flush=True)
    time.sleep(5)


def start_training(resume_ckpt=None):
    if train_pids():
        print("training already running, skip start", flush=True)
        return
    cfg = HYPIR_DIR / "configs" / "sd2_train_1x_v3.yaml"
    args = ["-u", "train.py", "--config", str(cfg)]
    if resume_ckpt:
        resume_cfg = HYPIR_DIR / "configs" / "sd2_train_1x_v3_resume_live.yaml"
        lines = []
        for ln in cfg.read_text(encoding="utf-8").splitlines():
            lines.append(f"resume_from_checkpoint: {resume_ckpt}" if ln.startswith("resume_from_checkpoint:") else ln)
        resume_cfg.write_text("\n".join(lines), encoding="utf-8")
        args = ["-u", "train.py", "--config", str(resume_cfg)]
    ps = ("Start-Process -FilePath 'D:\\Python\\python.exe' -ArgumentList " +
          "'" + "','".join(args) + "'" +
          " -WorkingDirectory '" + str(HYPIR_DIR).replace("'", "''") + "'" +
          " -RedirectStandardOutput 'D:\\models\\hypir\\e4v3_live.log'" +
          " -RedirectStandardError 'D:\\models\\hypir\\e4v3_live.err' -WindowStyle Hidden")
    subprocess.run(["powershell", "-Command", ps], capture_output=True)
    print("training started (detached)", flush=True)


last_newest = None
idle_since = time.time()
while True:
    ckpts = complete_ckpts()
    newest = ckpts[-1] if ckpts else None

    # 1) 门控最新未测 checkpoint
    gated = False
    if newest:
        tag, ck = newest
        if tag >= MIN_TAG and str(tag) not in done_tags:
            print(f"[gate] checkpoint-{tag} (最新) 暂停训练评测", flush=True)
            kill_training()
            done_tags.add(str(tag))
            with DONE.open("a", encoding="utf-8") as f:
                f.write(str(tag) + "\n")
            try:
                r = subprocess.run(
                    [sys.executable, str(WS / "tmp_e4_gate.py"), str(ck / "ema_state_dict.pth"), f"e4_{tag}"],
                    capture_output=True, text=True, timeout=GATE_TIMEOUT)
                print(r.stdout[-1000:], flush=True)
            except subprocess.TimeoutExpired:
                print(f"[gate] checkpoint-{tag} 超时, 跳过", flush=True)
                r = None
            if r is not None and r.returncode == 0:
                with PASS.open("a", encoding="utf-8") as f:
                    f.write(f"checkpoint-{tag} PASSED gate\n")
                print(f"[gate] checkpoint-{tag} PASSED! 训练停止, 等人工处理", flush=True)
                sys.exit(0)
            print(f"[gate] checkpoint-{tag} 未过门槛, 从该点续训", flush=True)
            start_training(str(ck))
            gated = True
            time.sleep(30)

    # 2) 自愈: 无训练进程 -> 从最新 checkpoint 重启
    if not gated:
        if not train_pids():
            if newest and newest[0] >= MIN_TAG:
                if last_newest != newest[0]:
                    last_newest = newest[0]
                    idle_since = time.time()
                if time.time() - idle_since > 90:
                    print("[selfheal] 无训练进程, 从最新 checkpoint 重启", flush=True)
                    start_training(str(newest[1]))
                    idle_since = time.time()
            else:
                if time.time() - idle_since > 90:
                    print("[selfheal] 无训练进程且无 checkpoint, 从头启动", flush=True)
                    start_training(None)
                    idle_since = time.time()
        else:
            idle_since = time.time()
    time.sleep(60)
