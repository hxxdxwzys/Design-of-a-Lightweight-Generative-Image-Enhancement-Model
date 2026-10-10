"""E4 门控守护 v2: 串行设计 — 检测到新 checkpoint 时:
  1) 杀死训练进程 (释放显存) → 2) 跑 4K 门槛评测 → 3) 用该 checkpoint 续训重启。
避免评测与训练抢 GPU (8GB 卡放不下两者并行)。
门槛通过 → 写 gate_PASS.txt 并停止训练 (等人工处理)。
用法: python tmp_e4_watch2.py   (用 PowerShell 脱离会话启动)
"""
import subprocess
import sys
import time
import re
from pathlib import Path

CKPT_DIR = Path(r"D:/models/hypir_ft_v2")
WS = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace")
HYPIR_DIR = Path(r"D:/models/hypir_code/HYPIR-main/HYPIR-main")
DONE = WS / "results" / "gate_done.txt"
PASS = WS / "results" / "gate_PASS.txt"
MIN_TAG = 250

done_tags = set()
if DONE.exists():
    done_tags = set(DONE.read_text(encoding="utf-8").split())

print("gate watcher v2 started; watching", CKPT_DIR, flush=True)


def find_train_pid():
    r = subprocess.run(
        ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine", "/format:list"],
        capture_output=True, text=True)
    pids = []
    cur = {}
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
    for pid in find_train_pid():
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        print(f"killed training PID {pid}", flush=True)
    time.sleep(5)


def start_training(resume_ckpt=None):
    cfg = HYPIR_DIR / "configs" / "sd2_train_1x_v2nos.yaml"
    args = ["-u", "train.py", "--config", str(cfg)]
    if resume_ckpt:
        # 临时 resume 配置 (字符串替换, 路径反斜杠安全)
        resume_cfg = HYPIR_DIR / "configs" / "sd2_train_1x_resume_live.yaml"
        lines = []
        for ln in cfg.read_text(encoding="utf-8").splitlines():
            if ln.startswith("resume_from_checkpoint:"):
                lines.append(f"resume_from_checkpoint: {resume_ckpt}")
            else:
                lines.append(ln)
        resume_cfg.write_text("\n".join(lines), encoding="utf-8")
        args = ["-u", "train.py", "--config", str(resume_cfg)]
    ps = ("Start-Process -FilePath 'D:\\Python\\python.exe' -ArgumentList " +
          "'" + "','".join(args) + "'" +
          " -WorkingDirectory '" + str(HYPIR_DIR).replace("'", "''") + "'" +
          " -RedirectStandardOutput 'D:\\models\\hypir\\e4_train_live.log'" +
          " -RedirectStandardError 'D:\\models\\hypir\\e4_train_live.err' -WindowStyle Hidden")
    subprocess.run(["powershell", "-Command", ps], capture_output=True)
    print("training restarted (detached)", flush=True)


while True:
    ckpts = sorted(CKPT_DIR.glob("checkpoint-*")) if CKPT_DIR.exists() else []
    for ck in ckpts:
        tag = ck.name.split("-")[1]
        if tag in done_tags or int(tag) < MIN_TAG:
            continue
        ema = ck / "ema_state_dict.pth"
        if not ema.exists():
            continue  # 还在写入
        done_tags.add(tag)
        with DONE.open("a", encoding="utf-8") as f:
            f.write(tag + "\n")
        print(f"[gate] checkpoint-{tag} complete; pausing training", flush=True)
        kill_training()
        r = subprocess.run(
            [sys.executable, str(WS / "tmp_e4_gate.py"), str(ema), f"e4_{tag}"],
            capture_output=True, text=True)
        print(r.stdout[-1200:], flush=True)
        if r.returncode == 0:
            with PASS.open("a", encoding="utf-8") as f:
                f.write(f"checkpoint-{tag} PASSED gate\n")
            print(f"[gate] checkpoint-{tag} PASSED! 训练停止, 等人工处理 (见 {PASS})", flush=True)
            sys.exit(0)
        print(f"[gate] checkpoint-{tag} 未过门槛 (exit {r.returncode}), 续训", flush=True)
        start_training(str(ck))
        time.sleep(30)
    time.sleep(120)
