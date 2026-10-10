"""E4 训练门控守护: 轮询 checkpoint 目录, 新 checkpoint 自动跑 4K 门槛评测。

用法: python tmp_e4_watch.py  (后台运行)
门槛: lpips < 0.3364 且 clipiqa >= 0.44 且 musiq >= 36.8 (tmp_e4_gate.py 内判断)
通过门槛的 checkpoint 会被记录到 results/gate_PASS.txt
"""
import subprocess
import sys
import time
from pathlib import Path

CKPT_DIR = Path(r"D:/models/hypir_ft")
MIN_TAG = 250  # 跳过早期写入中的 checkpoint
WS = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace")
DONE = WS / "results" / "gate_done.txt"
PASS = WS / "results" / "gate_PASS.txt"

done_tags = set()
if DONE.exists():
    done_tags = set(DONE.read_text(encoding="utf-8").split())

print("gate watcher started; watching", CKPT_DIR, flush=True)
while True:
    ckpts = sorted(CKPT_DIR.glob("checkpoint-*")) if CKPT_DIR.exists() else []
    for ck in ckpts:
        tag = ck.name.split("-")[1]
        if tag in done_tags or int(tag) < MIN_TAG:
            continue
        w = ck / "ema_state_dict.pth"
        if not w.exists():
            w = ck / "state_dict.pth"
        if not w.exists():
            continue  # checkpoint 还在写入
        done_tags.add(tag)
        with DONE.open("a", encoding="utf-8") as f:
            f.write(tag + "\n")
        print(f"[gate] running for checkpoint-{tag} ...", flush=True)
        r = subprocess.run(
            [sys.executable, str(WS / "tmp_e4_gate.py"), str(w), f"e4_{tag}"],
            capture_output=True, text=True)
        print(r.stdout[-1500:], flush=True)
        if r.returncode == 0:
            with PASS.open("a", encoding="utf-8") as f:
                f.write(f"checkpoint-{tag} PASSED gate\n")
            print(f"[gate] checkpoint-{tag} PASSED! 记录于 {PASS}", flush=True)
        else:
            print(f"[gate] checkpoint-{tag} 未过门槛 (exit {r.returncode})", flush=True)
    time.sleep(180)
