#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E4 训练实时进度条 (Windows Terminal / 任意 ANSI 终端)。

用法:
  python progress.py          # 单次快照
  python progress.py --watch  # 每 5 秒自动刷新

显示: 训练步数/总步数、速度、ETA、checkpoint、门控结果表、下次门控 ETA。
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

CKPT_DIR = Path(r"D:/models/hypir_ft_v3")
RESULTS = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\results")
LOGS = [
    Path(r"D:/models/hypir/e4v3_train.err"),
    Path(r"D:/models/hypir/e4v3_live.err"),
    Path(r"D:/models/hypir/e4v3_train.log"),
    Path(r"D:/models/hypir/e4_train_live.err"),
    Path(r"D:/models/hypir/e4v2_r600.err"),
]
WATCH_LOG = Path(r"D:/models/hypir/e4v2_watch.log")
BASE = dict(lpips=0.3364, clipiqa=0.4400, musiq=36.81, psnr=22.13)
CHECKPOINT_EVERY = 300
MAX_STEPS = 2000


def find_train_log():
    best, best_t = None, 0
    for p in LOGS:
        if p.exists() and p.stat().st_mtime > best_t:
            best, best_t = p, p.stat().st_mtime
    return best


def parse_pbar(path):
    """从日志提取最新 (step, max, elapsed_sec, s_per_it)"""
    if not path or not path.exists():
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    m = re.findall(r"(\d+)/(\d+) \[(\d+):(\d+)(?::(\d+))?<[^,]+, ([0-9.]+)s/it\]", txt)
    if not m:
        return None
    s, mx, h, mi, sec, spd = m[-1]
    # 无第三字段 -> mm:ss; 有 -> h:mm:ss
    elapsed = (int(h) * 3600 + int(mi) * 60 + int(sec)) if sec else (int(h) * 60 + int(mi))
    return dict(step=int(s), max=int(mx), elapsed=elapsed, spd=float(spd))


def gate_results():
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "gate_e4_*.json"))):
        t = int(f.split("_")[-1].split(".")[0])
        d = json.load(open(f, encoding="utf-8"))["mean"]
        rows.append((t, d))
    return rows


def fmt(sec):
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60)}m{int(sec % 60)}s"


def render():
    lines = []
    lines.append("=" * 78)
    lines.append("  E4 v2 训练进度 (HYPIR-LoRA 微调, 目标: lpips < 0.3364)")
    lines.append("=" * 78)

    # 训练状态
    info = parse_pbar(find_train_log())
    if info:
        pct = info["step"] / info["max"] * 100
        eta = (info["max"] - info["step"]) * info["spd"]
        bar_w = 40
        filled = int(bar_w * info["step"] / info["max"])
        bar = "#" * filled + "-" * (bar_w - filled)
        lines.append(f"  训练: [{bar}] {pct:5.1f}%  {info['step']}/{info['max']} 步")
        lines.append(f"  速度: {info['spd']:.1f}s/步 | 已用: {fmt(info['elapsed'])} | "
                     f"预计剩余: {fmt(eta)}")
        next_gate = ((info["step"] // CHECKPOINT_EVERY) + 1) * CHECKPOINT_EVERY
        if next_gate <= info["max"]:
            lines.append(f"  下次门控: step {next_gate} (约 {fmt((next_gate - info['step']) * info['spd'])} 后)")
    else:
        lines.append("  训练: 未运行 (GPU 空闲或正在加载)")
        next_gate = None

    # 门控结果
    rows = gate_results()
    if rows:
        lines.append("-" * 78)
        lines.append("  门控结果 (vs 基线 lpips 0.3364 / clipiqa 0.440 / musiq 36.81):")
        lines.append(f"  {'步数':>6}  {'lpips':>8}  {'clipiqa':>8}  {'musiq':>7}  {'判定':<8}")
        for t, d in rows:
            ok = d["lpips"] < BASE["lpips"] and d["musiq"] > 35.0 and d["clipiqa"] > 0.42
            verdict = "[PASS]" if ok else "[FAIL]"
            lines.append(f"  {t:>6}  {d['lpips']:>8.4f}  {d['clipiqa']:>8.4f}  {d['musiq']:>7.2f}  {verdict}")
    else:
        lines.append("  门控: 尚无结果 (首个在 step 300, 训练中)")

    # 其他前线
    lines.append("-" * 78)
    reserve = Path(r"D:/models/probe_fast3.zip")
    lines.append("  速度线: 204x 模型 (1.40ms) 已就绪" + (" [OK]" if reserve.exists() else " [缺失]"))
    lines.append("  官方分数: 37.52 (16.9x) → 42.21 (~36x) → 47.98 (~87x) | 目标: 54.73+ (前10)")
    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="每 5 秒自动刷新")
    args = ap.parse_args()
    if not args.watch:
        print(render())
        return
    try:
        while True:
            sys.stdout.write("\x1b[2J\x1b[H" + render())
            sys.stdout.flush()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
