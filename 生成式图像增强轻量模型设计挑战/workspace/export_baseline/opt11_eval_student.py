# -*- coding: utf-8 -*-
"""v18 评测：按与历史同口径（字典序 30 张、down=2）渲染并量 clipiqa/niqe/lapvar。
用法: python opt11_eval_student.py <tag> <ckpt>
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
tag = sys.argv[1]
ckpt = sys.argv[2]
out = Path(r"D:\models\opt11") / ("eval_" + tag)
out.mkdir(parents=True, exist_ok=True)

def run(name, cmd):
    print("[run] %s" % name, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (r.stdout or "").strip().splitlines()[-4:]
    print("   " + " | ".join(tail), flush=True)
    if r.returncode != 0:
        print("   [stderr]", (r.stderr or "")[-500:], flush=True)
    return r.returncode

rc = run("render_subset(lex,30)",
         [PY, "-u", str(T / "render_subset.py"), "--ckpt", ckpt, "--out", str(out), "--down", "2", "--n", "30", "--order", "lex"])
if rc == 0:
    run("measure_test30", [PY, "-u", str(T / "measure_test30.py"), "--dir", str(out), "--n", "30"])
    m = out.parent / "test30_metrics.json"
    if m.exists():
        import shutil
        rec = out.parent / ("test30_%s.json" % tag)
        shutil.copyfile(m, rec)
        print("[metrics] %s -> %s" % (tag, rec.read_text(encoding="utf-8").replace("\n", " ")), flush=True)
    else:
        print("[metrics] 未生成", flush=True)
