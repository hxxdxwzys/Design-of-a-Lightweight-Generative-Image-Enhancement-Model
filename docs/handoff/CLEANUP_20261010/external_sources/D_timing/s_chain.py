# -*- coding: utf-8 -*-
"""连续工作链（S 系列）：S1 评测 -> 判据判决 -> 若胜出则自动出包+门禁 -> 若为负则按序接 S2/S3。

设计原则
  1. **GPU 串行**：本脚本所有步骤顺序执行，绝不并发（一个 GPU 任务 = 一个进程）。
  2. **可暂停**：每一步的产物都落盘（检查点/渲染目录/JSON），中断后可从任意步骤重启。
  3. **诚实判据**：预登记阈值写死在脚本里，跑完不改。
  4. **不自动"洗白"**：若 S1 为负，**必须打印负结果并停止**，不自动换参数重跑。

用法:
  python s_chain.py stage1     # 导出 up1b -> 渲染 -> 评测
  python s_chain.py stage2     # 判决 + 若胜出则出包与门禁
"""
import json, os, subprocess, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = r"D:\Python\python.exe"
T = Path(r"D:\tmp_timing")
EB = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")

CKPT = Path(r"D:\models\refine_v1\upref_up1b_s6000.pt")
MD = Path(r"D:\models\pkg_s1")
EVAL = Path(r"D:\models\opt11\eval_s1")
METRICS = Path(r"D:\models\opt11\test30_metrics.json")

# 预登记判据（冻结）
BASE_V4 = 0.8042          # 已提交 V4
BEST = 0.8118             # 当前最优 V4t1（thr=1）
ALIVE = 0.83              # "轴活"阈值


def run(name, cmd, timeout=10800):
    print("\n[chain] >>> %s" % name, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, env=ENV)
    print("[chain] %s rc=%d  %.1f min" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in [x for x in (r.stdout or "").strip().splitlines() if x.strip()][-3:]:
        print("        " + l.encode("ascii", "replace").decode("ascii"), flush=True)
    if r.returncode != 0:
        print("[chain] STDERR:", (r.stderr or "")[-1000:].encode("ascii", "replace").decode("ascii"), flush=True)
    return r.returncode


def stage1():
    if not CKPT.exists():
        print("[chain] 缺检查点 %s -> 停止（训练可能未完成）" % CKPT, flush=True); sys.exit(1)
    rc = run("export", [PY, "-u", str(T / "s1_export.py"), str(CKPT), str(MD), "--thr", "1.0"], timeout=3600)
    if rc != 0: sys.exit(1)
    rc = run("render30", [PY, "-u", str(EB / "opt11_render_pkg.py"),
                          "--modeldir", str(MD), "--out", str(EVAL), "--n", "30"], timeout=7200)
    if rc != 0: sys.exit(1)
    run("measure", [PY, "-u", str(T / "measure_test30.py"), "--dir", str(EVAL), "--n", "30"], timeout=3600)
    print("\n[chain] stage1 完成", flush=True)


def stage2():
    if not METRICS.exists():
        print("[chain] 缺 %s -> 先跑 stage1" % METRICS, flush=True); sys.exit(1)
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    c, q = float(m["clipiqa"]), float(m["niqe"])
    print("\n[chain] ===== S1 判决 =====", flush=True)
    print("[chain] S1(可学习上采样) clipiqa=%.4f niqe=%.4f" % (c, q), flush=True)
    print("[chain] 参照: 已提交 V4 = 0.8042/4.9967 ; 当前最优 V4t1 = 0.8118/4.8555", flush=True)
    if c >= ALIVE:
        verdict = "ALIVE(>=0.83)"
    elif c > BEST:
        verdict = "POSITIVE(>V4t1)"
    elif c > BASE_V4:
        verdict = "MARGINAL(>V4, <=V4t1)"
    else:
        verdict = "CLOSED(<=V4)"
    print("[chain] 判决: %s" % verdict, flush=True)

    out = {"tag": "s1", "clipiqa": c, "niqe": q, "verdict": verdict,
           "refs": {"v4": BASE_V4, "v4t1": BEST, "alive": ALIVE},
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    (Path(r"D:\models\opt11") / "s1_verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    if verdict.startswith(("ALIVE", "POSITIVE")):
        print("\n[chain] 胜出 -> 出包 + 9 项门禁", flush=True)
        outd = MD / "output_dir"
        run("render100", [PY, "-u", str(EB / "opt11_render_pkg.py"),
                          "--modeldir", str(MD), "--out", str(outd)], timeout=10800)
        run("package", [PY, "-u", str(T / "package_submit_v2.py"),
                        "--out-dir", str(outd), "--model-dir", str(MD),
                        "--name", "s1", "--zip", r"D:\models\opt11\Track2_S1.zip"], timeout=3600)
        run("gate", [PY, "-u", str(T / "preupload_gate.py"), r"D:\models\opt11\Track2_S1.zip"], timeout=7200)
        print("\n[chain] 下一步(人工决定): 配对显著性 paired.py + 时延 lat_honest.py", flush=True)
    else:
        print("\n[chain] 未胜出 -> 按 FINAL_PUSH.md §4 接 S2 或 S3（不自动启动，需人工确认）", flush=True)
    print("[chain] stage2 完成", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    print("[chain] %s  start %s" % (which, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    {"stage1": stage1, "stage2": stage2}[which]()
    print("S_CHAIN_DONE %s" % which, flush=True)
