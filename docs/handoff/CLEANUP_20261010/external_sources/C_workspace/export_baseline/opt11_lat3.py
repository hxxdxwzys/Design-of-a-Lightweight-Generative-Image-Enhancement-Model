r"""严格对照时延台：交替测多个 model_dir，同一进程/同一热状态，消除热漂移。
用法: python opt11_lat3.py DIR1 [DIR2 ...] [--rounds 3] [--n 20] [--preheat 45]
输出: 每个 dir 的 median/min（合并所有轮次）=> D:\models\opt11\lat3.json
"""
import importlib.util, json, os, statistics, sys, time
from pathlib import Path
import torch
sys.path.insert(0, r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
import score01_bench as SB

args = sys.argv[1:]
def opt(name, dv):
    global args
    if name in args:
        i = args.index(name); v = type(dv)(args[i+1]); del args[i:i+2]; return v
    return dv
rounds = opt("--rounds", 3); n = opt("--n", 20); preheat = opt("--preheat", 45)
dirs = args
x = SB.load_input("c1_512")
R = {}
for d in dirs:
    spec = importlib.util.spec_from_file_location("lh_" + Path(d).name, os.path.join(d, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    R[d] = m.Runner(d)

with torch.no_grad():
    t0 = time.time()
    while time.time() - t0 < preheat:      # 预热：交替跑所有模型，稳定热状态
        for d in dirs: R[d].infer(x, prompt="")
    for d in dirs:                          # 各自 2 次 warmup
        for _ in range(2): R[d].infer(x, prompt="")
    allt = {d: [] for d in dirs}
    per_round = []
    for r in range(rounds):
        row = {}
        for d in dirs:
            ts = []
            for _ in range(n):
                t = time.perf_counter(); R[d].infer(x, prompt=""); ts.append((time.perf_counter() - t) * 1e3)
            allt[d] += ts
            row[d] = (round(statistics.median(ts), 2), round(min(ts), 2))
        per_round.append(row)
        print("ROUND%d %s" % (r, json.dumps(row, ensure_ascii=False)), flush=True)

def score(t): return (285.6 / (t + 2) / 0.78) ** 0.19
IT = {"V4": 35.645, "E8": 36.402}
out = {}
for d in dirs:
    s = sorted(allt[d]); med = statistics.median(s)
    out[d] = {"median": round(med, 2), "min": round(s[0], 2), "p90": round(s[min(len(s)-1, int(0.9*len(s)))], 2),
              "n": len(s), "fac_median": round(score(med), 4), "fac_min": round(score(s[0]), 4),
              "pred_E8": round(IT["E8"]*score(med), 2), "pred_V4": round(IT["V4"]*score(med), 2)}
Path(r"D:\models\opt11\lat3.json").write_text(json.dumps({"rounds": per_round, "summary": out}, indent=1, ensure_ascii=False), encoding="utf-8")
print("LAT3 " + json.dumps(out, ensure_ascii=False), flush=True)
print("LAT3_DONE", flush=True)
