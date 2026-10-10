r"""E4 测量协议研究：同一 V4 包在 (a) 空闲 (b) 并发负载 (c) 连续三轮 下的 512 时延分布。
输出: D:\models\opt11\e4_protocol.json + 表格
"""
import importlib.util, json, os, statistics, subprocess, sys, time
from pathlib import Path
import torch
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input("c1_512")
MD = r"D:\models\pkg_TinyRefV4\model_dir"

def clocks():
    try:
        return subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu","--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e: return str(e)

def load():
    spec = importlib.util.spec_from_file_location("e4r", os.path.join(MD, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.Runner(MD)

def session(R, n=20, warm=2, label=""):
    with torch.no_grad():
        for _ in range(warm): R.infer(x, prompt="")
        ts = []
        for _ in range(n):
            t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
    s = sorted(ts)
    st = {"label": label, "n": n, "min": s[0], "p10": s[max(0,int(0.1*n)-1)], "median": statistics.median(s),
          "p90": s[min(n-1,int(0.9*n))], "max": s[-1], "mean": statistics.mean(s), "clocks": clocks()}
    print("   %-22s median=%6.2f min=%6.2f p90=%6.2f max=%6.2f | %s" % (label, st["median"], st["min"], st["p90"], st["max"], st["clocks"]), flush=True)
    return st

out = {"model_dir": MD, "sessions": []}
print("== (a) 空闲态：连续三轮（同一进程）==", flush=True)
R = load()
for i in (1,2,3):
    out["sessions"].append(session(R, label="idle-round%d"%i))
print("== (b) 空闲态：新进程（跨会话方差）==", flush=True)
cmd = [r"D:\Python\python.exe", "-c",
       "import importlib.util,statistics,sys,time,torch;sys.path.insert(0,r'%s');import score01_bench as SB;"
       "x=SB.load_input('c1_512');spec=importlib.util.spec_from_file_location('r',r'%s');m=importlib.util.module_from_spec(spec);"
       "spec.loader.exec_module(m);R=m.Runner(r'%s');"
       "[R.infer(x,prompt='') for _ in range(2)];ts=[]\n"
       "for _ in range(20):\n t=time.perf_counter();R.infer(x,prompt='');ts.append((time.perf_counter()-t)*1e3)\n"
       "print('median=%.2f min=%.2f'%(statistics.median(ts),min(ts)))" % (str(HERE), os.path.join(MD,'runner.py'), MD)]
r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
print("   新进程:", (r.stdout or "").strip().splitlines()[-1] if r.stdout else r.stderr[-200:], flush=True)
out["sessions"].append({"label":"idle-fresh-process","raw":(r.stdout or "").strip().splitlines()[-1]})
print("== (c) 并发负载（另一进程持续算力占用）==", flush=True)
burner = subprocess.Popen([r"D:\Python\python.exe","-c",
   "import torch;x=torch.randn(2048,2048,device='cuda');\nwhile True:\n y=x@x"],
   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(20)
try:
    out["sessions"].append(session(R, label="contended"))
finally:
    burner.terminate()
    try: burner.wait(timeout=20)
    except Exception: burner.kill()
time.sleep(10)
print("== (d) 恢复后再测（热恢复）==", flush=True)
out["sessions"].append(session(R, label="idle-after-contention"))
Path(r"D:\models\opt11\e4_protocol.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("E4_DONE", flush=True)
