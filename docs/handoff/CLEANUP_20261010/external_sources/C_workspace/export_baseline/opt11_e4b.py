r"""E4b 测量协议：冷态首测 / 连续新进程 / 并发负载 / 恢复 —— 每项用独立进程测量（贴合"平台只跑一次"的场景）。"""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
MD = r"D:\models\pkg_TinyRefV4\model_dir"
PY = r"D:\Python\python.exe"
HELP = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline\opt11_lat1.py"
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
def clocks():
    return subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu","--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=20).stdout.strip()
def run(label):
    t0 = time.time()
    r = subprocess.run([PY, "-u", HELP, MD, "20"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=900)
    line = [l for l in (r.stdout or "").splitlines() if l.startswith("LAT")]
    print("   %-22s %s | clocks(before)=%s | %.0fs" % (label, line[0] if line else "FAIL " + (r.stderr or "")[-120:], clocks(), time.time()-t0), flush=True)
    return line[0] if line else None
print("== E4b：每项独立进程（模拟平台单次评测）==", flush=True)
print("-- 先静置 90 秒让 GPU 回落 --", flush=True)
time.sleep(90)
res = {}
res["cold_first"] = run("cold-first(fresh)")
res["warm2"] = run("fresh-2nd")
res["warm3"] = run("fresh-3rd")
print("-- 并发负载 --", flush=True)
burner = subprocess.Popen([PY, "-c", "import torch;x=torch.randn(2048,2048,device='cuda')\nwhile True: y=x@x"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(25)
try:
    res["contended"] = run("contended")
finally:
    burner.terminate()
    try: burner.wait(timeout=20)
    except Exception: burner.kill()
time.sleep(30)
res["recovered"] = run("after-contention")
Path(r"D:\models\opt11\e4b_protocol.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print("E4B_DONE", flush=True)
