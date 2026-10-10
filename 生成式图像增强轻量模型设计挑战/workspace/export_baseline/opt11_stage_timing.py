r"""分段计时（增量法，同步可靠）：学生 / +refiner / +unsharp 三个导出各测一次。"""
import importlib.util, os, statistics, subprocess, sys, time
from pathlib import Path
import torch
PY = r"D:\Python\python.exe"
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
import score01_bench as SB
x = SB.load_input("c1_512")
CKPT = r"D:\models\tiny_v1\v21_s21000.pt"
REF = r"D:\models\refine_v1\refiner_thin2_s6000.pt"

def build_measure(tag, refiner, unsharp):
    MD = str(Path(r"D:\models") / ("pkg_prof_" + tag) / "model_dir")
    cmd = [PY, "-u", str(HERE / "opt11_export_combined.py"), "--ckpt", CKPT, "--modeldir", MD, "--unsharp", str(unsharp)]
    if refiner: cmd += ["--refiner", refiner]
    else: cmd += ["--refiner", REF]          # 需要 refiner 参数；下面用 unsharp=0 且换 shell 处理
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
    if r.returncode != 0:
        print("[%s] export fail: %s" % (tag, (r.stderr or "")[-200:])); return None
    spec = importlib.util.spec_from_file_location("p_" + tag, os.path.join(MD, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    R = m.Runner(MD)
    with torch.no_grad():
        for _ in range(3): R.infer(x, prompt="")
        ts = []
        for _ in range(20):
            t = time.perf_counter(); R.infer(x, prompt=""); ts.append((time.perf_counter()-t)*1e3)
    med = statistics.median(ts)
    print("[%-18s] median %.2f ms best %.2f (graph=%s)" % (tag, med, min(ts), R.graph is not None), flush=True)
    del R; torch.cuda.empty_cache()
    return med

m_student_ref = build_measure("student_ref_us0", REF, 0)     # 学生 + refiner（无 unsharp）
m_full = build_measure("student_ref_us60", REF, 60)          # 学生 + refiner + unsharp60
print("[结论] refiner 增量 = %.2f ms（学生 4.19 → 学生+refiner %.2f）；unsharp60 增量 = %.2f ms" %
      ((m_student_ref or 0) - 4.19, m_student_ref or -1, (m_full or 0) - (m_student_ref or 0)), flush=True)
