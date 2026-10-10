import importlib.util, statistics, sys, time, os
from pathlib import Path
import torch
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input('c1_512')

def load(md):
    spec = importlib.util.spec_from_file_location('cm', str(Path(md) / 'runner.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.Runner(md)

for name in ('pkg_TinyV11b', 'pkg_TinyV7gan'):
    md = os.path.join(r'D:\models', name, 'model_dir')
    R = load(md)
    with torch.no_grad():
        for _ in range(2): R.infer(x, prompt='')
        g = []
        for _ in range(8):
            t = time.perf_counter(); R.infer(x, prompt=''); g.append((time.perf_counter()-t)*1e3)
    R.graph = None                      # 强制 eager 回退
    with torch.no_grad():
        R.infer(x, prompt='')
        e = []
        for _ in range(8):
            t = time.perf_counter(); R.infer(x, prompt=''); e.append((time.perf_counter()-t)*1e3)
    print('%-13s graph %.2f ms (%.1fx) | eager %.2f ms (%.1fx)  回落代价 %.2fx' %
          (name, statistics.median(g), 285.6/statistics.median(g),
           statistics.median(e), 285.6/statistics.median(e), statistics.median(e)/statistics.median(g)), flush=True)
    del R; torch.cuda.empty_cache()
