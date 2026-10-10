import importlib.util, statistics, sys, time, os
from pathlib import Path
import torch
HERE = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
sys.path.insert(0, str(HERE))
import score01_bench as SB
x = SB.load_input('c1_512')

def load(md):
    spec = importlib.util.spec_from_file_location('cm_' + os.path.basename(os.path.dirname(md)),
                                                  str(Path(md) / 'runner.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.Runner(md)

names = ['pkg_TinyV4g', 'pkg_TinyV7gan', 'pkg_TinyV11b']
Rs = {n: load(os.path.join(r'D:\models', n, 'model_dir')) for n in names}
res = {n: [] for n in names}
with torch.no_grad():
    for n in names:                      # 预热
        for _ in range(2): Rs[n].infer(x, prompt='')
    for rnd in range(3):                 # 交替 3 轮，每轮每模型 10 次
        for n in names:
            ts = []
            for _ in range(10):
                t = time.perf_counter(); Rs[n].infer(x, prompt=''); ts.append((time.perf_counter()-t)*1e3)
            res[n].extend(ts)
        print('  轮%d 完成' % (rnd + 1), flush=True)
print()
for n in names:
    v = res[n]
    print('%-13s median %.2f ms (%.1fx)  best %.2f  p10 %.2f  p90 %.2f  n=%d' %
          (n, statistics.median(v), 285.6/statistics.median(v), min(v),
           sorted(v)[len(v)//10], sorted(v)[int(len(v)*0.9)], len(v)), flush=True)
import json
Path(r'D:\models\opt11\decoup_ab.json').write_text(json.dumps({n: res[n] for n in names}, indent=1), encoding='utf-8')
print('[done] -> D:\\models\\opt11\\decoup_ab.json')
