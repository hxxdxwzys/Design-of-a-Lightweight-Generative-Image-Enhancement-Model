import sys, importlib.util, numpy as np, time
spec = importlib.util.spec_from_file_location('tt', r'D:/tmp_timing/train_tiny.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
t0 = time.time()
for _ in range(10):
    b = m.degrade(a)
print('degrade OK', b.shape, b.dtype, f'{(time.time()-t0)/10*1000:.1f} ms/张')
import inspect
src = inspect.getsource(m.degrade)
print('残留 rng. 调用:', [l.strip() for l in src.splitlines() if 'rng.' in l])
