
import inspect, os, sys
import torch
md = os.path.abspath(sys.argv[1])
sys.path.insert(0, md)
import runner as R
sig_init = str(inspect.signature(R.Runner.__init__))
sig_infer = str(inspect.signature(R.Runner.infer))
print('SIG init ', sig_init)
print('SIG infer', sig_infer)
obj = R.Runner(md)
x = torch.rand(1, 3, 512, 512) * 2 - 1          # [-1,1] fp32, 与官方 harness 同约定
with torch.no_grad():
    y = obj.infer(x)
print('OUT shape=%s dtype=%s min=%.3f max=%.3f std=%.3f' % (
    tuple(y.shape), y.dtype, y.min().item(), y.max().item(), y.std().item()))
ok = (tuple(y.shape) == (1, 3, 512, 512) and torch.isfinite(y).all()
      and y.std().item() > 0.05 and y.min().item() >= -1.01 and y.max().item() <= 1.01)
print('IFACE_' + ('PASS' if ok else 'FAIL'))
