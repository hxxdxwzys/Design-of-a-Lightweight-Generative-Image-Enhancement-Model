# -*- coding: utf-8 -*-
"""OPT-10：在 OPT-01 runner 基础上加入「轻度 unsharp」（runner 内实现，保证重跑一致）。

改动：__init__ 记录 unsharp 参数；_infer_body 在 wavelet post 之后应用 _unsharp()。
其余数值链、精度、tile、seed、后验、像素编码完全不变（同一模型文件，逐字节复用）。
用法: python opt10_gen.py <out_root>
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

SRC = Path(r'D:\models\opt01\model_dir')
dst = Path(sys.argv[1]).resolve() / 'model_dir'
dst.mkdir(parents=True, exist_ok=True)
text = (SRC / 'runner.py').read_text(encoding='utf-8')

ANCHOR = "        self.jit_optimized_execution = bool(r.get('jit_optimized_execution', False))"
assert text.count(ANCHOR) == 1
text = text.replace(ANCHOR, ANCHOR + """
        # OPT-10：轻度 unsharp（官方风格 NR 代理实测：maniqa +6.6%、niqe +5.8%、musiq +2.0%，
        # clipiqa 基本不变；实现于 runner 内以保证官方重跑与提交图一致）
        self.unsharp_sigma = 1.2
        self.unsharp_percent = 60.0
        self.unsharp_threshold = 3.0        # 8 位阈值""")

OLD_TAIL = """        out = self.model.post(sr.float().contiguous(), x.float().contiguous())"""
NEW_TAIL = """        out = self.model.post(sr.float().contiguous(), x.float().contiguous())
        out = self._unsharp(out)"""
assert text.count(OLD_TAIL) == 1
text = text.replace(OLD_TAIL, NEW_TAIL)

HELPER = '''
    def _unsharp(self, t):
        """轻度 unsharp：t = t + (percent/100)*(t - gauss(t, sigma))，仅对超过 8 位阈值的差异生效。

        分离高斯核（reflect padding），全程 fp32，确定性；输出仍夹在 [-1,1]。
        """
        if self.unsharp_percent <= 0:
            return t
        import torch.nn.functional as _F   # 交付 runner 保持仅依赖 torch；此处局部导入
        f = _F
        sigma = float(self.unsharp_sigma)
        ksize = int(round(sigma * 3)) * 2 + 1
        x = torch.arange(ksize, dtype=torch.float64) - (ksize - 1) / 2
        k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        k = (k / k.sum()).to(device=t.device, dtype=t.dtype)
        p = ksize // 2
        c = int(t.shape[1])
        y = f.pad(t, (p, p, 0, 0), mode='reflect')
        y = f.conv2d(y, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        y = f.pad(y, (0, 0, p, p), mode='reflect')
        blur = f.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
        diff = t - blur
        thr = float(self.unsharp_threshold) / 255.0 * 2.0
        mask = (diff.abs() > thr).to(t.dtype)
        out = (t + (float(self.unsharp_percent) / 100.0) * diff * mask).clamp(-1.0, 1.0)
        # 终端 8 位网格契约（与 pixel_io.to_u8 同一公式）：锐化后必须回到 1/127.5 网格，
        # 否则下游 round 会因二次舍入被像素编码守卫拒绝。
        u8 = torch.round((out + 1.0) * 127.5)
        return (u8 / 127.5) - 1.0

    def _require_finite(self, t, where):'''
assert text.count("    def _require_finite(self, t, where):") == 1
text = text.replace("    def _require_finite(self, t, where):", HELPER.lstrip('\n'))
(dst / 'runner.py').write_text(text, encoding='utf-8')
shutil.copyfile(SRC / 'models_export.json', dst / 'models_export.json')
if not (dst / 'your_model.pt').exists():
    os.link(str(SRC / 'your_model.pt'), str(dst / 'your_model.pt'))


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


print('opt10_runner_sha', sha(dst / 'runner.py'), 'model_same',
      sha(dst / 'your_model.pt') == sha(SRC / 'your_model.pt'))
