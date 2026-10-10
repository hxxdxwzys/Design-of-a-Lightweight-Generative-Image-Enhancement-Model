# -*- coding: utf-8 -*-
"""修 avg2 runner 的缩进 bug + 重打包 + 重跑门禁 (2026-09-13 05:1x)
Bug: 2 样本补丁被插进 `if ph or pw:` 块内部, 512x512 时 ph=pw=0 -> 该块不执行 -> sr 未定义。
本脚本重写整个 infer 方法 (干净缩进), 然后重新打包 + 门禁。图不用重渲。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
MODELS = Path(r'D:/models')
MD = MODELS / 'final_compliant_M24avg2/model_dir'
OUTDIR = MODELS / 'final_compliant_M24avg2/output_dir'

NEW_INFER = '''    def infer(self, image_tensor, prompt=""):
        """image_tensor: [1,3,H,W] float in [-1,1]; 返回同尺寸 [-1,1] (fp32)
        2 样本平均 (速度探针 2026-09-13): 同权重、推理两次、逐像素平均。"""
        x = image_tensor.to(self.device, dtype=torch.float16)
        b, _, h, w = x.shape
        ph = (-h) % 8
        pw = (-w) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        zs = [torch.randn(b, 4, (h + ph) // 8, (w + pw) // 8,
                          device=self.device, dtype=torch.float16) for _ in range(2)]
        with torch.no_grad():
            acc = None
            for zz in zs:
                o = self.model(x, zz)
                acc = o if acc is None else acc + o
            sr = acc / len(zs)
        if ph or pw:
            sr = sr[:, :, :h, :w]
        return sr.float().clamp(-1.0, 1.0)
'''


def main():
    rp = MD / 'runner.py'
    src = rp.read_text(encoding='utf-8')
    i = src.find('    def infer(')
    if i < 0:
        print('!! 未找到 infer 方法'); return
    new = src[:i] + NEW_INFER
    rp.write_text(new, encoding='utf-8')
    print('[fix] infer 方法已重写 (缩进修正)', flush=True)

    import py_compile
    py_compile.compile(str(rp), doraise=True)
    print('[fix] 语法 OK', flush=True)

    # 删掉旧的失败 zip
    old = MODELS / 'Track2_M24avg2.zip'
    if old.exists():
        old.unlink()
    up = MODELS / 'upload/Track2_M24avg2.zip'
    if up.exists():
        up.unlink()
        print('[fix] 已删除 upload/ 里门禁未过的旧包', flush=True)

    for script, tag, args in (
            ('package_generic.py', 'pack',
             ['--name', 'Track2_M24avg2', '--outdir', str(OUTDIR), '--modeldir', str(MD)]),
            ('preupload_gate.py', 'gate',
             ['--stage-upload', '--upload-name', 'Track2_M24avg2.zip', str(old)])):
        with (T / f'fix_avg2_{tag}.log').open('w', encoding='utf-8') as fo:
            subprocess.run([PY, '-u', str(T / script)] + args, stdout=fo,
                           stderr=subprocess.STDOUT, text=True, cwd=r'C:/models/finals')
        txt = (T / f'fix_avg2_{tag}.log').read_text(encoding='utf-8', errors='replace')
        if tag == 'gate':
            for ln in txt.splitlines():
                if any(k in ln for k in ('✅', '❌', '==>')):
                    print('   ' + ln.strip(), flush=True)
            ok = 'PREUPLOAD_ALL_PASS' in txt
            (T / 'build_M24avg2.json').write_text(json.dumps(
                {'tag': 'M24avg2', 'zip': str(old), 'gate_pass': ok}, indent=1), encoding='utf-8')
            print(f'   总体: {"✅ 通过" if ok else "❌ 未过"}', flush=True)
    print('FIX_AVG2_DONE', flush=True)


if __name__ == '__main__':
    main()
