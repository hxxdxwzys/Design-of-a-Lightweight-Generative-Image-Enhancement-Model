# -*- coding: utf-8 -*-
"""2x val 收尾 (2026-08-26): 补渲染 u1/case3_lq (1x) + 重跑全部 eval 写最终 json
在 b_2x_val.py 主任务完成后运行。
用法: D:/Python/python.exe D:/tmp_timing/b_2x_finish.py
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, r'D:/tmp_timing')
from b_2x_val import OUT_ROOT, evaluate, render, downsample  # noqa: E402

ENV = dict(os.environ)
ENV['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
ENV['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def main():
    u1 = OUT_ROOT / 'u1'
    if not (u1 / 'case3_lq.png').exists():
        # 单独渲染 case3 1x 到临时目录再移入
        tmp = OUT_ROOT / 'u1_case3_tmp'
        tmp.mkdir(parents=True, exist_ok=True)
        src = Path(r'D:/tmp_timing/val_lq_only/case3_lq.jpg')
        cmd = [r'D:/Python/python.exe', r'D:/tmp_timing/vosr_lora_infer.py',
               '-c', r'C:/models/finals/preset/ckpts/VOSR_0.5B_os',
               '--lora', r'D:/models/vosr_lora/v7/lora_s5000.pt', '--lora-scale', '0.8',
               '-i', str(src), '-o', str(tmp), '-u', '1', '--seed', '8',
               '--align_method', 'wavelet', '--tile_size', '512',
               '--vae_tile_size', '1024', '--vae_tile_overlap', '64']
        r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
        if r.returncode != 0 or not (tmp / 'case3_lq.png').exists():
            print('[FAIL] case3 1x render', flush=True)
            print(r.stdout[-3000:], flush=True)
            print(r.stderr[-3000:], flush=True)
            sys.exit(1)
        (tmp / 'case3_lq.png').replace(u1 / 'case3_lq.png')
        print('[ok] u1/case3_lq.png', flush=True)

    device = 'cuda'
    res = {}
    res['u1'] = evaluate(u1, 'u1_1x', OUT_ROOT / 'b_2x_u1.json', device)
    res['u2_bicubic'] = evaluate(OUT_ROOT / 'u2_bicubic', 'u2_bicubic', OUT_ROOT / 'b_2x_u2b.json', device)
    res['u2_lanczos'] = evaluate(OUT_ROOT / 'u2_lanczos', 'u2_lanczos', OUT_ROOT / 'b_2x_u2l.json', device)
    (OUT_ROOT / 'b_2x_val.json').write_text(
        __import__('json').dumps(res, indent=1), encoding='utf-8')
    print('[done] b_2x_val.json FINAL', flush=True)


if __name__ == '__main__':
    main()
