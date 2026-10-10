# -*- coding: utf-8 -*-
"""HYPIR 断点续跑 (2026-09-13): 只跑 out_test100/result 里还缺的那些 case
用法: python hypir_resume.py
"""
from __future__ import annotations
import shutil, subprocess, sys, time
from pathlib import Path

PY = r'D:\Python\python.exe'
BASE = r'D:/models/hypir_code/HYPIR-main/HYPIR-main'
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
OUT = Path(r'D:/models/hypir/out_test100')
RES = OUT / 'result'
LQ_REST = Path(r'D:/models/hypir/lq_rest')
WT = r'D:/models/hypir/HYPIR_sd2.pth'

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    have = set()
    if RES.exists():
        from PIL import Image as _I
        for p in RES.glob('*.png'):
            try:
                with _I.open(p) as _im:
                    _im.load()
                have.add(p.stem)
            except Exception:
                print(f'  [跳过损坏] {p.name}', flush=True)
    allc = {f'case{i}' for i in range(1, 101)}
    missing = sorted(allc - have, key=lambda s: int(s[4:]))
    print(f'已完成 {len(have)}/100, 缺 {len(missing)}: {missing[:10]}{" ..." if len(missing) > 10 else ""}', flush=True)
    if not missing:
        print('HYPIR_RESUME_NOTHING_TO_DO'); return
    if LQ_REST.exists():
        shutil.rmtree(LQ_REST)
    LQ_REST.mkdir(parents=True)
    from PIL import Image
    for c in missing:
        Image.open(TEST / f'{c}.jpg').convert('RGB').save(LQ_REST / f'{c}.png')
    out_rest = Path(r'D:/models/hypir/out_rest')
    if (out_rest / 'result').exists():
        shutil.rmtree(out_rest / 'result')      # 清掉上次被杀留下的残缺产物
    cmd = [PY, 'test.py', '--base_model_type', 'sd2', '--base_model_path', r'D:/models/sd21base',
           '--model_t', '200', '--coeff_t', '200', '--lora_rank', '256',
           '--lora_modules', 'to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj',
           '--weight_path', WT, '--patch_size', '512', '--stride', '256',
           '--lq_dir', str(LQ_REST), '--upscale', '1', '--captioner', 'empty',
           '--output_dir', str(out_rest), '--device', 'cuda']
    t0 = time.time()
    # 坑(2026-09-13 实测): 不指定 encoding 时 Windows 用 locale(GBK) 解码子进程 UTF-8 输出 ->
    # UnicodeDecodeError, 白跑 40 分钟。必须显式 utf-8 + errors='replace'。
    r = subprocess.run(cmd, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding='utf-8', errors='replace')
    print(f'[run] rc={r.returncode} {(time.time()-t0)/60:.1f} min', flush=True)
    src = out_rest / 'result'
    n = 0
    if src.exists():
        from PIL import Image as _I2
        RES.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob('*.png')):
            try:                                  # 只搬能完整解码的
                with _I2.open(p) as _im:
                    _im.load()
            except Exception:
                print(f'  [残缺丢弃] {p.name}', flush=True)
                continue
            shutil.move(str(p), str(RES / p.name)); n += 1
    print(f'[moved] {n} -> {RES}, 现在共 {len(list(RES.glob("*.png")))}/100', flush=True)
    print('HYPIR_RESUME_DONE')

if __name__ == '__main__':
    main()
