# -*- coding: utf-8 -*-
"""四方保真对照 (2026-09-13) —— LQ / M24 / HYPIR / GT
判据 (先写死):
  若 HYPIR 在"对 GT 的保真"上明显强于我们 (PSNR +1.5dB 或 LPIPS -0.1) -> 走 HYPIR 路线
  若 HYPIR 也不比我们强 -> 手上所有图像源都跳不出平台
内存纪律: 4K 的 LPIPS/DISTS 会吃 20+GB (已两次翻车) -> 保真重指标在 1024 上算, 五方同口径;
           PSNR/SSIM 在原生 4K 上流式计算。
同时输出肉眼对照图 (512 原生裁剪) -> D:/tmp_timing/compare_crops/
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
SRC = {
    'LQ':      lambda i: VAL / f'case{i}_lq.jpg',
    'M24':     lambda i: Path(rf'D:/models/vosr_lora/v7/val_out_v7s5000_sc08_seed8_wavelet/case{i}_lq.png'),
    'HYPIR1x': lambda i: Path(rf'D:/models/hypir/out_val3_2x/result_1x/case{i}.png'),
    'HYPIRct': lambda i: Path(rf'D:/models/hypir/out_val3_ct150/result/case{i}_lq.png'),
}
GT = lambda i: next(VAL.glob(f'case{i}_gt.*'))
OUTJSON = Path(r'D:/tmp_timing/fidelity_compare.json')
CROPS = Path(r'D:/tmp_timing/compare_crops')
RES_LP = 1024          # 保真重指标的工作分辨率
CROP = 640             # 肉眼对照裁剪边长 (原生像素)


class MS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_gb():
    st = MS(); st.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 1024 ** 3


def main():
    import numpy as np
    import torch
    import pyiqa
    from PIL import Image

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lp = pyiqa.create_metric('lpips', device=dev)
    ds = pyiqa.create_metric('dists', device=dev)
    nq = pyiqa.create_metric('niqe', device=dev)
    cq = pyiqa.create_metric('clipiqa', device=dev)
    CROPS.mkdir(parents=True, exist_ok=True)

    res = {}
    for i in (1, 2, 3):
        g = GT(i)
        gt_img = Image.open(g).convert('RGB')
        gw, gh = gt_img.size
        print(f'\n=== case{i}: GT {gw}x{gh} ({g.name}) | 可用内存 {free_gb():.1f} GB ===', flush=True)
        ga = np.asarray(gt_img).astype(np.float32)
        rec = {}
        crops = []
        for name, fn in SRC.items():
            p = fn(i)
            if not p.exists():
                print(f'  [{name}] 缺文件 {p}'); continue
            im = Image.open(p).convert('RGB')
            if im.size != (gw, gh):
                im = im.resize((gw, gh), Image.BICUBIC)
            a = np.asarray(im).astype(np.float32)
            mse = float(((a - ga) ** 2).mean())
            psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))
            # 全局 SSIM (与项目既有口径一致的简化式)
            mp, mg = a.mean(), ga.mean()
            vp, vg = a.var(), ga.var()
            cov = float(((a - mp) * (ga - mg)).mean())
            c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            ssim = ((2 * mp * mg + c1) * (2 * cov + c2)) / ((mp ** 2 + mg ** 2 + c1) * (vp + vg + c2))
            # 降分辨率算重指标
            sm = im.resize((RES_LP, int(RES_LP * gh / gw)), Image.BICUBIC)
            sg = gt_img.resize((RES_LP, int(RES_LP * gh / gw)), Image.BICUBIC)
            t = torch.from_numpy(np.asarray(sm).copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
            tg = torch.from_numpy(np.asarray(sg).copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
            with torch.no_grad():
                lpv = float(lp(t, tg).item())
                dsv = float(ds(t, tg).item())
            tn = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
            with torch.no_grad():
                nqv = float(nq(tn).item())
                cqv = float(cq(tn).item())
            rec[name] = {'psnr': round(psnr, 3), 'ssim': round(ssim, 4),
                         f'lpips@{RES_LP}': round(lpv, 4), f'dists@{RES_LP}': round(dsv, 4),
                         'niqe': round(nqv, 4), 'clipiqa': round(cqv, 4)}
            print(f'  {name:<9} PSNR {psnr:6.3f}  SSIM {ssim:.4f}  LPIPS {lpv:.4f}  '
                  f'DISTS {dsv:.4f}  NIQE {nqv:.3f}  CLIPIQA {cqv:.3f}', flush=True)
            # 裁剪: 取中心偏上的一块 (纹理区)
            x0 = max(0, gw // 2 - CROP // 2); y0 = max(0, gh // 3)
            crops.append((name, im.crop((x0, y0, x0 + CROP, y0 + CROP))))
            del a, t, tg, tn, im
            gc.collect(); torch.cuda.empty_cache()
        if crops:
            sheet = Image.new('RGB', (CROP * len(crops), CROP + 24), (255, 255, 255))
            from PIL import ImageDraw
            dr = ImageDraw.Draw(sheet)
            for j, (nm, c) in enumerate(crops):
                sheet.paste(c, (j * CROP, 24))
                dr.text((j * CROP + 8, 6), nm, fill=(0, 0, 0))
            sheet.save(CROPS / f'case{i}_compare.png')
            print(f'  对照图 -> {CROPS / f"case{i}_compare.png"}', flush=True)
        del ga, crops
        gc.collect()
        res[f'case{i}'] = rec

    # 汇总
    print('\n=== 三张平均 ===')
    names = list(SRC)
    print(f'  {"源":<9}{"PSNR":>8}{"SSIM":>9}{"LPIPS":>9}{"DISTS":>9}{"NIQE":>8}{"CLIPIQA":>9}')
    avg = {}
    for n in names:
        rows = [res[f'case{i}'][n] for i in (1, 2, 3) if n in res.get(f'case{i}', {})]
        if not rows:
            continue
        avg[n] = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}
        a = avg[n]
        print(f'  {n:<9}{a["psnr"]:>8.3f}{a["ssim"]:>9.4f}{a["lpips@1024"]:>9.4f}'
              f'{a["dists@1024"]:>9.4f}{a["niqe"]:>8.3f}{a["clipiqa"]:>9.4f}')

    m, h = avg.get('M24'), avg.get('HYPIR1x')
    if m and h:
        print('\n=== 判据 (HYPIR1x vs M24) ===')
        dp = h['psnr'] - m['psnr']; dl = h['lpips@1024'] - m['lpips@1024']
        print(f'  ΔPSNR = {dp:+.3f} dB   ΔLPIPS = {dl:+.4f}')
        ok = (dp >= 1.5) or (dl <= -0.1)
        print(f'  → {"✅ 达到判据 (HYPIR 明显更保真): 走 HYPIR 路线" if ok else "❌ 未达判据: 手上图像源跳不出平台"}')
    OUTJSON.write_text(json.dumps({'res': res, 'avg': avg, 'res_lpips_dists': RES_LP},
                                  ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n已存 {OUTJSON}')


if __name__ == '__main__':
    main()
