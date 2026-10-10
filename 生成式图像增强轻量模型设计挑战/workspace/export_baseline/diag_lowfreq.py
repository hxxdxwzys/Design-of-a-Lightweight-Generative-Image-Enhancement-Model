# -*- coding: utf-8 -*-
"""诊断 2：4K 情况下 fp16 与 fp32 输出差异的来源（低频是否来自输入图）。仅 CPU 读盘。"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, str(Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\pipeline_baseline')))
import core as PBcore  # noqa: E402

RUN = Path(r'C:\models\exports\EXPORT01_RUN_20260917_135055')
SRC = RUN / 'evidence' / 'src_outputs'
RB = Path('D:/models/pipeline_baseline/PIPE01RB_20260917_021500/attempt_01')
TEST = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集')


def to_t(arr):
    return (arr.astype(np.float32) / 255.0) * 2.0 - 1.0


def blur(t):
    return PBcore.wavelet_color_fix_tensor.__globals__ and None  # 不用；下面直接用张量实现


def lowpass_np(a, sigma=5.0):
    import cv2
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)


for name, img in (('case1_full', 'case1.jpg'), ('case7_full', 'case7.jpg')):
    arr = np.asarray(Image.open(TEST / img).convert('RGB'))
    x = to_t(arr)
    y16 = np.load(SRC / ('%s_src.npy' % name), allow_pickle=False)[0].transpose(1, 2, 0)
    y32 = np.load(RB / ('attempt_01_072821_A_%s_r%s.npy'
                        % (name, '2' if name.startswith('case1') else '3')),
                  allow_pickle=False)[0].transpose(1, 2, 0)
    print('==', name, 'input', arr.shape, 'fp16', y16.shape, 'fp32', y32.shape)
    print('   input_sha(pixels)=%s' % __import__('hashlib').sha256(arr.tobytes()).hexdigest()[:16])
    lx, l16, l32 = lowpass_np(x), lowpass_np(y16), lowpass_np(y32)
    print('   low-freq: |y16-lx| mean=%.5f   |y32-lx| mean=%.5f   |y16-y32| mean=%.5f'
          % (np.abs(l16 - lx).mean(), np.abs(l32 - lx).mean(), np.abs(y16 - y32).mean()))
    print('   high-freq energy: x=%.5f  y16=%.5f  y32=%.5f'
          % (np.abs(x - lx).mean(), np.abs(y16 - l16).mean(), np.abs(y32 - l32).mean()))
    print('   full: |y16-y32| mean=%.5f  corr=%.4f'
          % (np.abs(y16 - y32).mean(),
             float(np.corrcoef(y16.ravel()[::37], y32.ravel()[::37])[0, 1])))
    # 直方图/全局色彩
    print('   mean per channel: x=%s y16=%s y32=%s'
          % (np.round(x.reshape(-1, 3).mean(0), 4).tolist(),
             np.round(y16.reshape(-1, 3).mean(0), 4).tolist(),
             np.round(y32.reshape(-1, 3).mean(0), 4).tolist()))
