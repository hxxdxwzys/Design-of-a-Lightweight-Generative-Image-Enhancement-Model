# -*- coding: utf-8 -*-
"""score_pred.py (2026-09-15) — 机械判分: 预测分 = IT(观感) + S(加速比)

IT 曲线 = 由【已实测官方分】反推的 (native-4K clipiqa -> 图像项) 折线, 只用 M24 家族 + 微模型家族
(1.4B 家族的两个点系统性低于曲线 ~3-5 分, 作为家族外离群点不参与插值, 见 HANDOFF 21)。
S(r): 以 0.78x 为 0 点; r>=0.78x 用 2.63 分/翻倍(TinyV2->V2g 同图实测), r<0.78x 用 5.0 分/翻倍
(M24->avg2 6.84 / M24->Slow8 4.99 的两点内插)。
用法: python score_pred.py <clipiqa> <ratio>
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PTS = [(0.3143, 16.49), (0.3897, 16.47), (0.4902, 25.14), (0.6448, 37.63), (0.7700, 43.28), (0.9014, 43.6951)]


def IT(x):
    if x <= PTS[0][0]:
        return PTS[0][1]
    for (x0, y0), (x1, y1) in zip(PTS, PTS[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return PTS[-1][1]


def S(r):
    import math
    return (2.63 if r >= 0.78 else 5.0) * math.log2(r / 0.78)


if __name__ == '__main__':
    cq, ratio = float(sys.argv[1]), float(sys.argv[2])
    print('观感 %.4f -> IT %.2f ; 加速比 %.1fx -> S %+.2f ; 预测分 %.2f' % (cq, IT(cq), ratio, S(ratio), IT(cq) + S(ratio)))
    print('参照: TinyV9b(0.6409@12.4x) -> %.2f | TinyV7gan(0.5059@61.9x) -> %.2f | M24(0.9014@0.78x) -> %.2f'
          % (IT(0.6409) + S(12.4), IT(0.5059) + S(61.9), IT(0.9014) + S(0.78)))
