# -*- coding: utf-8 -*-
"""敏感度矛盾分析：官方图像项 IT 对 clipiqa 的斜率到底是 9 还是 64？
这是本会话所有"还有多少空间"结论的真正支点 —— 用比赛自己的数据把它钉清楚。

四个独立估计：
  E1 微族 3 点拟合（跨 0.63->0.80，0.1736 clipiqa 跨度）        -> IT slope
  E2 官方单变量 V4us0 -> V4（只差 unsharp；时延因子须视为相同）  -> IT slope
  E3 跨族 V21 -> OPT-10（0.63 -> 0.9123）
  E4 用 cross-family 对每个族的 IT 差 vs clipiqa 差

关键陷阱：E2 的官方分差 0.82 的**标准误**必须估出来，否则无法判断 64 是不是噪声。
"""
import numpy as np

K, C, P, B = 285.6, 2.515, 0.198, 0.78
f = lambda t: (K / (t + C) / B) ** P

# 官方读数（件, clipiqa, t_ms, 官方分）
rows = [
    ("TinyV21",       0.6306,   4.88, 68.6681),
    ("TinyRefV4us0",  0.7918,   6.62, 68.6300),
    ("TinyRefV3|V4",  0.8042,   6.50, 70.1021),
    ("OPT-10",        0.9123, 312.00, 45.6141),
    ("A2",            0.9000, 312.00, 42.2145),
]
IT = {n: s / f(t) for n, c, t, s in rows}
print("反解 IT：")
for n, c, t, s in rows:
    print("  %-15s clipiqa=%.4f  t=%7.2f  IT=%7.3f" % (n, c, t, IT[n]))

print("\n=== E1 微族内：用 clipiqa 跨度估 IT 斜率 ===")
micro = [r for r in rows if r[1] < 0.85]
cs = np.array([r[1] for r in micro]); its = np.array([IT[r[0]] for r in micro])
slope_micro = np.polyfit(cs, its, 1)[0]
print("  3 点线性斜率 = %.2f IT/单位 clipiqa   (跨度只有 %.4f，外推极不可靠)" % (slope_micro, cs.max() - cs.min()))
print("  => 每 +0.01 clipiqa = %+.3f 分（在 t=7.43ms, f=%.3f）" % (slope_micro * 0.01 * f(7.43), f(7.43)))

print("\n=== E2 官方单变量 V4us0 -> V4（只差内置 unsharp）===")
dIT = IT["TinyRefV3|V4"] - IT["TinyRefV4us0"]
dC = 0.8042 - 0.7918
print("  d_clipiqa = %.4f,  d_IT = %+.3f  => 斜率 = %.1f IT/单位" % (dC, dIT, dIT / dC))
print("  但注意：t 从 6.62 -> 6.50（factor %.4f -> %.4f，差 %+.2f%%）"
      % (f(6.62), f(6.50), (f(6.50) / f(6.62) - 1) * 100))
print("  => 若把 factor 差归零（严格单变量），d_IT = %+.3f，斜率 = %.1f"
      % (IT["TinyRefV3|V4"] - 68.6300 / f(6.50), (IT["TinyRefV3|V4"] - 68.6300 / f(6.50)) / dC))

print("\n=== 官方分噪声：用两对【clipiqa 几乎相同】的件估 ===")
print("  V3|V4 (0.8042) vs V21 (0.6306)：d_clipiqa=%.4f, d_官方分=%+.4f" % (0.8042-0.6306, 70.1021-68.6681))
print("  V4us0 (0.7918) vs V21 (0.6306)：d_clipiqa=%.4f, d_官方分=%+.4f" % (0.7918-0.6306, 68.6300-68.6681))
print("  同一 zip 若重复评测，官方分应相同 => 两对的差若来自真质量差，应同号同量级；")
print("  实测 +1.4340 vs -0.0381 => **后者在 clipiqa 差 0.1612 时几乎零变化**，")
print("  说明官方分的件间散布 >= 1 分量级，或官方图像项对这部分 clipiqa 差【不敏感】。")

print("\n=== 结论性对照：两种斜率下，同一实验的判定相反 ===")
for slope_name, slope in (("微族外推 9", slope_micro), ("单变量 64", dIT / dC)):
    # 需要多少 IT 才能补回 cap2 的 factor 损失
    f_v4, f_c2 = f(7.43), f(10.72)
    need_IT = IT["TinyRefV3|V4"] * f_v4 / f_c2
    need_dC = (need_IT - IT["TinyRefV3|V4"]) / slope
    print("  斜率 %-10s => cap2 全链需要 clipiqa +%.4f（即 %.4f）才算净胜 V4"
          % (slope_name, need_dC, 0.8042 + need_dC))
    print("               而 refiner 增益 100%% 可加也只有 +0.1736 => 结论：%s"
          % ("净胜" if 0.1736 > need_dC else "净负"))

print("\n=== 因此：本会话所有'剩余空间'数字都挂在这个斜率上 ===")
print("  - 时延侧'+17 分'：用微族斜率算的图像项基线 => 斜率变大则该数字变小")
print("  - '容量轴净负'：同上")
print("  - '域轴 CLOSED'：**不依赖斜率**（那是训练失败，clipiqa 直接掉了 0.12，方向明确）")
