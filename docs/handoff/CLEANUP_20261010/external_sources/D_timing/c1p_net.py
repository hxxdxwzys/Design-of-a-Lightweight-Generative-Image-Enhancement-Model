# -*- coding: utf-8 -*-
"""C1' 的净值判据：容量+refiner 全链要多少 clipiqa 才【净胜】已提交 V4？
考虑 cap2 的时延代价（10.72 ms vs V4 的 7.43 ms，同会话实测）。
"""
K, C, P, B = 285.6, 2.515, 0.198, 0.78
f = lambda t: (K / (t + C) / B) ** P

t_v4, t_cap2 = 7.43, 10.72
f_v4, f_c2 = f(t_v4), f(t_cap2)
print("V4   t=%.2f ms -> factor %.4f" % (t_v4, f_v4))
print("cap2 t=%.2f ms -> factor %.4f   (ratio %.4f, 即 factor 降 %.2f%%)"
      % (t_cap2, f_c2, f_c2 / f_v4, (1 - f_c2 / f_v4) * 100))

# 微族斜率（官方 3 点拟合）：IT = 35.047 * clipiqa^0.2181
A, b = 35.047, 0.2181
IT = lambda c: A * c ** b
c_v4 = 0.8042
print("\n微族 IT 曲线 IT = %.3f * c^%.4f" % (A, b))
print("V4   clipiqa=%.4f -> IT=%.3f -> score=%.2f (反推，仅供内部一致)" % (c_v4, IT(c_v4), IT(c_v4) * f_v4))

print("\n要让 cap2 全链【净胜 V4】，需要：")
print("  IT_new * f_cap2 > IT_v4 * f_v4")
need_IT = IT(c_v4) * f_v4 / f_c2
need_c = (need_IT / A) ** (1 / b)
print("  => IT_new > %.3f  =>  clipiqa > %.4f" % (need_IT, need_c))
print("  => 即全链 clipiqa 必须达到 %.4f（比 V4 的 %.4f 高 %.4f）" % (need_c, c_v4, need_c - c_v4))

print("\n各档全链 clipiqa 下的净值（相对 V4，单位：分）：")
for c in (0.8042, 0.82, 0.83, 0.8392, 0.85, 0.87):
    net = IT(c) * f_c2 - IT(c_v4) * f_v4
    print("   clipiqa=%.4f  =>  净值 %+.2f" % (c, net))

print("\n参照：若 refiner 增益在 cap2 上【完全可加】(+0.1736)：全链 = %.4f" % (0.6656 + 0.1736))
print("      若只迁移 50%%：全链 = %.4f" % (0.6656 + 0.1736 * 0.5))
print("      若只迁移 70%%：全链 = %.4f" % (0.6656 + 0.1736 * 0.7))
