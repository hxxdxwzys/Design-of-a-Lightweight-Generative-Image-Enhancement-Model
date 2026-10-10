r"""E7/D1-D2：用修正模型推算（a）达到第 10 名线/榜首所需的 (musiq, clipiqa, t) 组合；（b）本族可达上限。"""
import math
A, b1, b2 = 0.0447, 0.247, 1.661
factor = lambda t: (285.6/(t+2)/0.78)**0.19
IT = lambda c, m: A * (c**b1) * (m**b2)
print("== 参考点 ==")
for name, c, m, t in [("V4(已交)", .8042, 57.67, 7.00), ("V3(已交)", .7946, 55.85, 6.68),
                      ("VOSR/OPT10 图", .9123, 65.86, 324.0), ("M24 图", .9023, 62.81, 366.2),
                      ("r256c", .679, 55.88, 5.69)]:
    it = IT(c, m); print("   %-14s IT=%6.2f factor=%.3f score=%6.2f  (t=%.2f ms)" % (name, it, factor(t), it*factor(t), t))

print()
print("== 达到目标分所需（括号内为对应 musiq）==")
for target in (83.62, 98.96, 120.0, 217.21):
    print("   目标 %.2f：" % target)
    for t in (2.0, 3.0, 5.0, 7.0, 15.0):
        need_it = target / factor(t)
        # 固定 clipiqa=0.82，解所需 musiq
        m_need = (need_it / (A * (0.82**b1)))**(1/b2)
        print("      t=%5.1f ms (factor %.3f) -> 需 IT=%6.2f -> 需 musiq=%6.2f%s" %
              (t, factor(t), need_it, m_need, "  ← VOSR 档(65.9)可达" if m_need <= 66 else ""))
print()
print("== 本族上限推算（乐观但仍在数据范围内）==")
best_it = IT(0.85, 66.0)     # clipiqa 0.85 + musiq 66（≈VOSR 观感）
print("   若快线做到 clipiqa 0.85 / musiq 66：IT=%.2f" % best_it)
for t in (2.0, 2.5, 3.0, 5.0, 6.7, 10.0):
    print("      t=%5.1f ms -> factor %.3f -> score %6.2f" % (t, factor(t), best_it*factor(t)))
print()
print("== 结论 ==")
print("   ① 10 名线 98.96 需要「musiq≈65 且 t≤3 ms」或「musiq≈69 且 t≈6.7 ms」")
print("   ② 我们当前最好 70.10 离它差 ~29 分；缺口 ~2/3 在图像项（musiq 58 → 65），~1/3 在时延（6.7 → 3 ms）")
print("   ③ 榜首 217 需 IT≈80（musiq>100）⇒ 超出本模型适用范围【UNKNOWN】，不能用我们的资产解释")
print("E7_DONE")
