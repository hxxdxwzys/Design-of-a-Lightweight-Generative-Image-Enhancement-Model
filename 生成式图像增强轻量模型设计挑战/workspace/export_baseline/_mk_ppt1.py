# -*- coding: utf-8 -*-
"""生成「一页 PPT 方案简介」（16:9 单页）。"""
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

OUT = Path(r"D:\models\finals_20260927\Track2_方案简介_一页.pptx")
DARK = RGBColor(0x1F, 0x2A, 0x37)
ACC = RGBColor(0x1B, 0x6E, 0xC2)
GREY = RGBColor(0x5A, 0x64, 0x72)
LIGHT = RGBColor(0xEC, 0xF2, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "微软雅黑"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
s = prs.slides.add_slide(prs.slide_layouts[6])

def box(x, y, w, h, fill=None, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.06
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh

def text(x, y, w, h, runs, size=11, color=DARK, bold=False, align=PP_ALIGN.LEFT, space=2, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, item in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space)
        if isinstance(item, str):
            item = [item]
        for el in item:
            t, st = (el, {}) if isinstance(el, str) else el
            r = p.add_run(); r.text = t
            r.font.size = Pt(st.get("size", size)); r.font.bold = st.get("bold", bold)
            r.font.color.rgb = st.get("color", color); r.font.name = FONT
    return tb

# 标题
bar = box(0, 0, 13.333, 0.92, fill=DARK)
text(0.45, 0.12, 9.2, 0.7, [[("赛道二 · 轻量一步扩散图像增强", {"size": 21, "bold": True, "color": WHITE})],
                             [("LiteDiff-Enhance：0.87M 一步扩散 + 细节合成，512×512 实测约 6.5 ms", {"size": 11, "color": RGBColor(0xC9, 0xD8, 0xE8)})]], space=1)
text(9.9, 0.2, 3.0, 0.6, [["方案简介（一页）"]], size=12, color=RGBColor(0xC9, 0xD8, 0xE8), align=PP_ALIGN.RIGHT)

# 左列：流水线
box(0.35, 1.12, 5.0, 4.35, fill=LIGHT)
text(0.55, 1.22, 4.6, 0.3, [["① 推理流水线（同一主体、同一代码路径）"]], size=12.5, bold=True, color=ACC)
steps = [
    ("4K 低质输入 (JPEG+噪声+模糊)", "同分辨率 1×，非超分"),
    ("512 分块 + 32 重叠 + 羽化融合", "6×8 块，显存友好"),
    ("下采样至 256 域", "降低算力，保留全图上下文"),
    ("一步扩散 UNet（0.87M）", "ε-预测 @ t=500；显式 x₀，单次前向 (1-NFE)"),
    ("上采样回 512", "回到原始分辨率"),
    ("细节合成器（残差 CNN, 11K）", "对齐 0.5B 教师模型的观感分布"),
    ("内置 unsharp（60%）", "零时延成本的观感增益"),
]
y = 1.62
for i, (a, b) in enumerate(steps):
    box(0.62, y, 4.5, 0.44, fill=WHITE, line=RGBColor(0xC3, 0xD4, 0xE8))
    text(0.74, y + 0.03, 4.3, 0.2, [[(a, {"size": 10.5, "bold": True})]], space=0)
    text(0.74, y + 0.21, 4.3, 0.2, [[(b, {"size": 8.5, "color": GREY})]], space=0)
    y += 0.53
text(0.62, 5.12, 4.5, 0.28, [["加速手段：整链 TorchScript trace + CUDA Graph（规则允许）｜全程 fp16"]], size=8.5, color=GREY)

# 中列：关键数据
box(5.55, 1.12, 3.9, 4.35, fill=WHITE, line=RGBColor(0xC3, 0xD4, 0xE8))
text(5.75, 1.22, 3.5, 0.3, [["② 关键实测数据"]], size=12.5, bold=True, color=ACC)
rows = [
    ("512×512 单卡时延", "约 6.5 ms（同会话 median；区间 6.2–7.1）"),
    ("模型规模", "0.87M（学生）+ 11K（合成器），单文件 2.0 MB"),
    ("无参考观感（30 张测试子集）", "CLIPIQA 0.8042 / MUSIQ 57.7 / MANIQA 0.387 / NIQE 4.997"),
    ("有参考保真（验证集 3 对，4K）", "PSNR 23.15 / SSIM 0.9547 / LPIPS 0.4057"),
    ("时延构成（同会话对照）", "空跑 1.6 ms ｜ 纯学生 5.1 ms ｜ 合成器 +1.4 ms"),
    ("训练成本", "4250 张高清图 + 退化匹配合成；单卡 RTX 4060 Laptop 8 GB"),
]
yy = 1.62
for a, b in rows:
    text(5.75, yy, 3.5, 0.28, [[(a, {"size": 9.5, "bold": True})]], space=0)
    text(5.75, yy + 0.17, 3.5, 0.42, [[(b, {"size": 8.5, "color": GREY})]], space=0)
    yy += 0.62

# 右列：方法与合规
box(9.62, 1.12, 3.36, 4.35, fill=WHITE, line=RGBColor(0xC3, 0xD4, 0xE8))
text(9.82, 1.22, 3.0, 0.3, [["③ 方法与合规"]], size=12.5, bold=True, color=ACC)
text(9.82, 1.60, 3.0, 1.5, [
    [("• 一步扩散蒸馏思路：", {"bold": True, "size": 9.5}), ("用显式 x₀ 公式替代多步采样，1 次前向完成去噪", {"size": 9.5, "color": GREY})],
    [("• 退化匹配训练：", {"bold": True, "size": 9.5}), ("按测试集实测的 JPEG/噪声/模糊分布合成配对，避免数据错位", {"size": 9.5, "color": GREY})],
    [("• 观感对齐：", {"bold": True, "size": 9.5}), ("细节合成器学习 0.5B 教师模型的输出观感（L1+LPIPS+GAN）", {"size": 9.5, "color": GREY})],
], space=5)
box(9.82, 3.35, 3.0, 1.9, fill=LIGHT)
text(9.95, 3.44, 2.75, 1.75, [
    [("合规要点", {"bold": True, "size": 10, "color": ACC})],
    "• 主体为一步扩散（ε-预测 + 显式 x₀），非纯 CNN 复原网络",
    "• 输入输出同为 4K（1×），强制 fp16",
    "• 保留官方 Runner(model_dir).infer(x, prompt=\"\") 接口",
    "• 无任何外部服务/修图软件；权重与代码随包提供",
    "• 随包 selfcheck.py 可复现已提交出图（逐像素比对）",
], size=8.5, space=3)

# 底部
box(0.35, 5.62, 12.63, 1.55, fill=LIGHT)
text(0.55, 5.72, 6.0, 1.35, [
    [("④ 我们理解的关键权衡（评委关注点）", {"bold": True, "size": 11, "color": ACC})],
    "• 感知-保真权衡：增强越强，无参考观感越好、有参考保真越低；本项目给出可调工作点与完整实测。",
    "• 时延-质量权衡：时延收益在毫秒量级被 (t+2) 项压缩，故不以牺牲质量为代价换取时延。",
    "• 可复现性：全部结论附同会话交替测量协议（预热 45 s + 3 轮 × 20 次取中位），避免热漂移误判。",
], size=9, space=3)
text(6.9, 5.72, 5.9, 1.35, [
    [("⑤ 复现与联系", {"bold": True, "size": 11, "color": ACC})],
    "• 工程：python infer.py --timing（512 时延）／ --input <4K目录>（出图）／ python selfcheck.py（自检）",
    "• 依赖：requirements.txt（torch / numpy / Pillow；runner.py 仅依赖 torch）",
    "• 队伍：＿＿＿＿＿＿　联系方式：＿＿＿＿＿＿　日期：2026-09-2＿",
], size=9, space=3)

prs.save(str(OUT))
print("PPT 已生成:", OUT, OUT.stat().st_size, "B")
