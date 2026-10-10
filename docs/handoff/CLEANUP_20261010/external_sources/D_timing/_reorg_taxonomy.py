# -*- coding: utf-8 -*-
"""子目录判据落地: 重写 handoff/INDEX.md; 更新 project_map 与 AGENTS 的目录说明."""
import io, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BT = chr(96)
ROOT = Path(r'C:\生成式图像增强轻量模型设计挑战')

# ---------- (a) handoff/INDEX.md 重写 ----------
handoff = '''# docs/handoff 索引

## 本目录的收录判据（只收这一类）

> **按时间产生、不再维护的历史记录**——内容当时为真、至今**仍可引用**，但**不再更新**。

判定问句：**"这是一次会话/一个阶段的产物吗？"** —— 是 ⇒ 放这里。
（与之相对：**前提已失效**的文档进 ''' + BT + '''../archive/''' + BT + '''；**仍在使用**的知识进 ''' + BT + '''../reference/''' + BT + '''。）

**不放这里**：仍在被持续更新的规则/分数/手册（→ reference）；已证伪的旧结论（→ archive）；六份核心文档该管的内容。

## 文件（按时间倒序）

| 文件 | 日期 | 性质 | 一句话 |
|---|---|---|---|
| ''' + BT + '''NEXT_ROUND_PROMPT_20260915NIGHT.md''' + BT + ''' | 09-15 夜 | 交接提示 | 分数模型 v3.1（C=2.515 / 上限 113.6x）、runner 拆解、_NATIVE 扫描、越界普查、Q 驱动分析 |
| ''' + BT + '''NEXT_ROUND_PROMPT_20260915PM.md''' + BT + ''' | 09-15 晚 | 交接提示 | 分数模型 v3 + Q 表 + Q 三性质（**其 §1 参数已被 NIGHT 版修正**） |
| ''' + BT + '''NEXT_ROUND_PROMPT_20260916.md''' + BT + ''' | 09-15 | 交接提示 | **已被 PM/NIGHT 取代**（加法分数模型；文件顶部有横幅） |
| ''' + BT + '''Q_CAMPAIGN_20260915.md''' + BT + ''' | 09-15 | 作战计划 | Q 战场：渲染配置重扫 / runner 提速 / 链 M（事实仍被引用，计划本身已被本轮实测改写） |
| ''' + BT + '''NEXT_ROUND_PROMPT_20260915.md''' + BT + ''' | 09-15 | 交接提示 | 已证伪轴清单 + 质量项形状修正 + PixRestore |
| ''' + BT + '''P0_AUDIT_20260914.md''' + BT + ''' | 09-14 | 自审（review 输入） | 逐条核对官方要求 vs 我们的包；**结论至今有效**（唯一形式偏差：M24 的 pt 非 JIT） |
| ''' + BT + '''MORNING_NOTE_20260914.md''' + BT + ''' | 09-14 | 晨报 | 速度轴钉死、微扩散 v1 训完、早上两件事 |
| ''' + BT + '''NEXT_ROUND_PROMPT_20260913.md''' + BT + ''' | 09-13 | 交接提示 | 榜单 / 配方 / 6 个新读数 / 5 次失败 |
| ''' + BT + '''NEXT_ROUND_PROMPT_20260911.md''' + BT + ''' | 09-11 | 交接提示 | 四发失败根因与修复、9 项门禁、弹药清单 |
| ''' + BT + '''NEXT_ROUND_PROMPT_20260910.md''' + BT + ''' | 09-10 | 交接提示 | 通道开放 + 4 发已提交 + 判读预登记 |
| ''' + BT + '''NEXT_ROUND_PROMPT_20260909.md''' + BT + ''' | 09-09 | 交接提示 | 战略 + 四包资产 + 行动序列 |
| ''' + BT + '''NEXT_ROUND_PROMPT_20260904.md''' + BT + ''' | 09-04 | 交接提示 | 1.4B 门控数据 + 决策树 |
| ''' + BT + '''NEXT_ROUND_PROMPT.md''' + BT + ''' | 08-31 | 交接提示 | 终极冲刺（服务器 1.4B 路线） |
| ''' + BT + '''FINAL_PLAN.md''' + BT + ''' | 08-31 | 决赛预案快照 | 出图方案/答辩主线/命令速查；**实战版已迁入 ''' + BT + '''../current_task.md''' + BT + ''' 任务 C** |
| ''' + BT + '''HANDOFF.md''' + BT + ''' | 08-13 起持续 | **全程记录** | 278.9 KB / 2419 行 / 234 节；项目最完整的历史（早期公式回归、速度线、事故复盘） |
| ''' + BT + '''CLAUDE_20260901_原始快照.md''' + BT + ''' | 09-01 | 规则快照 | 原 ''' + BT + '''CLAUDE.md''' + BT + ''' 全文（已并入 ''' + BT + '''../../AGENTS.md''' + BT + '''；重命名以防被自动加载） |
| ''' + BT + '''AGENTS_20260901_原始快照.md''' + BT + ''' | 09-01 | 规则快照 | 原 ''' + BT + '''AGENTS.md''' + BT + ''' 全文（同上） |
'''
io.open(ROOT / 'docs/handoff/INDEX.md', 'w', encoding='utf-8').write(handoff)
print('[ok] handoff/INDEX.md 重写,', len(handoff.splitlines()), '行')

# ---------- (b) project_map.md 修订 ----------
p = ROOT / 'docs/project_map.md'
t = io.open(p, encoding='utf-8').read()
reps = [
    ('### docs/reference/（仍在使用的长文参考，6 个）',
     '### docs/reference/（长期有效的知识材料，5 个 + INDEX.md）'),
    ('### docs/archive/（已作废/被取代，8 个）',
     '### docs/archive/（前提已失效、引用会误导，8 个 + INDEX.md）'),
    ('| ' + BT + 'FINAL_PLAN.md' + BT + ' | 决赛预案（出图方案/答辩主线/命令速查，日期 08-31，部分前提已过期） |\n', ''),
    ('### docs/handoff/（历史交接，16 个）',
     '### docs/handoff/（时间序列历史记录，18 个，含 FINAL_PLAN 与 INDEX）'),
]
for a, b in reps:
    if a not in t:
        print('[MISS]', a[:46]); continue
    t = t.replace(a, b, 1)
# 在 §8 末尾追加判据提示
t = t.rstrip() + '''

### 子目录收录判据（**放文件前先对照，不要当垃圾桶用**）

| 子目录 | 只收什么 | 判定问句 | 明确的排除项 |
|---|---|---|---|
| %%BT%%docs/handoff/%%BT%% | 按时间产生、**不再维护**的历史记录（内容仍可引用） | "这是一次会话/一个阶段的产物吗？" | 仍在更新的知识 → reference；已证伪 → archive |
| %%BT%%docs/reference/%%BT%% | **长期有效、会被反复查阅并持续更新**的项目知识 | "半年后还会被引用/更新吗？" | 一次性交接 → handoff；前提失效 → archive |
| %%BT%%docs/archive/%%BT%% | 前提**已被证伪**或路线**已废弃** ⇒ 引用会误导 | "照它做会不会走回死路？" | 只是"旧但有效" → handoff；**不得当作临时堆放处** |

判据全文写在各自的 %%BT%%INDEX.md%%BT%% 里（handoff / reference / archive 各一份）。
'''.replace('%%BT%%', BT)
io.open(p, 'w', encoding='utf-8').write(t)
print('[ok] project_map.md 修订')

# ---------- (c) AGENTS.md §7 增补判据 ----------
a = ROOT / 'AGENTS.md'
at = io.open(a, encoding='utf-8').read()
anchor = '2. 新文档先判断归属（map/architecture/task/decisions/progress/review/handoff/reference/archive），**不新建"平级杂物"文件**。'
add = anchor + '''
   - **子目录各有硬判据**（全文见各自 INDEX.md）：%%BT%%handoff/%%BT%% = 时间序列历史记录（内容仍有效）；%%BT%%reference/%%BT%% = 长期知识库（会被反复查阅/更新）；%%BT%%archive/%%BT%% = **前提已失效**（照它做会走回死路）。
   - **四问定归属**：① 是六份核心文档该管的内容吗？② 半年后还会被引用/更新吗（→ reference）？③ 是一次会话/阶段的产物吗（→ handoff）？④ 照它做会走回已证伪的路吗（→ archive）？**四问都不成立 ⇒ 不要新建文件**，先并进现有文档。'''.replace('%%BT%%', BT)
if anchor in at:
    at = at.replace(anchor, add, 1)
    io.open(a, 'w', encoding='utf-8').write(at)
    print('[ok] AGENTS.md §7 增补四问定归属')
else:
    print('[MISS] AGENTS anchor')
