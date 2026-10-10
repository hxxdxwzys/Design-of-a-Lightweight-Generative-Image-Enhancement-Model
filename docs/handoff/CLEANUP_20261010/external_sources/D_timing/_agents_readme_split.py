# -*- coding: utf-8 -*-
"""AGENTS/README 分工落地: 还原反引号 + 同步 project_map/progress 的说明与指针."""
import io, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BT = chr(96); FENCE = BT * 3
ROOT = Path(r'C:\生成式图像增强轻量模型设计挑战')
LANGS = {'cpp','py','bash','json','python','powershell','ps1','text','yaml'}

def fix(rel):
    p = ROOT / rel
    t = io.open(p, encoding='utf-8').read()
    n = t.count('%%BT%%')
    t = t.replace('%%BT%%', BT)
    out, nf = [], 0
    for ln in t.splitlines():
        if ln.strip() in LANGS and not ln.startswith(BT):
            out.append(FENCE); nf += 1
        else:
            out.append(ln)
    io.open(p, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    print('[fix] %-14s 反引号 %3d 处, 围栏 %d 处' % (rel, n, nf))

fix('AGENTS.md'); fix('README.md')

# project_map 同步
p = ROOT / 'docs/project_map.md'
t = io.open(p, encoding='utf-8').read()
reps = [
    ('├── AGENTS.md                              ← 协作规则（唯一权威，见 §5）',
     '├── AGENTS.md                              ← 双 Agent 协作宪法（角色/权限/纪律）'),
    ('├── README.md                              ← 项目入口说明（保留在根目录）',
     '├── README.md                              ← 项目入口：任务/约束/评分/成绩/命令/坑'),
    ('| ' + BT + 'AGENTS.md' + BT + ' | 我们怎么协作（角色、权限、纪律） | 用户 |',
     '| ' + BT + 'AGENTS.md' + BT + ' | **双 Agent 协作宪法**：角色、权限、升级、文档规则、纪律、Git | 用户 |'),
    ('| ' + BT + 'README.md' + BT + ' | 项目入口说明（对外/对新会话） | 用户 |',
     '| ' + BT + 'README.md' + BT + ' | **项目入口**：任务、硬约束、评分、当前成绩、命令、易踩坑、文档地图 | 用户 |'),
]
for a, b in reps:
    if a in t: t = t.replace(a, b, 1); print('[ok] project_map:', a[:34])
    else: print('[MISS] project_map:', a[:34])
io.open(p, 'w', encoding='utf-8').write(t)

# progress.md 指向 README 的旧世界表, 不重复
p2 = ROOT / 'docs/progress.md'
t2 = io.open(p2, encoding='utf-8').read()
anchor = '## 3. 已完成（里程碑）'
add = ('## 2.5 旧世界成绩（历史，含已作废件）\n\n'
       '旧世界（账号 A/B/C 早期）的成绩表统一放在 ' + BT + '../README.md' + BT + ' §10，**此处不重复维护**。\n'
       '要点：旧世界最好 256.3045（M24）；MetricNet1105x / MetricV7 两发**已被自动判欺诈作废**。\n\n')
if anchor in t2 and '## 2.5 旧世界成绩' not in t2:
    t2 = t2.replace(anchor, add + anchor, 1)
    io.open(p2, 'w', encoding='utf-8').write(t2)
    print('[ok] progress.md 增加 §2.5 指针')
else:
    print('[skip] progress.md')
