# -*- coding: utf-8 -*-
"""清理过时无用件: 3 个旧世界榜单 xlsx + 5 份已作废文档; 同步更新索引与交叉引用."""
import io, subprocess, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BT = chr(96)
ROOT = Path(r'C:\生成式图像增强轻量模型设计挑战')

DEL = [
    '初赛榜单8.8.xlsx', '初赛榜单8.11.xlsx', '初赛榜单8.12.xlsx',
    'docs/archive/COMMERCIAL_API_PLAN.md',
    'docs/archive/THEORETICAL_CEILING_FRAMEWORK.md',
    'docs/archive/SERVER_SPEC.md',
    'docs/archive/SERVER_PHASE0.md',
    'docs/archive/SERVER_P100.md',
    'docs/handoff/NEXT_ROUND_PROMPT_20260916.md',
]

# ---------- 1) 先保全残值: decisions.md 追加 D16 ----------
d = ROOT / 'docs/decisions.md'
t = io.open(d, encoding='utf-8').read().rstrip() + '''

## D16 清理已作废文档（2026-09-15 夜，用户授权）
- **决定人**：用户（"把过时的，无用的东西删除"）｜**执行**：DeepSeek
- **删除清单（9 个）**：
  1. 旧世界榜单快照 3 份 xlsx（8.8 / 8.11 / 8.12，2026-08-07~08-12）——新世界榜单已另行记录（%%BT%%docs/reference/新世界模型分数对应表.md%%BT%%）。
  2. %%BT%%COMMERCIAL_API_PLAN.md%%BT%% —— **文件自身即标注"已作废（用户 2026-08-17 决定：不合规）"**；残值保留在本条：**商业 API 生成器路线被用户判定为不合规，不再考虑。**
  3. %%BT%%THEORETICAL_CEILING_FRAMEWORK.md%%BT%% —— "理论高度"论证框架，**前提是已被证伪的加法评分公式**（见 D4）；照它推演会得到错误的天花板。
  4. %%BT%%SERVER_SPEC.md%%BT%% / %%BT%%SERVER_PHASE0.md%%BT%% / %%BT%%SERVER_P100.md%%BT%% —— 云端算力采购/部署清单，**路线已由用户放弃**（D2：全部本地 4060）。残值：SSH/SFTP 与机器兼容性经验仍保存在 %%BT%%docs/handoff/CLAUDE_20260901_原始快照.md%%BT%%。
  5. %%BT%%NEXT_ROUND_PROMPT_20260916.md%%BT%% —— 加法分数模型时代的交接提示，**已被 15PM / 15NIGHT 明确取代**；其仍有效的读数（同速 +8.70、四项有参反证、分辨率杠杆）已收录于 %%BT%%docs/reference/新世界模型分数对应表.md%%BT%% 一·D/一·E 与 D4。
- **可恢复性**：以上文件均曾提交进 git，**历史版本可随时 %%BT%%git checkout <commit> -- <path>%%BT%% 取回**；本仓库不做破坏性历史改写。
- **未删除但列为候选**（待用户一句话确认）：5 份中间期交接提示（%%BT%%NEXT_ROUND_PROMPT_20260904/09/10/11/13%%BT%%）、2 份规则快照（%%BT%%AGENTS_20260901_原始快照.md%%BT%% / %%BT%%CLAUDE_20260901_原始快照.md%%BT%%）。
'''
io.open(d, 'w', encoding='utf-8').write(t + '\n')
print('[ok] decisions.md += D16')

# ---------- 2) 删除文件 ----------
for rel in DEL:
    r = subprocess.run(['git', 'rm', '-q', '--', rel], cwd=str(ROOT), capture_output=True, text=True, errors='replace')
    ok = not (ROOT / rel).exists()
    print('%-58s %s' % (rel, 'DELETED' if ok else ('FAIL: ' + r.stderr.strip()[:60])))

# ---------- 3) 更新索引与引用 ----------
def drop_lines(path, keys, note=None):
    p = ROOT / path
    lines = io.open(p, encoding='utf-8').read().splitlines()
    kept, dropped = [], 0
    for ln in lines:
        if any(k in ln for k in keys):
            dropped += 1
            continue
        kept.append(ln)
    if note:
        kept.append('')
        kept.append(note)
    io.open(p, 'w', encoding='utf-8').write('\n'.join(kept) + '\n')
    print('[ok] %-32s 删除行 %d' % (path, dropped))

drop_lines('docs/archive/INDEX.md',
           ['COMMERCIAL_API_PLAN', 'THEORETICAL_CEILING_FRAMEWORK', 'SERVER_SPEC', 'SERVER_PHASE0', 'SERVER_P100'],
           '> **2026-09-15 删除（用户授权）**：%%BT%%COMMERCIAL_API_PLAN.md%%BT%%（自身标注已作废）、'
           '%%BT%%THEORETICAL_CEILING_FRAMEWORK.md%%BT%%（前提为已证伪的加法公式）、'
           '%%BT%%SERVER_SPEC/SERVER_PHASE0/SERVER_P100.md%%BT%%（云端路线已弃用）。'
           '残值与恢复方式见 %%BT%%../decisions.md%%BT%% D16；git 历史保留原文。'.replace('%%BT%%', BT))

drop_lines('docs/handoff/INDEX.md', ['NEXT_ROUND_PROMPT_20260916'],
           '> **2026-09-15 删除（用户授权）**：%%BT%%NEXT_ROUND_PROMPT_20260916.md%%BT%%（加法分数模型时代，已被 15PM/15NIGHT 取代）'
           '；余下 5 份中间期交接提示与 2 份规则快照**待用户确认后亦可删除**。原见 %%BT%%../decisions.md%%BT%% D16。'.replace('%%BT%%', BT))

drop_lines('docs/project_map.md', ['SERVER_SPEC', 'SERVER_PHASE0', 'SERVER_P100', 'COMMERCIAL_API_PLAN', 'THEORETICAL_CEILING_FRAMEWORK'],
           '（2026-09-15 删除 5 份：商业 API 计划、旧公式论证框架、服务器规格/就绪/P100 清单 —— 见 %%BT%%../decisions.md%%BT%% D16）'.replace('%%BT%%', BT))

# architecture.md 里的 SERVER_* 引用
a = ROOT / 'docs/architecture.md'
at = io.open(a, encoding='utf-8').read()
old = '（腾讯云 V100 / 教学 P100 均关机，仅存档，见 ' + BT + 'docs/archive/SERVER_*.md' + BT + '）'
new = '（腾讯云 V100 / 教学 P100 均关机；服务器文档已于 2026-09-15 删除，SSH/兼容性经验留在 ' + BT + 'docs/handoff/CLAUDE_20260901_原始快照.md' + BT + '，见 decisions D16）'
if old in at:
    at = at.replace(old, new, 1); io.open(a, 'w', encoding='utf-8').write(at); print('[ok] architecture.md 引用已更新')
else:
    print('[MISS] architecture 引用')
# project_map 里 handoff 计数
p = ROOT / 'docs/project_map.md'
pt = io.open(p, encoding='utf-8').read()
pt = pt.replace('### docs/handoff/（时间序列历史记录，18 个，含 FINAL_PLAN 与 INDEX）',
                '### docs/handoff/（时间序列历史记录，17 个，含 FINAL_PLAN 与 INDEX）')
pt = pt.replace('### docs/archive/（前提已失效、引用会误导，8 个 + INDEX.md）',
                '### docs/archive/（前提已失效、引用会误导，3 个 + INDEX.md）')
io.open(p, 'w', encoding='utf-8').write(pt)
print('[ok] project_map.md 计数更新')
