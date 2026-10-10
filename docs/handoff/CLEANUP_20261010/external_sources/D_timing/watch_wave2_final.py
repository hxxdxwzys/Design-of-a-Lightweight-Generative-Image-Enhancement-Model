# -*- coding: utf-8 -*-
"""等待波 2 的在跑任务落地, 然后一次性打印最终结果 (2026-09-11)

等待两个条件同时满足 (或超时 3h):
  A. wave2_overnight.json 里出现 m0s1_s100 且带 test_clip (第 4 个候选渲染+直证完成)
  B. pack_wave2.py 进程退出 (第一个波 2 zip 打包完成)
然后输出:
  1. 四个候选的 val / test clip 汇总
  2. 第一个波 2 zip 的结构 + 语法审计 + IQA 遮蔽审计
"""
from __future__ import annotations

import json
import subprocess
import time
import zipfile
from pathlib import Path

PY = r'D:/Python/python.exe'
OUT = Path(r'D:/tmp_timing/wave2_overnight.json')
ZIP = Path(r'D:/models/Track2_14B_M0S7_SC097.zip')
DEADLINE = time.time() + 3 * 3600
PS = ['powershell', '-NoProfile', '-Command',
      "(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
      "Where-Object {$_.CommandLine -like '*pack_wave2*'}).ProcessId"]


def s100_done():
    try:
        d = json.loads(OUT.read_text(encoding='utf-8'))
    except Exception:
        return False
    e = d.get('m0s1_s100')
    return bool(e and e.get('test_clip'))


def pack_running():
    r = subprocess.run(PS, capture_output=True, text=True)
    return bool((r.stdout or '').strip())


while time.time() < DEADLINE:
    if s100_done() and not pack_running():
        break
    time.sleep(60)

print('==================== WAVE2_FINAL_BEGIN ====================')
print('s100_done=%s pack_running=%s elapsed_until=%s'
      % (s100_done(), pack_running(), time.strftime('%H:%M:%S')))

# 1. 候选汇总
try:
    d = json.loads(OUT.read_text(encoding='utf-8'))
except Exception as e:
    d = {}
    print('!! wave2_overnight.json 读取失败:', e)
print('\n--- 候选汇总 (val / test100) ---')
for tag, e in d.items():
    v = e.get('val') or {}
    t = e.get('test_clip') or {}
    print('%-12s %-7s val=(%s, %s) d=(%s, %s)  test_clip mean=%s std=%s n=%s'
          % (tag, e.get('branch'), v.get('clip'), v.get('lpips'),
             v.get('d_clip'), v.get('d_lpips'),
             t.get('mean'), t.get('std'), t.get('n')))

# 2. zip 验证 (非 GPU 项)
print('\n--- 波 2 zip ---')
if not ZIP.exists():
    print('!! 不存在:', ZIP)
else:
    print('size GB: %.3f' % (ZIP.stat().st_size / 1e9))
    try:
        z = zipfile.ZipFile(ZIP)
        names = z.namelist()
        print('entries:', len(names), '| jpg:', sum(1 for n in names if n.lower().endswith('.jpg')),
              '| runner.py:', any(n.endswith('model_dir/runner.py') for n in names),
              '| your_model.pt:', any(n.endswith('model_dir/your_model.pt') for n in names))
        for i in z.infolist():
            if i.filename.endswith('model_dir/runner.py'):
                txt = z.read(i.filename).decode('utf-8', 'replace')
                print('runner patched (_PATH_BEFORE):', '_PATH_BEFORE' in txt)
        print('ZIP_VALID')
    except Exception as e:
        print('!! zip 无效:', type(e).__name__, e)
    for tool in ('audit_py_compat.py', 'harness_sim_zip.py'):
        p = Path(r'D:/tmp_timing') / tool
        if not p.exists():
            continue
        try:
            r = subprocess.run([PY, str(p), str(ZIP)], capture_output=True, text=True, timeout=3600)
            out = ((r.stdout or '') + (r.stderr or '')).strip()
            print('\n[%s] rc=%s' % (tool, r.returncode))
            print(out[-900:])
        except Exception as e:
            print('[%s] 异常 %s' % (tool, e))

sim_md = Path(r'D:/models/_harness_zip_sim/model_dir')
if sim_md.exists():
    r = subprocess.run([PY, r'D:/tmp_timing/audit_iqa_shadow.py', str(sim_md)],
                       capture_output=True, text=True, timeout=1800)
    txt = ((r.stdout or '') + (r.stderr or '')).strip()
    print('\n[audit_iqa_shadow] rc=%s' % r.returncode)
    print(txt[-700:])

print('==================== WAVE2_FINAL_END ====================')
