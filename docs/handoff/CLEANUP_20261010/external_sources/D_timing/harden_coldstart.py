# -*- coding: utf-8 -*-
"""runner 冷启动加固 (2026-09-11)
① 在 import torch 之前显式 TORCHDYNAMO_DISABLE=1 (实测首次调用 6.2s -> 2.9s)
② torch.load(..., mmap=True) 惰性映射 (实测 880ms -> 43ms), 带类型/异常回退
用法: python harden_coldstart.py [--apply] <model_dir> [...]
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

APPLY = '--apply' in sys.argv

NEW_ENV = ("os.environ.setdefault('TORCHDYNAMO_SUPPRESS_ERRORS', '1')\n"
           "# 2026-09-11 冷启动加固: 显式关闭 dynamo, 避免首次调用触发编译尝试(实测省 3s+)\n"
           "os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')\n")

OLD_LOAD = ("pkg = torch.load(os.path.join(model_dir, 'your_model.pt'),\n"
            "                             map_location='cpu', weights_only=False)")
NEW_LOAD = ("pkg = torch.load(os.path.join(model_dir, 'your_model.pt'),\n"
            "                             map_location='cpu', weights_only=False, mmap=True)")
OLD_LOAD2 = ("pkg = torch.load(os.path.join(model_dir, 'your_model.pt'), map_location='cpu', weights_only=False)")
NEW_LOAD2 = ("pkg = torch.load(os.path.join(model_dir, 'your_model.pt'), map_location='cpu',\n"
             "                             weights_only=False, mmap=True)")

FALLBACK = """        except TypeError:  # torch<2.0 无 weights_only 参数
            pkg = torch.load(os.path.join(model_dir, 'your_model.pt'), map_location='cpu')
        except Exception:  # mmap 不可用(非 zip 格式/旧版 torch) -> 退回普通加载
            pkg = torch.load(os.path.join(model_dir, 'your_model.pt'),
                             map_location='cpu', weights_only=False)"""


def harden(md: Path):
    rp = md / 'runner.py'
    if not rp.exists():
        return ['!! 无 runner.py']
    text = rp.read_text(encoding='utf-8')
    log = []
    if 'TORCHDYNAMO_DISABLE' in text:
        log.append('已在加固状态')
        return log
    # ① env
    old_env = "os.environ.setdefault('TORCHDYNAMO_SUPPRESS_ERRORS', '1')\n"
    if old_env in text:
        text = text.replace(old_env, NEW_ENV, 1)
        log.append('已插入 TORCHDYNAMO_DISABLE')
    else:
        log.append('!! 未找到 SUPPRESS_ERRORS 锚点')
    # ② mmap
    if OLD_LOAD in text:
        text = text.replace(OLD_LOAD, NEW_LOAD, 1)
        log.append('torch.load -> mmap=True')
    elif OLD_LOAD2 in text:
        text = text.replace(OLD_LOAD2, NEW_LOAD2, 1)
        log.append('torch.load -> mmap=True (单行版)')
    else:
        log.append('!! 未找到 torch.load 锚点')
    # ③ 回退分支
    if "except TypeError:  # torch<2.0 无 weights_only 参数" in text and 'mmap 不可用' not in text:
        text = text.replace("""        except TypeError:  # torch<2.0 无 weights_only 参数
            pkg = torch.load(os.path.join(model_dir, 'your_model.pt'), map_location='cpu')""",
                            FALLBACK, 1)
        log.append('已加 mmap 回退分支')
    if APPLY:
        rp.write_text(text, encoding='utf-8', newline='')
        try:
            py_compile.compile(str(rp), doraise=True)
            log.append('py_compile OK')
        except Exception as e:
            log.append(f'!! 编译失败: {e}')
    return log


dirs = [Path(a) for a in sys.argv[1:] if not a.startswith('--')]
if not dirs:
    dirs = [Path(r'D:/models/final_compliant/model_dir'),
            Path(r'D:/models/final_compliant_14b/model_dir'),
            Path(r'D:/models/final_compliant_m0s7/model_dir'),
            Path(r'D:/models/final_compliant_g3w18s1/model_dir'),
            Path(r'D:/models/wave2_m0s7_s097/pack_model_dir')]
for d in dirs:
    print(f'== {d}')
    for ln in harden(d):
        print('   ' + ln)
print('\n(干跑; 加 --apply 生效)' if not APPLY else '\n(已写入)')
