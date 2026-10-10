# -*- coding: utf-8 -*-
"""RCON-01 工作副本 — 上传前总门禁（失败传播窄修复）

来源（只读原件，未修改）:
    D:/tmp_timing/preupload_gate.py
    sha256 = acff56c784e703a2cb8513519098098cfc08ead8c0047c15840e84b1014c8478
    2026-09-16 读取；与 docs/review.md 记录的哈希一致（无版本漂移）。

状态: **待审查工具**（docs/current_task.md §2）。D:/tmp_timing 的原调用入口**未切换**；
      本副本不接管上传流程，也不构成任何“可以上传”的许可。

本副本相对原件只做 current_task §4 指定的窄修复，保留原 10 项检查意图与全部业务判据
（ASCII/结构/体积/出图尺寸/语法兼容/接口/两种 IQA 遮蔽/官方格式）:

  1. 每个子工具必须同时满足【退出码 0】+【对应终态成功标记】+【无明确失败标记】；
     harness 采用最终 HARNESS_ZIP_SIM_PASS，不再接受中间态 "infer OK"。
  2. 超时/异常/缺工具/缺依赖/未执行 = FAIL 或 NOT_RUN，绝不计 PASS；
     任一必需项 FAIL 或 NOT_RUN 时，总体不得返回“全部通过”。
  3. harness 未在本包成功运行时，拒绝复用共享 SIM_DIR 残留的遮蔽检查结果（③④ 记 NOT_RUN）。
  4. 子进程执行器与工具路径可注入（--tool-root/--python/--sim-dir/--timeout），便于 CPU 测试；
     默认使用显式解释器，不隐式调用 GPU 检查。
  5. --stage-upload 被明确拒绝，且拒绝后【立即停止】：不写入、不归位、不覆盖任何真实包，
     也不调用任何子工具；最终文案只报告“已执行的技术检查”，已移除“可以上传”的自动许可。
  5b. 【返修】默认【计划模式】：不加 --execute-tools 时绝不调用任何子工具，因为真实链路有
     副作用（harness 会 rmtree+重建解压目录，并在其中 import runner 做 512 前向）。
  5c. 【返修/R3】③④ 检查的目录必须绑定到本包本次运行：目录需已被重建（mtime 变化）、harness
     自报解压数量与包内 output_dir 成员数一致，且【解压目录 model_dir 与本 ZIP 的 model_dir
     文件集合与内容哈希完全一致】；任一条不成立即 NOT_RUN。仅凭前两条无法排除串包。
  5d. 【返修】成功标记要求【整行相等】（NOT_X / X_FAILED 不再被判 PASS）；失败标记按整词。
  6. 未改动业务接口、输出张量、质量阈值、依赖或导出格式；其它缺陷只登记不混修。

用法:
    python preupload_gate_rcon.py <zip> [more.zip ...]
    python preupload_gate_rcon.py <zip> --tool-root <目录> --python <解释器> --sim-dir <目录>

退出码: 0 仅当所有包的全部必需检查 PASS 且无 NOT_RUN；否则 1。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PY = r'D:/Python/python.exe'
TOOLS = Path(r'D:/tmp_timing')
SIM_DIR = Path(r'D:/models/_harness_zip_sim')
# harness_sim_zip.py 第 16 行把解压目录硬编码为本路径，且不接受目录参数。
# ⇒ --sim-dir 只有在等于该固定路径时才可能与本包产物绑定；否则必须拒绝执行遮蔽检查。
HARNESS_FIXED_SIM_DIR = Path(r'D:/models/_harness_zip_sim')
MAX_GB = 10.0
HASH_CHUNK = 1 << 23
# model_dir 内容哈希预算（默认 8 GiB）：超出即 NOT_RUN，绝不因“太大”而跳过核对。
MODEL_HASH_MAX_BYTES = 8 * 1024 ** 3
ASCII_RE = re.compile(r'^[\x20-\x7e]+$')

PASS, FAIL, NOT_RUN = 'PASS', 'FAIL', 'NOT_RUN'
ICON = {PASS: '✅', FAIL: '❌', NOT_RUN: '⏸'}
TRACEBACK = 'Traceback (most recent call last)'
UPLOAD_DIR = Path(r'D:/models/upload')   # 仅用于在拒绝归位时报告路径，本副本从不写入


# ───────────────────────── 原有本地检查（未改判据） ─────────────────────────
# 以下四个函数与原件逐字一致，业务判据未调整。

def check_ascii_name(p: Path):
    bad = [c for c in p.name if ord(c) > 126]
    if bad:
        return False, f'文件名含非 ASCII 字符: {"".join(bad)!r} -> 会重演「文件名含中文」失败'
    return True, f'文件名纯 ASCII: {p.name}'


def check_inner_names(p: Path):
    with zipfile.ZipFile(p) as z:
        bad = [n for n in z.namelist() if not ASCII_RE.match(n)]
    if bad:
        return False, f'zip 内有 {len(bad)} 个非 ASCII 路径 (前3: {bad[:3]})'
    return True, 'zip 内路径全部 ASCII'


def check_structure(p: Path):
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        bad = z.testzip()
    if bad is not None:
        return False, f'zip 损坏 @ {bad}'
    jpgs = [n for n in names if n.startswith('output_dir/') and n.lower().endswith('.jpg')]
    mds = [n for n in names if n.startswith('model_dir/')]
    problems = []
    if len(jpgs) != 100:
        problems.append(f'output_dir 有 {len(jpgs)} 张 jpg (需 100)')
    for must in ('model_dir/runner.py', 'model_dir/your_model.pt'):
        if must not in names:
            problems.append(f'缺 {must}')
    if problems:
        return False, '; '.join(problems)
    return True, f'testzip OK; output_dir {len(jpgs)} jpg; model_dir {len(mds)} 项'


def check_size(p: Path):
    gb = p.stat().st_size / 2 ** 30
    if gb > MAX_GB:
        return False, f'{gb:.2f} GiB > {MAX_GB} GB 上限'
    return True, f'{gb:.2f} GiB (上限 {MAX_GB} GB)'


def check_jpg_dims(p: Path, n=3):
    """抽 n 张出图检查尺寸(应为 4K)与可解码性

    RCON-01 改动（唯一一处本地检查改动）: 缺 PIL 依赖时返回 None -> 记为 NOT_RUN。
    原实现返回 True/"SKIP"，会把缺依赖当成 PASS（current_task §4.2）。尺寸阈值 min>=2000 未改。
    """
    try:
        from PIL import Image
    except Exception as e:
        return None, f'NOT_RUN 缺依赖 PIL 不可用: {e}'
    with zipfile.ZipFile(p) as z:
        jpgs = sorted(n_ for n_ in z.namelist()
                      if n_.startswith('output_dir/') and n_.lower().endswith('.jpg'))
        if not jpgs:
            return False, '无 jpg 可查'
        picks = [jpgs[0], jpgs[len(jpgs) // 2], jpgs[-1]][:n]
        info = []
        for nm in picks:
            import io
            with Image.open(io.BytesIO(z.read(nm))) as im:
                w, h = im.size
                info.append(f'{Path(nm).name} {w}x{h}')
                if min(w, h) < 2000:
                    return False, f'{Path(nm).name} 尺寸 {w}x{h} 不是 4K 量级'
        return True, '; '.join(info)


def default_local_checks():
    """原 ①组 的 5 个子检查，名称与顺序保持与原件一致。"""
    return (
        ('① 文件名 ASCII', check_ascii_name),
        ('   内部路径 ASCII', check_inner_names),
        ('   结构与完整性', check_structure),
        ('   体积', check_size),
        ('   出图尺寸', check_jpg_dims),
    )


# ───────────────────────── 子工具判定（RCON-01 修复 1/2） ─────────────────────────

class ToolSpec:
    """一个外置子工具的判定契约（RCON-01 返修：完整终态协议，非子串）。

    success_lines: 【完整终态协议】要求逐行 strip 后【整行相等】，且列出的行全部出现。
        子串匹配被明确禁止：NOT_HARNESS_ZIP_SIM_PASS 与 HARNESS_ZIP_SIM_PASS_FAILED
        都不等于 HARNESS_ZIP_SIM_PASS，因此判 FAIL（fail-closed）。
    failure_tokens: 失败标记按【整词】匹配（前后不得为 A-Za-z0-9_），允许出现在行内，
        例如 "  ==> HARD_AUDIT_FAIL" 与 "IQA_SHADOW_AUDIT_FAIL  (原因...)"。
    """

    __slots__ = ('name', 'script', 'success_lines', 'failure_tokens')

    def __init__(self, name, script, success_lines, failure_tokens=()):
        self.name = name
        self.script = script
        self.success_lines = tuple(success_lines)
        self.failure_tokens = tuple(failure_tokens)


# 终态标记取自各子工具源码（2026-09-16 静态核对；本次未修改这些工具）：
#   audit_py_compat.py      末行 "合计有风险文件数: N"；缺文件时打 "!! 缺文件"
#   harness_sim_zip.py      末行 "HARNESS_ZIP_SIM_PASS"（"infer OK" 只是中间态）
#   audit_iqa_shadow.py     "IQA_SHADOW_AUDIT_PASS" / "IQA_SHADOW_AUDIT_FAIL"
#   audit_iqa_shadow_hard.py "HARD_ALL_PASS" / "HARD_ANY_FAIL"（逐目录另有 HARD_AUDIT_*）
#   audit_official_format.py "FORMAT_ALL_PASS" / "FORMAT_ANY_FAIL"（逐包另有 FORMAT_*）
TOOL_SPECS = {
    'compat': ToolSpec('② 语法兼容 (|None)', 'audit_py_compat.py',
                       ('合计有风险文件数: 0',), ('!! 缺文件',)),
    'harness': ToolSpec('   接口/512 推理', 'harness_sim_zip.py',
                        ('HARNESS_ZIP_SIM_PASS',), ()),
    'shadow': ToolSpec('③ IQA 遮蔽', 'audit_iqa_shadow.py',
                       ('IQA_SHADOW_AUDIT_PASS',), ('IQA_SHADOW_AUDIT_FAIL',)),
    'shadow_hard': ToolSpec('④ IQA 遮蔽(对抗)', 'audit_iqa_shadow_hard.py',
                            ('HARD_ALL_PASS',), ('HARD_ANY_FAIL', 'HARD_AUDIT_FAIL')),
    'format': ToolSpec('⑤ 官方格式严格审计', 'audit_official_format.py',
                       ('FORMAT_ALL_PASS',), ('FORMAT_ANY_FAIL', 'FORMAT_FAIL')),
}
TOOL_ORDER = ('compat', 'harness', 'shadow', 'shadow_hard', 'format')


class ExecResult:
    """子进程执行结果。error 非 None 表示【未正常完成】，一律不得判 PASS。"""

    __slots__ = ('rc', 'out', 'error')

    def __init__(self, rc=None, out='', error=None):
        self.rc = rc
        self.out = out or ''
        self.error = error


class SubprocessExecutor:
    """默认执行器：显式解释器 + 显式工具目录。可被测试用的假执行器替换。"""

    def __init__(self, tool_root, python=PY, timeout=3600):
        self.tool_root = Path(tool_root)
        self.python = str(python)
        self.timeout = timeout

    def __call__(self, script, args, cwd=None):
        tool = self.tool_root / script
        if not tool.is_file():
            return ExecResult(error='missing_tool:%s' % tool)
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        try:
            r = subprocess.run([self.python, str(tool), *map(str, args)],
                               capture_output=True, text=True, encoding='utf-8',
                               errors='replace', timeout=self.timeout, cwd=cwd, env=env)
        except subprocess.TimeoutExpired as e:
            tail = _tail(_as_text(e.stdout) + _as_text(e.stderr))
            return ExecResult(error='timeout', out=tail)
        except FileNotFoundError as e:
            return ExecResult(error='missing_dependency:%s' % e)
        except OSError as e:
            return ExecResult(error='exception:%s' % e)
        return ExecResult(rc=r.returncode, out=(r.stdout or '') + (r.stderr or ''))


def _as_text(v):
    if v is None:
        return ''
    return v if isinstance(v, str) else v.decode('utf-8', 'replace')


def _tail(text, n=6, width=160):
    lines = [ln.strip()[:width] for ln in (text or '').splitlines() if ln.strip()]
    return ' | '.join(lines[-n:])


def _lines(text):
    return [ln.strip() for ln in (text or '').splitlines()]


def _token_hits(text, tokens):
    """整词匹配：前后不得为 A-Za-z0-9_（避免 NOT_X / X_FAILED 之类被误判）。"""
    lines = _lines(text)
    hits = []
    for tok in tokens:
        rx = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(tok) + r'(?![A-Za-z0-9_])')
        if any(rx.search(ln) for ln in lines):
            hits.append(tok)
    return hits


def _dir_state(p: Path):
    """共享解压目录的可核验指纹：不存在 -> None，否则 mtime_ns。"""
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return None


def _sha256_file(p):
    h = hashlib.sha256()
    n = 0
    with open(p, 'rb') as f:
        while True:
            b = f.read(HASH_CHUNK)
            if not b:
                break
            n += len(b)
            h.update(b)
    return h.hexdigest(), n


def _zip_dir_digest(zf, prefix):
    """{相对路径: (未压缩字节, sha256)}：直接流式读取 zip 成员，不解压落盘。"""
    out = {}
    total = 0
    for info in zf.infolist():
        if info.is_dir() or not info.filename.startswith(prefix):
            continue
        h = hashlib.sha256()
        with zf.open(info) as f:
            while True:
                b = f.read(HASH_CHUNK)
                if not b:
                    break
                h.update(b)
        out[info.filename[len(prefix):]] = (info.file_size, h.hexdigest())
        total += info.file_size
    return out, total


def _disk_dir_digest(root: Path):
    out = {}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            h, n = _sha256_file(p)
            out[str(p.relative_to(root)).replace('\\', '/')] = (n, h)
    return out


def _model_binding(zip_path, sim_dir, max_bytes):
    """核对解压目录的 model_dir 与本 ZIP 的 model_dir：文件集合 + 内容哈希。

    只靠“目录 mtime 变化 + 图片数量相同”无法排除串包（另一个包同样 100 张图）。
    返回 None 表示绑定成立；否则返回 NOT_RUN 原因。超预算或读不到一律 NOT_RUN。
    """
    md = Path(sim_dir) / 'model_dir'
    try:
        with zipfile.ZipFile(zip_path) as z:
            want, total = _zip_dir_digest(z, 'model_dir/')
    except Exception as e:
        return '无法读取包内 model_dir 成员: %s: %s' % (type(e).__name__, e)
    if not want:
        return '包内没有 model_dir/ 成员，无法核对模型归属'
    if total > max_bytes:
        return ('包内 model_dir 未压缩体积 %.1f MiB 超出哈希预算 %.1f MiB，'
                '无法完成内容核对' % (total / 2 ** 20, max_bytes / 2 ** 20))
    try:
        got = _disk_dir_digest(md)
    except Exception as e:
        return '无法读取解压目录 model_dir: %s: %s' % (type(e).__name__, e)
    if set(got) != set(want):
        return ('解压目录 model_dir 文件集合与包内不一致（缺 %s / 多 %s）'
                % (sorted(set(want) - set(got))[:3], sorted(set(got) - set(want))[:3]))
    diff = [k for k in sorted(want) if got[k] != want[k]]
    if diff:
        return ('解压目录 model_dir 内容哈希与包内不一致（示例 %s，共 %d 个文件）'
                % (diff[:2], len(diff)))
    return None


def _zip_output_dir_members(p):
    with zipfile.ZipFile(p) as z:
        return sum(1 for n in z.namelist() if n.startswith('output_dir/'))


def _parse_extract_count(out):
    """harness 第 [1] 步自报的解压数量：'[1] 解压 OK: output_dir=N ...'"""
    for ln in _lines(out):
        m = re.match(r'^\[1\] 解压 OK: output_dir=(\d+)', ln)
        if m:
            return int(m.group(1))
    return None


def evaluate_tool(spec: ToolSpec, res: ExecResult):
    """把一次子工具执行换算成 PASS / FAIL / NOT_RUN + 理由。

    判定顺序：执行异常 -> Traceback -> 明确失败标记 -> 退出码 -> 终态成功标记行。
    成功要求【整行相等】；失败按【整词】。二者同时出现时必须 FAIL（§5 判据）。
    """
    if res.error:
        kind, _, detail = res.error.partition(':')
        if kind == 'missing_tool':
            return NOT_RUN, 'NOT_RUN 缺工具 %s' % detail
        if kind == 'missing_dependency':
            return NOT_RUN, 'NOT_RUN 缺依赖 %s' % detail
        if kind == 'timeout':
            return FAIL, '超时未完成 (timeout) %s' % _tail(res.out, 3)
        return FAIL, '执行异常 %s' % res.error

    if TRACEBACK in res.out:
        return FAIL, '子工具抛异常 (Traceback) %s' % _tail(res.out, 3)

    hits = _token_hits(res.out, spec.failure_tokens)
    if hits:
        return FAIL, '出现明确失败标记 %s' % hits

    if res.rc != 0:
        return FAIL, '退出码 %s (需 0)' % res.rc

    lines = _lines(res.out)
    missing = [m for m in spec.success_lines if m not in lines]
    if missing:
        return FAIL, ('rc=0 但缺终态标记行（需整行相等，非子串）: %s'
                      % ' / '.join(repr(m) for m in missing))
    return PASS, 'rc=0 + 终态标记行齐全 %s' % [m for m in spec.success_lines]


def _sim_binding(zip_path, sim_dir, pre_state, out, max_bytes=MODEL_HASH_MAX_BYTES):
    """把 ③④ 要检查的目录绑定到【本包本次运行】的解压产物。

    绑定证据（四条全部满足才算成立，fail-closed，绝不退回“目录存在即用”）：
      1) 该目录下存在 model_dir；
      2) 该目录在本次 harness 调用期间被重建（mtime 变化；原先不存在则视为新建）；
      3) harness 自报的 output_dir 解压数量与本包实际成员数一致；
      4) 【R3】解压目录 model_dir 与本 ZIP 的 model_dir 文件集合与内容哈希完全一致
         —— 仅凭 2)+3) 无法排除“另一个同样 100 张图的模型包”串包。
    注意：真实 harness_sim_zip.py 把解压目录硬编码为 D:/models/_harness_zip_sim 且不接受
    目录参数，因此把 --sim-dir 指到别处时第 2 条通常不成立 -> 直接 NOT_RUN（不会检查旧目录）。
    """
    md = Path(sim_dir) / 'model_dir'
    if not md.is_dir():
        return '本次 harness 未产出 %s' % md
    post = _dir_state(Path(sim_dir))
    if pre_state is not None and post is not None and pre_state == post:
        return ('共享解压目录在本次运行中 mtime 未变（未被重建），'
                '无法证明它属于本包本次运行，拒绝复用')
    try:
        want = _zip_output_dir_members(zip_path)
    except Exception as e:
        return '无法读取包内 output_dir 成员数: %s: %s' % (type(e).__name__, e)
    got = _parse_extract_count(out)
    if got is None:
        return 'harness 输出缺少 [1] 解压 OK: output_dir=N 行，无法核对解压归属'
    if got != want:
        return 'harness 报告 output_dir=%d 而本包实际 %d 个成员，解压产物与包不匹配' % (got, want)
    # 关键：图片数量相同不能证明其中的模型来自本 ZIP（两个不同模型包都含 100 张图）
    return _model_binding(zip_path, sim_dir, max_bytes)


# ───────────────────────── 主流程 ─────────────────────────

def run_gate(zips, *, executor, local_checks, sim_dir=SIM_DIR, stage_upload=False,
             execute_tools=True, model_hash_max_bytes=MODEL_HASH_MAX_BYTES,
             out=print, timeout_note=None):
    """execute_tools=False 为【计划模式】：只做无副作用的本地只读检查，绝不调用子工具。"""
    if stage_upload:
        # 返修：拒绝归位时必须【立即停止】，不得继续调用任何子工具。
        out('  ⛔ --stage-upload 未获授权: 本工作副本不写入、不归位、不覆盖任何真实包。')
        out('     拒绝原因: RCON-01 §4.5；真正归位仍由用户决定的上传流程负责。')
        out('STAGE_UPLOAD_UNAUTHORIZED')
        out('  已拒绝并立即停止：未执行任何检查、未调用任何子工具、未触碰任何目录。')
        out('PREUPLOAD_ANY_FAIL')
        return 2

    ok_all = True
    if not execute_tools:
        out('  [i] 计划模式：只做本地只读检查，不调用任何子工具（真实子工具链有副作用）。')

    for zp in zips:
        src = Path(zp)
        out('=' * 74)
        out(f'门禁(工作副本): {src}')
        out('=' * 74)
        if not src.exists():
            out('  ❌ 文件不存在')
            ok_all = False
            continue
        if timeout_note:
            out(f'  [i] {timeout_note}')

        results = []   # (name, verdict, detail)

        # ① 本地五项
        for name, fn in local_checks:
            try:
                ok, msg = fn(src)
            except Exception as e:
                ok, msg = False, f'{type(e).__name__}: {e}'
            verdict = NOT_RUN if ok is None else (PASS if ok else FAIL)
            results.append((name, verdict, msg))
            out(f'  {ICON[verdict]} {name}: {msg}')

        # ②③④⑤ + 接口
        harness_verdict = NOT_RUN
        sim_binding = 'harness 未运行'
        for key in TOOL_ORDER:
            spec = TOOL_SPECS[key]
            if not execute_tools:
                verdict, detail = NOT_RUN, ('NOT_RUN 计划模式未执行（需 --execute-tools；'
                                            '真实子工具链会删除/重建解压目录并运行模型前向）')
            elif key in ('shadow', 'shadow_hard'):
                # 只允许【本包本次 harness 成功且解压归属已绑定】的产物进入遮蔽检查
                if harness_verdict != PASS:
                    verdict = NOT_RUN
                    detail = ('NOT_RUN 本包 harness 未成功运行 -> 拒绝复用共享解压目录 '
                              '(%s) 的旧结果' % sim_dir)
                elif sim_binding is not None:
                    verdict, detail = NOT_RUN, 'NOT_RUN 解压归属未绑定: %s' % sim_binding
                else:
                    verdict, detail = evaluate_tool(
                        spec, executor(spec.script, [Path(sim_dir) / 'model_dir']))
            else:
                pre_state = _dir_state(Path(sim_dir)) if key == 'harness' else None
                res = executor(spec.script, [src])
                verdict, detail = evaluate_tool(spec, res)
                if key == 'harness':
                    harness_verdict = verdict
                    if verdict == PASS:
                        sim_binding = _sim_binding(src, Path(sim_dir), pre_state, res.out,
                                                   max_bytes=model_hash_max_bytes)
                        if sim_binding is not None:
                            out('     [i] 解压归属未绑定: %s' % sim_binding)
            results.append((spec.name, verdict, detail))
            out(f'  {ICON[verdict]} {spec.name}: {detail}')

        fails = [n for n, v, _ in results if v == FAIL]
        notrun = [n for n, v, _ in results if v == NOT_RUN]
        if fails:
            verdict = 'PREUPLOAD_GATE_FAIL (%s)' % ', '.join(fails)
        elif notrun:
            verdict = 'PREUPLOAD_GATE_INCOMPLETE (%s)' % ', '.join(notrun)
        else:
            verdict = 'PREUPLOAD_GATE_PASS'
        out(f'  ==> {verdict}')
        ok_all = ok_all and not fails and not notrun

    out('=' * 74)
    if not execute_tools:
        out('SIDE_EFFECTS_NOT_EXECUTED（计划模式：未调用任何子工具）')
    if ok_all:
        out('总体: ✅ 已执行的技术检查全部通过')
        out('      （这是技术检查范围的结果，不是官方合规认证，也不构成上传许可）')
    else:
        out('总体: ❌ 存在问题、存在未运行项，或归位未获授权')
    out('PREUPLOAD_ALL_PASS' if ok_all else 'PREUPLOAD_ANY_FAIL')
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description='RCON-01 门禁工作副本（失败传播窄修复）')
    ap.add_argument('zips', nargs='+')
    ap.add_argument('--tool-root', default=str(TOOLS), help='子工具目录（默认原件目录，只读调用）')
    ap.add_argument('--python', default=PY, help='子进程解释器')
    ap.add_argument('--sim-dir', default=str(SIM_DIR), help='harness 解压目录（只读，不删除）')
    ap.add_argument('--timeout', type=float, default=3600, help='单个子工具超时秒数')
    ap.add_argument('--model-hash-max-mb', type=float, default=MODEL_HASH_MAX_BYTES / 2 ** 20,
                    help='核对解压目录 model_dir 内容哈希的预算(MiB)；超出或无法核对应判 NOT_RUN')
    ap.add_argument('--execute-tools', action='store_true',
                    help='真正执行外置子工具链。默认【不执行】：harness_sim_zip.py 会 '
                         'rmtree+重建解压目录，并在其中 import runner 做 512 前向（可能用 CUDA）')
    ap.add_argument('--stage-upload', action='store_true',
                    help='保留兼容；本副本明确拒绝，且拒绝后立即停止、不调用任何子工具')
    ap.add_argument('--upload-name', default=None, help='保留兼容；本副本不使用')
    a = ap.parse_args(argv)

    if Path(a.sim_dir).resolve() != HARNESS_FIXED_SIM_DIR.resolve():
        print('[i] 注意：真实 harness_sim_zip.py 把解压目录硬编码为 %s，不接受目录参数。'
              % HARNESS_FIXED_SIM_DIR)
        print('    --sim-dir 指向别处时，若该目录未被本次运行重建，③④ 将判 NOT_RUN（fail-closed）。')

    if a.execute_tools:
        print('[!] 已启用 --execute-tools：将执行真实子工具链，已知副作用：')
        print('    - 工具目录 %s 下的 harness_sim_zip.py 会删除并重建解压目录 %s'
              % (a.tool_root, a.sim_dir))
        print('    - 该目录内会 import runner 并做 512x512 前向（加载权重，可能使用 CUDA）')
        print('    - ③④ 遮蔽检查依赖该目录，且仅在 harness 本次成功且解压归属绑定时执行')
        print('    若只需无副作用检查，请去掉该参数（默认计划模式）。')

    ex = SubprocessExecutor(a.tool_root, python=a.python, timeout=a.timeout)
    return run_gate(a.zips, executor=ex, local_checks=default_local_checks(),
                    sim_dir=a.sim_dir, stage_upload=a.stage_upload,
                    execute_tools=a.execute_tools,
                    model_hash_max_bytes=a.model_hash_max_mb * 2 ** 20)


if __name__ == '__main__':
    sys.exit(main())
