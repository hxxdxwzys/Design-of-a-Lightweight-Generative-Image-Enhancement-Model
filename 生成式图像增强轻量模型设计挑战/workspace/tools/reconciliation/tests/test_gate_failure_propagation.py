# -*- coding: utf-8 -*-
"""RCON-01 返修后 CPU 单元测试 — 门禁失败传播 + 边界（预登记判据见 docs/current_task.md §5）

原 18 项判据全部保留；返修新增以下负测试（对应 Architect REQUEST_CHANGES 四项）:
  A. 只读复跑副作用: 计划模式(默认)不得调用任何子工具; --stage-upload 拒绝后必须立即停止。
  B. --sim-dir 未绑定真实产物: 目录未被本次运行重建 / 解压数量与包不符 -> ③④ 必须 NOT_RUN。
  C. 成功标记误识别: NOT_HARNESS_ZIP_SIM_PASS、HARNESS_ZIP_SIM_PASS_FAILED 必须 FAIL。
  D. 证据对齐: 相关引用与哈希由 evidence/ 的生成器自校验（不在本文件内断言）。
  E. [R3] 串包防护: 同数量不同权重 / model_dir 文件集合不符 / 只改目录时间 / 超哈希预算
     -> ③④ 必须 NOT_RUN，且不得调用遮蔽脚本。

边界（强制）: 全部临时合成输入 + mock 执行器 + stub 子工具；不 import 真实 runner、不 import torch、
不调用 CUDA、不运行完整门禁、不触碰 D:/models/_harness_zip_sim 与 D:/models/upload。
因此本测试的 PASS 只代表【模拟覆盖范围】内判定逻辑正确。
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / 'preupload_gate_rcon.py'
UPLOAD_DIR = Path(r'D:/models/upload')
REAL_SIM_DIR = Path(r'D:/models/_harness_zip_sim')


def load_gate():
    spec = importlib.util.spec_from_file_location('preupload_gate_rcon', GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = load_gate()
ExecResult = G.ExecResult
ALL_SCRIPTS = tuple(s.script for s in G.TOOL_SPECS.values())


def make_zip(path: Path, n_out=100):
    """合成最小可解析包：100 个 output_dir 成员 + model_dir 两个必备文件。"""
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('model_dir/runner.py', '# stub\n')
        z.writestr('model_dir/your_model.pt', b'\x00' * 16)
        for i in range(1, n_out + 1):
            z.writestr('output_dir/case%d.jpg' % i, b'not-a-real-jpeg')
    return path


def extract_model_dir(zip_path: Path, sim_dir: Path, tamper=None):
    """模拟真实 harness：删除并重建解压目录，并真的把本包的 model_dir 解压出来。

    tamper: None | 'different_weights' | 'extra_file'
        用于构造负测试：同数量不同权重 / 文件集合不符（串包场景）。
    """
    shutil.rmtree(sim_dir, ignore_errors=True)
    sim_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if n.startswith('model_dir/'):
                z.extract(n, sim_dir)
    if tamper == 'different_weights':
        (sim_dir / 'model_dir' / 'your_model.pt').write_bytes(b'DIFFERENT-MODEL-WEIGHTS')
    elif tamper == 'extra_file':
        (sim_dir / 'model_dir' / 'stub_not_in_zip.py').write_text('# x\n', encoding='utf-8')
    os.utime(sim_dir, None)


def backdate(path: Path, seconds=120):
    t = time.time() - seconds
    os.utime(path, (t, t))


def success_map(sim_dir, n_out=100, touch=True, report=None, tamper=None):
    """全部子工具的“成功”返回；harness 会把本包 model_dir 真的解压到 sim_dir（除非 touch=False）。"""
    def harness(script, args):
        if touch:
            extract_model_dir(Path(args[0]), Path(sim_dir), tamper=tamper)
        cnt = n_out if report is None else report
        return ExecResult(rc=0, out='[1] 解压 OK: output_dir=%d model_dir files=3\n'
                                    '[3] infer OK: (1, 3, 512, 512)\n'
                                    'HARNESS_ZIP_SIM_PASS' % cnt)
    return {
        'audit_py_compat.py': ExecResult(rc=0, out='风险统计\n合计有风险文件数: 0'),
        'harness_sim_zip.py': harness,
        'audit_iqa_shadow.py': ExecResult(rc=0, out='PROBE pass=True\nIQA_SHADOW_AUDIT_PASS'),
        'audit_iqa_shadow_hard.py': ExecResult(rc=0, out='HARD pass=True\nHARD_AUDIT_PASS\nHARD_ALL_PASS'),
        'audit_official_format.py': ExecResult(rc=0, out='FORMAT_PASS\nFORMAT_ALL_PASS'),
    }


def make_exec(script_map=None, missing=()):
    script_map = dict(script_map or {})
    calls = []

    def ex(script, args, cwd=None):
        calls.append((script, tuple(str(a) for a in args)))
        if script in missing:
            return ExecResult(error='missing_tool:/stub/%s' % script)
        r = script_map.get(script)
        if r is None:
            return ExecResult(error='missing_tool:/stub/%s' % script)
        if callable(r):
            return r(script, args)
        return r

    ex.calls = calls
    return ex


def ok_checks(n=5):
    def mk(i):
        return ('chk%d' % i, lambda p: (True, 'ok'))
    return tuple(mk(i) for i in range(n))


def verdict_row(out, name):
    for ln in out.splitlines():
        if name + ':' in ln:
            for icon, word in (('✅', G.PASS), ('❌', G.FAIL), ('⏸', G.NOT_RUN)):
                if icon in ln:
                    return word
    return None


def upload_snapshot():
    if not UPLOAD_DIR.exists():
        return None
    return sorted((x.name, x.stat().st_mtime_ns, x.stat().st_size) for x in UPLOAD_DIR.iterdir())


class GateCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='rcon01r2_')
        self.tmp = Path(self._tmp.name)
        self.pkg = make_zip(self.tmp / 'pkg.zip')
        self.sim = self.tmp / 'sim'          # 故意【不】预建，交给 harness 行为决定

    def tearDown(self):
        self._tmp.cleanup()

    def run_gate(self, script_map=None, missing=(), checks=None, zips=None,
                 sim_dir=None, stage_upload=False, execute_tools=True,
                 model_hash_max_bytes=None):
        buf = []
        sim = sim_dir if sim_dir is not None else self.sim
        ex = make_exec(script_map if script_map is not None else success_map(sim), missing)
        kw = {} if model_hash_max_bytes is None else {'model_hash_max_bytes': model_hash_max_bytes}
        rc = G.run_gate(zips or [str(self.pkg)], executor=ex,
                        local_checks=checks if checks is not None else ok_checks(),
                        sim_dir=str(sim), stage_upload=stage_upload,
                        execute_tools=execute_tools, out=buf.append, **kw)
        return rc, '\n'.join(buf), ex


class TestFailurePropagation(GateCase):

    # 判据 1
    def test_01_all_success_mock_scope_passes(self):
        rc, out, _ = self.run_gate()
        self.assertEqual(rc, 0, out)
        self.assertIn('PREUPLOAD_ALL_PASS', out)
        self.assertNotIn('PREUPLOAD_ANY_FAIL', out)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.PASS)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.PASS)
        self.assertNotIn('可以上传', out)
        self.assertIn('不是官方合规认证', out)

    # 判据 2
    def test_02_infer_ok_then_assert_fail_is_fail(self):
        m = success_map(self.sim)
        m['harness_sim_zip.py'] = ExecResult(
            rc=1, out='[1] 解压 OK: output_dir=100\n[3] infer OK: (1, 3, 512, 512)\n'
                      'Traceback (most recent call last):\nAssertionError')
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.FAIL)
        self.assertNotIn('PREUPLOAD_ALL_PASS', out)

    # 判据 3
    def test_03_success_marker_with_nonzero_rc_is_fail(self):
        m = success_map(self.sim)
        m['audit_py_compat.py'] = ExecResult(rc=1, out='合计有风险文件数: 0')
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '② 语法兼容 (|None)'), G.FAIL)

    # 判据 4a
    def test_04a_rc0_without_terminal_marker_is_fail(self):
        m = success_map(self.sim)
        m['harness_sim_zip.py'] = ExecResult(rc=0, out='[1] 解压 OK: output_dir=100\n[3] infer OK: fine')
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.FAIL)

    # 判据 4b
    def test_04b_rc0_with_both_markers_is_fail(self):
        m = success_map(self.sim)
        m['audit_iqa_shadow.py'] = ExecResult(
            rc=0, out='IQA_SHADOW_AUDIT_PASS\nIQA_SHADOW_AUDIT_FAIL')
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.FAIL)

    # 判据 5
    def test_05a_timeout_is_fail(self):
        m = success_map(self.sim)
        m['audit_official_format.py'] = ExecResult(error='timeout', out='partial output')
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '⑤ 官方格式严格审计'), G.FAIL)

    def test_05b_exception_is_fail(self):
        m = success_map(self.sim)
        m['audit_py_compat.py'] = ExecResult(error='exception:[WinError 2] not found')
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '② 语法兼容 (|None)'), G.FAIL)

    def test_05c_missing_tool_is_not_run_and_overall_not_pass(self):
        rc, out, _ = self.run_gate(missing=ALL_SCRIPTS)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '② 语法兼容 (|None)'), G.NOT_RUN)
        self.assertNotIn('PREUPLOAD_ALL_PASS', out)
        self.assertIn('PREUPLOAD_ANY_FAIL', out)

    def test_05d_local_check_exception_is_fail(self):
        def boom(p):
            raise RuntimeError('synthetic')
        rc, out, _ = self.run_gate(checks=(('① 合成检查', boom),))
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '① 合成检查'), G.FAIL)

    def test_05e_local_check_none_is_not_run(self):
        rc, out, _ = self.run_gate(checks=(('① 缺依赖检查', lambda p: (None, 'NOT_RUN 缺依赖')),))
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '① 缺依赖检查'), G.NOT_RUN)

    # 判据 6
    def test_06_stale_sim_dir_not_reused_after_harness_failure(self):
        extract_model_dir(self.pkg, self.sim)       # 上一次运行残留
        m = success_map(self.sim, touch=False)
        m['harness_sim_zip.py'] = ExecResult(rc=1, out='[1] 解压 OK: output_dir=100\nboom')
        rc, out, ex = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)
        self.assertEqual(verdict_row(out, '④ IQA 遮蔽(对抗)'), G.NOT_RUN)
        called = [s for s, _ in ex.calls]
        self.assertNotIn('audit_iqa_shadow.py', called)
        self.assertNotIn('audit_iqa_shadow_hard.py', called)

    def test_06b_sim_dir_missing_when_harness_passed_is_not_run(self):
        empty = self.tmp / 'no_sim'
        empty.mkdir()
        rc, out, _ = self.run_gate(success_map(empty, touch=False), sim_dir=empty)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)

    # 判据 7
    def test_07_multipackage_mixed_is_nonzero(self):
        p1 = make_zip(self.tmp / 'a.zip')
        p2 = make_zip(self.tmp / 'b.zip')
        base = success_map(self.sim)

        def harness(script, args):
            if str(args[0]).endswith('a.zip'):
                return ExecResult(rc=1, out='[1] 解压 OK: output_dir=100\nboom')
            return base['harness_sim_zip.py'](script, args)

        m = dict(base)
        m['harness_sim_zip.py'] = harness
        buf = []
        ex = make_exec(m)
        rc = G.run_gate([str(p1), str(p2)], executor=ex, local_checks=ok_checks(),
                        sim_dir=str(self.sim), out=buf.append)
        out = '\n'.join(buf)
        self.assertEqual(rc, 1, out)
        self.assertIn('PREUPLOAD_ANY_FAIL', out)
        self.assertNotIn('PREUPLOAD_ALL_PASS', out)
        self.assertIn('PREUPLOAD_GATE_FAIL', out)
        self.assertIn('PREUPLOAD_GATE_PASS', out)

    # 判据 8
    def test_08_stage_upload_refused_without_writes(self):
        before = upload_snapshot()
        rc, out, _ = self.run_gate(stage_upload=True)
        self.assertEqual(rc, 2, out)
        self.assertIn('STAGE_UPLOAD_UNAUTHORIZED', out)
        self.assertNotIn('PREUPLOAD_ALL_PASS', out)
        self.assertEqual(upload_snapshot(), before,
                         '本副本不得对 D:/models/upload 产生任何写入')


class TestReworkBoundaries(GateCase):
    """返修新增：A 副作用 / B 解压绑定 / C 标记协议。"""

    # C：成功标记必须整行相等，不得子串匹配
    def test_20_success_marker_prefix_is_not_pass(self):
        m = success_map(self.sim)
        m['harness_sim_zip.py'] = ExecResult(
            rc=0, out='[1] 解压 OK: output_dir=100\nNOT_HARNESS_ZIP_SIM_PASS')
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.FAIL)

    def test_20b_success_marker_with_suffix_is_not_pass(self):
        m = success_map(self.sim)
        m['harness_sim_zip.py'] = ExecResult(
            rc=0, out='[1] 解压 OK: output_dir=100\nHARNESS_ZIP_SIM_PASS_FAILED')
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.FAIL)

    def test_20c_failure_token_inside_longer_word_is_not_a_failure(self):
        # 反向保护：NOT_HARD_ALL_PASS 这类“更长单词”不应被当成成功标记；
        # 若只有它出现，则 rc=0 但缺终态行 -> FAIL（不得 PASS）。
        m = success_map(self.sim)
        m['audit_iqa_shadow_hard.py'] = ExecResult(rc=0, out='NOT_HARD_ALL_PASS')
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '④ IQA 遮蔽(对抗)'), G.FAIL)

    # B：解压归属未绑定 -> ③④ NOT_RUN
    def test_21_sim_dir_not_rebuilt_is_not_bound(self):
        extract_model_dir(self.pkg, self.sim)        # 旧目录存在但本次不会被重建
        m = success_map(self.sim, touch=False)       # harness 报成功但不触碰目录
        rc, out, ex = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.PASS)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)
        self.assertEqual(verdict_row(out, '④ IQA 遮蔽(对抗)'), G.NOT_RUN)
        called = [s for s, _ in ex.calls]
        self.assertNotIn('audit_iqa_shadow.py', called)

    def test_22_extract_count_mismatch_is_not_bound(self):
        m = success_map(self.sim, report=99)         # 包内实际 100
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)
        self.assertIn('解压产物与包不匹配', out)

    def test_22b_extract_line_missing_is_not_bound(self):
        def harness(script, args):
            extract_model_dir(self.pkg, self.sim)
            return ExecResult(rc=0, out='HARNESS_ZIP_SIM_PASS')     # 无 [1] 行
        m = success_map(self.sim)
        m['harness_sim_zip.py'] = harness
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)

    # A：副作用边界
    def test_23_stage_upload_invokes_no_tools(self):
        rc, out, ex = self.run_gate(stage_upload=True)
        self.assertEqual(rc, 2, out)
        self.assertEqual(ex.calls, [], '拒绝归位后不得调用任何子工具')
        self.assertIn('STAGE_UPLOAD_UNAUTHORIZED', out)
        self.assertIn('立即停止', out)

    def test_24_plan_mode_invokes_no_tools(self):
        rc, out, ex = self.run_gate(execute_tools=False)
        self.assertEqual(rc, 1, out)
        self.assertEqual(ex.calls, [], '计划模式不得调用任何子工具')
        self.assertIn('SIDE_EFFECTS_NOT_EXECUTED', out)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.NOT_RUN)
        self.assertEqual(verdict_row(out, '② 语法兼容 (|None)'), G.NOT_RUN)


class TestCrossPackageBinding(GateCase):
    """R3：图片数量相同 + 目录时间变化，都不能证明其中的模型来自本 ZIP。"""

    def test_28_same_count_different_weights_is_not_bound(self):
        m = success_map(self.sim, tamper='different_weights')   # 100 张图 + 别人的权重
        rc, out, ex = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.PASS)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)
        self.assertEqual(verdict_row(out, '④ IQA 遮蔽(对抗)'), G.NOT_RUN)
        self.assertIn('内容哈希与包内不一致', out)
        called = [s for s, _ in ex.calls]
        self.assertNotIn('audit_iqa_shadow.py', called)
        self.assertNotIn('audit_iqa_shadow_hard.py', called)

    def test_28b_file_set_mismatch_is_not_bound(self):
        m = success_map(self.sim, tamper='extra_file')
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)
        self.assertIn('文件集合与包内不一致', out)

    def test_28c_renamed_model_file_is_not_bound(self):
        # 同内容改名 -> 集合不符，仍须 NOT_RUN
        def harness(script, args):
            extract_model_dir(Path(args[0]), self.sim)
            src = self.sim / 'model_dir' / 'your_model.pt'
            src.rename(self.sim / 'model_dir' / 'your_model_v2.pt')
            os.utime(self.sim, None)
            return ExecResult(rc=0, out='[1] 解压 OK: output_dir=100 model_dir files=3\n'
                                        '[3] infer OK\nHARNESS_ZIP_SIM_PASS')
        m = success_map(self.sim)
        m['harness_sim_zip.py'] = harness
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)

    def test_29_directory_time_only_change_is_not_bound(self):
        extract_model_dir(self.pkg, self.sim, tamper='different_weights')
        backdate(self.sim)                        # 旧目录：先做旧
        m = success_map(self.sim, touch=False)

        def harness(script, args):
            os.utime(self.sim, None)              # 只改目录时间，不重新解压
            return ExecResult(rc=0, out='[1] 解压 OK: output_dir=100 model_dir files=3\n'
                                        '[3] infer OK\nHARNESS_ZIP_SIM_PASS')
        m['harness_sim_zip.py'] = harness
        rc, out, _ = self.run_gate(m)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.PASS)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)
        # 由 model_dir 内容核对兜住，而不是只靠目录时间
        self.assertIn('内容哈希与包内不一致', out)

    def test_30_hash_budget_exceeded_is_not_run(self):
        rc, out, _ = self.run_gate(model_hash_max_bytes=1)
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)
        self.assertIn('超出哈希预算', out)


class TestCliInjection(GateCase):
    """端到端：可注入工具路径 + 退出码/标记传播 + 无副作用默认（仍为 stub，不跑真实工具）。"""

    SENTINEL = 'stub_ran.txt'

    def _stub_tools(self, tools_dir, sim_dir, harness_body=None):
        tools_dir.mkdir(parents=True, exist_ok=True)
        sentinel = tools_dir / self.SENTINEL
        head = ("import os\n"
                "open(r'%s', 'a', encoding='utf-8').write(os.path.basename(__file__) + chr(10))\n"
                % sentinel)
        default_harness = (
            "import os, shutil, sys, zipfile\n"
            "sim = r'%s'\n"
            "shutil.rmtree(sim, ignore_errors=True)\n"
            "os.makedirs(sim)\n"
            "with zipfile.ZipFile(sys.argv[1]) as z:\n"
            "    for n in z.namelist():\n"
            "        if n.startswith('model_dir/'):\n"
            "            z.extract(n, sim)\n"
            "os.utime(sim, None)\n"
            "print('[1] 解压 OK: output_dir=100 model_dir files=3')\n"
            "print('[3] infer OK: (1, 3, 512, 512)')\n"
            "print('HARNESS_ZIP_SIM_PASS')\n" % sim_dir)
        bodies = {
            'audit_py_compat.py': "print('合计有风险文件数: 0')\n",
            'harness_sim_zip.py': harness_body or default_harness,
            'audit_iqa_shadow.py': "print('IQA_SHADOW_AUDIT_PASS')\n",
            'audit_iqa_shadow_hard.py': "print('HARD_ALL_PASS')\n",
            'audit_official_format.py': "print('FORMAT_ALL_PASS')\n",
        }
        for name, body in bodies.items():
            (tools_dir / name).write_text(head + body, encoding='utf-8')
        return tools_dir, sentinel

    def _run_cli(self, args):
        p = subprocess.run([sys.executable, str(GATE)] + [str(a) for a in args],
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=300)
        return p.returncode, (p.stdout or '') + (p.stderr or '')

    def test_09_cli_stub_tools_end_to_end(self):
        tools_dir, _ = self._stub_tools(self.tmp / 'tools', self.sim)
        pkg = make_zip(self.tmp / 'cli.zip')
        rc, out = self._run_cli([pkg, '--tool-root', tools_dir, '--python', sys.executable,
                                 '--sim-dir', self.sim, '--execute-tools'])
        self.assertEqual(rc, 1, out)          # 合成包的 jpg 不可解码 -> ①出图尺寸 FAIL
        self.assertEqual(verdict_row(out, '   出图尺寸'), G.FAIL)
        self.assertEqual(verdict_row(out, '② 语法兼容 (|None)'), G.PASS)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.PASS)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.PASS)
        self.assertEqual(verdict_row(out, '④ IQA 遮蔽(对抗)'), G.PASS)
        self.assertEqual(verdict_row(out, '⑤ 官方格式严格审计'), G.PASS)

    def test_09b_cli_harness_failure_blocks_shadow_checks(self):
        tools_dir, _ = self._stub_tools(
            self.tmp / 'tools2', self.sim,
            harness_body="print('[3] infer OK')\nraise SystemExit(1)\n")
        pkg = make_zip(self.tmp / 'cli2.zip')
        rc, out = self._run_cli([pkg, '--tool-root', tools_dir, '--python', sys.executable,
                                 '--sim-dir', self.sim, '--execute-tools'])
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.FAIL)
        self.assertEqual(verdict_row(out, '③ IQA 遮蔽'), G.NOT_RUN)
        self.assertEqual(verdict_row(out, '④ IQA 遮蔽(对抗)'), G.NOT_RUN)

    def test_10_cli_missing_tools_never_pass(self):
        empty = self.tmp / 'empty_tools'
        empty.mkdir()
        pkg = make_zip(self.tmp / 'cli3.zip')
        rc, out = self._run_cli([pkg, '--tool-root', empty, '--python', sys.executable,
                                 '--sim-dir', self.sim, '--execute-tools'])
        self.assertEqual(rc, 1, out)
        self.assertEqual(verdict_row(out, '② 语法兼容 (|None)'), G.NOT_RUN)
        self.assertNotIn('PREUPLOAD_ALL_PASS', out)

    def test_11_cli_stage_upload_no_write_and_no_execution(self):
        tools_dir, sentinel = self._stub_tools(self.tmp / 'tools3', self.sim)
        pkg = make_zip(self.tmp / 'cli4.zip')
        before = upload_snapshot()
        rc, out = self._run_cli([pkg, '--tool-root', tools_dir, '--python', sys.executable,
                                 '--sim-dir', self.sim, '--execute-tools', '--stage-upload',
                                 '--upload-name', 'RCON01_SHOULD_NOT_EXIST.zip'])
        self.assertNotEqual(rc, 0, out)
        self.assertIn('STAGE_UPLOAD_UNAUTHORIZED', out)
        self.assertFalse(sentinel.exists(), '拒绝归位后不得执行任何 stub 子工具')
        self.assertEqual(upload_snapshot(), before, '--stage-upload 不得写入上传目录')
        if UPLOAD_DIR.exists():
            self.assertFalse((UPLOAD_DIR / 'RCON01_SHOULD_NOT_EXIST.zip').exists())

    def test_25_cli_plan_mode_does_not_execute_stubs(self):
        tools_dir, sentinel = self._stub_tools(self.tmp / 'tools4', self.sim)
        pkg = make_zip(self.tmp / 'cli5.zip')
        rc, out = self._run_cli([pkg, '--tool-root', tools_dir, '--python', sys.executable,
                                 '--sim-dir', self.sim])          # 无 --execute-tools
        self.assertEqual(rc, 1, out)
        self.assertIn('SIDE_EFFECTS_NOT_EXECUTED', out)
        self.assertFalse(sentinel.exists(), '默认计划模式不得执行任何子工具')
        self.assertEqual(verdict_row(out, '   接口/512 推理'), G.NOT_RUN)

    def test_26_cli_execute_tools_actually_runs_stubs(self):
        tools_dir, sentinel = self._stub_tools(self.tmp / 'tools5', self.sim)
        pkg = make_zip(self.tmp / 'cli6.zip')
        rc, out = self._run_cli([pkg, '--tool-root', tools_dir, '--python', sys.executable,
                                 '--sim-dir', self.sim, '--execute-tools'])
        self.assertEqual(rc, 1, out)
        self.assertTrue(sentinel.exists(), '--execute-tools 时 stub 应被实际执行')
        self.assertEqual(verdict_row(out, '② 语法兼容 (|None)'), G.PASS)

    def test_27_cli_repo_defaults_do_not_touch_real_shared_dirs(self):
        # 断言测试进程从未改写真实共享解压目录（只做存在性与 mtime 记录，不创建/不删除）
        state = G._dir_state(REAL_SIM_DIR)
        tools_dir, _ = self._stub_tools(self.tmp / 'tools6', self.sim)
        pkg = make_zip(self.tmp / 'cli7.zip')
        self._run_cli([pkg, '--tool-root', tools_dir, '--python', sys.executable,
                       '--sim-dir', self.sim, '--execute-tools'])
        self.assertEqual(G._dir_state(REAL_SIM_DIR), state,
                         '测试不得改动真实共享解压目录')


if __name__ == '__main__':
    unittest.main(verbosity=2)
