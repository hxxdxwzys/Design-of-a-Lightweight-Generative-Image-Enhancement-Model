# -*- coding: utf-8 -*-
"""EXPORT-01R worker：在独立子进程内执行复验（总账/锁/attempt 由本进程独占）。

用法: python r_worker.py <run_dir> <ledger_path> <prereg_path> <model_dir> <exp_dir>
退出码：0=OK，1=FAIL（判据或异常），2=启动/协议失败。
"""
from __future__ import annotations

import os

# 冻结执行模式：必须在 CUDA 初始化（导入 torch 后首次使用 CUDA）之前设置
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r_ledger  # noqa: E402
import r_verify  # noqa: E402


def main():
    run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) if i < 4 else Path(p)
                                                             for i, p in enumerate(sys.argv[1:6])]
    exp_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    flags = {'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
             'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
             'cudnn_benchmark': torch.backends.cudnn.benchmark,
             'cudnn_deterministic': torch.backends.cudnn.deterministic,
             'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
             'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
             'torch': torch.__version__, 'pid': os.getpid()}
    (exp_dir / 'execution_flags.json').write_text(json.dumps(flags, indent=1), encoding='utf-8')
    print('[worker] flags %s' % json.dumps(flags), flush=True)

    led = r_ledger.Ledger(ledger_path)
    try:
        led.acquire_lock('EXPORT01R attempt_01')
    except r_ledger.LockHeld as e:
        print('[worker] LOCK_HELD %s' % e, flush=True)
        return 2
    status = 'FAIL'
    detail = ''
    try:
        prereg = json.loads(prereg_path.read_text(encoding='utf-8'))
        led.open_attempt('attempt_01', run_dir,
                         prereg_sha256=r_verify.sha256_file(prereg_path),
                         source_sha256=r_verify.sha256_file(HERE / 'r_verify.py'),
                         model_sha256=None)
        try:
            if not (Path(model_dir) / 'your_model.pt').is_file():
                ledger_note = led.note('build_start', {'model_dir': str(model_dir)})
                import r_build
                build_info = r_build.build(model_dir, run_dir, led, exp_dir, None, HERE)
                led.note('build_done', {'model_sha256': build_info['model_sha256'],
                                        'component_forwards': build_info['trace_component_forwards']})
                print('[worker] built %s' % build_info['model_sha256'], flush=True)
                # EXP-R13：保存后由全新进程加载 .pt 复测组件（9 次真实前向，先扣后跑）
                for _i in range(9):
                    led.reserve('component', 'fresh_process_component_%d' % (_i + 1))
                gate = subprocess.run(
                    [sys.executable, '-B', str(HERE / 'r3_component_gate.py'), 'check',
                     str(run_dir), str(exp_dir), str(prereg_path)],
                    capture_output=True, text=True, timeout=1200)
                led.note('component_gate', {'rc': gate.returncode,
                                            'tail': (gate.stdout or '')[-300:]})
                print('[worker] component_gate rc=%d %s'
                      % (gate.returncode, (gate.stdout or '').strip()[-200:]), flush=True)
                if gate.returncode != 0:
                    raise RuntimeError('组件门槛未通过（不得开始完整 infer）：%s'
                                       % ((gate.stderr or gate.stdout or '')[-500:]))
            readings = r_verify.run_verification(run_dir, led, prereg_path, model_dir,
                                                 prereg['tolerances'], exp_dir)
            readings['status'] = 'OK'
            (exp_dir / 'readings.json').write_text(
                json.dumps(readings, indent=1, ensure_ascii=False), encoding='utf-8')
            status = 'OK'
        except Exception as e:
            detail = '%s: %s' % (type(e).__name__, e)
            # EXP-R14：保留失败前的结构化 partial（run_verification 逐次落盘），只叠加失败信息
            partial = {}
            rp = exp_dir / 'readings.json'
            if rp.is_file():
                try:
                    partial = json.loads(rp.read_text(encoding='utf-8'))
                except Exception:
                    partial = {}
            partial.update({'schema': 'EXPORT01R-readings/1', 'status': 'FAIL', 'error': detail,
                            'traceback': traceback.format_exc()[-4000:],
                            'ledger': led.summary(),
                            'elapsed_s': round(led.elapsed_s(), 3)})
            rp.write_text(json.dumps(partial, indent=1, ensure_ascii=False), encoding='utf-8')
            print('[worker] FAIL %s' % detail, flush=True)
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
