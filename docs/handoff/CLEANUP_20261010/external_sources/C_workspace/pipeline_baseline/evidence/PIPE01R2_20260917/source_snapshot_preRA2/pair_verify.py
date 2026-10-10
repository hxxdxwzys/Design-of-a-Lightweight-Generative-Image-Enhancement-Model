# -*- coding: utf-8 -*-
"""PIPE-01R 阶段B 成对验证驱动（固定样本/顺序/硬预算；不做候选选择）。

相对 PIPE-01 首版的修复（对应 review 的 PIPE-R03/R04/R06/R08）：
  R03  两个实例都保存**完整身份**（配置/资产/源码/实现/有效参数/模块文件/环境），
       并强制逐字段相等；不一致即 FAIL，即使输出全零。运行前强制核对预登记与当前状态。
  R04  **持久化预算总账**：attempt 开始即登记；真实 infer 之前先扣额度并落盘；
       失败/中断不退回；墙钟跨 attempt 累计；0 余额不启动；拒绝并发与非空 run 目录复用。
  R06  每次调用唯一 call_id（NPY/PNG 不覆盖）；A/B 采用**同一 infer 计时边界**
       （core.infer_core 全程），输入解码/写图单独列。
  R08  每次调用（含失败）写入生成 manifest：输入/输出哈希、shape、seed、身份、预算、时间。

协议（任务单 §6）：每实例按 [crop512, case1, case7, crop512]，两实例先后加载；
实例A=tensor 入口，实例B=离线文件入口；总计 <= 总账 max_forwards。

用法:
  D:/Python/python.exe pair_verify.py --run-dir <dir> --prereg <dir>/prereg.json \
      --case1 <case1.jpg> --case7 <case7.jpg> [--budget-ledger <json>] [--init-ledger]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import core
import offline
from runner import Runner

ORDER = ('case1_crop512', 'case1_full', 'case7_full', 'case1_crop512')
UNIQUE = ('case1_crop512', 'case1_full', 'case7_full')


def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


class BudgetStop(RuntimeError):
    """预算/并发/复用等硬约束触发；调用方必须在启动任何前向之前停下。"""


def _pid_alive(pid):
    """只读判定进程是否存活；无法判定时保守返回 True（拒绝并发）。"""
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    except Exception:
        return True


class BudgetLedger:
    """跨进程持久化预算总账。所有扣减先落盘再执行，失败不退回。"""

    def __init__(self, path, max_forwards=None, max_minutes=None, max_gib=None, init=False):
        self.path = Path(path)
        self.lock_path = None
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
            for k, v in (('max_forwards', max_forwards), ('max_minutes', max_minutes),
                         ('max_gib', max_gib)):
                if v is not None and float(self.data.get(k, -1)) != float(v):
                    raise BudgetStop('总账 %s 的 %s=%s 与本次预登记 %s 不一致；拒绝改写配额'
                                     % (self.path, k, self.data.get(k), v))
        elif init:
            self.data = {'task': 'PIPE-01', 'created_at': _now(),
                         'max_forwards': int(max_forwards), 'max_minutes': float(max_minutes),
                         'max_gib': float(max_gib), 'forwards_reserved': 0,
                         'wall_seconds_total': 0.0, 'attempts': [], 'open_attempt': None,
                         'reservations': [], 'stale_locks': []}
            self._write()
        else:
            raise BudgetStop('预算总账不存在: %s（需 make_evidence 初始化或显式 --init-ledger）'
                             % self.path)

    # ── 持久化 ──
    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, self.path)

    # ── 并发 ──
    def acquire_lock(self):
        if self.lock_path is None:
            self.lock_path = self.path.with_suffix('.lock')
        if self.lock_path.exists():
            try:
                info = json.loads(self.lock_path.read_text(encoding='utf-8'))
            except Exception:
                info = {}
            if _pid_alive(info.get('pid', -1)):
                raise BudgetStop('检测到并发 PIPE-01 attempt（pid=%s）；拒绝并发运行'
                                 % info.get('pid'))
            self.data.setdefault('stale_locks', []).append(
                {'stale': info, 'taken_over_at': _now()})
            self._write()
        self.lock_path.write_text(json.dumps({'pid': os.getpid(), 'started_at': _now()}),
                                  encoding='utf-8')

    def release_lock(self):
        try:
            if self.lock_path and self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass

    # ── attempt ──
    def begin_attempt(self, attempt_id, run_dir):
        oa = self.data.get('open_attempt')
        if oa:
            elapsed = time.monotonic() - float(oa['started_monotonic'])
            self.data['wall_seconds_total'] = float(self.data['wall_seconds_total']) + elapsed
            self.data['attempts'].append({
                'attempt_id': oa['attempt_id'], 'run_dir': oa['run_dir'],
                'started_at': oa['started_at'], 'status': 'INTERRUPTED',
                'note': '上次 attempt 未正常结束；时间按保守方式计入', 'elapsed_s': elapsed})
        self.data['open_attempt'] = {'attempt_id': attempt_id, 'run_dir': str(run_dir),
                                     'started_at': _now(), 'started_monotonic': time.monotonic()}
        self._write()

    def end_attempt(self, status, note=''):
        oa = self.data.get('open_attempt')
        if not oa:
            return
        elapsed = time.monotonic() - float(oa['started_monotonic'])
        self.data['wall_seconds_total'] = float(self.data['wall_seconds_total']) + elapsed
        self.data['attempts'].append({'attempt_id': oa['attempt_id'], 'run_dir': oa['run_dir'],
                                      'started_at': oa['started_at'], 'status': status,
                                      'note': note, 'elapsed_s': elapsed})
        self.data['open_attempt'] = None
        self._write()

    # ── 额度 ──
    def reserve(self, where):
        used = int(self.data['forwards_reserved'])
        cap = int(self.data['max_forwards'])
        if used >= cap:
            raise BudgetStop('前向额度已用尽（%d/%d）：拒绝在 %s 启动新的真实前向'
                             % (used, cap, where))
        self.data['forwards_reserved'] = used + 1
        self.data.setdefault('reservations', []).append(
            {'where': where, 'used_after': used + 1, 'at': _now()})
        self._write()                      # 先落盘，再执行
        return used + 1

    def wall_used(self):
        total = float(self.data.get('wall_seconds_total', 0.0))
        oa = self.data.get('open_attempt')
        if oa:
            total += time.monotonic() - float(oa['started_monotonic'])
        return total

    def check_wall(self):
        limit = float(self.data['max_minutes']) * 60.0
        if self.wall_used() > limit:
            raise BudgetStop('墙钟累计触顶 %.1f 分钟（含所有 attempt 与加载）' % (limit / 60.0))

    def check_disk(self, run_dir):
        limit = float(self.data['max_gib']) * (1024 ** 3)
        rd = Path(run_dir)
        used = sum(p.stat().st_size for p in rd.rglob('*') if p.is_file()) if rd.exists() else 0
        if used > limit:
            raise BudgetStop('新增输出触顶 %.2f GiB（已 %.2f GiB）'
                             % (limit / 1024 ** 3, used / 1024 ** 3))
        return used

    def snapshot(self):
        return {k: v for k, v in self.data.items() if k != 'open_attempt'}


def verify_prereg(pq, cfg):
    """模型加载前强制核对预登记（R03）：任何不符立即阻断。"""
    checks = {
        'config_sha256': (pq.get('config_sha256'), cfg['_config_sha256']),
        'asset_fingerprint': (pq.get('asset_fingerprint'), core.asset_fingerprint(cfg)),
        'source_fingerprint': (pq.get('source_fingerprint'), core.source_fingerprint(cfg)),
        'implementation_fingerprint': (pq.get('implementation_fingerprint'),
                                       core.implementation_fingerprint()),
    }
    bad = [k for k, (a, b) in checks.items() if a != b]
    if bad:
        raise BudgetStop('PREREG_MISMATCH: 预登记与当前状态不符（模型加载前阻断）: %s' % bad)
    return True


def build_inputs(case1, case7, crop, in_dir):
    import hashlib
    from PIL import Image
    in_dir.mkdir(parents=True, exist_ok=True)
    im1 = Image.open(str(case1)).convert('RGB')
    im7 = Image.open(str(case7)).convert('RGB')
    w, h = im1.size
    left, top = (w - crop) // 2, (h - crop) // 2
    box = (left, top, left + crop, top + crop)
    crop_path = in_dir / ('case1_crop%d.png' % crop)
    im1.crop(box).save(crop_path)
    items = {
        'case1_crop512': {'path': crop_path, 'crop_box': list(box), 'src_wh': [w, h],
                          'tensor': core.pil_to_tensor(Image.open(crop_path).convert('RGB'))},
        'case1_full': {'path': Path(case1), 'crop_box': None, 'src_wh': [w, h],
                       'tensor': core.pil_to_tensor(im1)},
        'case7_full': {'path': Path(case7), 'crop_box': None, 'src_wh': list(im7.size),
                       'tensor': core.pil_to_tensor(im7)},
    }
    for k, v in items.items():
        t = v['tensor']
        v['tensor_shape'] = list(t.shape)
        v['tensor_sha256'] = hashlib.sha256(
            t.detach().float().cpu().numpy().tobytes()).hexdigest()
        v['input_file_sha256'] = core.sha256_file(v['path'])
    return items


def _diff(a, b):
    import torch
    if tuple(a.shape) != tuple(b.shape):
        return {'shape_mismatch': [list(a.shape), list(b.shape)]}
    d = (a.float() - b.float()).abs()
    return {'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()),
            'n': int(d.numel())}


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str),
                          encoding='utf-8')


def run_instance(entry, model_dir, cfg, inputs, run_dir, ledger, manifest, device, attempt_id):
    """执行一个独立实例的全部固定调用；失败也登记，且不退回已扣额度。"""
    import numpy as np
    import torch

    is_tensor = (entry == 'A')
    t_load = time.perf_counter()
    if is_tensor:
        r = Runner(model_dir, device=device)      # 显式同设备，保证 A/B 身份可比
        r._ensure()
        assets = r.assets
        identity = r.identity(entry='tensor_runner')
    else:
        assets = core.Assets(cfg, device)
        identity = assets.identity(entry='offline_file', model_dir=model_dir, device=device)
    load_s = time.perf_counter() - t_load
    manifest['instances'][entry] = {'load_s': load_s, 'identity': identity,
                                    'peak_vram_bytes': None}
    _write_json(run_dir / 'manifest.json', manifest)

    for idx, tag in enumerate(ORDER, start=1):
        cid = '%s_%s_%s_r%d' % (attempt_id, entry, tag, idx)
        ledger.check_wall()
        ledger.check_disk(run_dir)
        n = ledger.reserve('%s:%s' % (entry, cid))
        timing = {}
        row = {'call_id': cid, 'entry': entry, 'tag': tag, 'run_index': idx,
               'reserve_n': n, 'status': 'OK', 'error': None, 'timing': timing}
        t0 = time.perf_counter()
        try:
            if is_tensor:
                x = inputs[tag]['tensor'].to(device)
                timing['to_device_s'] = time.perf_counter() - t0
                out = r.infer(x, prompt='', timing=timing)
                npy = run_dir / ('%s.npy' % cid)
                np.save(npy, out.detach().float().cpu().numpy())
                png = run_dir / ('%s.png' % cid)
                core.tensor_to_pil(out).save(png)
                row.update({'shape': list(out.shape), 'dtype': str(out.dtype),
                            'min': float(out.min()), 'max': float(out.max()),
                            'finite': bool(torch.isfinite(out).all()),
                            'tensor_npy': str(npy), 'png_path': str(png),
                            'png_bytes': png.stat().st_size,
                            'png_sha256': core.sha256_file(png),
                            'tensor_sha256': core.sha256_file(npy)})
            else:
                rec = offline.process_image(assets, inputs[tag]['path'], run_dir, run_dir,
                                            call_id=cid, timing=timing)
                row.update({k: v for k, v in rec.items()
                            if k not in ('entry', 'timing', 'call_id')})
        except Exception as e:
            row['status'] = 'FAILED'
            row['error'] = '%s: %s' % (type(e).__name__, e)
        row['entry_wall_s'] = time.perf_counter() - t0
        manifest['calls'].append(row)
        _write_json(run_dir / 'manifest.json', manifest)

    try:
        import torch as _t
        manifest['instances'][entry]['peak_vram_bytes'] = (
            int(_t.cuda.max_memory_allocated()) if _t.cuda.is_available() else None)
    except Exception:
        pass
    if is_tensor:
        r.release()
        del r
        gc.collect()
    return assets, identity, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description='PIPE-01R 阶段B 成对验证驱动')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--prereg', required=True, help='预登记 manifest（不存在则拒绝执行）')
    ap.add_argument('--case1', required=True)
    ap.add_argument('--case7', required=True)
    ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--budget-ledger', default=None,
                    help='持久化预算总账（默认 <run-dir>/../budget_ledger.json）')
    ap.add_argument('--init-ledger', action='store_true',
                    help='总账不存在时按预登记预算创建（首次运行）')
    ap.add_argument('--attempt-id', default=None)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    pre = Path(args.prereg)
    if not pre.is_file():
        raise SystemExit('拒绝执行：缺少预登记 manifest %s（不得先跑后登记）' % pre)
    leftover = []
    if run_dir.exists():
        leftover = [p.name for p in run_dir.iterdir() if p.name != 'prereg.json']
    if leftover:
        raise SystemExit('拒绝复用非空 run 目录: %s（已存在: %s；禁止覆盖既有产物）'
                         % (run_dir, leftover[:5]))
    pq = json.loads(pre.read_text(encoding='utf-8'))
    cfg = core.load_config()
    ledger_path = Path(args.budget_ledger) if args.budget_ledger else (
        run_dir.parent / 'budget_ledger.json')
    try:
        verify_prereg(pq, cfg)
        ledger = BudgetLedger(ledger_path, pq['budget']['max_forwards'],
                              pq['budget']['max_minutes'], pq['budget']['max_output_gib'],
                              init=args.init_ledger)
        ledger.acquire_lock()
    except BudgetStop as e:
        raise SystemExit(str(e))

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    attempt_id = args.attempt_id or ('%s_%s' % (run_dir.name, time.strftime('%H%M%S')))
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(args.case1, args.case7, args.crop, run_dir / 'inputs')
    model_dir = args.model_dir or cfg['base_dir']
    crit = pq['criteria']
    manifest = {'task': 'PIPE-01', 'schema': 'PIPE01-generation-manifest/1',
                'attempt_id': attempt_id, 'run_dir': str(run_dir), 'device': device,
                'prereg_sha256': core.sha256_file(pre), 'started_at': _now(),
                'instances': {}, 'calls': [],
                'inputs': {k: {kk: vv for kk, vv in v.items() if kk != 'tensor'}
                           for k, v in inputs.items()}}
    _write_json(run_dir / 'manifest.json', manifest)

    ledger.begin_attempt(attempt_id, run_dir)
    status = 'OK'
    note = ''
    try:
        _a, idA, manifest = run_instance('A', model_dir, cfg, inputs, run_dir,
                                         ledger, manifest, device, attempt_id)
        gc.collect()
        _b, idB, manifest = run_instance('B', model_dir, cfg, inputs, run_dir,
                                         ledger, manifest, device, attempt_id)
    except BudgetStop as e:
        status, note = 'BUDGET_STOP', str(e)
    except Exception as e:
        status, note = 'FAILED', '%s: %s' % (type(e).__name__, e)
    finally:
        ledger.check_disk(run_dir)

    import numpy as np
    result = {'status': status, 'note': note, 'attempt_id': attempt_id,
              'wall_used_total_s': ledger.wall_used(),
              'forwards_reserved_total': int(ledger.data['forwards_reserved']),
              'criteria': crit, 'criteria_detail': {}, 'pairs': {}, 'repeat_512': {},
              'disk_used_bytes': ledger.check_disk(run_dir),
              'ledger': ledger.snapshot()}
    if status == 'OK':
        try:
            calls = {c['call_id']: c for c in manifest['calls']}
            detail = {}
            idA_c = core.identity_comparable(manifest['instances']['A']['identity'])
            idB_c = core.identity_comparable(manifest['instances']['B']['identity'])
            detail['identity_equal'] = bool(idA_c == idB_c)
            if not detail['identity_equal']:
                diff = [k for k in idA_c if idA_c.get(k) != idB_c.get(k)]
                detail['identity_diff_keys'] = diff
            pairs = {}
            for tag in UNIQUE:
                a = [c for c in manifest['calls']
                     if c['entry'] == 'A' and c['tag'] == tag and c['status'] == 'OK']
                b = [c for c in manifest['calls']
                     if c['entry'] == 'B' and c['tag'] == tag and c['status'] == 'OK']
                if not a or not b:
                    pairs[tag] = {'missing': True}
                    continue
                ar, br = a[0], b[0]
                ta = torch.from_numpy(np.load(ar['tensor_npy'], allow_pickle=False))
                tb = torch.from_numpy(np.load(br['tensor_npy'], allow_pickle=False))
                d = _diff(ta, tb)
                d['pass_max'] = d.get('max_abs_diff', 9e9) <= crit['max_abs_diff']
                d['pass_mean'] = d.get('mean_abs_diff', 9e9) <= crit['mean_abs_diff']
                from PIL import Image
                pa = np.array(Image.open(ar['png_path']).convert('RGB'))
                pb = np.array(Image.open(br['png_path']).convert('RGB'))
                d['png_identical'] = bool(np.array_equal(pa, pb))
                d['pass_png'] = bool(d['png_identical'])
                d['a_png_sha256'] = ar['png_sha256']
                d['b_png_sha256'] = br['png_sha256']
                d['a_tensor_npy'] = ar['tensor_npy']
                d['b_tensor_npy'] = br['tensor_npy']
                pairs[tag] = d
            result['pairs'] = pairs
            rep = {}
            for inst in ('A', 'B'):
                cs = [c for c in manifest['calls']
                      if c['entry'] == inst and c['tag'] == 'case1_crop512']
                if len(cs) >= 2 and all(c['status'] == 'OK' for c in cs[:2]):
                    a1 = torch.from_numpy(np.load(cs[0]['tensor_npy'], allow_pickle=False))
                    a2 = torch.from_numpy(np.load(cs[1]['tensor_npy'], allow_pickle=False))
                    d = _diff(a1, a2)
                    d['pass_max'] = d.get('max_abs_diff', 9e9) <= crit['max_abs_diff']
                    d['pass_mean'] = d.get('mean_abs_diff', 9e9) <= crit['mean_abs_diff']
                    d['first_npy'] = cs[0]['tensor_npy']
                    d['second_npy'] = cs[1]['tensor_npy']
                    rep['%s_crop512_repeat' % inst] = d
            result['repeat_512'] = rep

            all_calls_ok = all(c['status'] == 'OK' for c in manifest['calls'])
            detail['all_calls_ok'] = bool(all_calls_ok)
            detail['n_calls'] = len(manifest['calls'])
            detail['pairs_ok'] = bool(result['pairs']) and all(
                v.get('pass_max') and v.get('pass_mean') and v.get('pass_png')
                for v in result['pairs'].values())
            detail['repeat_ok'] = bool(result['repeat_512']) and all(
                v.get('pass_max') and v.get('pass_mean')
                for v in result['repeat_512'].values())
            detail['budget_ok'] = bool(
                int(ledger.data['forwards_reserved']) <= int(ledger.data['max_forwards'])
                and ledger.wall_used() <= float(ledger.data['max_minutes']) * 60.0)
            detail['prereg_verified'] = True
            detail['no_unsafe_load'] = bool(
                'weights_only=True' in json.dumps(manifest['instances']['A']['identity']
                                                  ['load_report'], ensure_ascii=False)
                or manifest['instances']['A']['identity']['load_report'].get(
                    'light_decoder_weights_only') is True)
            detail['dino_loader'] = manifest['instances']['A']['identity'][
                'load_report'].get('dino_loader')
            result['criteria_detail'] = detail
            if not (detail['identity_equal'] and all_calls_ok and detail['pairs_ok']
                    and detail['repeat_ok'] and detail['budget_ok']
                    and detail['prereg_verified'] and detail['no_unsafe_load']):
                status = 'CRITERIA_FAIL'
                result['status'] = status
        except Exception as e:
            status = 'CRITERIA_FAIL'
            result['status'] = status
            result['criteria_eval_error'] = '%s: %s' % (type(e).__name__, e)

    result['criteria_all_pass'] = bool(
        result.get('criteria_detail') and all(
            result['criteria_detail'].get(k) for k in
            ('identity_equal', 'all_calls_ok', 'pairs_ok', 'repeat_ok', 'budget_ok',
             'prereg_verified', 'no_unsafe_load')))
    result['finished_at'] = _now()
    _write_json(run_dir / 'paired_readings.json', result)
    manifest['finished_at'] = _now()
    manifest['final_status'] = result['status']
    _write_json(run_dir / 'manifest.json', manifest)
    ledger.end_attempt(result['status'], note)
    ledger.release_lock()

    print('[pair] status=%s calls=%d forwards_reserved=%d wall_total=%.1fs'
          % (result['status'], len(manifest['calls']),
             result['forwards_reserved_total'], result['wall_used_total_s']))
    if result.get('criteria_detail'):
        print('[criteria] %s' % json.dumps(result['criteria_detail'], ensure_ascii=False))
    for k, v in result['pairs'].items():
        print('[pair] %-14s max=%.3e mean=%.3e png_identical=%s'
              % (k, v.get('max_abs_diff', -1), v.get('mean_abs_diff', -1),
                 v.get('png_identical')))
    return 0 if result['status'] == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
