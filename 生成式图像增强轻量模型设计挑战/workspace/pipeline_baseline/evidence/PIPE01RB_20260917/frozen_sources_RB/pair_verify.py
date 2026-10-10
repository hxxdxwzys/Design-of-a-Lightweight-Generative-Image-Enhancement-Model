# -*- coding: utf-8 -*-
"""PIPE-01R3 成对验证驱动（监督者/工作者单所有者模型）。

修复 review 的 PIPE-RB01–RB03：
  RB01 所有权：**worker 独占** run 目录、锁与 attempt 生命周期；监督者只读总账、不建目录、
       不抢锁、不写总账；worker 日志写在 run 目录之外，避免被当作残留。
       监督者只做：只读预检 → 派生 → 按 deadline 监督 → 子进程退出确认 → 事后恢复与合并判决。
  RB02 陈旧覆盖：监督者在任何写入前 **reload** 总账；只在子进程未收尾时按最新数据做中断记账；
       只清理属于该子进程 pid 的锁。worker 端 finally 覆盖比较/写盘/记账全链，并在末次写盘后做终态核验。
  RB03 协议与判决：协议为**严格 schema**（缺字段/空集合/缺哈希一律阻断）；最终判决为
       worker 判据 AND supervisor 成功 AND 账本一致 AND 最终产物核验（各自落盘证据）。

用法（生产默认 = 监督模式）:
  python pair_verify.py --run-dir <dir> --prereg <dir>/prereg.json --case1 <jpg> --case7 <jpg> \
      --budget-ledger <json> [--init-ledger]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import core
import offline
from runner import Runner

ORDER = ('case1_crop512', 'case1_full', 'case7_full', 'case1_crop512')
UNIQUE = ('case1_crop512', 'case1_full', 'case7_full')
A = 'A'
B = 'B'
REQUIRED_SAMPLE_KEYS = ('input_wh', 'input_file_sha256')
CROP_TOLERANCE_S = 5.0          # 轮询间隔带来的收尾容差（仅用于报告，不放大预算）


def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


class BudgetStop(RuntimeError):
    """预算/并发/复用/协议不符等硬约束：必须在加载或前向之前停下。"""


class StopExecution(RuntimeError):
    """运行中必须立即停止（前向失败/身份不符/超阈值/超时）。"""


# ───────────────────────── 进程状态 ─────────────────────────

def _pid_state(pid):
    """三态：True=存活，False=确定不存在，None=无法判定（调用方 fail closed）。"""
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))
        if h:
            k32.CloseHandle(h)
            return True
        err = k32.GetLastError()
        if err == 87:
            return False
        if err == 5:
            return True
        return None
    except Exception:
        return None


# ───────────────────────── 预算总账 ─────────────────────────

def read_ledger_ro(path):
    p = Path(path)
    if not p.is_file():
        raise BudgetStop('预算总账不存在: %s' % p)
    return json.loads(p.read_text(encoding='utf-8'))


class BudgetLedger:
    def __init__(self, path, max_forwards=None, max_minutes=None, max_gib=None,
                 init=False, prereg_sha256=None):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix('.lock')
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
            for k, v in (('max_forwards', max_forwards), ('max_minutes', max_minutes),
                         ('max_gib', max_gib)):
                if v is not None and float(self.data.get(k, -1)) != float(v):
                    raise BudgetStop('总账 %s 的 %s=%s 与本次预登记 %s 不一致；拒绝改写配额'
                                     % (self.path, k, self.data.get(k), v))
            if prereg_sha256 and self.data.get('prereg_sha256') != prereg_sha256:
                raise BudgetStop('总账绑定的预登记哈希与本次不符: %s != %s'
                                 % (self.data.get('prereg_sha256'), prereg_sha256))
        elif init:
            self.data = {'task': 'PIPE-01', 'schema': 'PIPE01-budget-ledger/2',
                         'created_at': _now(), 'max_forwards': int(max_forwards),
                         'max_minutes': float(max_minutes), 'max_gib': float(max_gib),
                         'forwards_reserved': 0, 'wall_seconds_total': 0.0,
                         'disk_bytes_total': 0, 'attempts': [], 'open_attempt': None,
                         'reservations': [], 'stale_locks': [], 'prereg_sha256': prereg_sha256}
            self._write()
        else:
            raise BudgetStop('预算总账不存在: %s' % self.path)

    def reload(self):
        """从磁盘重读（RB02）：任何写入之前必须先刷新，避免用陈旧内存覆盖他人扣款。"""
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
        return self.data

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, self.path)

    def acquire_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({'pid': os.getpid(), 'started_at': _now()}).encode('utf-8')
        for _ in (1, 2):
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, payload)
                os.close(fd)
                self.reload()
                return
            except FileExistsError:
                try:
                    info = json.loads(self.lock_path.read_text(encoding='utf-8'))
                except Exception:
                    info = {}
                state = _pid_state(info.get('pid', -1))
                if state is None:
                    raise BudgetStop('锁文件存在且进程状态无法判定（fail closed）: %s' % info)
                if state is True:
                    raise BudgetStop('检测到并发 attempt（pid=%s）；拒绝并发' % info.get('pid'))
                self.reload()
                self.data.setdefault('stale_locks', []).append(
                    {'stale': info, 'taken_over_at': _now()})
                self._write()
                try:
                    self.lock_path.unlink()
                except Exception:
                    pass
        raise BudgetStop('无法获取预算总账锁: %s' % self.lock_path)

    def release_lock(self):
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass

    def begin_attempt(self, attempt_id, run_dir):
        oa = self.data.get('open_attempt')
        if oa:
            elapsed = time.monotonic() - float(oa['started_monotonic'])
            self.data['wall_seconds_total'] = float(self.data['wall_seconds_total']) + elapsed
            self.data['attempts'].append({
                'attempt_id': oa['attempt_id'], 'run_dir': oa['run_dir'],
                'started_at': oa['started_at'], 'status': 'INTERRUPTED',
                'note': '上次 attempt 未正常结束；时间按保守方式计入', 'elapsed_s': elapsed,
                'disk_bytes': oa.get('disk_bytes', 0)})
        self.data['open_attempt'] = {'attempt_id': attempt_id, 'run_dir': str(run_dir),
                                     'started_at': _now(), 'started_monotonic': time.monotonic(),
                                     'disk_bytes': 0}
        self._write()

    def end_attempt(self, status, note='', run_dir=None):
        oa = self.data.get('open_attempt')
        if not oa:
            return False
        elapsed = time.monotonic() - float(oa['started_monotonic'])
        disk = int(oa.get('disk_bytes', 0))
        if run_dir is not None:
            disk = max(disk, _dir_bytes(run_dir))
        self.data['wall_seconds_total'] = float(self.data['wall_seconds_total']) + elapsed
        self.data['attempts'].append({'attempt_id': oa['attempt_id'], 'run_dir': oa['run_dir'],
                                      'started_at': oa['started_at'], 'status': status,
                                      'note': note, 'elapsed_s': elapsed,
                                      'disk_bytes': int(disk)})
        self.data['disk_bytes_total'] = int(self.data.get('disk_bytes_total', 0)) + int(disk)
        self.data['open_attempt'] = None
        self._write()
        return True

    def reserve(self, where):
        used = int(self.data['forwards_reserved'])
        cap = int(self.data['max_forwards'])
        if used >= cap:
            raise BudgetStop('前向额度已用尽（%d/%d）：拒绝在 %s 启动新的真实前向'
                             % (used, cap, where))
        self.data['forwards_reserved'] = used + 1
        self.data.setdefault('reservations', []).append(
            {'where': where, 'used_after': used + 1, 'at': _now()})
        self._write()
        return used + 1

    def wall_used(self):
        total = float(self.data.get('wall_seconds_total', 0.0))
        oa = self.data.get('open_attempt')
        if oa:
            total += time.monotonic() - float(oa['started_monotonic'])
        return total

    def remaining_seconds(self):
        return float(self.data['max_minutes']) * 60.0 - self.wall_used()

    def check_startable(self):
        if int(self.data['forwards_reserved']) >= int(self.data['max_forwards']):
            raise BudgetStop('前向额度为 0 可用（%d/%d）：拒绝加载与前向'
                             % (self.data['forwards_reserved'], self.data['max_forwards']))
        if self.remaining_seconds() <= 0:
            raise BudgetStop('墙钟已无余额（剩余 %.1fs）: 拒绝加载' % self.remaining_seconds())

    def check_wall(self):
        if self.remaining_seconds() <= 0:
            raise BudgetStop('墙钟累计触顶（%.1f 分钟，含所有 attempt 与加载）'
                             % float(self.data['max_minutes']))

    def disk_used(self, run_dir):
        prev = int(self.data.get('disk_bytes_total', 0))
        cur = _dir_bytes(run_dir) if run_dir else 0
        return prev + cur

    def check_disk(self, run_dir):
        limit = float(self.data['max_gib']) * (1024 ** 3)
        used = self.disk_used(run_dir)
        if used > limit:
            raise BudgetStop('累计输出触顶 %.2f GiB（已 %.2f GiB）'
                             % (limit / 1024 ** 3, used / 1024 ** 3))
        return used

    def snapshot(self):
        return {k: v for k, v in self.data.items() if k != 'open_attempt'}


def _dir_bytes(root):
    rd = Path(root)
    if not rd.exists():
        return 0
    return sum(p.stat().st_size for p in rd.rglob('*') if p.is_file())


# ───────────────────────── 监督（RB01/RB02 的硬墙钟） ─────────────────────────

def run_supervised(cmd, deadline_s, poll=0.5, log_path=None):
    """派生并监督**本任务自建**的工作进程；只终止该子进程。返回监督记录。"""
    rec = {'deadline_s': float(deadline_s), 'poll_s': poll, 'started_at': _now(),
           'started_monotonic': time.monotonic(), 'terminate_requested_at': None,
           'kill_at': None, 'child_pid': None, 'child_exit_code': None,
           'child_exit_confirmed': False, 'timed_out': False, 'wall_s': None,
           'worker_log': str(log_path) if log_path else None}
    out = open(log_path, 'ab') if log_path else None
    try:
        p = subprocess.Popen(cmd, stdout=out or subprocess.DEVNULL,
                             stderr=subprocess.STDOUT if out else subprocess.DEVNULL)
    except Exception as e:
        if out:
            out.close()
        raise StopExecution('无法启动工作进程: %s: %s' % (type(e).__name__, e))
    rec['child_pid'] = p.pid
    try:
        while True:
            if p.poll() is not None:
                break
            if time.monotonic() - rec['started_monotonic'] > float(deadline_s):
                rec['timed_out'] = True
                rec['terminate_requested_at'] = _now()
                rec['terminate_requested_monotonic'] = time.monotonic()
                p.terminate()
                try:
                    p.wait(timeout=10)
                except Exception:
                    rec['kill_at'] = _now()
                    p.kill()
                    try:
                        p.wait(timeout=10)
                    except Exception:
                        pass
                break
            time.sleep(poll)
    finally:
        if out:
            out.close()
    rec['child_exit_code'] = p.returncode
    rec['child_exit_confirmed'] = (p.poll() is not None)
    rec['wall_s'] = time.monotonic() - rec['started_monotonic']
    rec['finished_at'] = _now()
    return rec


# ───────────────────────── 预登记 / 协议（RB03 严格 schema） ─────────────────────────

def verify_prereg(pq, cfg):
    checks = {
        'config_sha256': (pq.get('config_sha256'), cfg['_config_sha256']),
        'asset_fingerprint': (pq.get('asset_fingerprint'), core.asset_fingerprint(cfg)),
        'source_fingerprint': (pq.get('source_fingerprint'), core.source_fingerprint(cfg)),
        'implementation_fingerprint': (pq.get('implementation_fingerprint'),
                                       core.implementation_fingerprint()),
        'env': (pq.get('env'), core.env_fingerprint()),
    }
    bad = [k for k, (a, b) in checks.items() if a != b]
    if bad:
        raise BudgetStop('PREREG_MISMATCH: 预登记与当前状态不符（加载前阻断）: %s' % bad)
    return True


def verify_protocol(pq, inputs, crop, order, instances):
    """严格协议 schema：缺字段/空集合/缺必需哈希一律阻断（RB03）。"""
    proto = pq.get('protocol')
    bad = []
    if not isinstance(proto, dict):
        raise BudgetStop('PROTOCOL_MISSING: 预登记缺少 protocol（拒绝真实运行）')
    samples = proto.get('samples')
    if not isinstance(samples, dict) or not samples:
        raise BudgetStop('PROTOCOL_MISSING: protocol.samples 缺失或为空')
    missing_samples = [s for s in UNIQUE if s not in samples]
    if missing_samples:
        bad.append('缺少必需样本: %s' % missing_samples)
    extra = [s for s in samples if s not in UNIQUE]
    if extra:
        bad.append('出现未登记样本: %s' % extra)
    if int(proto.get('crop', -1)) != int(crop):
        bad.append('crop %s != %s' % (proto.get('crop'), crop))
    if list(proto.get('order_per_instance', [])) != list(order):
        bad.append('order 不一致')
    if list(proto.get('instances', [])) != list(instances):
        bad.append('instances 不一致')
    if int(proto.get('warmups', -1)) != 0:
        bad.append('warmups 必须为 0')
    if int(proto.get('forwards_total', -1)) != len(order) * len(instances):
        bad.append('forwards_total 与顺序/实例数不符')
    for name in UNIQUE:
        want = samples.get(name) or {}
        got = inputs.get(name) or {}
        absent = [k for k in REQUIRED_SAMPLE_KEYS if k not in want]
        if absent:
            bad.append('%s 缺必需字段: %s' % (name, absent))
            continue
        if want.get('input_wh') is None:
            bad.append('%s.input_wh 为空' % name)
        elif want['input_wh'] != got.get('input_wh'):
            bad.append('%s.input_wh 不符（%s != %s）' % (name, want['input_wh'],
                                                        got.get('input_wh')))
        fh = want.get('input_file_sha256')
        if fh is None:
            src = want.get('source')
            src_sha = want.get('source_sha256')
            if not src or not src_sha:
                bad.append('%s 缺 input_file_sha256 且未给出 source/source_sha256' % name)
            elif (inputs.get(src) or {}).get('input_file_sha256') != src_sha:
                bad.append('%s.source_sha256 不符' % name)
        elif fh != got.get('input_file_sha256'):
            bad.append('%s.input_file_sha256 不符' % name)
        if name == 'case1_crop512' and want.get('crop_box') != got.get('crop_box'):
            bad.append('%s.crop_box 不符（%s != %s）' % (name, want.get('crop_box'),
                                                        got.get('crop_box')))
    if bad:
        raise BudgetStop('PROTOCOL_MISMATCH: 实际输入/协议与预登记不符（加载前阻断）: %s' % bad)
    return True


def verify_ledger_binding(ledger, prereg_sha256):
    if ledger.data.get('prereg_sha256') != prereg_sha256:
        raise BudgetStop('LEDGER_BINDING_MISMATCH: 总账绑定的预登记哈希与本次不符')
    return True


def build_inputs(case1, case7, crop, in_dir):
    import hashlib
    from PIL import Image
    in_dir.mkdir(parents=True, exist_ok=True)
    im1 = Image.open(str(case1)).convert('RGB')
    im7 = Image.open(str(case7)).convert('RGB')
    w, h = im1.size
    left, top = (w - crop) // 2, (h - crop) // 2
    box = [left, top, left + crop, top + crop]
    crop_path = in_dir / ('case1_crop%d.png' % crop)
    im1.crop(tuple(box)).save(crop_path)
    items = {
        'case1_crop512': {'path': crop_path, 'crop_box': box, 'src_wh': [w, h],
                          'source': 'case1_full', 'input_wh': [int(crop), int(crop)],
                          'tensor': core.pil_to_tensor(Image.open(crop_path).convert('RGB'))},
        'case1_full': {'path': Path(case1), 'crop_box': None, 'src_wh': [w, h],
                       'input_wh': list(im1.size),
                       'tensor': core.pil_to_tensor(im1)},
        'case7_full': {'path': Path(case7), 'crop_box': None, 'src_wh': list(im7.size),
                       'input_wh': list(im7.size),
                       'tensor': core.pil_to_tensor(im7)},
    }
    for k, v in items.items():
        t = v['tensor']
        v['tensor_shape'] = list(t.shape)
        v['tensor_content_sha256'] = hashlib.sha256(
            t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        v['tensor_byte_protocol'] = 'float32-little-endian-raw-C-contiguous'
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


def _hash_tensor(t, npy_path):
    import hashlib
    return {'tensor_content_sha256': hashlib.sha256(
                t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),
            'tensor_byte_protocol': 'float32-little-endian-raw-C-contiguous',
            'tensor_shape': list(t.shape), 'tensor_dtype': str(t.dtype),
            'tensor_npy': str(npy_path),
            'tensor_npy_sha256': core.sha256_file(npy_path)}


class Runner_:
    """单实例（A 或 B）的构造与逐次调用。"""

    def __init__(self, entry, model_dir, cfg, device, run_dir, ledger, manifest, attempt_id):
        self.entry = entry
        self.model_dir = model_dir
        self.cfg = cfg
        self.device = device
        self.run_dir = Path(run_dir)
        self.ledger = ledger
        self.manifest = manifest
        self.attempt_id = attempt_id
        self.inputs = None
        self.runner = None
        self.assets = None

    def build(self):
        t0 = time.perf_counter()
        if self.entry == A:
            self.runner = Runner(self.model_dir, device=self.device)
            self.runner._ensure()
            self.assets = self.runner.assets
            identity = self.runner.identity(entry='tensor_runner')
        else:
            self.assets = core.Assets(self.cfg, self.device)
            identity = self.assets.identity(entry='offline_file', model_dir=self.model_dir,
                                            device=self.device)
        load_s = time.perf_counter() - t0
        self.manifest['instances'][self.entry] = {'load_s': load_s, 'identity': identity}
        _write_json(self.run_dir / 'manifest.json', self.manifest)
        return identity

    def run_calls(self, after=None):
        import numpy as np
        import torch
        for idx, tag in enumerate(ORDER, start=1):
            cid = '%s_%s_%s_r%d' % (self.attempt_id, self.entry, tag, idx)
            self.ledger.check_wall()
            self.ledger.check_disk(self.run_dir)
            n = self.ledger.reserve('%s:%s' % (self.entry, cid))
            timing = {}
            row = {'call_id': cid, 'entry': self.entry, 'tag': tag, 'run_index': idx,
                   'reserve_n': n, 'status': 'OK', 'error': None, 'timing': timing,
                   'artifacts': {'status': 'NOT_PRODUCED'}}
            t0 = time.perf_counter()
            try:
                if self.entry == A:
                    out = self.runner.infer(self.inputs[tag]['tensor'], prompt='', timing=timing)
                    npy = self.run_dir / ('%s.npy' % cid)
                    np.save(npy, out.detach().float().cpu().numpy())
                    png = self.run_dir / ('%s.png' % cid)
                    core.tensor_to_pil(out).save(png)
                    row.update({'shape': list(out.shape), 'dtype': str(out.dtype),
                                'min': float(out.min()), 'max': float(out.max()),
                                'finite': bool(torch.isfinite(out).all()),
                                'png_path': str(png), 'png_bytes': png.stat().st_size,
                                'png_sha256': core.sha256_file(png)})
                    row.update(_hash_tensor(out, npy))
                else:
                    rec = offline.process_image(self.assets, self.inputs[tag]['path'],
                                                self.run_dir, self.run_dir,
                                                call_id=cid, timing=timing)
                    row.update({k: v for k, v in rec.items()
                                if k not in ('entry', 'timing', 'call_id')})
                row['artifacts'] = {'status': 'PRODUCED',
                                    'tensor_npy': row.get('tensor_npy'),
                                    'tensor_npy_sha256': row.get('tensor_npy_sha256'),
                                    'tensor_content_sha256': row.get('tensor_content_sha256'),
                                    'png': row.get('png_path'),
                                    'png_sha256': row.get('png_sha256')}
            except Exception as e:
                row['status'] = 'FAILED'
                row['error'] = '%s: %s' % (type(e).__name__, e)
                row['artifacts'] = {'status': 'NOT_PRODUCED', 'reason': row['error']}
            row['entry_wall_s'] = time.perf_counter() - t0
            self.manifest['calls'].append(row)
            _write_json(self.run_dir / 'manifest.json', self.manifest)
            if row['status'] != 'OK':
                raise StopExecution('CALL_FAILED %s: %s' % (cid, row['error']))
            if after is not None:
                after(tag, idx)
        try:
            import torch as _t
            self.manifest['instances'][self.entry]['peak_vram_bytes'] = (
                int(_t.cuda.max_memory_allocated()) if _t.cuda.is_available() else None)
        except Exception:
            pass

    def release(self):
        if getattr(self, 'runner', None) is not None:
            self.runner.release()
            self.runner.assets = None
            self.runner = None
        if getattr(self, 'assets', None) is not None:
            self.assets.release()
            self.assets = None
        self.inputs = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def compare_tag(manifest, tag, crit):
    import numpy as np
    from PIL import Image
    a = [c for c in manifest['calls'] if c['entry'] == A and c['tag'] == tag
         and c['status'] == 'OK']
    b = [c for c in manifest['calls'] if c['entry'] == B and c['tag'] == tag
         and c['status'] == 'OK']
    if not a or not b:
        return None
    ar, br = a[0], b[0]
    ta = __import__('torch').from_numpy(np.load(ar['tensor_npy'], allow_pickle=False))
    tb = __import__('torch').from_numpy(np.load(br['tensor_npy'], allow_pickle=False))
    d = _diff(ta, tb)
    d['pass_max'] = d.get('max_abs_diff', 9e9) <= crit['max_abs_diff']
    d['pass_mean'] = d.get('mean_abs_diff', 9e9) <= crit['mean_abs_diff']
    pa = np.array(Image.open(ar['png_path']).convert('RGB'))
    pb = np.array(Image.open(br['png_path']).convert('RGB'))
    d['png_identical'] = bool(np.array_equal(pa, pb))
    d['pass_png'] = bool(d['png_identical'])
    d['a_tensor_npy'] = ar['tensor_npy']
    d['b_tensor_npy'] = br['tensor_npy']
    d['a_png_sha256'] = ar['png_sha256']
    d['b_png_sha256'] = br['png_sha256']
    return d


def compare_repeat(manifest, entry, crit):
    import numpy as np
    cs = [c for c in manifest['calls'] if c['entry'] == entry
          and c['tag'] == 'case1_crop512' and c['status'] == 'OK']
    if len(cs) < 2:
        return None
    a1 = __import__('torch').from_numpy(np.load(cs[0]['tensor_npy'], allow_pickle=False))
    a2 = __import__('torch').from_numpy(np.load(cs[1]['tensor_npy'], allow_pickle=False))
    d = _diff(a1, a2)
    d['pass_max'] = d.get('max_abs_diff', 9e9) <= crit['max_abs_diff']
    d['pass_mean'] = d.get('mean_abs_diff', 9e9) <= crit['mean_abs_diff']
    d['first_npy'] = cs[0]['tensor_npy']
    d['second_npy'] = cs[1]['tensor_npy']
    return d


def identity_checks(manifest):
    """两侧身份 + 安全加载：必须都满足（RB03：不能只看 A 的 decoder 布尔值）。"""
    out = {'identity_equal': False, 'identity_diff_keys': None, 'no_unsafe_load': False,
           'loaders': {}}
    ia = (manifest['instances'].get(A) or {}).get('identity')
    ib = (manifest['instances'].get(B) or {}).get('identity')
    if not ia or not ib:
        return out
    ca, cb = core.identity_comparable(ia), core.identity_comparable(ib)
    out['identity_equal'] = bool(ca == cb)
    if not out['identity_equal']:
        out['identity_diff_keys'] = [k for k in ca if ca.get(k) != cb.get(k)]
    ok = True
    for entry, idy in ((A, ia), (B, ib)):
        lr = idy.get('load_report') or {}
        dino = lr.get('dino') or {}
        out['loaders'][entry] = {'light_decoder_weights_only': lr.get('light_decoder_weights_only'),
                                 'dino_loader': dino.get('loader'),
                                 'dino_weights_only': dino.get('weights_only'),
                                 'dino_strict': dino.get('strict'),
                                 'dino_uses_torch_hub': dino.get('uses_torch_hub')}
        if not (lr.get('light_decoder_weights_only') is True
                and dino.get('weights_only') is True
                and dino.get('strict') is True
                and dino.get('uses_torch_hub') is False):
            ok = False
    out['no_unsafe_load'] = bool(ok)
    return out


def timing_events(manifest):
    ev = {}
    for c in manifest['calls']:
        if c['status'] == 'OK':
            ev.setdefault(c['entry'], []).append((c['tag'], c['timing'].get('events')))
    if not ev.get(A) or not ev.get(B):
        return False, {'reason': '缺少事件序列'}
    if any(not e[1] for e in ev[A] + ev[B]):
        return False, {'reason': '事件序列为空'}
    n = min(len(ev[A]), len(ev[B]))
    same = all(ev[A][i][1] == ev[B][i][1] for i in range(n))
    return bool(same and len(ev[A]) == len(ev[B])), {'A': [e[1] for e in ev[A]],
                                                     'B': [e[1] for e in ev[B]]}


def verify_final_artifacts(run_dir, manifest, crit):
    """最终产物核验：从磁盘重算全部哈希与成对/重复判据（RB03）。"""
    import hashlib
    import numpy as np
    from PIL import Image
    run_dir = Path(run_dir)
    res = {'checked_calls': 0, 'errors': [], 'pairs': {}, 'repeat': {}, 'ok': False}
    ok_calls = [c for c in manifest['calls'] if c['status'] == 'OK']
    for c in ok_calls:
        try:
            npy, png = Path(c['tensor_npy']), Path(c['png_path'])
            if not npy.is_file() or not png.is_file():
                res['errors'].append('%s: 产物缺失' % c['call_id'])
                continue
            arr = np.load(npy, allow_pickle=False)
            if hashlib.sha256(arr.tobytes()).hexdigest() != c.get('tensor_content_sha256'):
                res['errors'].append('%s: 张量内容哈希不符' % c['call_id'])
            if core.sha256_file(npy) != c.get('tensor_npy_sha256'):
                res['errors'].append('%s: NPY 文件哈希不符' % c['call_id'])
            if core.sha256_file(png) != c.get('png_sha256'):
                res['errors'].append('%s: PNG 哈希不符' % c['call_id'])
            im = np.array(Image.open(png).convert('RGB'))
            if list(im.shape) != [int(arr.shape[2]), int(arr.shape[3]), 3]:
                res['errors'].append('%s: PNG 尺寸与张量不符' % c['call_id'])
            res['checked_calls'] += 1
        except Exception as e:
            res['errors'].append('%s: %s: %s' % (c['call_id'], type(e).__name__, e))
    for tag in UNIQUE:
        d = compare_tag(manifest, tag, crit)
        if d:
            res['pairs'][tag] = d
    for e in (A, B):
        d = compare_repeat(manifest, e, crit)
        if d:
            res['repeat']['%s_crop512_repeat' % e] = d
    res['ok'] = bool(
        not res['errors'] and res['checked_calls'] == len(ok_calls) == len(ORDER) * 2
        and len(res['pairs']) == len(UNIQUE)
        and all(v.get('pass_max') and v.get('pass_mean') and v.get('pass_png')
                for v in res['pairs'].values())
        and len(res['repeat']) == 2
        and all(v.get('pass_max') and v.get('pass_mean') for v in res['repeat'].values()))
    return res


def ledger_consistency(ledger_path, manifest, prereg_sha256, expect_forwards):
    """账本一致：终态闭合、扣款数与调用数一致、绑定的预登记一致、锁已释放（RB02/RB03）。"""
    data = read_ledger_ro(ledger_path)
    lock = Path(str(ledger_path)).with_suffix('.lock')
    n_calls = len(manifest.get('calls', []))
    out = {
        'attempts_closed': data.get('open_attempt') is None,
        'forwards_reserved': int(data.get('forwards_reserved', -1)),
        'expected_forwards': int(expect_forwards),
        'n_calls': n_calls,
        'n_reservations': len(data.get('reservations', [])),
        'prereg_bound': data.get('prereg_sha256') == prereg_sha256,
        'lock_released': not lock.exists(),
        'last_attempt_status': (data.get('attempts') or [{}])[-1].get('status'),
        'wall_seconds_total': float(data.get('wall_seconds_total', 0.0)),
        'disk_bytes_total': int(data.get('disk_bytes_total', 0)),
    }
    out['reserved_matches_calls'] = (out['forwards_reserved'] == n_calls
                                     == out['n_reservations'])
    out['ok'] = bool(out['attempts_closed'] and out['reserved_matches_calls']
                     and out['prereg_bound'] and out['lock_released'])
    return out


def _worker_argv(args, attempt_id, deadline_s=None):
    cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
           '--run-dir', str(args.run_dir), '--prereg', str(args.prereg),
           '--case1', str(args.case1), '--case7', str(args.case7),
           '--crop', str(args.crop), '--budget-ledger', str(args.budget_ledger),
           '--attempt-id', attempt_id]
    if args.model_dir:
        cmd += ['--model-dir', str(args.model_dir)]
    if args.device:
        cmd += ['--device', str(args.device)]
    if deadline_s is not None:
        cmd += ['--deadline-s', '%.1f' % float(deadline_s)]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description='PIPE-01R3 成对验证驱动（默认监督模式）')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--prereg', required=True)
    ap.add_argument('--case1', required=True)
    ap.add_argument('--case7', required=True)
    ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--budget-ledger', default=None)
    ap.add_argument('--init-ledger', action='store_true')
    ap.add_argument('--attempt-id', default=None)
    ap.add_argument('--worker', action='store_true', help='内部：工作进程')
    ap.add_argument('--no-supervise', action='store_true',
                    help='内部/CPU 测试：同进程运行（GPU 批次禁止使用）')
    ap.add_argument('--deadline-s', type=float, default=None)
    args = ap.parse_args(argv)
    if args.budget_ledger is None:
        args.budget_ledger = str(Path(args.run_dir).parent / 'budget_ledger.json')
    if args.worker:
        return _run_worker(args)
    return _supervise(args) if not args.no_supervise else _run_worker(args)


def _supervise(args):
    """监督者：只读预检 → 派生 worker → 硬墙钟监督 → 事后恢复 → 合并判决。"""
    pre = Path(args.prereg)
    if not pre.is_file():
        raise SystemExit('拒绝执行：缺少预登记 manifest %s' % pre)
    prereg_sha = core.sha256_file(pre)
    try:
        ro = read_ledger_ro(args.budget_ledger)
        if ro.get('prereg_sha256') != prereg_sha:
            raise BudgetStop('LEDGER_BINDING_MISMATCH: 总账绑定的预登记哈希与本次不符')
        if int(ro.get('forwards_reserved', 0)) >= int(ro.get('max_forwards', 0)):
            raise BudgetStop('前向额度为 0 可用（%d/%d）：拒绝启动'
                             % (ro.get('forwards_reserved'), ro.get('max_forwards')))
        if ro.get('open_attempt'):
            raise BudgetStop('存在未结束的 attempt（%s）：本批不自动接管'
                             % ro['open_attempt'].get('attempt_id'))
        remaining = float(ro['max_minutes']) * 60.0 - float(ro.get('wall_seconds_total', 0.0))
        if remaining <= 0:
            raise BudgetStop('墙钟无余额（剩余 %.1fs）' % remaining)
    except BudgetStop as e:
        raise SystemExit(str(e))

    attempt_id = args.attempt_id or ('%s_%s' % (Path(args.run_dir).name,
                                                time.strftime('%H%M%S')))
    log = Path(str(Path(args.run_dir)) + '.worker_log.txt')     # 目录外（RB01）
    rec = run_supervised(_worker_argv(args, attempt_id, remaining),
                         deadline_s=remaining, log_path=log)
    rec['attempt_id'] = attempt_id
    rec['run_dir'] = str(args.run_dir)
    rec['prereg_sha256'] = prereg_sha

    # 事后恢复（RB02）：先 reload 最新总账，再按需记中断；只清理本子进程的锁
    led = BudgetLedger(args.budget_ledger, init=False, prereg_sha256=prereg_sha)
    led.reload()
    recovered = False
    if led.data.get('open_attempt'):
        recovered = led.end_attempt('TIMEOUT_STOP' if rec['timed_out'] else 'WORKER_INTERRUPTED',
                                    'supervisor recovery: worker 未自行收尾',
                                    run_dir=args.run_dir)
    lock = Path(str(args.budget_ledger)).with_suffix('.lock')
    lock_cleaned = False
    if lock.exists():
        try:
            info = json.loads(lock.read_text(encoding='utf-8'))
        except Exception:
            info = {}
        if info.get('pid') == rec.get('child_pid'):
            try:
                lock.unlink()
                lock_cleaned = True
            except Exception:
                pass
    rec['recovery_applied'] = bool(recovered)
    rec['lock_cleaned_by_supervisor'] = bool(lock_cleaned)

    run_dir = Path(args.run_dir)
    worker_result = None
    if (run_dir / 'paired_readings.json').is_file():
        try:
            worker_result = json.loads((run_dir / 'paired_readings.json').read_text(
                encoding='utf-8'))
        except Exception:
            worker_result = None
    manifest = None
    if (run_dir / 'manifest.json').is_file():
        try:
            manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        except Exception:
            manifest = None

    worker_checks_pass = bool(worker_result and worker_result.get('criteria_all_pass'))
    led_reload = read_ledger_ro(args.budget_ledger)
    batch_wall = float(led_reload.get('wall_seconds_total', 0.0))
    budget_seconds = float(led_reload.get('max_minutes', 0.0)) * 60.0
    supervisor_ok = bool(rec['child_exit_confirmed'] and not rec['timed_out']
                         and rec['child_exit_code'] == 0
                         and batch_wall <= budget_seconds)
    ledger_ok = None
    artifacts_ok = None
    if manifest is not None:
        ledger_ok = ledger_consistency(args.budget_ledger, manifest, prereg_sha,
                                       expect_forwards=len(ORDER) * 2)
        crit = (worker_result or {}).get('criteria') or {'max_abs_diff': 1e-5,
                                                         'mean_abs_diff': 1e-6}
        artifacts_ok = verify_final_artifacts(run_dir, manifest, crit)
    merged = {
        'schema': 'PIPE01-attempt-result/1',
        'attempt_id': attempt_id, 'run_dir': str(run_dir),
        'worker_checks_pass': worker_checks_pass,
        'supervisor_success': supervisor_ok,
        'ledger_consistent': bool(ledger_ok and ledger_ok.get('ok')),
        'final_artifacts_verified': bool(artifacts_ok and artifacts_ok.get('ok')),
        'supervisor': rec,
        'ledger': led_reload,
        'ledger_consistency': ledger_ok,
        'artifacts': {k: v for k, v in (artifacts_ok or {}).items() if k != 'pairs'},
        'worker_status': (worker_result or {}).get('status'),
        'finished_at': _now(),
    }
    merged['all_pass'] = bool(merged['worker_checks_pass'] and merged['supervisor_success']
                              and merged['ledger_consistent']
                              and merged['final_artifacts_verified'])
    if run_dir.exists():
        _write_json(run_dir / 'supervisor.json', rec)
        _write_json(run_dir / 'attempt_result.json', merged)
    print('[supervise] attempt=%s rc=%s timed_out=%s wall=%.1fs recovered=%s lock_cleaned=%s'
          % (attempt_id, rec['child_exit_code'], rec['timed_out'], rec['wall_s'],
             rec['recovery_applied'], rec['lock_cleaned_by_supervisor']))
    print('[verdict] worker=%s supervisor=%s ledger=%s artifacts=%s ALL=%s'
          % (merged['worker_checks_pass'], merged['supervisor_success'],
             merged['ledger_consistent'], merged['final_artifacts_verified'],
             merged['all_pass']))
    return 0 if merged['all_pass'] else 1


def _run_worker(args):
    run_dir = Path(args.run_dir)
    pre = Path(args.prereg)
    if not pre.is_file():
        raise SystemExit('拒绝执行：缺少预登记 manifest %s（不得先跑后登记）' % pre)
    leftover = [p.name for p in run_dir.iterdir() if p.name != 'prereg.json'] \
        if run_dir.exists() else []
    if leftover:
        raise SystemExit('拒绝复用非空 run 目录: %s（已存在: %s）' % (run_dir, leftover[:5]))
    pq = json.loads(pre.read_text(encoding='utf-8'))
    prereg_sha = core.sha256_file(pre)
    cfg = core.load_config()
    ledger = None
    try:
        verify_prereg(pq, cfg)
        ledger = BudgetLedger(args.budget_ledger, pq['budget']['max_forwards'],
                              pq['budget']['max_minutes'], pq['budget']['max_output_gib'],
                              init=args.init_ledger, prereg_sha256=prereg_sha)
        ledger.acquire_lock()
        verify_ledger_binding(ledger, prereg_sha)
        ledger.check_startable()
    except BudgetStop as e:
        raise SystemExit(str(e))

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    attempt_id = args.attempt_id or ('%s_%s' % (run_dir.name, time.strftime('%H%M%S')))
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(args.case1, args.case7, args.crop, run_dir / 'inputs')
    try:
        verify_protocol(pq, inputs, args.crop, ORDER,
                        ['A=tensor_runner', 'B=offline_file_entry'])
    except BudgetStop as e:
        ledger.end_attempt('PROTOCOL_MISMATCH', str(e), run_dir=run_dir)
        ledger.release_lock()
        raise SystemExit(str(e))
    model_dir = args.model_dir or cfg['base_dir']
    crit = pq['criteria']
    manifest = {'task': 'PIPE-01', 'schema': 'PIPE01-generation-manifest/2',
                'attempt_id': attempt_id, 'run_dir': str(run_dir), 'device': device,
                'prereg_sha256': prereg_sha, 'started_at': _now(),
                'instances': {}, 'calls': [],
                'inputs': {k: {kk: vv for kk, vv in v.items() if kk != 'tensor'}
                           for k, v in inputs.items()}}
    _write_json(run_dir / 'manifest.json', manifest)
    ledger.begin_attempt(attempt_id, run_dir)

    deadline = ledger.remaining_seconds()
    if args.deadline_s is not None:
        deadline = min(deadline, float(args.deadline_s))
    stop_flag = threading.Event()
    status, note = 'OK', ''
    pairs, repeat, identity, idchk, events_ok, events_detail = {}, {}, {}, {}, False, {}
    inst_a = inst_b = None

    def _watchdog():
        while not stop_flag.wait(0.5):
            if time.monotonic() - t_start > deadline:
                stop_flag.set()
                return

    def _check_stop():
        if stop_flag.is_set():
            raise StopExecution('WATCHDOG: 运行中墙钟触顶')

    t_start = time.monotonic()
    th = threading.Thread(target=_watchdog, daemon=True)
    th.start()
    try:
        try:
            inst_a = Runner_(A, model_dir, cfg, device, run_dir, ledger, manifest, attempt_id)
            inst_a.inputs = inputs
            idA = inst_a.build()
            inst_a.run_calls()
            d = compare_repeat(manifest, A, crit)
            if d:
                repeat['A_crop512_repeat'] = d
                if not (d['pass_max'] and d['pass_mean']):
                    raise StopExecution('REPEAT_FAIL A: %s' % d)
            _check_stop()
            inst_a.release()
            inst_a = None
            gc.collect()

            inst_b = Runner_(B, model_dir, cfg, device, run_dir, ledger, manifest, attempt_id)
            inst_b.inputs = inputs
            idB = inst_b.build()
            identity = {'A': idA, 'B': idB}
            idchk = identity_checks(manifest)
            if not idchk['identity_equal']:
                raise StopExecution('IDENTITY_MISMATCH（在 B 首个前向之前阻断）: %s'
                                    % idchk['identity_diff_keys'])

            def _after_b(tag, idx):
                _check_stop()
                dd = compare_tag(manifest, tag, crit)
                if dd and not (dd['pass_max'] and dd['pass_mean'] and dd['pass_png']):
                    raise StopExecution('CRITERIA_EXCEEDED at %s: max=%.3e mean=%.3e png=%s'
                                        % (tag, dd['max_abs_diff'], dd['mean_abs_diff'],
                                           dd['png_identical']))
                if tag == 'case1_crop512':
                    dr = compare_repeat(manifest, B, crit)
                    if dr and not (dr['pass_max'] and dr['pass_mean']):
                        raise StopExecution('REPEAT_FAIL B: %s' % dr)
            inst_b.run_calls(after=_after_b)
            dr = compare_repeat(manifest, B, crit)
            if dr:
                repeat['B_crop512_repeat'] = dr
        except StopExecution as e:
            status, note = 'STOPPED', str(e)
        except BudgetStop as e:
            status, note = 'BUDGET_STOP', str(e)
        except Exception as e:
            status, note = 'FAILED', '%s: %s' % (type(e).__name__, e)
    finally:
        stop_flag.set()
        try:
            if inst_b is not None:
                inst_b.release()
                inst_b = None
            if inst_a is not None:
                inst_a.release()
                inst_a = None
            gc.collect()
        except Exception:
            pass
        # 比较/写盘/记账全链收尾（RB02）：任何异常都必须留下终态并释放自己的锁
        try:
            for tag in UNIQUE:
                d = compare_tag(manifest, tag, crit)
                if d:
                    pairs[tag] = d
            for e in (A, B):
                dr = compare_repeat(manifest, e, crit)
                if dr:
                    repeat['%s_crop512_repeat' % e] = dr
            events_ok, events_detail = timing_events(manifest)
            if not idchk:
                idchk = identity_checks(manifest)
            all_calls_ok = all(c['status'] == 'OK' for c in manifest['calls'])
            detail = {
                'identity_equal': bool(idchk.get('identity_equal')),
                'all_calls_ok': bool(all_calls_ok),
                'n_calls': len(manifest['calls']),
                'n_expected_calls': len(ORDER) * 2,
                'completed_protocol': bool(len(manifest['calls']) == len(ORDER) * 2
                                           and all_calls_ok),
                'pairs_ok': bool(pairs) and len(pairs) == len(UNIQUE) and all(
                    v.get('pass_max') and v.get('pass_mean') and v.get('pass_png')
                    for v in pairs.values()),
                'repeat_ok': len(repeat) == 2 and all(
                    v.get('pass_max') and v.get('pass_mean') for v in repeat.values()),
                'timing_events_equal': bool(events_ok),
                'no_unsafe_load': bool(idchk.get('no_unsafe_load')),
                'loaders': idchk.get('loaders'),
                'budget_ok': bool(int(ledger.data['forwards_reserved'])
                                  <= int(ledger.data['max_forwards'])
                                  and ledger.wall_used()
                                  <= float(ledger.data['max_minutes']) * 60.0),
                'disk_ok': bool(ledger.disk_used(run_dir)
                                <= float(ledger.data['max_gib']) * (1024 ** 3)),
                'prereg_verified': True,
                'protocol_verified': True,
            }
            ok_keys = ('identity_equal', 'completed_protocol', 'pairs_ok', 'repeat_ok',
                       'budget_ok', 'disk_ok', 'no_unsafe_load', 'timing_events_equal',
                       'prereg_verified', 'protocol_verified')
            result = {'status': status, 'note': note, 'attempt_id': attempt_id,
                      'wall_used_total_s': ledger.wall_used(),
                      'forwards_reserved_total': int(ledger.data['forwards_reserved']),
                      'disk_used_total_bytes': ledger.disk_used(run_dir),
                      'criteria': crit, 'criteria_detail': detail, 'pairs': pairs,
                      'repeat_512': repeat, 'timing_events': events_detail,
                      'identity_diff_keys': idchk.get('identity_diff_keys'),
                      'ledger': ledger.snapshot()}
            result['criteria_all_pass'] = bool(status == 'OK'
                                               and all(detail.get(k) for k in ok_keys))
            if status == 'OK' and not result['criteria_all_pass']:
                result['status'] = status = 'CRITERIA_FAIL'
            # 最终产物核验（末次写盘之前算，写盘之后由监督者再核验一次）
            final_art = verify_final_artifacts(run_dir, manifest, crit)
            result['final_artifacts'] = {k: v for k, v in final_art.items() if k != 'pairs'}
            result['final_artifacts_ok'] = bool(final_art['ok'])
            result['finished_at'] = _now()
            _write_json(run_dir / 'paired_readings.json', result)
            manifest['finished_at'] = _now()
            manifest['final_status'] = status
            _write_json(run_dir / 'manifest.json', manifest)
            # 终态核验：末次写盘之后重读磁盘，确认可解析且状态一致
            re_read = json.loads((run_dir / 'paired_readings.json').read_text(encoding='utf-8'))
            result['post_write_check'] = {
                'paired_readings_reparse_ok': True,
                'status_match': re_read.get('status') == status,
                'manifest_final_status': manifest.get('final_status'),
            }
        except Exception as e:
            result = {'status': 'FAILED', 'note': '收尾阶段异常: %s: %s'
                      % (type(e).__name__, e), 'criteria_all_pass': False,
                      'post_write_check': {'paired_readings_reparse_ok': False}}
            _write_json(run_dir / 'paired_readings.json', result)
        finally:
            try:
                ledger.end_attempt(result.get('status', 'UNKNOWN'), result.get('note', ''),
                                   run_dir=run_dir)
            finally:
                ledger.release_lock()
    print('[pair] status=%s calls=%d forwards_reserved=%d wall=%.1fs'
          % (result.get('status'), len(manifest['calls']),
             int(ledger.data['forwards_reserved']), ledger.wall_used()))
    return 0 if result.get('criteria_all_pass') else 1


if __name__ == '__main__':
    sys.exit(main())
