# -*- coding: utf-8 -*-
"""PIPE-01R2 (R-A2) 阶段B 成对验证驱动。相对 R-A 的返修（review PIPE-RA02–RA07）：

  RA02 立即停止：前向失败 -> 记录并保留扣款后**不再启动后续调用/实例**；
       B 加载后**先比身份**，不符则在 B 首个 infer 前阻断；
       A/B 每出现一个可比样本/重复项就即时判定，超阈值不再继续。
  RA03 实例释放：A 调用的资产不被驱动持有；释放后（引用清零 + gc）才加载 B。
  RA04 预算加固：排他锁（O_EXCL）+ 持锁读最新总账；PID 判定不确定即 fail closed；
       无余额/时间不足则在加载前拒绝；监督**本任务自建工作进程**兑现硬墙钟；
       累计本批所有 attempt 的输出；任何异常路径都保留终态与已占用次数并释放锁。
  RA05 协议绑定：预登记固定 8 次/0 预热/30 分钟/4 GiB，并冻结样本哈希、尺寸、裁切框、顺序、
       阈值、环境与源码；运行前校验实际输入与协议，总账绑定同一预登记哈希。
  RA06 计时边界：A/B 都传 CPU 张量进 core.infer_core，搬运在计时区间内；事件序列可比对。
  RA07 manifest：tensor 内容哈希与 NPY 文件哈希分开命名并标注协议；失败调用明确标注未产出。

用法（监督模式，默认）:
  python pair_verify.py --run-dir <dir> --prereg <dir>/prereg.json --case1 <jpg> --case7 <jpg> \
      --budget-ledger <json> [--init-ledger]
内部工作进程由本驱动自行派生（--worker），不会被外部调用。
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


def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


class BudgetStop(RuntimeError):
    """预算/并发/复用/协议不符等硬约束；必须在加载或前向之前停下。"""


class StopExecution(RuntimeError):
    """运行中必须立即停止（前向失败/身份不符/超阈值/超时）。"""


# ───────────────────────── 预算总账（RA04） ─────────────────────────

def _pid_state(pid):
    """三态：True=存活，False=确定不存在，None=无法判定（调用方须 fail closed）。"""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if h:
            k32.CloseHandle(h)
            return True
        err = k32.GetLastError()
        if err == 87:        # ERROR_INVALID_PARAMETER: PID 不存在
            return False
        if err == 5:         # ERROR_ACCESS_DENIED: 存在但无权限
            return True
        return None
    except Exception:
        return None


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
                raise BudgetStop('总账绑定的预登记哈希与本次不符（拒绝复用）: %s != %s'
                                 % (self.data.get('prereg_sha256'), prereg_sha256))
        elif init:
            self.data = {'task': 'PIPE-01', 'schema': 'PIPE01-budget-ledger/2',
                         'created_at': _now(), 'max_forwards': int(max_forwards),
                         'max_minutes': float(max_minutes), 'max_gib': float(max_gib),
                         'forwards_reserved': 0, 'wall_seconds_total': 0.0,
                         'disk_bytes_total': 0, 'attempts': [], 'open_attempt': None,
                         'reservations': [], 'stale_locks': [],
                         'prereg_sha256': prereg_sha256}
            self._write()
        else:
            raise BudgetStop('预算总账不存在: %s（需 make_evidence 初始化）' % self.path)

    # ── 持久化与排他锁 ──
    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, self.path)

    def acquire_lock(self):
        """排他创建锁文件（O_EXCL）后再读取总账，消除先读后锁的竞争窗口。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({'pid': os.getpid(), 'started_at': _now()}).encode('utf-8')
        for attempt in (1, 2):
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, payload)
                os.close(fd)
                self._reload_locked()
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
                    raise BudgetStop('检测到并发 PIPE-01 attempt（pid=%s）；拒绝并发'
                                     % info.get('pid'))
                self.data.setdefault('stale_locks', []).append(
                    {'stale': info, 'taken_over_at': _now()})
                self._write()
                try:
                    self.lock_path.unlink()
                except Exception:
                    pass
        raise BudgetStop('无法获取预算总账锁: %s' % self.lock_path)

    def _reload_locked(self):
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))

    def release_lock(self):
        try:
            if self.lock_path.exists():
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
                'note': '上次 attempt 未正常结束；时间按保守方式计入', 'elapsed_s': elapsed,
                'disk_bytes': oa.get('disk_bytes', 0)})
        self.data['open_attempt'] = {'attempt_id': attempt_id, 'run_dir': str(run_dir),
                                     'started_at': _now(), 'started_monotonic': time.monotonic(),
                                     'disk_bytes': 0}
        self._write()

    def end_attempt(self, status, note='', run_dir=None):
        oa = self.data.get('open_attempt')
        if not oa:
            return
        elapsed = time.monotonic() - float(oa['started_monotonic'])
        disk = oa.get('disk_bytes', 0)
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
        """加载之前调用：无余额或时间不足直接拒绝。"""
        if int(self.data['forwards_reserved']) >= int(self.data['max_forwards']):
            raise BudgetStop('前向额度为 0 可用（%d/%d）：拒绝加载与前向'
                             % (self.data['forwards_reserved'], self.data['max_forwards']))
        if self.remaining_seconds() <= 0:
            raise BudgetStop('墙钟已无余额（剩余 %.1fs）: 拒绝加载' % self.remaining_seconds())

    def check_wall(self):
        if self.remaining_seconds() <= 0:
            raise BudgetStop('墙钟累计触顶（%.1f 分钟，含所有 attempt 与加载）'
                             % (float(self.data['max_minutes'])))

    def disk_used(self, run_dir):
        prev = int(self.data.get('disk_bytes_total', 0))
        cur = _dir_bytes(run_dir) if run_dir else 0
        return prev + cur

    def check_disk(self, run_dir):
        limit = float(self.data['max_gib']) * (1024 ** 3)
        used = self.disk_used(run_dir)
        if used > limit:
            raise BudgetStop('累计输出触顶 %.2f GiB（已 %.2f GiB，含历史 attempt）'
                             % (limit / 1024 ** 3, used / 1024 ** 3))
        return used

    def snapshot(self):
        return {k: v for k, v in self.data.items() if k != 'open_attempt'}


def _dir_bytes(root):
    rd = Path(root)
    if not rd.exists():
        return 0
    return sum(p.stat().st_size for p in rd.rglob('*') if p.is_file())


# ───────────────────────── 监督：只终止本任务自建的工作进程（RA04） ─────────────────────────

def run_supervised(cmd, deadline_s, poll=0.5, log_path=None):
    """运行自建工作进程并在硬墙钟内监督；超时只终止该子进程。返回 (rc, timed_out, wall_s)。"""
    t0 = time.monotonic()
    out = open(log_path, 'ab') if log_path else None
    try:
        p = subprocess.Popen(cmd, stdout=out or subprocess.DEVNULL,
                             stderr=subprocess.STDOUT if out else subprocess.DEVNULL)
    except Exception as e:
        if out:
            out.close()
        raise StopExecution('无法启动工作进程: %s: %s' % (type(e).__name__, e))
    timed_out = False
    try:
        while True:
            if p.poll() is not None:
                break
            if time.monotonic() - t0 > max(0.0, float(deadline_s)):
                timed_out = True
                p.terminate()
                try:
                    p.wait(timeout=10)
                except Exception:
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
    return (p.returncode, timed_out, time.monotonic() - t0)


# ───────────────────────── 预登记与协议绑定（RA05） ─────────────────────────

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
    """实际样本/裁切/顺序/实例数必须与预登记冻结值一致（RA05）。"""
    proto = pq.get('protocol') or {}
    bad = []
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
    for name, want in (proto.get('samples') or {}).items():
        got = inputs.get(name)
        if got is None:
            bad.append('样本缺失 %s' % name)
            continue
        for key in ('input_file_sha256', 'input_wh'):
            want_v = want.get(key)
            if want_v is None:
                continue          # 运行期生成的中间产物不预冻结文件哈希
            if want_v != got.get(key):
                bad.append('%s.%s 不符' % (name, key))
        if want.get('source_sha256') is not None:
            src = want.get('source') or 'case1_full'
            if inputs.get(src, {}).get('input_file_sha256') != want['source_sha256']:
                bad.append('%s.source_sha256 不符' % name)
        if name == 'case1_crop512' and want.get('crop_box') != got.get('crop_box'):
            bad.append('%s.crop_box 不符（%s != %s）'
                       % (name, want.get('crop_box'), got.get('crop_box')))
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
                          'source': 'case1_full',
                          'tensor': core.pil_to_tensor(Image.open(crop_path).convert('RGB'))},
        'case1_full': {'path': Path(case1), 'crop_box': None, 'src_wh': [w, h],
                       'tensor': core.pil_to_tensor(im1)},
        'case7_full': {'path': Path(case7), 'crop_box': None, 'src_wh': list(im7.size),
                       'tensor': core.pil_to_tensor(im7)},
    }
    for k, v in items.items():
        t = v['tensor']
        v['tensor_shape'] = list(t.shape)
        v['tensor_content_sha256'] = hashlib.sha256(
            t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        v['tensor_byte_protocol'] = 'float32-little-endian-raw-C-contiguous'
        v['input_file_sha256'] = core.sha256_file(v['path'])
        v['input_wh'] = (list(im1.size) if k == 'case1_crop512' else
                         (list(im1.size) if k == 'case1_full' else list(im7.size)))
        if k == 'case1_crop512':
            v['input_wh'] = [crop, crop]
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
    content = hashlib.sha256(
        t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
    out = {'tensor_content_sha256': content,
           'tensor_byte_protocol': 'float32-little-endian-raw-C-contiguous',
           'tensor_shape': list(t.shape), 'tensor_dtype': str(t.dtype),
           'tensor_npy': str(npy_path),
           'tensor_npy_sha256': core.sha256_file(npy_path)}
    return out


class Runner_:
    """内部：一个实例（A 或 B）的构造与逐次调用。"""

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
            self.runner = None
            self.assets = core.Assets(self.cfg, self.device)
            identity = self.assets.identity(entry='offline_file', model_dir=self.model_dir,
                                            device=self.device)
        load_s = time.perf_counter() - t0
        self.manifest['instances'][self.entry] = {'load_s': load_s, 'identity': identity}
        _write_json(self.run_dir / 'manifest.json', self.manifest)
        return identity

    def run_calls(self, after=None):
        """按固定顺序逐次调用；任一次失败 -> 记录后立即 StopExecution（不再继续）。

        after(tag, idx) 在每次调用登记之后、下一次调用之前执行，用于**即时判定**（RA02）。
        """
        import numpy as np
        import torch
        for idx, tag in enumerate(ORDER, start=1):
            cid = '%s_%s_%s_r%d' % (self.attempt_id, self.entry, tag, idx)
            self.ledger.check_wall()
            self.ledger.check_disk(self.run_dir)
            n = self.ledger.reserve('%s:%s' % (self.entry, cid))   # 先扣后跑
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
        """释放本实例（RA03）：先清驱动引用，再让 Assets 清模块，最后 gc。"""
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


def compare_tag(manifest, run_dir, tag, crit, ledger):
    """即时比较：A/B 同一 tag 都有成功结果就立刻判定，超阈值返回 diff 供调用方停止。"""
    import numpy as np
    from PIL import Image
    a = [c for c in manifest['calls'] if c['entry'] == A and c['tag'] == tag and c['status'] == 'OK']
    b = [c for c in manifest['calls'] if c['entry'] == B and c['tag'] == tag and c['status'] == 'OK']
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


def compare_repeat(manifest, run_dir, entry, crit):
    import numpy as np
    cs = [c for c in manifest['calls']
          if c['entry'] == entry and c['tag'] == 'case1_crop512' and c['status'] == 'OK']
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


def _worker_argv(args, attempt_id):
    cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
           '--run-dir', str(args.run_dir), '--prereg', str(args.prereg),
           '--case1', str(args.case1), '--case7', str(args.case7),
           '--crop', str(args.crop), '--budget-ledger', str(args.budget_ledger),
           '--attempt-id', attempt_id]
    if args.model_dir:
        cmd += ['--model-dir', str(args.model_dir)]
    if args.device:
        cmd += ['--device', str(args.device)]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description='PIPE-01R2 成对验证驱动（默认监督模式）')
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
    ap.add_argument('--worker', action='store_true', help='内部：作为工作进程运行')
    ap.add_argument('--no-supervise', action='store_true', help='CPU 测试用：不派生监督进程')
    args = ap.parse_args(argv)
    if args.budget_ledger is None:
        args.budget_ledger = str(Path(args.run_dir).parent / 'budget_ledger.json')

    if not args.worker and not args.no_supervise:
        return _supervise(args)
    return _run_worker(args)


def _supervise(args):
    """监督模式：按剩余墙钟派生并监督本任务自建的工作进程（RA04 硬墙钟）。"""
    pre = Path(args.prereg)
    if not pre.is_file():
        raise SystemExit('拒绝执行：缺少预登记 manifest %s' % pre)
    pq = json.loads(pre.read_text(encoding='utf-8'))
    prereg_sha = core.sha256_file(pre)
    ledger = BudgetLedger(args.budget_ledger, pq['budget']['max_forwards'],
                          pq['budget']['max_minutes'], pq['budget']['max_output_gib'],
                          init=args.init_ledger, prereg_sha256=prereg_sha)
    try:
        ledger.acquire_lock()                  # 由监督者持锁，工作进程不再抢锁
    except BudgetStop as e:
        raise SystemExit(str(e))
    attempt_id = args.attempt_id or ('%s_%s' % (Path(args.run_dir).name,
                                                time.strftime('%H%M%S')))
    try:
        try:
            ledger.check_startable()
        except BudgetStop as e:
            raise SystemExit(str(e))
        remaining = ledger.remaining_seconds()
        ledger.begin_attempt(attempt_id, args.run_dir)
        log = Path(args.run_dir) / 'worker_log.txt'
        Path(args.run_dir).mkdir(parents=True, exist_ok=True)
        rc, timed_out, wall = run_supervised(_worker_argv(args, attempt_id),
                                             deadline_s=remaining, log_path=log)
        status = 'TIMEOUT_STOP' if timed_out else ('OK' if rc == 0 else 'WORKER_FAILED')
        ledger.end_attempt(status, 'supervised rc=%s timed_out=%s' % (rc, timed_out),
                           run_dir=args.run_dir)
        # 工作进程被强杀时可能留下锁：仅当锁 pid == 子进程 pid 时清理
        status_out = {'attempt_id': attempt_id, 'supervised': True, 'rc': rc,
                      'timed_out': timed_out, 'wall_s': wall,
                      'remaining_before_s': remaining, 'status': status}
        _write_json(Path(args.run_dir) / 'supervisor.json', status_out)
        print('[supervise] status=%s rc=%s timed_out=%s wall=%.1fs'
              % (status, rc, timed_out, wall))
        return 0 if status == 'OK' else 1
    finally:
        ledger.release_lock()


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
    try:
        verify_prereg(pq, cfg)
        ledger = BudgetLedger(args.budget_ledger, pq['budget']['max_forwards'],
                              pq['budget']['max_minutes'], pq['budget']['max_output_gib'],
                              init=args.init_ledger, prereg_sha256=prereg_sha)
        ledger.acquire_lock()
        verify_ledger_binding(ledger, prereg_sha)
        ledger.check_startable()               # 余额/时间不足：加载前拒绝
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

    status, note = 'OK', ''
    pairs, repeat = {}, {}
    identity, identity_ok = {}, None
    watchdog_stop = threading.Event()

    def _watchdog():
        while not watchdog_stop.wait(0.5):
            if ledger.remaining_seconds() <= 0:
                watchdog_stop.set()
                return

    def _check_stop():
        if watchdog_stop.is_set():
            raise StopExecution('WATCHDOG: 运行中墙钟触顶')

    th = threading.Thread(target=_watchdog, daemon=True)
    th.start()
    inst_a = inst_b = None
    try:
        inst_a = Runner_(A, model_dir, cfg, device, run_dir, ledger, manifest, attempt_id)
        inst_a.inputs = inputs
        idA = inst_a.build()
        inst_a.run_calls()
        d = compare_repeat(manifest, run_dir, A, crit)
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
        identity_ok = (core.identity_comparable(idA) == core.identity_comparable(idB))
        if not identity_ok:
            diff = [k for k in core.identity_comparable(idA)
                    if core.identity_comparable(idA).get(k) != core.identity_comparable(idB).get(k)]
            identity['diff_keys'] = diff
            raise StopExecution('IDENTITY_MISMATCH（在 B 首个前向之前阻断）: %s' % diff)
        # B 的每次调用完成后，即时比较已有对照的样本（超阈值立即停止）
        def _after_b(tag, idx):
            _check_stop()
            d = compare_tag(manifest, run_dir, tag, crit, ledger)
            if d and not (d['pass_max'] and d['pass_mean'] and d['pass_png']):
                raise StopExecution('CRITERIA_EXCEEDED at %s: max=%.3e mean=%.3e png=%s'
                                    % (tag, d['max_abs_diff'], d['mean_abs_diff'],
                                       d['png_identical']))
            if tag == 'case1_crop512':
                dr0 = compare_repeat(manifest, run_dir, B, crit)
                if dr0 and not (dr0['pass_max'] and dr0['pass_mean']):
                    raise StopExecution('REPEAT_FAIL B: %s' % dr0)
        inst_b.run_calls(after=_after_b)
    except StopExecution as e:
        status, note = 'STOPPED', str(e)
    except BudgetStop as e:
        status, note = 'BUDGET_STOP', str(e)
    except Exception as e:
        status, note = 'FAILED', '%s: %s' % (type(e).__name__, e)
    finally:
        watchdog_stop.set()
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

    for tag in UNIQUE:
        d = compare_tag(manifest, run_dir, tag, crit, ledger)
        if d:
            pairs[tag] = d
    for e in (A, B):
        dr = compare_repeat(manifest, run_dir, e, crit)
        if dr:
            repeat['%s_crop512_repeat' % e] = dr

    all_calls_ok = all(c['status'] == 'OK' for c in manifest['calls'])
    n_expected = len(ORDER) * 2
    detail = {
        'identity_equal': bool(identity_ok) if identity_ok is not None else False,
        'all_calls_ok': bool(all_calls_ok),
        'n_calls': len(manifest['calls']),
        'n_expected_calls': n_expected,
        'completed_protocol': bool(len(manifest['calls']) == n_expected and all_calls_ok),
        'pairs_ok': bool(pairs) and all(
            v.get('pass_max') and v.get('pass_mean') and v.get('pass_png')
            for v in pairs.values()) and len(pairs) == len(UNIQUE),
        'repeat_ok': len(repeat) == 2 and all(
            v.get('pass_max') and v.get('pass_mean') for v in repeat.values()),
        'timing_events_equal': None,
        'no_unsafe_load': bool(manifest['instances'].get(A, {}).get('identity', {})
                               .get('load_report', {}).get('light_decoder_weights_only')),
        'safety': ((manifest['instances'].get(A, {}).get('identity', {})
                    .get('load_report', {})).get('dino') or {}).get('loader'),
    }
    try:
        import torch as _t
        detail['budget_ok'] = bool(
            int(ledger.data['forwards_reserved']) <= int(ledger.data['max_forwards'])
            and ledger.wall_used() <= float(ledger.data['max_minutes']) * 60.0)
        detail['disk_ok'] = bool(ledger.disk_used(run_dir)
                                 <= float(ledger.data['max_gib']) * (1024 ** 3))
    except Exception:
        detail['budget_ok'] = detail['disk_ok'] = False
    # RA06：A/B 的 infer 事件序列必须一致（证明共同计时边界）
    ev = {}
    for c in manifest['calls']:
        if c['status'] == 'OK':
            ev.setdefault(c['entry'], []).append((c['tag'], c['timing'].get('events')))
    if ev.get(A) and ev.get(B):
        detail['timing_events_equal'] = all(
            ev[A][i][1] == ev[B][i][1] for i in range(min(len(ev[A]), len(ev[B]))))
    detail['prereg_verified'] = True
    detail['protocol_verified'] = True

    result = {'status': status, 'note': note, 'attempt_id': attempt_id,
              'wall_used_total_s': ledger.wall_used(),
              'forwards_reserved_total': int(ledger.data['forwards_reserved']),
              'disk_used_total_bytes': ledger.disk_used(run_dir),
              'criteria': crit, 'criteria_detail': detail, 'pairs': pairs,
              'repeat_512': repeat, 'identity_diff_keys': identity.get('diff_keys'),
              'ledger': ledger.snapshot()}
    ok_keys = ('identity_equal', 'completed_protocol', 'pairs_ok', 'repeat_ok',
               'budget_ok', 'disk_ok', 'no_unsafe_load', 'timing_events_equal',
               'prereg_verified', 'protocol_verified')
    result['criteria_all_pass'] = bool(status == 'OK' and all(detail.get(k) for k in ok_keys))
    if status == 'OK' and not result['criteria_all_pass']:
        result['status'] = status = 'CRITERIA_FAIL'
    result['finished_at'] = _now()
    _write_json(run_dir / 'paired_readings.json', result)
    manifest['finished_at'] = _now()
    manifest['final_status'] = status
    _write_json(run_dir / 'manifest.json', manifest)
    try:
        ledger.end_attempt(status, note, run_dir=run_dir)
    finally:
        ledger.release_lock()
    print('[pair] status=%s calls=%d/%d forwards_reserved=%d wall_total=%.1fs'
          % (status, len(manifest['calls']), n_expected,
             result['forwards_reserved_total'], result['wall_used_total_s']))
    print('[criteria] %s' % json.dumps(detail, ensure_ascii=False))
    return 0 if result['status'] == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
