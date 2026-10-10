# -*- coding: utf-8 -*-
"""EXPORT-01R 总账与排他锁（先扣后跑、失败不退、跨进程累计、硬墙钟）。

设计沿用已验收的 pipeline_baseline/pair_verify 语义，最小实现：
  - 所有计数在**调用之前**持久化（reserve），没有任何退款路径；
  - 跨进程：worker 每次 reserve 都从磁盘 reload 后再写，防止旧内存覆盖；
  - 排他锁：O_EXCL 创建，含 pid/启动时间；陈旧锁不自动接管，直接 fail-closed；
  - 绝对 deadline：attempt 开始时刻 + 墙钟上限，超限即抛 BudgetExhausted。

用法（库）：Ledger(path).open_attempt(...) / reserve(...) / close_attempt(...)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

TOP_LIMIT = 12
TOP4K_LIMIT = 4
COMPONENT_LIMIT = 48
WALL_LIMIT_S = 1800.0
DISK_LIMIT_GIB = 8.0


class BudgetExhausted(RuntimeError):
    pass


class LockHeld(RuntimeError):
    pass


def _limits_from_disk(path):
    """DELIV-02 补充 2/54：唯一 ledger 绝对路径 + 全字段；缺文件/缺字段一律拒绝（不回落默认）。"""
    p = Path(path).resolve()
    if not p.is_file():
        raise BudgetExhausted("唯一持久总账缺失：%s（禁止回落默认额度）" % p)
    d = json.loads(p.read_text(encoding="utf-8"))
    need = ("top", "top4k", "component", "wall_s", "disk_gib")
    lim = d.get("limits") or {}
    miss = [k for k in need if k not in lim]
    if miss:
        raise BudgetExhausted("总账缺字段 %s：%s" % (miss, p))
    prior = d.get("prior_batch")
    if prior is None:
        raise BudgetExhausted("总账缺 prior_batch 历史基数：%s" % p)
    for k in ("prior_top", "prior_top4k", "prior_component"):
        if k not in prior:
            raise BudgetExhausted("历史基数缺字段 %s：%s" % (k, p))
    return p, d, lim, prior


def _pid_alive(pid: int) -> bool:
    """进程存活探测（Windows 安全）。

    重要语义陷阱：os.kill(pid, 0) 在 Windows 上不是 POSIX 的"探测"——CPython 会走
    OpenProcess + TerminateProcess(handle, sig)，sig=0 时**静默终止目标进程**，对自身 pid
    调用会直接杀掉自己（无 traceback、退出码 0）。因此 Windows 改用 OpenProcess(
    QUERY_LIMITED_INFORMATION) + GetExitCodeProcess 判定，绝不调用 os.kill。
    """
    pid = int(pid)
    if pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k = ctypes.windll.kernel32
        h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not k.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


class Ledger:
    @classmethod
    def from_file(cls, path):
        """DELIV-02 补充：严格从唯一持久总账构造（缺文件/缺字段/缺历史基数即拒绝，不回落默认）。"""
        p, d, lim, prior = _limits_from_disk(path)
        obj = cls(p, top=lim['top'], top4k=lim['top4k'], component=lim['component'],
                  wall_s=lim['wall_s'], disk_gib=lim['disk_gib'], prior=prior)
        obj.base_used = dict(d.get('used') or {})
        obj.supervision_used_s = float(d.get('supervision_used_s') or 0.0)
        return obj

    def __init__(self, path, top=TOP_LIMIT, top4k=TOP4K_LIMIT, component=COMPONENT_LIMIT,
                 wall_s=WALL_LIMIT_S, disk_gib=DISK_LIMIT_GIB, prior=None):
        self.path = Path(path)
        self.limits = {'top': int(top), 'top4k': int(top4k), 'component': int(component),
                       'wall_s': float(wall_s), 'disk_gib': float(disk_gib)}
        self.prior = dict(prior) if prior else None   # 旧批次关联（哈希/扣款），不改旧账
        # DELIV-02：累计硬上限 = 本次额度 + 历史消耗（历史只增不减，不冲销旧账）
        self.hard_caps = {'top': int(top) + int((prior or {}).get('prior_top', 0)),
                          'top4k': int(top4k) + int((prior or {}).get('prior_top4k', 0)),
                          'component': int(component) + int((prior or {}).get('prior_component', 0))}
        self.lock_path = Path(str(self.path) + '.lock')

    # ── 持久化 ──
    def _read(self):
        if not self.path.is_file():
            return {'schema': 'EXPORT01R-ledger/1', 'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'limits': dict(self.limits), 'prior_batch': self.prior,
                    'used': {'top': 0, 'top4k': 0, 'component': 0},
                    'events': [], 'attempts': [], 'open_attempt': None}
        return json.loads(self.path.read_text(encoding='utf-8'))

    def _write(self, d):
        tmp = Path(str(self.path) + '.tmp')
        tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding='utf-8')
        os.replace(str(tmp), str(self.path))

    # ── 锁 ──
    def acquire_lock(self, label=''):
        try:
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info = {}
            try:
                info = json.loads(self.lock_path.read_text(encoding='utf-8'))
            except Exception:
                pass
            pid = int(info.get('pid') or -1)
            raise LockHeld('锁已存在（不自动接管）：%s pid=%s alive=%s'
                           % (self.lock_path, pid, _pid_alive(pid) if pid > 0 else 'UNKNOWN'))
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({'pid': os.getpid(), 'label': label,
                       'at': time.strftime('%Y-%m-%d %H:%M:%S')}, f)
        return self.lock_path

    def release_lock(self):
        if self.lock_path.is_file():
            self.lock_path.unlink()

    def lock_owner(self):
        if not self.lock_path.is_file():
            return None
        try:
            return json.loads(self.lock_path.read_text(encoding='utf-8'))
        except Exception:
            return {'pid': None}

    # ── attempt ──
    def open_attempt(self, attempt_id, run_dir, prereg_sha256=None, source_sha256=None,
                     model_sha256=None):
        d = self._read()
        if d.get('open_attempt'):
            raise BudgetExhausted('已有未关闭 attempt：%s' % d['open_attempt'])
        # DELIV-02：同一持久总账下允许不同 ID 的**阶段** attempt（A/B、C 等），
        # 但同一 ID 不得重复；额度仍由统一计数与硬上限约束。
        if any(a.get('attempt_id') == attempt_id for a in d['attempts']):
            raise BudgetExhausted('attempt ID 重复：%s' % attempt_id)
        d['open_attempt'] = {'attempt_id': attempt_id, 'run_dir': str(run_dir),
                             'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                             'started_monotonic': time.monotonic(),
                             'prereg_sha256': prereg_sha256,
                             'source_sha256': source_sha256, 'model_sha256': model_sha256,
                             'status': 'RUNNING'}
        d['attempts'].append(dict(d['open_attempt']))
        self._write(d)
        return d['open_attempt']

    def _deadline_ok(self, d):
        """DELIV-02 补充：deadline **跨 A/B/C 累计**（不因新 attempt 重新起算）。"""
        att = d.get('open_attempt') or {}
        if not att:
            return False
        wall = float((d.get('limits') or {}).get('wall_s', self.limits['wall_s']))
        return self._elapsed_from(d) <= wall

    def _elapsed_from(self, d):
        used = float(d.get('supervision_used_s') or 0.0)
        att = d.get('open_attempt') or {}
        if att:
            used += time.monotonic() - float(att['started_monotonic'])
        return used

    def elapsed_s(self):
        return self._elapsed_from(self._read())

    def append_allowance_change(self, new_limits, basis):
        """在**原总账**就地追加额度变更事件（保留历史 used/快照，不另开空账本）。"""
        d = self._read()
        old = dict(d.get('limits') or {})
        d.setdefault('allowance_changes', []).append(
            {'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'basis': basis,
             'before': old, 'after': dict(new_limits), 'used_at_change': dict(d.get('used') or {})})
        d['limits'] = {**old, **{k: new_limits[k] for k in new_limits}}
        self.limits = dict(d['limits'])
        self._write(d)
        return d['limits']

    def check_hard_caps(self):
        """DELIV-02：累计上限校验（本次已扣 + 历史消耗 ≤ 硬上限）。"""
        d = self._read()
        prior = (d.get('prior_batch') or {})
        hist = {'top': int(prior.get('prior_top', 0) or 0),
                'top4k': int(prior.get('prior_top4k', 0) or 0),
                'component': int(prior.get('prior_component', 0) or 0)}
        # DELIV-02：以**磁盘**的 limits/prior 为准（唯一持久总账），对象内值必须与磁盘一致
        limits = d.get('limits') or self.limits
        # DELIV-02 补充 2/54：比较**全部**额度字段（不只 top/component）
        for k in ('top', 'top4k', 'component'):
            if int(limits[k]) != int(self.limits[k]):
                raise BudgetExhausted('ledger 磁盘额度与对象额度不一致(%s)：disk=%s obj=%s'
                                      % (k, limits, self.limits))
        if abs(float(limits['wall_s']) - float(self.limits['wall_s'])) > 1e-6 or \
                abs(float(limits['disk_gib']) - float(self.limits['disk_gib'])) > 1e-6:
            raise BudgetExhausted('ledger wall/disk 额度不一致：disk=%s obj=%s'
                                  % (limits, self.limits))
        caps = {'top': int(limits['top']) + hist['top'],
                'top4k': int(limits['top4k']) + hist['top4k'],
                'component': int(limits['component']) + hist['component']}
        for k in ('top', 'top4k', 'component'):
            if d['used'][k] + hist[k] > caps[k]:
                raise BudgetExhausted('%s 累计超限：本次 %d + 历史 %d > 硬上限 %d'
                                      % (k, d['used'][k], hist[k], caps[k]))
        return True

    def reserve(self, kind, where, is_4k=False):
        """先扣后跑；任何失败都不退款。超限（含累计硬上限、重复 call ID）抛 BudgetExhausted。"""
        if kind not in ('top', 'component'):
            raise ValueError(kind)
        d = self._read()
        if not d.get('open_attempt'):
            raise BudgetExhausted('没有打开的 attempt，禁止调用')
        if not self._deadline_ok(d):
            raise BudgetExhausted('墙钟已超 %.0fs 上限（当前 %.1fs）'
                                  % (self.limits['wall_s'], self.elapsed_s()))
        # 只允许本次获准续跑的 revision 前缀；前缀仅标识授权续跑，不是无限重试
        prefix = os.environ.get('DELIV02_CALL_PREFIX', '')
        if prefix:
            where = prefix + where
        if any(e.get('where') == where and e.get('kind') == kind for e in d['events']):
            raise BudgetExhausted('重复调用 ID：%s（禁止双层预留/重复记账）' % where)
        self.check_hard_caps()
        used = d['used']
        if kind == 'top':
            if used['top'] + 1 > self.limits['top']:
                raise BudgetExhausted('顶层 infer 超限 %d' % self.limits['top'])
            if is_4k and used['top4k'] + 1 > self.limits['top4k']:
                raise BudgetExhausted('4K infer 超限 %d' % self.limits['top4k'])
        else:
            if used['component'] + 1 > self.limits['component']:
                raise BudgetExhausted('组件前向超限 %d' % self.limits['component'])
        used[kind] += 1
        if kind == 'top' and is_4k:
            used['top4k'] += 1
        d['events'].append({'seq': len(d['events']) + 1, 'kind': kind, 'is_4k': bool(is_4k),
                            'where': where, 'pid': os.getpid(),
                            'at': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'monotonic': time.monotonic(),
                            'after': dict(used)})
        self._write(d)
        return dict(used)

    def note(self, where, detail=None):
        d = self._read()
        d['events'].append({'seq': len(d['events']) + 1, 'kind': 'note', 'where': where,
                            'detail': detail, 'pid': os.getpid(),
                            'at': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'monotonic': time.monotonic(),
                            'after': dict(d['used'])})
        self._write(d)

    def close_attempt(self, status, note=''):
        d = self._read()
        att = d.get('open_attempt')
        if not att:
            return None
        att = dict(att)
        att.update({'status': status, 'note': note,
                    'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'elapsed_s': round(time.monotonic() - float(att['started_monotonic']), 3)})
        # 累计监督时间（跨 attempt 不可退款）
        d['supervision_used_s'] = round(self._elapsed_from(d), 3)
        d['attempts'][-1] = att
        d['open_attempt'] = None
        self._write(d)
        return att

    def summary(self):
        d = self._read()
        return {'limits': d['limits'], 'used': d['used'],
                'n_events': len(d['events']), 'open_attempt': d.get('open_attempt'),
                'attempts': d.get('attempts', [])}
