# -*- coding: utf-8 -*-
"""PIPE-01 阶段B 成对验证驱动（固定样本、固定顺序、硬预算；不做任何候选选择）。

协议（任务单 §6）：
  * 固定样本：case1 原图、case7 原图、case1 中心 512x512 裁切（记录左上角，不 resize）。
  * 两个**独立初始化**的实例先后加载（前一个释放后再加载下一个）：
      实例A = tensor 入口（runner.Runner）
      实例B = 离线文件入口（offline.process_image）
  * 每个实例按 [crop512, case1, case7, crop512] 顺序各执行一次 => 8 次真实前向；
    允许最多 2 次预登记 512 预热；总计 <= --max-forwards。
  * 硬预算：前向次数、GPU 墙钟、新增输出体积；触顶立即停止并记录，不换配方、不降精度。

用法:
  D:/Python/python.exe pair_verify.py --run-dir <dir> --prereg <dir>/prereg.json \
      --case1 <case1.jpg> --case7 <case7.jpg>
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import core
import offline
from runner import Runner

ORDER = ('case1_crop512', 'case1_full', 'case7_full', 'case1_crop512')
UNIQUE = ('case1_crop512', 'case1_full', 'case7_full')


def _cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def build_inputs(case1, case7, crop, in_dir):
    """构造固定输入：写一份中心裁切 PNG（文件入口要用文件），并给出对应张量。"""
    from PIL import Image
    in_dir.mkdir(parents=True, exist_ok=True)
    im1 = Image.open(str(case1)).convert('RGB')
    im7 = Image.open(str(case7)).convert('RGB')
    w, h = im1.size
    left, top = (w - crop) // 2, (h - crop) // 2
    box = (left, top, left + crop, top + crop)
    crop_path = in_dir / ('case1_crop%d.png' % crop)
    im1.crop(box).save(crop_path)          # PNG 无损：文件入口读回像素与张量一致
    items = {
        'case1_crop512': {'path': crop_path, 'crop_box': list(box),
                          'src_wh': [w, h], 'tensor': core.pil_to_tensor(Image.open(crop_path).convert('RGB'))},
        'case1_full': {'path': Path(case1), 'crop_box': None,
                       'src_wh': [w, h], 'tensor': core.pil_to_tensor(im1)},
        'case7_full': {'path': Path(case7), 'crop_box': None,
                       'src_wh': list(im7.size), 'tensor': core.pil_to_tensor(im7)},
    }
    import hashlib
    for k, v in items.items():
        t = v['tensor']
        v['tensor_shape'] = list(t.shape)
        v['tensor_sha256'] = hashlib.sha256(
            t.detach().float().cpu().numpy().tobytes()).hexdigest()
    return items


def _diff(a, b):
    import torch
    if tuple(a.shape) != tuple(b.shape):
        return {'shape_mismatch': [list(a.shape), list(b.shape)]}
    d = (a.float() - b.float()).abs()
    return {'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()),
            'n': int(d.numel())}


def _reset_peak():
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_vram():
    import torch
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None


def run_tensor_instance(model_dir, inputs, out_dir, ledger, budget, device='cuda'):
    """实例A：tensor 入口。"""
    import numpy as np
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    r = Runner(model_dir)
    _reset_peak()
    t_load0 = time.time()
    r._ensure()
    ledger.instances['A'].append({'event': 'load', 'wall_s': time.time() - t_load0})
    for tag in ORDER:
        budget.check()
        x = inputs[tag]['tensor'].to(device)
        _cuda_sync()
        t0 = time.time()
        with torch.no_grad():
            out = r.infer(x, prompt='')
        _cuda_sync()
        dt = time.time() - t0
        budget.tick('A', tag, dt)
        arr = out.detach().float().cpu().numpy()
        np.save(out_dir / ('%s.npy' % tag), arr)
        core.tensor_to_pil(out).save(out_dir / ('%s.png' % tag))
        ledger.note_repeat('A', tag, arr, ledger.crit)
        ledger.records.append({'instance': 'A', 'tag': tag, 'wall_s': dt,
                               'shape': list(out.shape), 'min': float(out.min()),
                               'max': float(out.max()), 'finite': bool(torch.isfinite(out).all()),
                               'tensor_npy': str(out_dir / ('%s.npy' % tag)),
                               'png': str(out_dir / ('%s.png' % tag))})
    ledger.instances['A_identity'] = r.identity()
    ledger.instances['A_peak_vram_bytes'] = _peak_vram()
    r.release()
    del r
    gc.collect()
    return ledger


def run_offline_instance(cfg, inputs, out_dir, tensor_dir, ledger, budget, device='cuda'):
    """实例B：离线文件入口（独立初始化，A 已释放）。"""
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    _reset_peak()
    t0 = time.time()
    assets = core.Assets(cfg, device)
    ledger.instances['B'].append({'event': 'load', 'wall_s': time.time() - t0})
    for tag in ORDER:
        budget.check()
        _cuda_sync()
        t0 = time.time()
        rec = offline.process_image(assets, inputs[tag]['path'], out_dir, tensor_dir)
        _cuda_sync()
        dt = time.time() - t0
        budget.tick('B', tag, dt)
        import numpy as _np
        ledger.note_repeat('B', tag, _np.load(rec['tensor_npy']), ledger.crit)
        rec['instance'] = 'B'
        rec['tag'] = tag
        rec['wall_s'] = dt
        ledger.records.append(rec)
    ledger.instances['B_peak_vram_bytes'] = _peak_vram()
    return ledger


class Budget:
    def __init__(self, max_forwards, max_minutes, max_gib, run_dir):
        self.max_forwards = int(max_forwards)
        self.max_seconds = float(max_minutes) * 60.0
        self.max_bytes = float(max_gib) * (1024 ** 3)
        self.t0 = time.time()
        self.forwards = 0
        self.run_dir = Path(run_dir)
        self.stopped_reason = None

    def tick(self, inst, tag, dt):
        self.forwards += 1
        if self.forwards > self.max_forwards:
            self.stopped_reason = 'forward 次数触顶 %d' % self.max_forwards
            raise SystemExit('BUDGET_STOP: %s' % self.stopped_reason)

    def check(self):
        if time.time() - self.t0 > self.max_seconds:
            self.stopped_reason = 'GPU 墙钟触顶 %.0f 分钟' % (self.max_seconds / 60)
            raise SystemExit('BUDGET_STOP: %s' % self.stopped_reason)
        if self.run_dir.exists():
            used = sum(p.stat().st_size for p in self.run_dir.rglob('*') if p.is_file())
            if used > self.max_bytes:
                self.stopped_reason = '输出体积触顶 %.2f GiB' % (self.max_bytes / 1024 ** 3)
                raise SystemExit('BUDGET_STOP: %s' % self.stopped_reason)

    def elapsed(self):
        return time.time() - self.t0


class Ledger:
    def __init__(self):
        self.records = []
        self.instances = {'A': [], 'B': []}
        self.repeat = {}      # 同一实例内 crop512 的两次结果比较（局部 seed 顺序无关性）

    def note_repeat(self, inst, tag, arr, crit):
        """crop512 在顺序中出现两次：比较两次输出，验证局部随机协议不随顺序漂移。"""
        import numpy as np
        if tag != 'case1_crop512':
            return
        key = '%s_crop512_run' % inst
        if key not in self.repeat:
            self.repeat[key] = {'first_sha256': core.sha256_bytes(arr.tobytes())}
            self._pending = getattr(self, '_pending', {})
            self._pending[inst] = np.array(arr, copy=True)
            return
        prev = self._pending[inst]
        d = _diff(__import__('torch').from_numpy(prev), __import__('torch').from_numpy(arr))
        d['second_sha256'] = core.sha256_bytes(arr.tobytes())
        d['pass_max'] = d.get('max_abs_diff', 9e9) <= crit['max_abs_diff']
        d['pass_mean'] = d.get('mean_abs_diff', 9e9) <= crit['mean_abs_diff']
        self.repeat[key].update(d)


def main(argv=None):
    ap = argparse.ArgumentParser(description='PIPE-01 阶段B 成对验证驱动')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--prereg', required=True, help='预登记 manifest（不存在则拒绝执行）')
    ap.add_argument('--case1', required=True)
    ap.add_argument('--case7', required=True)
    ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--model-dir', default=None, help='tensor 入口的 model_dir（默认用基座目录）')
    ap.add_argument('--device', default=None, help='默认 cuda（可用时）；CPU 起搏测试用 cpu')
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    pre = Path(args.prereg)
    if not pre.is_file():
        raise SystemExit('拒绝执行：缺少预登记 manifest %s（不得先跑后登记）' % pre)
    pq = json.loads(pre.read_text(encoding='utf-8'))
    crit = pq['criteria']
    bud = pq['budget']
    cfg = core.load_config()
    run_dir.mkdir(parents=True, exist_ok=True)

    inputs = build_inputs(args.case1, args.case7, args.crop, run_dir / 'inputs')
    model_dir = args.model_dir or cfg['base_dir']
    budget = Budget(bud['max_forwards'], bud['max_minutes'], bud['max_output_gib'], run_dir)
    ledger = Ledger()
    ledger.crit = crit
    ledger.instances['prereg_sha256'] = core.sha256_file(pre)

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ledger.instances['device'] = device
    status = 'OK'
    try:
        run_tensor_instance(model_dir, inputs, run_dir / 'A_tensor', ledger, budget,
                            device=device)
        run_offline_instance(cfg, inputs, run_dir / 'B_offline',
                             run_dir / 'B_offline_tensor', ledger, budget, device=device)
    except SystemExit as e:
        status = 'BUDGET_STOP'
        ledger.instances['stop'] = str(e)
    except Exception as e:
        status = 'FAILED'
        ledger.instances['error'] = '%s: %s' % (type(e).__name__, e)

    result = {'status': status, 'elapsed_s': budget.elapsed(), 'forwards': budget.forwards,
              'criteria': crit, 'budget': bud, 'inputs': {
                  k: {kk: vv for kk, vv in v.items() if kk != 'tensor'} for k, v in inputs.items()}}
    if status == 'OK':
        import numpy as np
        import torch
        from PIL import Image
        recs = {}
        for r in ledger.records:
            if r.get('instance') == 'B':
                recs[r['tag']] = r
        pairs = {}
        for tag in UNIQUE:
            a_rec = [r for r in ledger.records if r.get('instance') == 'A' and r['tag'] == tag][0]
            a = torch.from_numpy(np.load(a_rec['tensor_npy']))
            b = torch.from_numpy(np.load(recs[tag]['tensor_npy']))
            d = _diff(a, b)
            d['pass_max'] = d.get('max_abs_diff', 9e9) <= crit['max_abs_diff']
            d['pass_mean'] = d.get('mean_abs_diff', 9e9) <= crit['mean_abs_diff']
            aa = np.array(Image.open(a_rec['png']).convert('RGB'))
            bb = np.array(Image.open(recs[tag]['png_path']).convert('RGB'))
            d['png_identical'] = bool(np.array_equal(aa, bb))
            d['png_shape'] = list(aa.shape)
            d['a_png_sha256'] = core.sha256_file(a_rec['png'])
            d['b_png_sha256'] = recs[tag]['png_sha256']
            d['pass_png'] = bool(d['png_identical'])
            pairs[tag] = d
        result['pairs'] = pairs
        result['repeat_512'] = ledger.repeat
        all_pass = all(v.get('pass_max') and v.get('pass_mean') and v.get('pass_png')
                       for v in pairs.values()) and all(
            v.get('pass_max') and v.get('pass_mean') for v in ledger.repeat.values())
        result['criteria_all_pass'] = bool(all_pass)
        if not all_pass:
            status = 'CRITERIA_FAIL'
    result['crit'] = crit
    result['status'] = status          # 判据失败会把 status 改为 CRITERIA_FAIL，这里同步
    result['records'] = ledger.records
    result['instances'] = ledger.instances
    (run_dir / 'paired_readings.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('[pair] status=%s forwards=%d elapsed=%.1fs' % (status, budget.forwards,
                                                          budget.elapsed()))
    if status == 'OK':
        for k, v in result['pairs'].items():
            print('[pair] %-14s max=%.3e mean=%.3e png_identical=%s pass=%s' % (
                k, v['max_abs_diff'], v['mean_abs_diff'], v['png_identical'],
                v['pass_max'] and v['pass_mean'] and v['pass_png']))
        if result.get('repeat_512'):
            for k, v in result['repeat_512'].items():
                print('[repeat] %-20s max=%.3e mean=%.3e pass=%s' % (
                    k, v.get('max_abs_diff', -1), v.get('mean_abs_diff', -1),
                    v.get('pass_max') and v.get('pass_mean')))
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
