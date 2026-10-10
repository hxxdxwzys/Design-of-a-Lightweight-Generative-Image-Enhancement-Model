# -*- coding: utf-8 -*-
"""CPU 故障注入替身：完整走通锁/attempt/读数/产物（满足 EXP-R10 必需集合），不加载模型。"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import r_ledger  # noqa: E402

run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
exp_dir.mkdir(parents=True, exist_ok=True)
led = r_ledger.Ledger(ledger_path)
led.acquire_lock('fake-ok')

ZERO = {'max_abs': 0.0, 'mean_abs': 0.0, 'png_pixel_identical': True}


def _npy(name):
    p = exp_dir / 'npy' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.zeros((1, 3, 4, 4), dtype=np.float32))
    return {'npy': str(p), 'npy_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


try:
    led.open_attempt('attempt_01', run_dir, prereg_sha256='x', source_sha256='y', model_sha256='z')
    led.reserve('top', 'fake_512_1')
    led.reserve('top', 'fake_4k_1', is_4k=True)
    led.reserve('component', 'fake_probe')
    rd = {'schema': 'EXPORT01R-readings/1', 'status': 'OK', 'phaseA': {}, 'phaseB': {},
          'pairs': [], 'migration': []}
    rd['phaseA']['released'] = True
    rd['phaseA']['call_4'] = dict({'call': 'A_511x513'}, **_npy('A_511x513.npy'))
    for i in (1, 2, 3):
        ar = {'call': 'A_512_%d' % i}
        if i > 1:
            ar['vs_first_A512'] = dict(ZERO)
        ar.update(_npy('A_512_r%d.npy' % i))
        rd['phaseA']['call_%d' % i] = ar
    rd['order'] = (['A_512_1', 'A_512_2', 'A_512_3', 'A_511x513',
                    'B_512_1', 'B_512_2', 'B_512_3', 'B_511x513',
                    'case1_full_A', 'case1_full_B', 'case7_full_A', 'case7_full_B'])
    for i in (1, 2, 3):
        rec = {'call': 'B_512_%d' % i, 'vs_A': dict(ZERO), 'vs_first_B512': dict(ZERO)}
        rec.update(_npy('B_512_r%d.npy' % i))
        rd['phaseB']['call_%d' % i] = rec
    rd['phaseB']['call_4'] = dict({'call': 'B_511x513', 'vs_A': dict(ZERO)},
                                  **_npy('B_511x513.npy'))
    for n in ('case1_full', 'case7_full'):
        rd['pairs'].append({'name': n, 'pair': dict(ZERO)})
    rd['migration'] = [{'name': 'case1_crop512', 'max_abs': 0.01, 'mean_abs': 0.001}]
    (exp_dir / 'readings.json').write_text(json.dumps(rd), encoding='utf-8')
    led.close_attempt('OK', 'fake')
finally:
    led.release_lock()
sys.exit(0)
