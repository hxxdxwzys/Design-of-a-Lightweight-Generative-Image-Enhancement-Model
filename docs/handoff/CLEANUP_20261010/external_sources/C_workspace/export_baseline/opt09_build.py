# -*- coding: utf-8 -*-
"""OPT-09 导出驱动：用 v6 s5000 LoRA（scale 0.8）重导出 all-fp16 交付件（其余配方与 OPT-01 完全相同）。

用法: python opt09_build.py <out_root>
说明：通过 monkeypatch PBcore.load_config 覆盖 lora_path/lora_scale（不改冻结源码）。
"""
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))
import core as PBcore  # noqa: E402
import frozen_mode  # noqa: E402
import r_build  # noqa: E402
import r_ledger  # noqa: E402

LORA = sys.argv[2] if len(sys.argv) > 2 else r'D:\models\vosr_lora\v6\lora_s5000.pt'
SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    (run / 'model_dir').mkdir(parents=True, exist_ok=True)
    (run / 'evidence').mkdir(parents=True, exist_ok=True)
    (run / 'npy').mkdir(parents=True, exist_ok=True)
    lp = run / 'ledger_opt09.json'
    if not lp.is_file():
        olds = [Path(r'D:\models\opt07\ledger_opt07.json'),
                Path(r'D:\models\opt01\ledger_opt01.json')]
        led = {'schema': 'EXPORT01R-ledger/1',
               'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'task': 'OPT-09',
               'limits': {'top': 250, 'top4k': 110, 'component': 24, 'wall_s': 14400.0,
                          'disk_gib': 6.0},
               'prior_batch': {'old_ledgers': [
                   {'ledger': str(p), 'sha256': sha(p),
                    'used': json.loads(p.read_text(encoding='utf-8'))['used'],
                    'note': '只读链接，额度不叠加'} for p in olds],
                   'note': 'OPT-09 独立池：top<=250、4K<=110、组件<=24、监督<=4h、磁盘<=6GiB',
                   'prior_top': 0, 'prior_top4k': 0, 'prior_component': 0},
               'used': {'top': 0, 'top4k': 0, 'component': 0}, 'events': [], 'attempts': [],
               'open_attempt': None}
        lp.write_text(json.dumps(led, indent=1, ensure_ascii=False), encoding='utf-8')
    led = r_ledger.Ledger.from_file(lp)
    flags = frozen_mode.apply_frozen_mode()
    # 覆盖 LoRA（不改冻结源码，仅运行时替换配置加载）
    orig_load = PBcore.load_config

    def patched(path=None):
        cfg = orig_load(path)
        cfg['lora_path'] = LORA
        cfg['lora_scale'] = SCALE
        return cfg

    PBcore.load_config = patched
    led.acquire_lock('OPT-09 build')
    status, detail = 'FAIL', ''
    try:
        led.open_attempt('opt09_build', run, source_sha256=sha(Path(__file__)))
        info = r_build.build(run / 'model_dir', run, led, run / 'evidence', None, HERE,
                             vae_fp32=False)
        (run / 'evidence' / 'opt09_build.json').write_text(
            json.dumps({'schema': 'OPT09-build/1', 'flags': flags, 'lora': LORA,
                        'lora_scale': SCALE, 'build': info}, indent=1, ensure_ascii=False),
            encoding='utf-8')
        status = 'OK'
        print('BUILD_OK sha=%s bytes=%s bundle=%s' % (info['model_sha256'], info['model_bytes'],
                                                      info['bundle_id']), flush=True)
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        print('BUILD_FAIL %s' % detail, flush=True)
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
