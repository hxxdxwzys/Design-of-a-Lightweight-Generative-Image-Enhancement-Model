# -*- coding: utf-8 -*-
"""测试集 2x 渲染 v2 (2026-08-26): M24 配置 (v7 lora_s5000 sc0.8 seed8 wavelet) @ -u 2
分块批量: 每 5 张一个子进程 (模型加载 1min 摊薄); 暂停续跑协议:
  D:/tmp_timing/b_2x_pause.flag 存在 -> 每块后检查, 干净退出;
  D:/tmp_timing/b_2x_test_state.json 记录已完成 case 列表, --resume 续跑。
用法: D:/Python/python.exe D:/tmp_timing/b_2x_test.py [--resume] [--skip-render]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TEST_LQ = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
OUT = Path(r'D:/models/test_out_2x')        # 禁与池目录同名 (pool_add_source rmtree 坑)
CHUNK_ROOT = Path(r'D:/tmp_timing/b_2x_chunks')
STATE = Path(r'D:/tmp_timing/b_2x_test_state.json')
PAUSE = Path(r'D:/tmp_timing/b_2x_pause.flag')
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
PY = r'D:/Python/python.exe'
INFER = r'D:/tmp_timing/vosr_lora_infer.py'
CHUNK = 5

ENV = dict(os.environ)
ENV['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
ENV['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--skip-render', action='store_true')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and STATE.exists():
        done = set(json.loads(STATE.read_text(encoding='utf-8')).get('done', []))
        print(f'[resume] {len(done)} done', flush=True)

    if not args.skip_render:
        lq_files = sorted(TEST_LQ.glob('*.jpg'))
        todo = [p for p in lq_files if p.stem not in done]
        print(f'[todo] {len(todo)}/{len(lq_files)}', flush=True)
        chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
        for ci, chunk in enumerate(chunks):
            if PAUSE.exists():
                print('[pause] flag found, clean exit', flush=True)
                break
            ch = CHUNK_ROOT / f'chunk_{ci}'
            if ch.exists():
                shutil.rmtree(ch)
            ch.mkdir(parents=True, exist_ok=True)
            for p in chunk:
                shutil.copy2(p, ch / p.name)
            cmd = [PY, INFER, '-c', CKPT, '--lora', LORA, '--lora-scale', '0.8',
                   '-i', str(ch), '-o', str(OUT), '-u', '2', '--seed', '8',
                   '--align_method', 'wavelet', '--tile_size', '512',
                   '--vae_tile_size', '1024', '--vae_tile_overlap', '64']
            t0 = time.time()
            r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
            ok = all((OUT / (p.stem + '.png')).exists() for p in chunk)
            if r.returncode != 0 or not ok:
                print(f'[FAIL] chunk {ci}: rc={r.returncode}', flush=True)
                Path(r'D:/tmp_timing/b_2x_test_err.log').write_text(
                    r.stdout[-5000:] + '\n---ERR---\n' + r.stderr[-5000:], encoding='utf-8')
                sys.exit(1)
            for p in chunk:
                done.add(p.stem)
            STATE.write_text(json.dumps({'done': sorted(done)}, ensure_ascii=False), encoding='utf-8')
            shutil.rmtree(ch, ignore_errors=True)
            print(f'[chunk {ci + 1}/{len(chunks)}] {(time.time() - t0) / 60:.1f}min done={len(done)}', flush=True)

    print(f'[done] {len(done)} images in {OUT}', flush=True)


if __name__ == '__main__':
    main()
