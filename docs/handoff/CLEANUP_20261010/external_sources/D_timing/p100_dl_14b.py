# -*- coding: utf-8 -*-
"""服务器下载 VOSR_1.4B_os + Qwen VAE (2026-09-03): hf-mirror + ModelScope 双源
1.4B 权重 ~6GB (CSWRY/VOSR 的 VOSR_1.4B_os/*), Qwen-Image-vae-2d ~1.5GB (ModelScope)
"""
import os
import sys
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download  # noqa: E402
import urllib.request  # noqa: E402

W = os.path.expanduser('~/p100/weights')


def try_dl(repo, local_dir, allow, tag):
    for attempt in range(8):
        try:
            snapshot_download(repo, local_dir=local_dir, allow_patterns=allow, max_workers=4)
            print(f'[{tag}] OK', flush=True)
            return True
        except Exception as e:
            print(f'[{tag}] attempt {attempt + 1} 失败: {type(e).__name__}: {str(e)[:120]}', flush=True)
            time.sleep(45 * (attempt + 1))
    return False


def ms_dl(path, dst):
    """ModelScope API 直下 (Range 续传)"""
    import requests
    url = f'https://www.modelscope.cn/api/v1/models/LULALULALU/VOSR_CKPT/repo?FilePath={path}&Revision=master'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.part'
    start = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {'User-Agent': 'Mozilla/5.0'}
    if start:
        headers['Range'] = f'bytes={start}-'
    for attempt in range(10):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=120)
            if r.status_code in (200, 206):
                mode = 'ab' if start else 'wb'
                with open(tmp, mode) as f:
                    for c in r.iter_content(1 << 20):
                        f.write(c)
                os.replace(tmp, dst)
                print(f'[ms] {path}: {os.path.getsize(dst) / 1e6:.1f} MB', flush=True)
                return
            else:
                print(f'[ms] {path}: http {r.status_code}', flush=True)
        except Exception as e:
            print(f'[ms] {path}: {e}', flush=True)
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f'ms fail {path}')


def main():
    # 1) hf-mirror: VOSR_1.4B_os
    ok = try_dl('CSWRY/VOSR', os.path.join(W, 'vosr_hf'),
                ['VOSR_1.4B_os/*'],
                'VOSR_1.4B')
    src = os.path.join(W, 'vosr_hf', 'VOSR_1.4B_os')
    dst = os.path.join(W, 'VOSR_1.4B_os')
    if ok and os.path.isdir(src) and not os.path.isdir(dst):
        os.rename(src, dst)
    # 2) ModelScope: Qwen-Image-vae-2d
    for f in ['Qwen-Image-vae-2d/config.json',
              'Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors']:
        ms_dl(f, os.path.join(W, f))
    print('=== 检查 ===', flush=True)
    for p in [os.path.join(W, 'VOSR_1.4B_os', 'checkpoints', 'ema_model.safetensors'),
              os.path.join(W, 'VOSR_1.4B_os', 'args.json'),
              os.path.join(W, 'Qwen-Image-vae-2d', 'config.json'),
              os.path.join(W, 'Qwen-Image-vae-2d', 'diffusion_pytorch_model.safetensors')]:
        print(('OK  ' if os.path.exists(p) else 'MISS') + ' ' + p, flush=True)
    print('DL14_DONE', flush=True)


if __name__ == '__main__':
    main()
