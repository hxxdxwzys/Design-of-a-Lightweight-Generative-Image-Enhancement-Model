# -*- coding: utf-8 -*-
"""P100 权重下载续传 v2 (2026-09-02): 低并发 + 多次重试, 应对 hf-mirror 429 限流
用法: python ~/p100/tools/p100_dl2.py   (自动续传, 完成后打 DL_DONE)
"""
import os
import shutil
import sys
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'

from huggingface_hub import snapshot_download  # noqa: E402

W = os.path.expanduser('~/p100/weights')
CACHE = os.path.expanduser('~/.cache/torch/hub')


def try_download(repo, local_dir, allow, ignore, workers, tag):
    for attempt in range(8):
        try:
            snapshot_download(repo, local_dir=local_dir, allow_patterns=allow,
                              ignore_patterns=ignore, max_workers=workers)
            print(f'[{tag}] attempt {attempt + 1} OK', flush=True)
            return True
        except Exception as e:
            print(f'[{tag}] attempt {attempt + 1} 失败: {type(e).__name__}: {str(e)[:120]}', flush=True)
            time.sleep(45 * (attempt + 1))
    return False


def main():
    ok = True
    # 1) 续传 CSWRY/VOSR (已完成文件自动跳过)
    print('=== [1/3] 续传 CSWRY/VOSR (低并发) ===', flush=True)
    ok &= try_download('CSWRY/VOSR', os.path.join(W, 'vosr_hf'),
                       ['VOSR_0.5B_os/*', 'sd21_lwdecoder.pth', 'torch_cache/*'],
                       ['*.git/*', '*/.github/*', '**/.github/**', '**/.git/**', '*.gitignore',
                        'torch_cache/facebookresearch_dinov2_main/dinov2/train/*',
                        'torch_cache/facebookresearch_dinov2_main/dinov2/train/**'],
                       2, 'VOSR')
    # 归位
    if os.path.isdir(os.path.join(W, 'vosr_hf', 'VOSR_0.5B_os')) and not os.path.isdir(os.path.join(W, 'VOSR_0.5B_os')):
        os.rename(os.path.join(W, 'vosr_hf', 'VOSR_0.5B_os'), os.path.join(W, 'VOSR_0.5B_os'))
    if os.path.isfile(os.path.join(W, 'vosr_hf', 'sd21_lwdecoder.pth')) and not os.path.isfile(os.path.join(W, 'sd21_lwdecoder.pth')):
        shutil.move(os.path.join(W, 'vosr_hf', 'sd21_lwdecoder.pth'), os.path.join(W, 'sd21_lwdecoder.pth'))
    if os.path.isdir(os.path.join(W, 'vosr_hf', 'torch_cache')) and not os.path.isdir(os.path.join(W, 'torch_cache')):
        os.rename(os.path.join(W, 'vosr_hf', 'torch_cache'), os.path.join(W, 'torch_cache'))

    # 2) SD2.1 VAE
    print('=== [2/3] SD2.1 VAE ===', flush=True)
    ok &= try_download('stabilityai/stable-diffusion-2-1-base', os.path.join(W, 'sd21base'),
                       ['vae/*'], [], 2, 'VAE')

    # 3) pyiqa 权重
    print('=== [3/3] IQA-PyTorch-Weights ===', flush=True)
    ok &= try_download('chaofengc/IQA-PyTorch-Weights', os.path.join(CACHE, 'pyiqa'),
                       [], [], 2, 'PYIQA')

    print('=== 检查 ===', flush=True)
    for path in [os.path.join(W, 'VOSR_0.5B_os', 'checkpoints', 'ema_model.safetensors'),
                 os.path.join(W, 'sd21_lwdecoder.pth'),
                 os.path.join(W, 'sd21base', 'vae', 'diffusion_pytorch_model.safetensors'),
                 os.path.join(CACHE, 'pyiqa', 'ckpt_koniq10k.pt'),
                 os.path.join(W, 'torch_cache', 'checkpoints', 'dinov2_vitb14_pretrain.pth'),
                 os.path.join(W, 'torch_cache', 'checkpoints', 'alexnet-owt-7be5be79.pth'),
                 os.path.join(W, 'torch_cache', 'clip', 'RN50.pt'),
                 os.path.join(W, 'torch_cache', 'facebookresearch_dinov2_main', 'hubconf.py')]:
        print(('OK  ' if os.path.exists(path) else 'MISS') + ' ' + path, flush=True)
    if ok:
        print('DL_DONE', flush=True)
    else:
        print('DL_PARTIAL_FAIL', flush=True)


if __name__ == '__main__':
    main()
