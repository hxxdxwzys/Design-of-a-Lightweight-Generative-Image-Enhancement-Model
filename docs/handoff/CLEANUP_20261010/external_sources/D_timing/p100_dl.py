# -*- coding: utf-8 -*-
"""P100 服务器侧权重/缓存下载 (2026-09-02): 全部走 hf-mirror, 比本地上传快一个数量级
用法 (miniconda base python): python ~/p100/tools/p100_dl.py
"""
import os
import shutil
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download  # noqa: E402

W = os.path.expanduser('~/p100/weights')
CACHE = os.path.expanduser('~/.cache/torch/hub')


def main():
    # 1) VOSR 0.5B + sd21_lwdecoder + torch_cache (HF 官方仓 CSWRY/VOSR)
    print('=== [1/4] CSWRY/VOSR (0.5B + lwdecoder + torch_cache) ===', flush=True)
    p = snapshot_download('CSWRY/VOSR', local_dir=os.path.join(W, 'vosr_hf'),
                          allow_patterns=['VOSR_0.5B_os/*', 'sd21_lwdecoder.pth', 'torch_cache/*'],
                          ignore_patterns=['*.git/*', '*/.github/*', '**/.github/**', '**/.git/**', '*.gitignore'],
                          max_workers=8)
    if os.path.isdir(os.path.join(W, 'vosr_hf', 'VOSR_0.5B_os')) and not os.path.isdir(os.path.join(W, 'VOSR_0.5B_os')):
        os.rename(os.path.join(W, 'vosr_hf', 'VOSR_0.5B_os'), os.path.join(W, 'VOSR_0.5B_os'))
    if os.path.isfile(os.path.join(W, 'vosr_hf', 'sd21_lwdecoder.pth')) and not os.path.isfile(os.path.join(W, 'sd21_lwdecoder.pth')):
        shutil.move(os.path.join(W, 'vosr_hf', 'sd21_lwdecoder.pth'), os.path.join(W, 'sd21_lwdecoder.pth'))
    if os.path.isdir(os.path.join(W, 'vosr_hf', 'torch_cache')) and not os.path.isdir(os.path.join(W, 'torch_cache')):
        os.rename(os.path.join(W, 'vosr_hf', 'torch_cache'), os.path.join(W, 'torch_cache'))

    # 2) SD2.1 VAE
    print('=== [2/4] SD2.1 VAE ===', flush=True)
    snapshot_download('stabilityai/stable-diffusion-2-1-base',
                      local_dir=os.path.join(W, 'sd21base'),
                      allow_patterns=['vae/*'], max_workers=8)

    # 3) pyiqa 权重 -> ~/.cache/torch/hub/pyiqa
    print('=== [3/4] IQA-PyTorch-Weights (pyiqa) ===', flush=True)
    snapshot_download('chaofengc/IQA-PyTorch-Weights',
                      local_dir=os.path.join(CACHE, 'pyiqa'), max_workers=8)

    # 4) dinov2 离线仓库目录兜底: 若 torch_cache 里没有 facebookresearch_dinov2_main,
    #    从 GitHub 拉不到时改用本地上传 (见 p100_upload.py); 这里只检查
    print('=== [4/4] 检查 ===', flush=True)
    for path in [os.path.join(W, 'VOSR_0.5B_os', 'checkpoints', 'ema_model.safetensors'),
                 os.path.join(W, 'sd21_lwdecoder.pth'),
                 os.path.join(W, 'sd21base', 'vae', 'diffusion_pytorch_model.safetensors'),
                 os.path.join(CACHE, 'pyiqa', 'ckpt_koniq10k.pt'),
                 os.path.join(W, 'torch_cache', 'checkpoints', 'dinov2_vitb14_pretrain.pth')]:
        print(('OK ' if os.path.exists(path) else 'MISS ') + path, flush=True)
    print('DL_DONE', flush=True)


if __name__ == '__main__':
    main()
