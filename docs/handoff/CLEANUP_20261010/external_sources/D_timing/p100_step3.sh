#!/bin/bash
# P100 step3: 等权重下载完成 -> 离线验证训练核心路径 -> 启动 Flickr2K 分块下载
# 用法: bash p100_step3.sh   (轮询直到 DL_DONE, 最长 ~1h)
set -e
export PATH="$HOME/miniconda3/bin:$PATH"

echo "=== [1/4] 等待权重下载 (hf-mirror) ==="
for i in $(seq 1 120); do
  if grep -q DL_DONE "$HOME/p100/logs/dl.log" 2>/dev/null; then
    echo "DL_DONE (${i}0s)"
    break
  fi
  if [ $i -eq 120 ]; then echo "TIMEOUT waiting dl.log"; tail -5 "$HOME/p100/logs/dl.log"; exit 1; fi
  sleep 10
done

echo "=== [2/4] 权重检查 ==="
ls -la "$HOME/p100/weights/VOSR_0.5B_os/checkpoints/ema_model.safetensors" 2>/dev/null || echo MISS_EMAMODEL
ls -la "$HOME/p100/weights/torch_cache/checkpoints/alexnet-owt-7be5be79.pth" 2>/dev/null || echo MISS_ALEXNET
ls -la "$HOME/p100/weights/torch_cache/clip/RN50.pt" 2>/dev/null || echo MISS_RN50
ls -la "$HOME/p100/weights/torch_cache/checkpoints/dinov2_vitb14_pretrain.pth" 2>/dev/null || echo MISS_DINOV2
ls -d "$HOME/p100/weights/torch_cache/facebookresearch_dinov2_main" 2>/dev/null || echo MISS_DINOV2_REPO
ls -la "$HOME/p100/weights/sd21_lwdecoder.pth" 2>/dev/null || echo MISS_LWDEC
ls -la "$HOME/p100/weights/sd21base/vae/diffusion_pytorch_model.safetensors" 2>/dev/null || echo MISS_VAE
ls -la "$HOME/.cache/torch/hub/pyiqa/ckpt_koniq10k.pt" 2>/dev/null || echo MISS_CLIPIQA_CKPT

echo "=== [3/4] 训练核心路径离线验证 (set_dir 先行) ==="
python - <<'PYEOF'
import os, torch, torchvision
import pyiqa
TC = os.path.expanduser('~/p100/weights/torch_cache')
torch.hub.set_dir(TC)
print('hub_dir =', torch.hub.get_dir())
lp = pyiqa.create_metric('lpips', device='cuda'); lp.as_loss = True
cq = pyiqa.create_metric('clipiqa', device='cuda'); cq.as_loss = True
a = torch.rand(1, 3, 256, 256, device='cuda').clamp(1e-4, 1 - 1e-4)
b = torch.rand(1, 3, 256, 256, device='cuda').clamp(1e-4, 1 - 1e-4)
print('lpips loss:', float(lp(a, b)))
print('clipiqa loss:', float((1 - cq(a)).mean()))
print('OFFLINE_LOSSES_OK')
PYEOF

echo "=== [4/4] 启动 Flickr2K 分块下载 (11.6GB) ==="
if pgrep -f "dl_chunked[.]py" > /dev/null 2>&1; then
  echo F2K_ALREADY_RUNNING
else
  DL_URL="https://hf-mirror.com/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip" \
  DL_TARGET="$HOME/p100/data/Flickr2K.zip" \
  DL_TMP="$HOME/p100/data/flickr2k_chunks" \
  DL_SIZE="11645004887" \
  DL_WORKERS="8" \
  nohup python "$HOME/p100/tools/dl_chunked.py" > "$HOME/p100/logs/f2k.log" 2>&1 &
  echo F2K_STARTED_PID_$!
fi
echo STEP3_DONE
