#!/bin/bash
# P100 权重齐备检查 (2026-09-02): 全部文件就位后 touch .weights_ready 标记
# 用法: bash p100_ready.sh
set -e
P="$HOME/p100"
W="$P/weights"
MISS=0
check() { # 路径 期望大小(可选)
  if [ -f "$1" ]; then
    if [ -n "$2" ] && [ "$(stat -c %s "$1")" != "$2" ]; then
      echo "SIZE_MISMATCH $1 ($(stat -c %s "$1") != $2)"; MISS=1
    else
      echo "OK  $1"; fi
  else
    echo "MISS $1"; MISS=1
  fi
}
check "$W/VOSR_0.5B_os/checkpoints/ema_model.safetensors" 1948337520
check "$W/VOSR_0.5B_os/args.json"
check "$W/sd21_lwdecoder.pth"
check "$W/sd21base/vae/config.json"
check "$W/sd21base/vae/diffusion_pytorch_model.safetensors" 334643276
check "$W/torch_cache/checkpoints/dinov2_vitb14_pretrain.pth" 346378731
check "$W/torch_cache/checkpoints/alexnet-owt-7be5be79.pth" 244408911
check "$W/torch_cache/clip/RN50.pt" 255827503
check "$W/torch_cache/facebookresearch_dinov2_main/hubconf.py"
check "$HOME/.cache/torch/hub/pyiqa/ckpt_koniq10k.pt"

if [ "$MISS" = "0" ]; then
  touch "$P/.weights_ready"
  echo "ALL_WEIGHTS_READY -> $P/.weights_ready"
else
  echo "WEIGHTS_INCOMPLETE"
  exit 1
fi
