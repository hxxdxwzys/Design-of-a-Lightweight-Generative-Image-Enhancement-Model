#!/bin/bash
# P100 全自动链 (2026-09-02): 权重下载 -> 离线验证 -> Flickr2K下载 -> 解压 -> 双卡训练
# 任何一步失败都会打日志退出; 幂等可重跑 (下载/训练均有 pgrep 防重)
set -e
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"

echo "=== [1/7] 等待权重齐备 (p100_ready.sh 打标记) ==="
for i in $(seq 1 180); do
  if [ -f "$P/.weights_ready" ]; then echo "WEIGHTS_READY (等待 ${i}0 秒)"; break; fi
  if grep -q DL_DONE "$P/logs/dl.log" 2>/dev/null; then echo "DL_DONE_LEGACY"; break; fi
  if [ $i -eq 180 ]; then echo "TIMEOUT: 权重未就绪"; tail -5 "$P/logs/dl2.log" 2>/dev/null; tail -5 "$P/logs/dl4.log" 2>/dev/null; exit 1; fi
  sleep 10
done

echo "=== [2/7] 离线验证训练核心路径 (set_dir 先行, 全程无网络) ==="
python - <<'PYEOF'
import os, torch, torchvision  # noqa
import pyiqa
TC = os.path.expanduser('~/p100/weights/torch_cache')
torch.hub.set_dir(TC)
lp = pyiqa.create_metric('lpips', device='cuda'); lp.as_loss = True
cq = pyiqa.create_metric('clipiqa', device='cuda'); cq.as_loss = True
a = torch.rand(1, 3, 256, 256, device='cuda').clamp(1e-4, 1 - 1e-4)
b = torch.rand(1, 3, 256, 256, device='cuda').clamp(1e-4, 1 - 1e-4)
print('lpips loss:', float(lp(a, b)))
print('clipiqa loss:', float((1 - cq(a)).mean()))
print('OFFLINE_LOSSES_OK')
PYEOF

echo "=== [3/7] 确保 Flickr2K 下载运行 ==="
if pgrep -f "dl_chunked[.]py" > /dev/null 2>&1; then
  echo "F2K_ALREADY_RUNNING"
else
  DL_URL="https://hf-mirror.com/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip" \
  DL_TARGET="$P/data/Flickr2K.zip" \
  DL_TMP="$P/data/flickr2k_chunks" \
  DL_SIZE="11645004887" \
  DL_WORKERS="8" \
  nohup python "$P/tools/dl_chunked.py" > "$P/logs/f2k.log" 2>&1 &
  echo "F2K_STARTED_PID_$!"
fi

echo "=== [4/7] 等待 Flickr2K 下载完成 (zip=11.6GB) ==="
for i in $(seq 1 300); do
  sz=$(stat -c %s "$P/data/Flickr2K.zip" 2>/dev/null || echo 0)
  if [ "$sz" = "11645004887" ]; then echo "F2K_DONE (等待 ${i}0 秒)"; break; fi
  [ $i -eq 300 ] && { echo "TIMEOUT: Flickr2K 下载超时"; tail -5 "$P/logs/f2k.log"; exit 1; }
  sleep 10
done

echo "=== [5/7] 解压 Flickr2K ==="
if [ -d "$P/data/flickr2k" ] && [ "$(find "$P/data/flickr2k" -name '*.png' | wc -l)" -gt 2000 ]; then
  echo "F2K_ALREADY_EXTRACTED: $(find "$P/data/flickr2k" -name '*.png' | wc -l) 张"
else
  python "$P/tools/extract_flickr2k.py" --zip "$P/data/Flickr2K.zip" --out "$P/data/flickr2k" --rm
fi
N=$(find "$P/data/flickr2k" -name '*.png' | wc -l)
echo "flickr2k png 数量: $N"
[ "$N" -gt 2000 ] || { echo "ERROR: flickr2k 数据不足"; exit 1; }

echo "=== [6/7] 数据/权重齐备检查 ==="
ls -la "$P/weights/VOSR_0.5B_os/checkpoints/ema_model.safetensors" > /dev/null && echo CKPT_OK
ls -d "$P/weights/torch_cache/facebookresearch_dinov2_main" > /dev/null && echo DINOV2_OK
ls -la "$P/weights/torch_cache/checkpoints/dinov2_vitb14_pretrain.pth" > /dev/null && echo DINOV2_W_OK
ls -la "$P/weights/sd21_lwdecoder.pth" > /dev/null && echo LWDEC_OK
ls -la "$P/weights/sd21base/vae/diffusion_pytorch_model.safetensors" > /dev/null && echo VAE_OK
if [ -f "$P/.cache/torch/hub/pyiqa/ckpt_koniq10k.pt" ]; then echo P100_PYIQA_PRESENT; fi
# pyiqa 缓存兜底: 若在 p100 下则拷到 ~/.cache (容错, set -e 安全)
mkdir -p "$HOME/.cache/torch/hub"
if [ -d "$P/.cache/torch/hub/pyiqa" ]; then
  cp -rn "$P/.cache/torch/hub/pyiqa" "$HOME/.cache/torch/hub/" || true
fi
if [ -f "$HOME/.cache/torch/hub/pyiqa/ckpt_koniq10k.pt" ]; then
  echo PYIQA_OK
else
  echo PYIQA_MISS
fi

echo "=== [7/7] 双卡训练启动 ==="
bash "$P/tools/p100_launch.sh"
echo AUTO_CHAIN_DONE
