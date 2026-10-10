#!/bin/bash
# OSEDiff 官方环境一键搭建 (Python 3.11 + diffusers 0.25 官方依赖)
# 用法: bash setup_osediff_env.sh
# 前提: 用户已手动安装 Python 3.11.9 (https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
set -e

PY311="/c/Users/Yuki Nagato/AppData/Local/Programs/Python/Python311/python.exe"
VENV="/d/venv_osediff311"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

if [ ! -x "$PY311" ]; then
  echo "❌ 找不到 Python 3.11: $PY311"
  echo "   请先安装: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
  exit 1
fi

# 1. 创建 venv (若不存在)
if [ ! -f "$VENV/Scripts/python.exe" ]; then
  echo "==> 创建 venv: $VENV"
  "$PY311" -m venv "$VENV"
else
  echo "==> venv 已存在: $VENV"
fi

VPY="$VENV/Scripts/python.exe"
VPIP="$VENV/Scripts/pip.exe"

# 2. torch 2.7 (复用本地 cu126 wheel, cp311 需确认; 若无则从清华源)
if ! "$VPY" -c "import torch" 2>/dev/null; then
  echo "==> 安装 torch 2.7 (cp311)"
  LOCAL_TORCH="/d/models/torch-2.7.0+cu126-cp311-cp311-win_amd64.whl"
  LOCAL_TV="/d/models/torchvision-0.22.0+cu126-cp311-cp311-win_amd64.whl"
  if [ -f "$LOCAL_TORCH" ]; then
    "$VPIP" install "$LOCAL_TORCH"
    [ -f "$LOCAL_TV" ] && "$VPIP" install "$LOCAL_TV"
  else
    echo "⚠️ 本地无 cp311 torch wheel, 尝试 pytorch 官方源:"
    "$VPIP" install -i https://download.pytorch.org/whl/cu126 torch torchvision --index-url https://download.pytorch.org/whl/cu126 || echo "❌ torch 安装失败, 需要手动下载 cp311 wheel"
  fi
else
  echo "==> torch 已装: $("$VPY" -c "import torch; print(torch.__version__)")"
fi

# 3. 官方依赖 (清华源; huggingface_hub 按官方 requirements.txt 锁 0.25.0)
echo "==> 安装 diffusers 0.25 / transformers 4.28 / peft 0.9 等"
"$VPIP" install -i "$MIRROR" \
  "diffusers==0.25.0" "transformers==4.28.1" "peft==0.9.0" \
  "loralib" "fairscale" "einops==0.7.0" "open-clip-torch==2.20.0" \
  "huggingface-hub==0.25.0" "accelerate==0.20.3" \
  "torchvision" 2>&1 | tail -3

# 4. 验证
echo "==> 验证:"
"$VPY" -c "
import torch, diffusers, transformers, peft, huggingface_hub
print('torch', torch.__version__)
print('diffusers', diffusers.__version__)
print('transformers', transformers.__version__)
print('peft', peft.__version__)
print('huggingface_hub', huggingface_hub.__version__)
import torch.cuda
print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
echo "✅ 环境就绪: $VENV"
echo "   新会话入口: $VPY"
