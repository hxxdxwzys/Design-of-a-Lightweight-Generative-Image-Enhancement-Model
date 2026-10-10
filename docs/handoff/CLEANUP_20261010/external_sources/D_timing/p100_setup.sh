#!/bin/bash
# P100 教学服务器环境引导 v3 (2026-09-02): 固定版 Miniconda (py3.12) + torch cu121 + 依赖
# v3: 用 Miniconda3-py312_24.7.1-0 (base=py3.12, torch 2.4.1+cu121 的 cp312 wheel 可用;
#     最新版 base=py3.14 没有对应 wheel)。绕开 conda create (新版 conda ToS 阻塞)。
# 用法: bash p100_setup.sh > ~/p100/logs/setup.log 2>&1
set -e
export PATH="$HOME/miniconda3/bin:$PATH"

echo "=== [1/4] miniconda (py312 固定版) ==="
if [ ! -x "$HOME/miniconda3/bin/python" ]; then
  curl -sL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
fi
python --version

echo "=== [2/4] pip 基础 ==="
pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple "setuptools<81" wheel

echo "=== [3/4] torch cu121 (含 sm_60 kernel) ==="
pip install -q torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121 \
  || pip install -q torch==2.4.1+cu121 torchvision==0.19.1+cu121 -f https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu121/ \
  || pip install -q torch==2.4.1+cu121 torchvision==0.19.1+cu121 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/

echo "=== [4/4] 其余依赖 ==="
pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple pyiqa peft safetensors pillow numpy requests diffusers

# 目录
mkdir -p ~/p100/{tools,VOSR,data/div2k,data/flickr2k,val,runs,logs,weights}
mkdir -p ~/.cache/torch/hub

# 验证
python -c "import torch; print('torch', torch.__version__); print('cuda ok:', torch.cuda.is_available()); print('gpu0:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('gpu1:', torch.cuda.get_device_name(1), torch.cuda.get_device_capability(1))"
python -c "
import torch, torchvision, pyiqa, peft, diffusers
x = torch.randn(64, 64, device='cuda', dtype=torch.float16)
y = torch.matmul(x, x)
torch.cuda.synchronize()
print('fp16 matmul OK:', float(y.abs().sum()) > 0)
print('pyiqa', pyiqa.__version__, '| peft', peft.__version__, '| diffusers', diffusers.__version__)
"
echo "SETUP_DONE"
