#!/bin/bash
# 云服务器环境安装：venv + PyTorch(CUDA) + 项目依赖
set -ex

# 1) venv 建在数据盘
python3 -m venv /data/env
/data/env/bin/pip install --upgrade pip wheel

# 2) PyTorch CUDA 构建 (cu126, 兼容驱动570/CUDA12.8)
/data/env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
/data/env/bin/python -c "import torch; print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"

# 3) 项目依赖 (clipiqa 需 setuptools<81)
/data/env/bin/pip install pyiqa lpips opencv-python-headless tqdm einops pyyaml scikit-image matplotlib accelerate diffusers transformers timm safetensors huggingface_hub "setuptools<81"

# 4) 验证导入
/data/env/bin/python -c "import torch,torchvision,pyiqa,lpips,cv2,transformers,diffusers,timm,accelerate,einops; print('ALL_IMPORTS_OK')"
echo PIP_INSTALL_DONE
