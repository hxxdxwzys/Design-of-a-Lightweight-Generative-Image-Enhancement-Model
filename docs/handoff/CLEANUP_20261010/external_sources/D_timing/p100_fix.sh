#!/bin/bash
# P100 环境修复: transformers 钉版 + 全栈验证 (2026-09-02)
set -e
export PATH="$HOME/miniconda3/bin:$PATH"

echo "=== transformers 钉版 4.56.2 ==="
pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers==4.56.2"

echo "=== 全栈导入验证 ==="
python - <<'PYEOF'
import torch, torchvision, pyiqa, peft, diffusers, transformers
print('IMPORTS OK')
print('torch', torch.__version__, '| transformers', transformers.__version__)
print('cuda:', torch.cuda.is_available())
print('gpu0:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('gpu1:', torch.cuda.get_device_name(1), torch.cuda.get_device_capability(1))
x = torch.randn(64, 64, device='cuda', dtype=torch.float16)
y = torch.matmul(x, x)
torch.cuda.synchronize()
print('FP16 MATMUL OK:', float(y.abs().sum()) > 0)
# 可微损失 (训练核心路径)
lpips_m = pyiqa.create_metric('lpips', device='cuda'); lpips_m.as_loss = True
cq_m = pyiqa.create_metric('clipiqa', device='cuda'); cq_m.as_loss = True
a = torch.rand(1, 3, 256, 256, device='cuda').clamp(1e-4, 1 - 1e-4)
b = torch.rand(1, 3, 256, 256, device='cuda').clamp(1e-4, 1 - 1e-4)
print('lpips loss:', float(lpips_m(a, b)))
print('clipiqa loss:', float((1 - cq_m(a)).mean()))
print('LOSSES OK')
PYEOF
echo FIX_DONE
