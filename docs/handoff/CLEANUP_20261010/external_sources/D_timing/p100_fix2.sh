#!/bin/bash
# P100 修复: diffusers 钉 0.31 (与 torch 2.4.1 兼容) + 重启双卡训练 (2026-09-02)
set -e
export PATH="$HOME/miniconda3/bin:$PATH"
P="$HOME/p100"

echo "=== [1/3] diffusers 0.31.0 ==="
pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple "diffusers==0.31.0"
python - <<'PYEOF'
import diffusers, torch
print('diffusers', diffusers.__version__, '| torch', torch.__version__)
from diffusers import AutoencoderKL
print('AutoencoderKL import OK')
PYEOF

echo "=== [2/3] 训练核心链路冒烟 (VAE+LoRA 加载) ==="
python - <<'PYEOF'
import os, torch
os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
import sys
sys.path.insert(0, os.path.expanduser('~/p100/VOSR'))
sys.path.insert(0, os.path.expanduser('~/p100/tools'))
from diffusers import AutoencoderKL
from models.lightningdit import LightningDiT
from models.light_decoder import LightDecoder
from safetensors.torch import load_file
import json
W = os.path.expanduser('~/p100/weights')
base = json.load(open(os.path.join(W, 'VOSR_0.5B_os', 'args.json')))
base['auxiliary_time_cond'] = True
A = type('A', (), base)()
torch.hub.set_dir(os.path.join(W, 'torch_cache'))
model = LightningDiT(
    input_size=A.resolution // 8, patch_size=A.patch_size,
    in_channels=2 * 4, out_channels=4,
    hidden_size=A.dim, depth=A.depth, num_heads=A.num_heads,
    mlp_ratio=A.mlp_ratio, z_dims=A.enc_dim, encdim_ratio=A.encdim_ratio,
    auxiliary_time_cond=True, use_qknorm=A.use_qknorm, use_swiglu=A.use_swiglu,
    use_rope=A.use_rope, use_rmsnorm=A.use_rmsnorm, wo_shift=A.wo_shift,
    num_fused_layers=len(A.layer_dinov2b_list))
sd = load_file(os.path.join(W, 'VOSR_0.5B_os', 'checkpoints', 'ema_model.safetensors'))
model.load_state_dict(sd, strict=False)
model.forward = model.forward_flexible
model = model.half().cuda().eval()
print('LightningDiT 0.5B 加载 OK, 参数:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
vae = AutoencoderKL.from_pretrained(os.path.join(W, 'sd21base'), subfolder='vae').to('cuda').eval()
print('VAE 加载 OK')
x = torch.randn(1, 3, 256, 256).cuda()
with torch.no_grad():
    z = vae.encode(x.half()).latent_dist.sample() * vae.config.scaling_factor
print('VAE encode OK, latent:', tuple(z.shape))
PYEOF

echo "=== [3/3] 重启双卡训练 ==="
bash "$P/tools/p100_launch.sh"
echo FIX2_DONE
