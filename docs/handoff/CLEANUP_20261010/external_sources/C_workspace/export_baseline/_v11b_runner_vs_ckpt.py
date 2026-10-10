import importlib.util, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

PKG = r"D:\models\pkg_TinyV11b"
CKPT = r"D:\models\tiny_v1\v11b_s25000.pt"
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
img = np.asarray(Image.open(TEST / "case1.jpg").convert("RGB"), dtype=np.uint8)[1792:2304, 1280:1792]  # 512 crop
x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div_(127.5).sub_(1)

spec = importlib.util.spec_from_file_location("r11", os.path.join(PKG, "model_dir", "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(os.path.join(PKG, "model_dir"))
with torch.no_grad():
    y_runner = R.infer(x, prompt="").float().cpu()

from diffusers import UNet2DModel
d = torch.load(CKPT, map_location="cpu", weights_only=False)
ch = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch, down_block_types=("DownBlock2D",) * len(ch),
                   up_block_types=("UpBlock2D",) * len(ch))
unet.load_state_dict(d["unet"]); net = unet.cuda().eval().half()
sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5
with torch.no_grad():
    t = F.interpolate(x.cuda(), size=(256, 256), mode="bicubic", align_corners=False).half()
    pred = net(t, torch.full((1,), 500, dtype=torch.long, device="cuda")).sample
    y = (t - sq_1a * pred.float()) / sq_a
    y_ckpt = F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False).clamp(-1, 1).cpu()

import torch as T
u8r = T.round((y_runner.clamp(-1,1)+1)*127.5)
u8c = T.round((y_ckpt.clamp(-1,1)+1)*127.5)
d_ = (u8r - u8c).abs()
print("runner vs ckpt: max u8 diff = %d, mean = %.4f, 不同像素比 = %.4f%%" %
      (int(d_.max()), float(d_.mean()), float((d_.max(dim=1).values > 0).float().mean()*100)))
print("runner sha:", T.tensor(list(u8r.numpy().tobytes()))[:0].numel() if False else __import__("hashlib").sha256(u8r.numpy().astype("uint8").tobytes()).hexdigest()[:16])
print("ckpt   sha:", __import__("hashlib").sha256(u8c.numpy().astype("uint8").tobytes()).hexdigest()[:16])
