import importlib.util, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
dev = "cuda"
def check(pkg, ckpt, img_path, box):
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)[box[0]:box[1], box[2]:box[3]].copy()
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div_(127.5).sub_(1)
    spec = importlib.util.spec_from_file_location("rr_" + os.path.basename(pkg), os.path.join(pkg, "model_dir", "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    R = m.Runner(os.path.join(pkg, "model_dir"))
    with torch.no_grad(): y_r = R.infer(x, prompt="").float().cpu()
    from diffusers import UNet2DModel
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    ch = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
    unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                       block_out_channels=ch, down_block_types=("DownBlock2D",)*len(ch), up_block_types=("UpBlock2D",)*len(ch))
    unet.load_state_dict(d["unet"]); net = unet.to(dev).eval().half()
    sq_a = alpha**0.5; sq_1a = (1-alpha)**0.5
    with torch.no_grad():
        xh = x.to(dev).half()
        xs = F.interpolate(xh, size=(256,256), mode="bicubic", align_corners=False)
        pred = net(xs, torch.full((1,), 500, dtype=torch.long, device=dev)).sample
        y_c = F.interpolate((xs - sq_1a*pred)/sq_a, size=(512,512), mode="bicubic", align_corners=False).float().clamp(-1,1).cpu()
    u8 = lambda z: torch.round((z.clamp(-1,1)+1)*127.5)
    dd = (u8(y_r) - u8(y_c)).abs()
    print("%-16s vs %-18s max u8 = %d  mean = %.5f" % (os.path.basename(pkg), os.path.basename(ckpt), int(dd.max()), float(dd.mean())), flush=True)
TESTS = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集"
check(r"D:\models\pkg_Tinyv20b", r"D:\models\tiny_v1\v20b_s21000.pt", os.path.join(TESTS, "case10.jpg"), (1280, 1792, 1792, 2304))
check(r"D:\models\pkg_Tinyv21",  r"D:\models\tiny_v1\v21_s21000.pt",  os.path.join(TESTS, "case3.jpg"),  (1792, 2304, 1280, 1792))
