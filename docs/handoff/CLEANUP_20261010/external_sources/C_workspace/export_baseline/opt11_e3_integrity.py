r"""E3 完整性核查：用包内 runner 重渲染 100 张，与提交 zip 内图逐字节比对；并做同输入重复确定性检验。
用法: python opt11_e3_integrity.py <zip> <model_dir> <tag>
"""
import hashlib, importlib.util, io, os, sys, time, zipfile
from pathlib import Path
import numpy as np, torch
from PIL import Image
zip_path, mdir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
TILE, OVER = 512, 32
spec = importlib.util.spec_from_file_location("r_%s" % tag, os.path.join(mdir, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(mdir)
mask = np.ones((TILE, TILE, 1), dtype=np.float32); r = OVER//2
ramp = np.linspace(0, 1, r, dtype=np.float32)
mask[:r,:,0] *= ramp[:,None]; mask[-r:,:,0] *= ramp[::-1,None]
mask[:,:r,0] *= ramp[None,:]; mask[:,-r:,0] *= ramp[::-1][None,:]
@torch.no_grad()
def run_tile(patch):
    t = torch.from_numpy(patch.copy()).permute(2,0,1)[None].float().div_(127.5).sub_(1)
    y = R.infer(t, prompt="")
    return np.clip((y[0].permute(1,2,0).float().cpu().numpy()+1)*127.5, 0, 255)
def render(p):
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    H, W = img.shape[:2]
    acc = np.zeros((H,W,3), dtype=np.float32); ws = np.zeros((H,W,1), dtype=np.float32)
    ys = sorted(set(list(range(0, max(1,H-TILE+1), TILE-OVER)) + [max(0,H-TILE)]))
    xs = sorted(set(list(range(0, max(1,W-TILE+1), TILE-OVER)) + [max(0,W-TILE)]))
    for y0 in ys:
        for x0 in xs:
            acc[y0:y0+TILE, x0:x0+TILE] += run_tile(img[y0:y0+TILE, x0:x0+TILE]) * mask
            ws[y0:y0+TILE, x0:x0+TILE] += mask
    return np.clip(acc/np.maximum(ws,1e-6), 0, 255).astype(np.uint8)
files = sorted(TEST.glob("case*.jpg"))
t0 = time.time(); same = 0; diff = []
with zipfile.ZipFile(zip_path) as zf:
    for i, f in enumerate(files, 1):
        arr = render(f)
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="JPEG", quality=98, subsampling=0)
        mine = buf.getvalue()
        try: theirs = zf.read("output_dir/%s" % f.name)
        except KeyError: diff.append((f.name, "missing")); continue
        if hashlib.sha256(mine).hexdigest() == hashlib.sha256(theirs).hexdigest(): same += 1
        else: diff.append((f.name, "byte-diff"))
        if i % 25 == 0: print("  %d/%d (%.1f s/张) 一致=%d 差异=%d" % (i, len(files), (time.time()-t0)/i, same, len(diff)), flush=True)
# 确定性：同输入两次
a1 = render(files[0]); a2 = render(files[0])
det = bool((a1 == a2).all())
print("[%s] 100 图逐字节一致 = %d/100；差异 = %s；同输入重复逐位一致 = %s (%.1f min)" %
      (tag, same, diff[:3], det, (time.time()-t0)/60), flush=True)
print("E3_DONE", flush=True)
