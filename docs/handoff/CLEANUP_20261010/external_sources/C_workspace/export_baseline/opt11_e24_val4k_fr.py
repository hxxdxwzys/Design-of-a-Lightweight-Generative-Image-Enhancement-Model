r"""E24: val 3 对 4K 全参考四项（PSNR/SSIM/LPIPS/DISTS），评估 V4 / clip1 / clip2 / E8。
独立于无参考代理的第二个轴：若 clip2 的全参考指标与 V4 相当，则其"高 clipiqa"不是靠破坏保真换来的。
"""
import importlib.util, json, os, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"): os.environ.pop(k, None)
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch
from PIL import Image
DEV = "cuda"; TILE, OVER = 512, 32
VAL = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集")
OUT = Path(r"D:\models\val4k"); OUT.mkdir(parents=True, exist_ok=True)
RES = Path(r"D:\models\opt11\e24_val4k_fr.json")
PKGS = {
 "v4":   r"D:\models\pkg_TinyRefV4\model_dir",
 "clip1":r"D:\models\pkg_clip1\model_dir",
 "clip2":r"D:\models\pkg_clip2\model_dir",
 "e8":   r"D:\models\pkg_E8vosr\model_dir",
}
todo = sys.argv[1].split(",") if len(sys.argv) > 1 else ["lq","v4","clip2","clip1"]
import lpips as lpips_mod, pyiqa
LP = lpips_mod.LPIPS(net="alex", verbose=False).to(DEV).eval()
DISTS = pyiqa.create_metric("dists", device=DEV)
to_t = lambda u8: torch.from_numpy(u8.copy()).permute(2,0,1)[None].to(DEV).float().div_(127.5).sub_(1)
def psnr(p, g):
    m = np.mean((p.astype(np.float64)-g.astype(np.float64))**2)
    return 99.0 if m <= 0 else float(10*np.log10(255.0**2/m))
def ssim_np(p, g):
    p = p.astype(np.float64); g = g.astype(np.float64)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    m1, m2 = p.mean(), g.mean(); v1, v2 = p.var(), g.var()
    cov = ((p-m1)*(g-m2)).mean()
    return float(((2*m1*m2+C1)*(2*cov+C2))/((m1**2+m2**2+C1)*(v1+v2+C2)))
def metrics4k(pred, gt):
    p, g = to_t(pred), to_t(gt)
    with torch.no_grad():
        l = float(LP(p, g).item()); d = float(DISTS((p+1)/2, (g+1)/2).item())
    return psnr(pred, gt), ssim_np(pred, gt), l, d
def make_runner(mdir):
    spec = importlib.util.spec_from_file_location("r_" + Path(mdir).parent.name, mdir + r"\runner.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.Runner(mdir)
mask = np.ones((TILE, TILE, 1), dtype=np.float32); r = OVER//2
ramp = np.linspace(0, 1, r, dtype=np.float32)
mask[:r,:,0] *= ramp[:,None]; mask[-r:,:,0] *= ramp[::-1,None]
mask[:,:r,0] *= ramp[None,:]; mask[:,-r:,0] *= ramp[::-1][None,:]
def render4k(runner, lq):
    H, W = lq.shape[:2]
    acc = np.zeros((H,W,3), dtype=np.float32); wsum = np.zeros((H,W,1), dtype=np.float32)
    ys = sorted(set(list(range(0, max(1,H-TILE+1), TILE-OVER)) + [max(0,H-TILE)]))
    xs = sorted(set(list(range(0, max(1,W-TILE+1), TILE-OVER)) + [max(0,W-TILE)]))
    for y in ys:
        for x in xs:
            t = torch.from_numpy(lq[y:y+TILE, x:x+TILE].copy()).permute(2,0,1)[None].to(DEV).float().div_(127.5).sub_(1)
            with torch.no_grad(): o = runner.infer(t)[0].permute(1,2,0).float().clamp(-1,1).cpu().numpy()
            acc[y:y+TILE, x:x+TILE] += np.clip((o+1)*127.5, 0, 255)*mask
            wsum[y:y+TILE, x:x+TILE] += mask
    return np.clip(acc/np.maximum(wsum,1e-6), 0, 255).astype(np.uint8)
res = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {}
for key in todo:
    if key in res: print("[skip]", key, flush=True); continue
    t0 = time.time(); runner = None if key == "lq" else make_runner(PKGS[key]); per = []
    for c in range(1, 4):
        lq = np.asarray(Image.open(next(VAL.glob("case%d_lq.*" % c))).convert("RGB"), dtype=np.uint8)
        gt = np.asarray(Image.open(next(VAL.glob("case%d_gt.*" % c))).convert("RGB"), dtype=np.uint8)
        pred = lq if runner is None else render4k(runner, lq)
        if runner is not None: Image.fromarray(pred).save(OUT / ("%s_case%d.jpg" % (key, c)), quality=95)
        p, ss, l, d = metrics4k(pred, gt)
        per.append({"psnr": p, "ssim": ss, "lpips": l, "dists": d})
        print("  %s case%d PSNR %.3f SSIM %.4f LPIPS %.4f DISTS %.4f" % (key, c, p, ss, l, d), flush=True)
    agg = {k: float(np.mean([q[k] for q in per])) for k in per[0]}
    res[key] = dict(per_case=per, **agg, elapsed_s=round(time.time()-t0, 1))
    RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %-6s PSNR %.3f SSIM %.4f LPIPS %.4f DISTS %.4f (%.0fs)" % (key, agg["psnr"], agg["ssim"], agg["lpips"], agg["dists"], time.time()-t0), flush=True)
print("E24_DONE", flush=True)
