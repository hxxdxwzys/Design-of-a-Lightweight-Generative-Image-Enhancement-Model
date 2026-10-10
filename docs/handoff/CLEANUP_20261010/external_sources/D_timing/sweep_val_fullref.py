# -*- coding: utf-8 -*-
"""val 512 中心裁切 有参四项 (PSNR/SSIM/LPIPS/DISTS) + 空闲态 512 时延 扫描 (2026-09-15)

配置键: lq(恒等基准) | m24 | s1 | b1 | b10 | l1 | l10 | xl1 | xl10   (字母=尺寸, 数字=采样步数)
输出: D:/tmp_timing/sweep_val_fullref.json  (增量写, 断点续跑)
用法: python sweep_val_fullref.py [--only b1,b10] [--lat-iters 50]
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
os.environ.setdefault('TORCHDYNAMO_SUPPRESS_ERRORS', '1')
for _k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(_k, None)

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

DEV = 'cuda'
VAL = Path('C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
REPO = Path('D:/models/newsrc/PixRestore_code')
CKPTS = Path('D:/models/newsrc/PixRestore')
DINOV2 = Path('D:/models/newsrc/dinov2')
DINOV2_REPO = 'D:/models/newsrc/dinov2_repo'
M24_DIR = 'D:/models/final_compliant/model_dir'
OUT = Path('D:/tmp_timing/sweep_val_fullref.json')

ALL = ['lq', 'm24', 's1', 'b1', 'b10', 'l1', 'l10', 'xl1', 'xl10', 'pkgv2', 'pkgv5', 'pkgv7',
       'hypir_ct150', 'hypir_2x1x']

ap = argparse.ArgumentParser()
ap.add_argument('--only', default='')
ap.add_argument('--lat-iters', type=int, default=50)
a = ap.parse_args()
todo = [c.strip() for c in a.only.split(',')] if a.only else list(ALL)
todo = [c for c in todo if c]


def log(*xs):
    print(*xs, flush=True)


# ---------------- 数据 ----------------
def load_val(n=3):
    pairs = []
    for c in range(1, n + 1):
        lq = Image.open(next(VAL.glob('case%d_lq.*' % c))).convert('RGB')
        gt = Image.open(next(VAL.glob('case%d_gt.*' % c))).convert('RGB')
        W, H = lq.size
        box = ((W - 512) // 2, (H - 512) // 2, (W - 512) // 2 + 512, (H - 512) // 2 + 512)
        pairs.append((np.asarray(lq.crop(box), dtype=np.uint8), np.asarray(gt.crop(box), dtype=np.uint8)))
    return pairs


def to_t(u8):
    return torch.from_numpy(u8.copy()).permute(2, 0, 1)[None].to(DEV).float().div_(127.5).sub_(1)


def to_u8(t):
    x = t[0].permute(1, 2, 0).float().clamp(-1, 1).cpu().numpy()
    return np.clip((x + 1) * 127.5, 0, 255).astype(np.uint8)


def psnr(p, g):
    m = np.mean((p.astype(np.float64) - g.astype(np.float64)) ** 2)
    return 99.0 if m <= 0 else float(10 * np.log10(255.0 ** 2 / m))


def ssim_np(p, g):
    p = p.astype(np.float64); g = g.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m1, m2 = p.mean(), g.mean(); v1, v2 = p.var(), g.var()
    cov = ((p - m1) * (g - m2)).mean()
    return float(((2 * m1 * m2 + C1) * (2 * cov + C2)) / ((m1 ** 2 + m2 ** 2 + C1) * (v1 + v2 + C2)))


# ---------------- 指标器 ----------------
import lpips as lpips_mod
import pyiqa
LP = lpips_mod.LPIPS(net='alex', verbose=False).to(DEV).eval()
DISTS = pyiqa.create_metric('dists', device=DEV)


def metrics(pred_u8, gt_u8):
    p, g = to_t(pred_u8), to_t(gt_u8)
    with torch.no_grad():
        l = float(LP(p, g).item())          # lpips 包: 期望 [-1,1]
        d = float(DISTS((p + 1) / 2, (g + 1) / 2).item())  # pyiqa: 期望 [0,1]
    return psnr(pred_u8, gt_u8), ssim_np(pred_u8, gt_u8), l, d


# ---------------- 模型 ----------------
_pix_cache = {}


def make_pix(size):
    if size in _pix_cache:
        return _pix_cache[size]
    # 防模块名冲突: 其它包的 vendored dinov2 会遮蔽真 repo 的 dinov2.hub
    for _m in [m for m in list(sys.modules) if m == 'dinov2' or m.startswith('dinov2.')]:
        del sys.modules[_m]
    sys.path.insert(0, str(REPO))
    from pixrestore import LightningDiT_PixelDiffusion, PixelDiffusion
    from pixrestore.vision import extract_layers, load_dinov2
    CK = CKPTS / ('pixrestore-' + size)
    cfg = json.loads((CK / 'config.json').read_text(encoding='utf-8'))
    enc_ck = {'dinov2s': 'dinov2_vits14_pretrain.pth', 'dinov2b': 'dinov2_vitb14_pretrain.pth',
              'dinov2l': 'dinov2_vitl14_pretrain.pth'}.get(cfg['encoder_type'], 'dinov2_vits14_pretrain.pth')
    cfg['dinov2_repository'] = DINOV2_REPO
    cfg['dinov2_checkpoint'] = str(DINOV2 / enc_ck)
    config = SimpleNamespace(**cfg)
    model = LightningDiT_PixelDiffusion(
        input_size=config.resolution, patch_size=config.patch_size, in_channels=6, out_channels=3,
        hidden_size=config.hidden_size, depth=config.depth, num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio, z_dims=config.encoder_dim if config.use_venc else None,
        encdim_ratio=config.encoder_dim_ratio, use_qknorm=config.use_qknorm, use_swiglu=config.use_swiglu,
        use_rope=config.use_rope, use_rmsnorm=config.use_rmsnorm, adain_single=True,
        num_fused_layers=len(config.encoder_layers), pca_dim=config.bottleneck_dim,
        use_bottleneck_patch_embed=True, use_dino_layer_router=config.use_dino_layer_router,
        dino_layer_router_mode=config.dino_layer_router_mode,
        same_condition_layer_weights=config.same_condition_layer_weights,
        gate_temperature=config.gate_temperature, feature_norm=config.feature_norm,
        use_lq_degradation_token=bool(getattr(config, 'use_lq_degradation_token', False)),
        register_local_gate_logit_scale=bool(getattr(config, 'register_local_gate_logit_scale', False)))
    st = load_file(str(CK / 'clean_weights' / 'ema_model.safetensors'))
    msg = model.load_state_dict(st, strict=False)
    model = model.to(DEV).eval().requires_grad_(False)
    flow = PixelDiffusion(flow_ratio=config.flow_ratio, time_dist=config.time_distribution, alpha=config.alpha,
                          z_start='noise', cfg_ratio=config.cfg_ratio, cfg_scale=config.cfg_scale,
                          image_size=config.resolution, channels=3, interp_type='lin', uncond_type='zero',
                          norm_p=config.adaptive_loss_power, accelerator=None, t_start=0, t_end=1,
                          use_cos=False, args=config)
    enc = load_dinov2(config.encoder_type, DEV, repository=config.dinov2_repository,
                      checkpoint=config.dinov2_checkpoint)
    npar = sum(p.numel() for p in model.parameters()) / 1e6
    log('[model] pixrestore-%s DiT %.1fM | encoder %s | missing %d unexpected %d'
        % (size, npar, config.encoder_type, len(msg.missing_keys), len(msg.unexpected_keys)))

    def restore(t, steps=1):
        fea = extract_layers(enc, t, config.encoder_layers, config.encoder_input_size)
        with torch.autocast('cuda', dtype=torch.float16):
            return flow.sample_multistep_fm(model, t, venc_fea=fea, n_steps=steps, schedule='linear')

    _pix_cache[size] = restore
    return restore


def make_pkg(key, mdir, label):
    spec = importlib.util.spec_from_file_location('pkg_runner_' + key, mdir + '/runner.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pkg_runner_' + key] = mod
    sys.path.insert(0, mdir)
    spec.loader.exec_module(mod)
    r = mod.Runner(mdir)
    log('[model] ' + label + ' (已提交包 runner 原样)')
    return lambda t: r.infer(t)


# 文件型配置: (目录, 文件名模板) —— 不跑模型, 直接对有参四项打分
FILE_CFGS = {
    'hypir_ct150': ('D:/models/hypir/out_val3_ct150/result', 'case{n}_lq.png'),
    'hypir_2x1x': ('D:/models/hypir/out_val3_2x/result_1x', 'case{n}.png'),
}

PKGS = {
    'pkgv5': ('D:/models/pkg_TinyV5d/model_dir', 'TinyV5d 0.87M (LFdiff 结构蒸馏, 官方 33.0382)'),
    'pkgv7': ('D:/models/pkg_TinyV7gan/model_dir', 'TinyV7gan 0.87M (GAN 蒸馏 M24 观感, 官方 41.7346)'),
    'pkgv2': ('D:/models/pkg_TinyV2g/model_dir', 'TinyV2g 0.87M (GT 目标 + CUDA Graph, 官方 33.2872)'),
}


def make_m24():
    return make_pkg('m24', M24_DIR, 'M24 VOSR 0.5B + v7_s5000 LoRA (合规 runner)')


def latency(fn, iters):
    """自适应: 先用 3 次探测, 慢配置只跑少量迭代 (总时长 ~<=4s)"""
    x = torch.randn(1, 3, 512, 512, device=DEV).clamp(-1, 1)
    with torch.no_grad():
        for _ in range(5):
            fn(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(3):
            fn(x)
        torch.cuda.synchronize()
        probe = (time.perf_counter() - t0) / 3 * 1000.0
        n = int(max(3, min(iters, 4000.0 / max(probe, 1e-6))))
        for _ in range(5):
            fn(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            fn(x)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000.0
    log('   [lat] probe %.1f ms -> %d 次迭代 -> %.1f ms' % (probe, n, ms))
    return ms


def main():
    res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {'configs': {}}
    val = load_val(3)
    log('[val] 3 对 512 中心裁切已载入')

    def save():
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')

    for key in todo:
        if key in res['configs']:
            log('[skip] %s 已有' % key)
            continue
        try:
            one(key, res, val, save)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res.setdefault('errors', {})[key] = '%s: %s' % (type(e).__name__, e)
            save()
            log('[FAIL] %s -> %s: %s' % (key, type(e).__name__, e))

    log('')
    log('| 配置 | 512 时延 | 加速比 | PSNR | SSIM | LPIPS | DISTS |')
    log('|---|---|---|---|---|---|---|')
    for k in ALL:
        if k in res['configs']:
            r = res['configs'][k]
            lat = '—' if k == 'lq' else '%.1f ms (%.2fx)' % (r['ms'], r['ratio'])
            log('| %s | %s | %.3f | %.4f | %.4f | %.4f |' % (k, lat, r['psnr'], r['ssim'], r['lpips'], r['dists']))
    log('SWEEP_DONE')


def one(key, res, val, save):
    if True:
        t_cfg = time.time()
        if key in FILE_CFGS:
            d, pat = FILE_CFGS[key]
            per = []
            for i, (lq_u8, gt_u8) in enumerate(val):
                im = Image.open(Path(d) / pat.format(n=i + 1)).convert('RGB')
                W, H = im.size
                box = ((W - 512) // 2, (H - 512) // 2, (W - 512) // 2 + 512, (H - 512) // 2 + 512)
                o_u8 = np.asarray(im.crop(box), dtype=np.uint8)
                p, s, l, dd = metrics(o_u8, gt_u8)
                per.append({'psnr': p, 'ssim': s, 'lpips': l, 'dists': dd})
                log('  case%d PSNR %.3f SSIM %.4f LPIPS %.4f DISTS %.4f' % (i + 1, p, s, l, dd))
            agg = {k: float(np.mean([c[k] for c in per])) for k in per[0]}
            res['configs'][key] = dict(size=key, steps=0, ms=float('nan'), ratio=float('nan'),
                                       per_case=per, src=d, elapsed_s=round(time.time() - t_cfg, 1), **agg)
            save()
            log('[done] %-12s (文件型, 无时延) | PSNR %.3f SSIM %.4f LPIPS %.4f DISTS %.4f | %.0fs'
                % (key, agg['psnr'], agg['ssim'], agg['lpips'], agg['dists'], time.time() - t_cfg))
            return
        if key == 'lq':
            fn = lambda t: t
            size, steps = 'identity', 0
        elif key == 'm24':
            fn = make_m24(); size, steps = 'm24', 1
        elif key in PKGS:
            mdir, label = PKGS[key]
            fn = make_pkg(key, mdir, label); size, steps = key, 1
        else:
            size = key[0]; steps = int(key[1:])
            fn = (lambda s: (lambda t: make_pix(s)(t, steps)))(size)
        x = torch.randn(1, 3, 512, 512, device=DEV).clamp(-1, 1)
        with torch.no_grad():
            fn(x)
        ms = latency(fn, a.lat_iters)
        per = []
        for i, (lq_u8, gt_u8) in enumerate(val):
            with torch.no_grad():
                out = fn(to_t(lq_u8))
            o_u8 = to_u8(out)
            p, s, l, d = metrics(o_u8, gt_u8)
            per.append({'psnr': p, 'ssim': s, 'lpips': l, 'dists': d})
            log('  case%d PSNR %.3f SSIM %.4f LPIPS %.4f DISTS %.4f' % (i + 1, p, s, l, d))
        agg = {k: float(np.mean([c[k] for c in per])) for k in per[0]}
        rec = dict(size=size, steps=steps, ms=ms, ratio=285.6 / ms, per_case=per,
                   elapsed_s=round(time.time() - t_cfg, 1), **agg)
        res['configs'][key] = rec
        save()
        log('[done] %-5s %8.1f ms (%6.2fx) | PSNR %.3f SSIM %.4f LPIPS %.4f DISTS %.4f | %.0fs'
            % (key, ms, 285.6 / ms, agg['psnr'], agg['ssim'], agg['lpips'], agg['dists'], time.time() - t_cfg))

    log('')
    log('| 配置 | 512 时延 | 加速比 | PSNR | SSIM | LPIPS | DISTS |')
    log('|---|---|---|---|---|---|---|')
    for k in ALL:
        if k in res['configs']:
            r = res['configs'][k]
            lat = '—' if k == 'lq' else '%.1f ms (%.2fx)' % (r['ms'], r['ratio'])
            log('| %s | %s | %.3f | %.4f | %.4f | %.4f |' % (k, lat, r['psnr'], r['ssim'], r['lpips'], r['dists']))
    log('SWEEP_DONE')


main()
