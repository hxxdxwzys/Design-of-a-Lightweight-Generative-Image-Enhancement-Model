# -*- coding: utf-8 -*-
"""verify_dinov2_shim.py
验证 D:/models/final_compliant/model_dir/dinov2/ 最小推理子集:
  import 链通 -> 权重加载 -> prepare_tokens_with_masks + forward_with_features 前向
  -> 与 "真实 dinov2 加载路径" (C:/models/finals 下 torch.hub 本地缓存 + load_dinov2) 输出一致

对比对象:
  A) 拷贝版: model_dir/dinov2 + your_model.pt pkg['venc'] (fp16)
  B) 真实路径: inference_vosr_onestep.load_dinov2 (hub 本地缓存 fp32 官方权重) -> .half()
比较: layer_8 MAE 与 x_norm MAE (判据 < 1e-2)
"""
import os
import sys
import time
import types
import math
import importlib.util

import torch

LOG_PATH = r"D:/tmp_timing/verify_dinov2_shim.log"
MODEL_DIR = r"D:/models/final_compliant/model_dir"
CKPT_PATH = os.path.join(MODEL_DIR, "your_model.pt")
VOSR_FILE = r"D:/models/finals/VOSR/inference_vosr_onestep.py"
C_FINALS = r"C:/models/finals"
REF_CACHE_REL = r"preset/ckpts/torch_cache"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAE_TOL = 1e-2


class _Log:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.f.write("=== verify_dinov2_shim run %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        self.f.flush()

    def __call__(self, msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


log = _Log(LOG_PATH)


def wait_for_gpu(min_free_gb=1.5, timeout_s=1200):
    """GPU 可能被占: 空闲显存不足则等待 (batch 1 仅需 <1.5GB)."""
    if not torch.cuda.is_available():
        log("cuda unavailable -> CPU fallback")
        return "cpu"
    free, total = torch.cuda.mem_get_info()
    log("cuda mem free %.1fGB / %.1fGB" % (free / 1e9, total / 1e9))
    t0 = time.time()
    while free < min_free_gb * 1e9:
        if time.time() - t0 > timeout_s:
            log("GPU busy too long -> CPU fallback")
            return "cpu"
        log("GPU busy (free %.2fGB), waiting 20s..." % (free / 1e9))
        time.sleep(20)
        free, total = torch.cuda.mem_get_info()
    return "cuda"


def patch_forward_with_features(model):
    """与 inference_vosr_onestep.py load_dinov2 中的 monkey-patch 完全一致 (不改原文件)."""
    def forward_with_features(self, x, masks=None):
        features = {}
        layer_indices = list(range(len(self.blocks)))
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layer_indices:
                features["layer_%d" % i] = x[:, 1:]
        x_norm = self.norm(x)
        return features, x_norm[:, 1:]

    model.forward_with_features = types.MethodType(forward_with_features, model)
    return model


def make_input(device, seed=1234):
    """随机 3x448x448 图, ImageNet mean/std 归一化, fp16 (与 VOSR preprocess 语义一致)."""
    torch.manual_seed(seed)
    img = torch.rand(1, 3, 448, 448)  # [0,1]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    xn = (img - mean) / std
    xn = xn.half().to(device)
    return xn


def load_pkg_venc():
    log("loading %s (weights_only=False) ..." % CKPT_PATH)
    pkg = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    assert isinstance(pkg, dict) and "venc" in pkg, "pkg['venc'] not found in checkpoint"
    sd = pkg["venc"]
    log("pkg keys: %s" % [k for k in pkg.keys() if not isinstance(pkg[k], dict)][:12])
    log("venc: %d keys, dtype=%s, pos_embed=%s"
        % (len(sd), set(str(sd[k].dtype) for k in sd), tuple(sd["pos_embed"].shape)))
    return pkg, sd


def infer_img_size_from_pos_embed(sd, patch_size):
    n = sd["pos_embed"].shape[1] - 1
    side = int(round(math.sqrt(n)))
    assert side * side == n, "pos_embed num_patches not square: %d" % n
    img_size = side * patch_size
    log("pos_embed -> %dx%d patches -> img_size=%d" % (side, side, img_size))
    return img_size


# ---------------------------------------------------------------- phase A: 拷贝版
def build_copy_model(sd, device):
    import dinov2.models.vision_transformer as vits  # from MODEL_DIR (copied subset)

    patch_size = 14
    img_size = infer_img_size_from_pos_embed(sd, patch_size)
    # 与官方 hub _make_dinov2_model(vitb14) 完全一致的构造参数
    model = vits.vit_base(
        img_size=img_size,
        patch_size=patch_size,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    log("copy model built: %s (%.2fM params)"
        % (type(model).__name__, sum(p.numel() for p in model.parameters()) / 1e6))
    model.half()
    res = model.load_state_dict(sd, strict=False)
    log("load_state_dict strict=False: missing=%d unexpected=%d"
        % (len(res.missing_keys), len(res.unexpected_keys)))
    if res.missing_keys:
        log("  missing: %s" % res.missing_keys[:10])
    if res.unexpected_keys:
        log("  unexpected: %s" % res.unexpected_keys[:10])
    model = patch_forward_with_features(model)
    return model.eval().to(device)


def phase_A_copy(device, xn):
    log("==== Phase A: 拷贝版 dinov2 (model_dir) ====")
    # 确保从 MODEL_DIR import 拷贝包, 且清除任何旧 dinov2 模块
    for m in list(sys.modules):
        if m == "dinov2" or m.startswith("dinov2."):
            del sys.modules[m]
    sys.path.insert(0, MODEL_DIR)
    # (导入可能被 torch hub 曾插入的路径干扰 -> 重建 sys.path 前面部分)
    try:
        import dinov2.models.vision_transformer as vits
    except Exception as e:
        log("IMPORT FAIL (copy): %r" % e)
        raise
    log("imported dinov2 from: %s" % os.path.dirname(os.path.dirname(vits.__file__)))
    pkg, sd = load_pkg_venc()
    model = build_copy_model(sd, device)
    with torch.no_grad():
        feats, x_norm = model.forward_with_features(xn)
    log("copy forward ok: layer keys=%s" % [k for k in feats if k.startswith("layer_")])
    outA = {
        "layer_8": feats["layer_8"].float().cpu(),
        "x_norm": x_norm.float().cpu(),
        "n_layers": len([k for k in feats if k.startswith("layer_")]),
    }
    log("copy layer_8 shape=%s x_norm shape=%s" % (tuple(outA["layer_8"].shape), tuple(outA["x_norm"].shape)))
    return outA, pkg, sd


# ---------------------------------------------------------------- phase B: 真实 hub 路径
def phase_B_reference(device, xn, sd_fp16, seed=1234):
    log("==== Phase B: 真实路径 load_dinov2 (torch.hub 本地缓存, C:/models/finals) ====")
    # 隔离: 移除拷贝版 dinov2, 避免与 hub 仓库的同名包冲突
    for m in list(sys.modules):
        if m == "dinov2" or m.startswith("dinov2.") or m == "hubconf":
            del sys.modules[m]
    sys.path = [p for p in sys.path if os.path.normcase(p) != os.path.normcase(MODEL_DIR)]
    os.chdir(C_FINALS)
    log("cwd=%s (hub cache rel: %s)" % (os.getcwd(), REF_CACHE_REL))
    if not os.path.isdir(os.path.join(REF_CACHE_REL, "facebookresearch_dinov2_main")):
        raise RuntimeError("local hub repo missing under %s" % REF_CACHE_REL)

    spec = importlib.util.spec_from_file_location("vosr_inference_mod", VOSR_FILE)
    vosr = importlib.util.module_from_spec(spec)
    sys.modules["vosr_inference_mod"] = vosr
    spec.loader.exec_module(vosr)  # 内部 torch.hub.set_dir('preset/ckpts/torch_cache') 以 cwd 解析
    log("inference_vosr_onestep imported (hub dir=%s)" % torch.hub.get_dir())

    args = types.SimpleNamespace(enc_type="dinov2b")
    enc = vosr.load_dinov2(args, device)  # fp32 hub 官方权重 -> cuda eval, forward_with_features monkey-patch 已打
    log("hub model: %s params=%.2fM dtype=%s" % (
        type(enc).__name__, sum(p.numel() for p in enc.parameters()) / 1e6,
        next(enc.parameters()).dtype))
    # 权重同源性诊断: hub fp32 -> half 应 == pkg['venc'] (fp16)
    ref_sd16 = {k: v.half() for k, v in enc.state_dict().items()}
    diff_keys = set(ref_sd16.keys()) ^ set(sd_fp16.keys())
    mae_weights = None
    if not diff_keys:
        mae_weights = max(
            (ref_sd16[k].float().cpu() - sd_fp16[k].float()).abs().mean().item() for k in ref_sd16
        )
    log("weight provenance vs pkg['venc']: keys_diff=%d max_mae=%s"
        % (len(diff_keys), ("%.3e" % mae_weights) if mae_weights is not None else "N/A"))

    enc = enc.half().eval()
    with torch.no_grad():
        feats, x_norm = enc.forward_with_features(xn)
    outB = {
        "layer_8": feats["layer_8"].float().cpu(),
        "x_norm": x_norm.float().cpu(),
        "n_layers": len([k for k in feats if k.startswith("layer_")]),
    }
    return outB


def compare(A, B):
    log("==== 对比 (fp16 前向, 相同随机输入) ====")
    mae_l8 = (A["layer_8"] - B["layer_8"]).abs().mean().item()
    max_l8 = (A["layer_8"] - B["layer_8"]).abs().max().item()
    mae_xn = (A["x_norm"] - B["x_norm"]).abs().mean().item()
    max_xn = (A["x_norm"] - B["x_norm"]).abs().max().item()
    log("layer_8  : MAE=%.4e  max_abs=%.4e  (tol %.1e)" % (mae_l8, max_l8, MAE_TOL))
    log("x_norm   : MAE=%.4e  max_abs=%.4e  (tol %.1e)" % (mae_xn, max_xn, MAE_TOL))
    log("n_layers A=%d B=%d" % (A["n_layers"], B["n_layers"]))
    ok = (mae_l8 < MAE_TOL) and (mae_xn < MAE_TOL) and (A["n_layers"] == B["n_layers"])
    log("RESULT: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    dev = wait_for_gpu()
    xn = make_input(dev)
    log("input: %s dtype=%s device=%s" % (tuple(xn.shape), xn.dtype, dev))

    outA, pkg, sd = None, None, None
    try:
        outA, pkg, sd = phase_A_copy(dev, xn)
    except Exception as e:
        log("PHASE A FAILED: %r" % e)
        import traceback
        log(traceback.format_exc())
        log("RESULT: FAIL")
        log.close()
        return 1

    try:
        outB = phase_B_reference(dev, xn, sd)
    except Exception as e:
        log("PHASE B FAILED: %r" % e)
        import traceback
        log(traceback.format_exc())
        log("RESULT: FAIL")
        log.close()
        return 1

    ok = compare(outA, outB)
    log("FINAL: %s" % ("PASS" if ok else "FAIL"))
    log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
