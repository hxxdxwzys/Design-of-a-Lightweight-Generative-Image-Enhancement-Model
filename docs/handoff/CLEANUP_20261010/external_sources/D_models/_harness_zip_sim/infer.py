# -*- coding: utf-8 -*-
"""赛道二 · 轻量一步扩散图像增强 —— 独立推理工程（4K 分块出图 + 512 测速）。

用法
----
# 1) 对 4K 图像目录出图（默认 512 块 + 32 重叠 + 羽化融合，保存 JPEG q98）
python infer.py --input  <输入目录> --output <输出目录> [--pattern "*.jpg"] [--limit N]

# 2) 官方口径 512x512 单卡时延自测（预热 + 多次取中位/最小）
python infer.py --timing [--iters 30 --warmup 5]

依赖：torch（GPU 版；无 CUDA 时自动回退 CPU，仅用于功能验证）、numpy、Pillow
模型：model_dir/your_model.pt（TorchScript, fp16）+ model_dir/runner.py（评测 harness 统一接口）
"""
from __future__ import annotations
import argparse, importlib.util, os, statistics, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

TILE, OVER = 512, 32
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model_dir"


def load_runner(model_dir: Path):
    spec = importlib.util.spec_from_file_location("submitted_runner", str(model_dir / "runner.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Runner(str(model_dir))


def make_feather(tile: int, over: int) -> np.ndarray:
    mask = np.ones((tile, tile, 1), dtype=np.float32)
    r = over // 2
    if r > 0:
        ramp = np.linspace(0.0, 1.0, r, dtype=np.float32)
        mask[:r, :, 0] *= ramp[:, None]
        mask[-r:, :, 0] *= ramp[::-1, None]
        mask[:, :r, 0] *= ramp[None, :]
        mask[:, -r:, 0] *= ramp[::-1][None, :]
    return mask


@torch.no_grad()
def render_4k(runner, lq: np.ndarray, mask: np.ndarray, device: str) -> np.ndarray:
    """整图分块推理 + 羽化融合（同一模型主体、同一代码路径）。"""
    H, W = lq.shape[:2]
    t = min(TILE, H, W)
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    m = mask if t == TILE else np.ones((t, t, 1), dtype=np.float32)
    ys = sorted(set(list(range(0, max(1, H - t + 1), max(1, t - OVER))) + [max(0, H - t)]))
    xs = sorted(set(list(range(0, max(1, W - t + 1), max(1, t - OVER))) + [max(0, W - t)]))
    for y0 in ys:
        for x0 in xs:
            patch = lq[y0:y0 + t, x0:x0 + t]
            x = torch.from_numpy(patch.copy()).permute(2, 0, 1)[None].to(device).float().div_(127.5).sub_(1.0)
            y = runner.infer(x, prompt="")
            out = y[0].permute(1, 2, 0).float().clamp(-1, 1).cpu().numpy()
            acc[y0:y0 + t, x0:x0 + t] += np.clip((out + 1.0) * 127.5, 0, 255) * m
            wsum[y0:y0 + t, x0:x0 + t] += m
    return np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)


def run_timing(runner, device: str, iters: int, warmup: int, preheat_s: float) -> int:
    """官方口径 512×512 时延自测。

    注意：**每次调用前后各做一次 GPU 同步**。若不同步，perf_counter 只能量到 kernel 入队时间
    （0.2 ms 量级），那是 CPU 提交耗时而不是计算耗时，会给出物理上不可能的最小值。
    """
    x = torch.rand(1, 3, 512, 512, device=device) * 2 - 1
    cuda = str(device).startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    with torch.no_grad():
        t0 = time.time()
        while time.time() - t0 < preheat_s:      # 预热：稳定 GPU 热状态
            runner.infer(x, prompt="")
        for _ in range(warmup):
            runner.infer(x, prompt="")
        ts = []
        for _ in range(iters):
            sync(); t = time.perf_counter()
            runner.infer(x, prompt="")
            sync(); ts.append((time.perf_counter() - t) * 1e3)
    s = sorted(ts)
    print("[timing] 512x512 device=%s  median=%.2f ms  min=%.2f ms  p90=%.2f ms  n=%d"
          % (device, statistics.median(s), s[0], s[min(len(s) - 1, int(0.9 * len(s)))], len(s)))
    print("[timing] 口径：预热 %.0f s + %d 次预热 + %d 次计时，每次调用前后各一次 GPU 同步"
          % (preheat_s, warmup, iters))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="", help="输入目录或单张图")
    ap.add_argument("--output", default="", help="输出目录")
    ap.add_argument("--pattern", default="*.jpg")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quality", type=int, default=98)
    ap.add_argument("--timing", action="store_true", help="只做 512 时延自测")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--preheat", type=float, default=30.0)
    a = ap.parse_args()

    if not MODEL_DIR.exists():
        print("[error] 缺少 model_dir/（应含 your_model.pt 与 runner.py）", file=sys.stderr)
        return 2
    runner = load_runner(MODEL_DIR)
    device = str(runner.device) if hasattr(runner, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    print("[env] torch=%s cuda=%s device=%s" % (torch.__version__, torch.cuda.is_available(), device), flush=True)

    if a.timing:
        return run_timing(runner, device, a.iters, a.warmup, a.preheat)

    src = Path(a.input)
    if not src.exists():
        print("[error] 输入不存在: %s" % src, file=sys.stderr)
        return 2
    files = [src] if src.is_file() else sorted(src.glob(a.pattern))
    if a.limit:
        files = files[:a.limit]
    if not files:
        print("[error] 未找到匹配图像: %s/%s" % (src, a.pattern), file=sys.stderr)
        return 2
    out_dir = Path(a.output or (HERE / "output_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = make_feather(TILE, OVER)
    t0 = time.time()
    for i, f in enumerate(files, 1):
        lq = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        res = render_4k(runner, lq, mask, device)
        Image.fromarray(res).save(out_dir / (f.stem + ".jpg"), quality=a.quality, subsampling=0)
        if i % 5 == 0 or i == len(files):
            print("  %d/%d  %.2f s/张  累计 %.0f s" % (i, len(files), (time.time() - t0) / i, time.time() - t0), flush=True)
    print("[done] %d 张 -> %s" % (len(files), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
