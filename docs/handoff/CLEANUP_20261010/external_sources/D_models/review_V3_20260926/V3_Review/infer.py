"""V3 review entry. Numerical path preserves opt11_render_pkg.py (historical).

CUDA only. Original model_dir/runner.py and your_model.pt are byte-preserved.
No reference image is read by this production entry.
"""
from pathlib import Path
import argparse
import importlib.util
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
TILE, OVER = 512, 32


def load_runner(model_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("V3 graph requires CUDA. CPU forward is not supported by this review package.")
    spec = importlib.util.spec_from_file_location("v3_original_runner", str(Path(model_dir) / "runner.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Runner(str(model_dir))


def positions(n):
    if n < TILE:
        raise ValueError("Historical renderer requires both dimensions >=512; no implicit resize/padding.")
    return sorted(set(list(range(0, max(1, n-TILE+1), TILE-OVER)) + [max(0, n-TILE)]))


def make_feather():
    mask = np.ones((TILE, TILE, 1), dtype=np.float32)
    r = OVER // 2
    ramp = np.linspace(0, 1, r, dtype=np.float32)
    mask[:r, :, 0] *= ramp[:, None]
    mask[-r:, :, 0] *= ramp[::-1, None]
    mask[:, :r, 0] *= ramp[None, :]
    mask[:, -r:, 0] *= ramp[::-1][None, :]
    return mask


@torch.no_grad()
def render_4k(runner, img):
    H, W = img.shape[:2]
    ys, xs = positions(H), positions(W)
    mask = make_feather()
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    for y0 in ys:
        for x0 in xs:
            patch = img[y0:y0+TILE, x0:x0+TILE]
            # Keep normalization on CPU, as in the historical V3 renderer.
            t = torch.from_numpy(patch).permute(2, 0, 1)[None].float().div_(127.5).sub_(1)
            y = runner.infer(t, prompt="")
            if not bool(torch.isfinite(y).all()):
                raise ValueError("Non-finite model output")
            o = y[0].permute(1, 2, 0).float().cpu().numpy()
            if o.shape != (TILE, TILE, 3):
                raise ValueError("Unexpected model output shape")
            acc[y0:y0+TILE, x0:x0+TILE] += np.clip((o+1)*127.5, 0, 255)*mask
            wsum[y0:y0+TILE, x0:x0+TILE] += mask
    # Deliberately retain original outer-edge zero weights and uint8 truncation.
    return np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    from selfcheck import verify_identity
    verify_identity(HERE)
    src, dst = Path(args.input), Path(args.output)
    files = [src] if src.is_file() else sorted(src.glob("*.jpg"))
    if not files:
        raise ValueError("No input images")
    dst.mkdir(parents=True, exist_ok=True)
    if any((dst/f.name).exists() for f in files):
        raise FileExistsError("Refusing to overwrite existing outputs")
    runner = load_runner(HERE / "model_dir")
    for f in files:
        arr = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        out = render_4k(runner, arr)
        Image.fromarray(out).save(dst/f.name, quality=98, subsampling=0)
        print(f.name, "DONE", flush=True)


if __name__ == "__main__":
    main()
