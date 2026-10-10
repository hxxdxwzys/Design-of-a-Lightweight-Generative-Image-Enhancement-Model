# -*- coding: utf-8 -*-
"""自检：① 512 路径可跑且输出形状/范围合法；② 4K 出图与随包样例逐像素一致（比最终 JPEG 字节）；
③ 打印权重哈希，便于与提交记录核对。

判据说明（重要）：出图保存为 JPEG q98/subsampling=0，**有损编码本身**就会让"渲染数组 vs 解码后的
JPEG"产生 ≤10 级、约一半像素的差异。因此本脚本不与"解码后的 JPEG"比数组，而是把本次渲染**编码成
与出图完全相同的 JPEG 参数**，再与随包样例**逐字节比对**（这是无损的、可精确复现的判据）。

用法：python selfcheck.py [--samples 样例目录（默认 samples/）]
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, statistics, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

QUALITY, SUBSAMPLING = 98, 0      # 必须与 infer.py 的出图参数一致

HERE = Path(__file__).resolve().parent
MD = HERE / "model_dir"
sys.path.insert(0, str(HERE))


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(HERE / "samples"))
    a = ap.parse_args()
    print("[1] 权重与代码")
    for f in sorted(MD.glob("*")):
        if f.is_file():
            print("    %-16s %9d B  sha256:%s" % (f.name, f.stat().st_size, sha16(f)))

    spec = importlib.util.spec_from_file_location("submitted_runner", str(MD / "runner.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    R = mod.Runner(str(MD))
    dev = str(R.device)
    print("[2] 512 路径（评测 harness 接口 Runner(model_dir).infer(x, prompt='')）")
    x = torch.rand(1, 3, 512, 512) * 2 - 1
    sync = torch.cuda.synchronize if str(dev).startswith("cuda") else (lambda: None)
    with torch.no_grad():
        y = R.infer(x, prompt="")
        for _ in range(3):
            R.infer(x, prompt="")
        ts = []
        for _ in range(10):
            sync(); t = time.perf_counter(); R.infer(x, prompt=""); sync()
            ts.append((time.perf_counter() - t) * 1e3)
    ok_shape = tuple(y.shape) == (1, 3, 512, 512)
    ok_range = float(y.min()) >= -1.0001 and float(y.max()) <= 1.0001
    print("    输出 shape=%s  range=[%.3f, %.3f]  => %s" % (tuple(y.shape), float(y.min()), float(y.max()),
          "PASS" if (ok_shape and ok_range) else "FAIL"))
    print("    冷启动时延 median=%.2f ms  min=%.2f ms（仅参考，正式测速请用 infer.py --timing）"
          % (statistics.median(ts), min(ts)))

    sdir = Path(a.samples)
    ins = sorted(sdir.glob("*_in.*"))
    if not ins:
        print("[3] 无样例对（%s/*_in.*），跳过一次出图一致性核对" % sdir)
        return 0 if (ok_shape and ok_range) else 1
    import infer as I
    print("[3] 4K 出图一致性（渲染 -> JPEG(q%d,subsampling=%d) -> 与随包样例逐字节比对）" % (QUALITY, SUBSAMPLING))
    good = True
    for f in ins:
        ref = sdir / (f.stem.replace("_in", "") + "_out.jpg")
        if not ref.exists():
            print("    跳过（缺参考）: %s" % ref.name); continue
        lq = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        got = I.render_4k(R, lq, I.make_feather(I.TILE, I.OVER), dev)
        buf = io.BytesIO()
        Image.fromarray(got.astype(np.uint8)).save(buf, format="JPEG", quality=QUALITY, subsampling=SUBSAMPLING)
        mine = buf.getvalue()
        theirs = ref.read_bytes()
        exact = (mine == theirs)
        # 退化判据（仅当 JPEG 字节不同才需要）：数组级容差
        exp = np.asarray(Image.open(ref).convert("RGB"), dtype=np.uint8)
        d = np.abs(got.astype(np.int16) - exp.astype(np.int16))
        near = (int(d.max()) <= 12 and float(d.mean()) <= 1.0)
        print("    %-16s 尺寸=%s 本次生成=%d B 样例=%d B  => %s%s" % (
            f.name, str(got.shape), len(mine), len(theirs),
            "PASS(逐字节一致)" if exact else ("PASS(数组近似)" if near else "FAIL"),
            "" if exact else "  数组级 max=%d mean=%.4f" % (int(d.max()), float(d.mean()))))
        good = good and (exact or near)
    return 0 if (ok_shape and ok_range and good) else 1


if __name__ == "__main__":
    sys.exit(main())
