# -*- coding: utf-8 -*-
"""DELIV-02 CPU 前置检查：8 位恢复语义、256 色阶穷举、真实数组残差、后处理参考重放、负例。

用法: python deliv02_cpu_checks.py <out_json> [<evidence_dir_with_npy>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_PB = HERE.parent / "pipeline_baseline"
if str(_PB) not in sys.path:
    sys.path.insert(0, str(_PB))
import pixel_io  # noqa: E402


def main():
    out = Path(sys.argv[1])
    ev = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    res = {"schema": "DELIV02-cpu-checks/1", "checks": {}}
    ok = True

    # 1) 256 色阶穷举往返：u8 -> [-1,1] -> u8 必须恒等
    levels = np.arange(256, dtype=np.uint8).reshape(1, 1, 256)
    y = (levels.astype(np.float64) / 127.5) - 1.0
    yt = torch.from_numpy(np.repeat(y, 3, axis=0)).unsqueeze(0).float()
    back = pixel_io.to_u8(yt)
    rt_ok = bool(np.array_equal(np.asarray(back)[0, :, 0], levels[0, 0]))
    res["checks"]["level_roundtrip_256"] = {"passed": rt_ok,
                                            "note": "u8->[-1,1]->u8 恒等（最近色阶）"}
    ok = ok and rt_ok

    # 2) 网格残差：已量化输入残差应为 0；任意连续输出残差大，必须被拒绝
    r1 = pixel_io.grid_residual(yt)
    off_grid = torch.full((1, 3, 4, 4), float(128.5 / 127.5 - 1.0))   # 恰好半色阶：非网格
    r2 = pixel_io.grid_residual(off_grid)
    rejected = False
    try:
        pixel_io.to_u8(off_grid)
    except pixel_io.PixelIOError:
        rejected = True
    res["checks"]["grid_residual"] = {"on_grid": r1, "off_grid": r2,
                                      "off_grid_rejected": rejected, "passed": rejected}
    ok = ok and rejected

    # 3) 非有限/越界负例
    bad = [torch.full((1, 3, 4, 4), float("nan")), torch.full((1, 3, 4, 4), 2.0)]
    bad_rejected = 0
    for b in bad:
        try:
            pixel_io.to_u8(b)
        except pixel_io.PixelIOError:
            bad_rejected += 1
    res["checks"]["nonfinite_and_range"] = {"rejected": bad_rejected, "of": len(bad),
                                            "passed": bad_rejected == len(bad)}
    ok = ok and bad_rejected == len(bad)

    # 4) 真实已存数组：网格残差与 A/B 差异（若提供证据目录）
    real = {}
    if ev is not None and (ev / "npy").is_dir():
        for name in ("A_512_r1", "A_512_r2", "A_512_r3", "A_511x513"):
            p = ev / "npy" / ("%s.npy" % name)
            if not p.is_file():
                continue
            arr = torch.from_numpy(np.load(p, allow_pickle=False))
            real[name] = pixel_io.grid_residual(arr)
        pair = {}
        if (ev / "npy" / "A_512_r1.npy").is_file() and (ev / "npy" / "A_512_r2.npy").is_file():
            a = pixel_io.to_u8(torch.from_numpy(np.load(ev / "npy" / "A_512_r1.npy",
                                                  allow_pickle=False)))
            b = pixel_io.to_u8(torch.from_numpy(np.load(ev / "npy" / "A_512_r2.npy",
                                                  allow_pickle=False)))
            pair = pixel_io.u8_delta(a, b)
        res["checks"]["real_arrays"] = {"grid_residual": real, "A1_vs_A2_u8": pair}
        same_ok = bool(pair.get("identical"))
        res["checks"]["real_arrays"]["passed"] = same_ok
        ok = ok and same_ok

    # 5) 后处理参考重放：同一 decode_raw + 同一 source 必须产生相同 u8；
    #    两个不同 decode_raw 的差异必须在量化前后都被记录
    import core as PBcore
    from chain import WaveletFix
    wav = WaveletFix(5.0)
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    src = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
    d1 = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
    d2 = (d1 + 1e-6).clamp(-1, 1)
    o1 = wav(d1, src)
    o1b = wav(d1, src)
    o2 = wav(d2, src)
    rep = {"same_input_identical": bool(torch.equal(o1, o1b)),
           "perturbed_decode_fp32_max": float((o1 - o2).abs().max()),
           "perturbed_u8_identical": bool(np.array_equal(pixel_io.to_u8(o1), pixel_io.to_u8(o2))),
           "u8_delta": pixel_io.u8_delta(pixel_io.to_u8(o1), pixel_io.to_u8(o2))}
    rep["passed"] = rep["same_input_identical"]
    res["checks"]["postprocess_replay"] = rep
    ok = ok and rep["passed"]

    res["verdict"] = "PASS" if ok else "FAIL"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"verdict": res["verdict"],
                      "checks": {k: v.get("passed") for k, v in res["checks"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
