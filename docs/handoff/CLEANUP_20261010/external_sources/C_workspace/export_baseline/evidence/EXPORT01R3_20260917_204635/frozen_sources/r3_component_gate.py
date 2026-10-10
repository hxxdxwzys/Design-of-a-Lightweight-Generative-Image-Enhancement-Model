# -*- coding: utf-8 -*-
"""EXPORT-01R3 组件门槛：先存"原参考"，再由**全新进程**加载 .pt 实测同一批组件。

用法：
  python r3_component_gate.py ref <ref_dir>        # 由构建进程调用（eager 侧参考）
  python r3_component_gate.py check <run_dir> <exp_dir> <prereg>   # 全新进程实测
判据：组件对照 max≤1e-5、mean≤1e-6、finite（t/r 嵌入经 step 的三个 (t_cur,t_nxt) 组合覆盖）。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_PB = HERE.parent / "pipeline_baseline"
if str(_PB) not in sys.path:
    sys.path.insert(0, str(_PB))

STEP_PAIRS = [(1.0, 0.0), (0.5, 0.0), (1.0, 0.5)]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_inputs(dev, dtype):
    """确定性组件输入（CPU 端生成后落盘，供全新进程复用）。"""
    g = torch.Generator(device="cpu")
    g.manual_seed(4242)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.2).to(dtype).to(dev)

    ins = {"enc_512": rnd(1, 3, 512, 512), "enc_1024": rnd(1, 3, 1024, 1024),
           "dino_512": rnd(1, 3, 512, 512), "latent": rnd(1, 4, 64, 64),
           "noise": rnd(1, 4, 64, 64), "feats": rnd(1, 1024, 768),
           "dec_64": rnd(1, 4, 64, 64), "dec_128": rnd(1, 4, 128, 128),
           "wav_t": rnd(1, 3, 64, 64), "wav_s": rnd(1, 3, 64, 64)}
    return ins


def capture_references(wraps, dev, ref_dir, ledger, dtype):
    """构建进程内：计算 eager 参考并落盘（每次真实前向先扣后跑）。"""
    ref_dir = Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    ins = make_inputs(dev, dtype)
    for k, v in ins.items():
        np.save(ref_dir / ("in_%s.npy" % k), v.detach().cpu().float().numpy())
    enc, dec, dino, dit, wav = wraps
    refs = {}
    calls = [("enc_512", lambda: enc(ins["enc_512"])),
             ("enc_1024", lambda: enc(ins["enc_1024"])),
             ("dino_512", lambda: dino(ins["dino_512"])),
             ("dec_64", lambda: dec(ins["dec_64"])),
             ("dec_128", lambda: dec(ins["dec_128"])),
             ("wavelet", lambda: wav(ins["wav_t"], ins["wav_s"]))]
    for name, fn in calls:
        ledger.reserve("component", "ref_%s" % name)
        with torch.no_grad():
            refs[name] = fn().detach().cpu().float().numpy()
        np.save(ref_dir / ("ref_%s.npy" % name), refs[name])
    for (tc, tn) in STEP_PAIRS:
        name = "step_t%s_r%s" % (str(tc).replace(".", "p"), str(tn).replace(".", "p"))
        ledger.reserve("component", "ref_%s" % name)
        t_cur = torch.full((1,), tc, device=dev, dtype=dtype)
        t_nxt = torch.full((1,), tn, device=dev, dtype=dtype)
        with torch.no_grad():
            refs[name] = dit(ins["latent"], ins["noise"], t_cur, t_nxt, ins["feats"]) \
                .detach().cpu().float().numpy()
        np.save(ref_dir / ("ref_%s.npy" % name), refs[name])
    return {"inputs": sorted(ins), "refs": sorted(refs),
            "ref_sha256": {k: _sha(ref_dir / ("ref_%s.npy" % k)) for k in refs}}


def check(run_dir, exp_dir, tol):
    """全新进程：加载 .pt，用相同输入与执行模式实测并对照。"""
    run_dir, exp_dir = Path(run_dir), Path(exp_dir)
    ref_dir = exp_dir / "component_refs"
    recipe = json.loads((run_dir / "model_dir" / "models_export.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda")
    optimize = bool(recipe.get("runtime", {}).get("jit_optimized_execution", False))
    model = torch.jit.load(str(run_dir / "model_dir" / "your_model.pt"), map_location=dev).eval()
    dtype = torch.float16 if recipe["weight_dtype"] == "float16" else torch.float32
    ins = {p.stem[3:]: torch.from_numpy(np.load(p, allow_pickle=False)).to(dev).to(dtype)
           for p in sorted(ref_dir.glob("in_*.npy"))}
    res = {"schema": "EXPORT01R3-component-gate/1", "checks": {}, "diffs": {},
           "execution_mode": {"jit_optimized_execution": optimize},
           "model_sha256": _sha(run_dir / "model_dir" / "your_model.pt")}
    ok = True

    def cmp(name, got, ref):
        nonlocal ok
        g = got.detach().float().cpu().numpy()
        d = np.abs(g - ref)
        finite = bool(np.isfinite(g).all())
        rec = {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "finite": finite,
               "shape": list(g.shape)}
        passed = (rec["max_abs"] <= tol["same_precision_max_abs"] and
                  rec["mean_abs"] <= tol["same_precision_mean_abs"] and finite)
        rec["passed"] = passed
        res["diffs"][name] = rec
        ok = ok and passed

    with torch.jit.optimized_execution(optimize):
        with torch.no_grad():
            cmp("enc_512", model.encode(ins["enc_512"], 512),
                np.load(ref_dir / "ref_enc_512.npy", allow_pickle=False))
            cmp("enc_1024", model.encode(ins["enc_1024"], 1024),
                np.load(ref_dir / "ref_enc_1024.npy", allow_pickle=False))
            cmp("dino_512", model.features(ins["dino_512"]),
                np.load(ref_dir / "ref_dino_512.npy", allow_pickle=False))
            cmp("dec_64", model.decode(ins["dec_64"], 64),
                np.load(ref_dir / "ref_dec_64.npy", allow_pickle=False))
            cmp("dec_128", model.decode(ins["dec_128"], 128),
                np.load(ref_dir / "ref_dec_128.npy", allow_pickle=False))
            cmp("wavelet", model.post(ins["wav_t"], ins["wav_s"]),
                np.load(ref_dir / "ref_wavelet.npy", allow_pickle=False))
            first = None
            for (tc, tn) in STEP_PAIRS:
                name = "step_t%s_r%s" % (str(tc).replace(".", "p"), str(tn).replace(".", "p"))
                t_cur = torch.full((1,), tc, device=dev, dtype=dtype)
                t_nxt = torch.full((1,), tn, device=dev, dtype=dtype)
                out = model.step(ins["latent"], ins["noise"], t_cur, t_nxt, ins["feats"])
                cmp(name, out, np.load(ref_dir / ("ref_%s.npy" % name), allow_pickle=False))
                if first is None:
                    first = out.detach().clone()
                else:
                    dd = (out.detach().float() - first.float()).abs()
                    res["diffs"][name + "_vs_first_call"] = {
                        "max_abs": float(dd.max()), "mean_abs": float(dd.mean()),
                        "passed": bool(float(dd.max()) <= tol["same_precision_max_abs"])}
                    ok = ok and res["diffs"][name + "_vs_first_call"]["passed"]
    res["finite_all"] = all(v.get("finite", True) for v in res["diffs"].values())
    res["verdict"] = "PASS" if ok else "FAIL"
    (exp_dir / "component_gate.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                 encoding="utf-8")
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        tol = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))["tolerances"]
        r = check(sys.argv[2], sys.argv[3], tol)
        print(json.dumps({"verdict": r["verdict"],
                          "max_abs": max(v.get("max_abs", 0.0) for v in r["diffs"].values())},
                         ensure_ascii=False))
        sys.exit(0 if r["verdict"] == "PASS" else 1)
    raise SystemExit("unknown mode")
