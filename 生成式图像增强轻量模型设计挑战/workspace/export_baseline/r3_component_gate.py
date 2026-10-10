# -*- coding: utf-8 -*-
"""EXPORT-01R3 组件门槛：先存"原参考"，再由**全新进程**加载 .pt 实测同一批组件。

用法：
  python r3_component_gate.py ref <ref_dir>        # 由构建进程调用（eager 侧参考）
  python r3_component_gate.py check <run_dir> <exp_dir> <prereg>   # 全新进程实测
判据：组件对照 max≤1e-5、mean≤1e-6、finite（t/r 嵌入经 step 的三个 (t_cur,t_nxt) 组合覆盖）。
"""
import hashlib
import json
import os
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
    import frozen_mode
    meta = {"inputs": sorted(ins), "refs": sorted(refs),
            "ref_sha256": {k: _sha(ref_dir / ("ref_%s.npy" % k)) for k in refs},
            "inputs_sha256": {k: hashlib.sha256(
                v.detach().cpu().float().numpy().tobytes()).hexdigest() for k, v in ins.items()},
            "flags": frozen_mode.read_flags(), "step_pairs": STEP_PAIRS,
            "criteria": {"max_abs": 1e-5, "mean_abs": 1e-6, "finite": True}}
    (ref_dir / "ref_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    return meta


def _repeat_selection():
    """可选削减同输入重复项（默认 None = 全量重复，必须自证重复性）。"""
    raw = os.environ.get("DELIV_GATE_REPEATS", "").strip()
    if not raw:
        return None
    return {s.strip() for s in raw.split(",") if s.strip()}


def check(run_dir, exp_dir, tol):
    """全新进程：加载 .pt，用相同输入与执行模式实测并对照。"""
    run_dir, exp_dir = Path(run_dir), Path(exp_dir)
    ref_dir = exp_dir / "component_refs"
    sub = os.environ.get("DELIV_MODEL_SUBDIR", "model_dir")   # DELIV-02 补充：FP32 回退候选目录
    recipe = json.loads((run_dir / sub / "models_export.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda")
    import frozen_mode
    cur_flags = frozen_mode.apply_frozen_mode()
    ref_meta = json.loads((ref_dir / "ref_meta.json").read_text(encoding="utf-8"))
    same, diff = frozen_mode.flags_match(ref_meta.get("flags", {}), cur_flags)
    res = {"schema": "DELIV01-component-gate/1", "checks": {}, "diffs": {},
           "execution_mode": {"jit_optimized_execution": recipe.get("runtime", {}).get(
               "jit_optimized_execution", False), "flags": cur_flags,
               "reference_flags": ref_meta.get("flags", {}), "flags_match": same,
               "flags_diff": diff},
           "model_subdir": sub,
           "model_sha256": _sha(run_dir / sub / "your_model.pt")}
    if not same:
        res["verdict"] = "INVALID_PROTOCOL"
        (exp_dir / "component_gate.json").write_text(
            json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        raise SystemExit("INVALID_PROTOCOL: 执行开关不一致 %s" % diff)
    optimize = bool(res["execution_mode"]["jit_optimized_execution"])
    model = torch.jit.load(str(run_dir / sub / "your_model.pt"), map_location=dev).eval()
    dtype = torch.float16 if recipe["weight_dtype"] == "float16" else torch.float32
    ins = {p.stem[3:]: torch.from_numpy(np.load(p, allow_pickle=False)).to(dev).to(dtype)
           for p in sorted(ref_dir.glob("in_*.npy"))}
    cur_fp = {k: hashlib.sha256(v.detach().cpu().float().numpy().tobytes()).hexdigest()
              for k, v in ins.items()}
    res["input_fingerprints"] = cur_fp
    bad_inputs = sorted(k for k, v in cur_fp.items()
                        if ref_meta.get("inputs_sha256", {}).get(k) != v)
    if bad_inputs:
        res["verdict"] = "INVALID_PROTOCOL"
        (exp_dir / "component_gate.json").write_text(
            json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        raise SystemExit("INVALID_PROTOCOL: 输入指纹不一致 %s" % bad_inputs)
    ok = True

    ck_dir = exp_dir / "component_checks"
    ck_dir.mkdir(parents=True, exist_ok=True)

    def cmp(name, got, ref):
        nonlocal ok
        g = got.detach().float().cpu().numpy()
        np.save(ck_dir / ("%s_check.npy" % name), g)       # §0E：检查侧原始输出必须留档
        np.save(ck_dir / ("%s_ref.npy" % name), np.asarray(ref, dtype=np.float32))
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
            # §0E：重复性必须覆盖 enc/dec（不能只用 step 代表）
            sel = _repeat_selection()
            res["repeat_selection"] = (sorted(sel) if sel is not None else "ALL")
            for key, rname, fn in (("enc_512", "enc_512_repeat_same_input",
                                    lambda: model.encode(ins["enc_512"], 512)),
                                   ("enc_1024", "enc_1024_repeat_same_input",
                                    lambda: model.encode(ins["enc_1024"], 1024)),
                                   ("dec_64", "dec_64_repeat_same_input",
                                    lambda: model.decode(ins["dec_64"], 64)),
                                   ("dec_128", "dec_128_repeat_same_input",
                                    lambda: model.decode(ins["dec_128"], 128)),
                                   ("dino", "dino_repeat_same_input",
                                    lambda: model.features(ins["dino_512"])),
                                   ("wavelet", "wavelet_repeat_same_input",
                                    lambda: model.post(ins["wav_t"], ins["wav_s"]))):
                if sel is not None and key not in sel:
                    res.setdefault("skipped_repeats", []).append(rname)
                    continue
                a = fn()
                b = fn()
                dd = (b.detach().float() - a.detach().float()).abs()
                rep = {"max_abs": float(dd.max()), "mean_abs": float(dd.mean()),
                       "same_input": True, "passed": bool(
                           float(dd.max()) <= tol["same_precision_max_abs"] and
                           float(dd.mean()) <= tol["same_precision_mean_abs"])}
                res["diffs"][rname] = rep
                ok = ok and rep["passed"]
            for (tc, tn) in STEP_PAIRS:
                name = "step_t%s_r%s" % (str(tc).replace(".", "p"), str(tn).replace(".", "p"))
                t_cur = torch.full((1,), tc, device=dev, dtype=dtype)
                t_nxt = torch.full((1,), tn, device=dev, dtype=dtype)
                out1 = model.step(ins["latent"], ins["noise"], t_cur, t_nxt, ins["feats"])
                cmp(name, out1, np.load(ref_dir / ("ref_%s.npy" % name), allow_pickle=False))
                if sel is not None and "step" not in sel:
                    res.setdefault("skipped_repeats", []).append(name + "_repeat_same_input")
                    continue
                out2 = model.step(ins["latent"], ins["noise"], t_cur, t_nxt, ins["feats"])
                dd = (out2.detach().float() - out1.detach().float()).abs()
                rep = {"max_abs": float(dd.max()), "mean_abs": float(dd.mean()),
                       "same_input": True, "passed": bool(
                           float(dd.max()) <= tol["same_precision_max_abs"] and
                           float(dd.mean()) <= tol["same_precision_mean_abs"])}
                res["diffs"][name + "_repeat_same_input"] = rep
                ok = ok and rep["passed"]
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
