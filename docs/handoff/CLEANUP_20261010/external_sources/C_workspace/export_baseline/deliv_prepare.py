# -*- coding: utf-8 -*-
"""DELIV-01 准备（CPU）：只读复用 R3 的 .pt，形成本包新 bundle 身份与预登记。"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R3_PT = Path("C:/models/exports/EXPORT01R3_20260917_204635/model_dir/your_model.pt")
R3_SHA = "c3946096bcda7513d1a100975f73e08aa8f6ce24f4c7837642484038dd3f543d"


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    ev = Path(sys.argv[2]).resolve()
    mdl = run / "model_dir"
    mdl.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "DELIV01-prepare/1", "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    src_sha, src_bytes = sha256_file(R3_PT), R3_PT.stat().st_size
    if src_sha != R3_SHA:
        raise SystemExit("R3 模型身份不符：%s" % src_sha)
    dst = mdl / "your_model.pt"
    shutil.copyfile(R3_PT, dst)
    if (sha256_file(dst), dst.stat().st_size) != (src_sha, src_bytes):
        raise SystemExit("复制后字节不一致")
    shutil.copyfile(HERE / "delivery_runner.py", mdl / "runner.py")
    base_recipe = json.loads((R3_PT.parent / "models_export.json").read_text(encoding="utf-8"))
    recipe = dict(base_recipe)
    recipe.setdefault("runtime", {})
    recipe["runtime"].update({"jit_optimized_execution": False,
                              "deterministic_algorithms": "True(warn_only=False)",
                              "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                              "cudnn_benchmark": False, "cudnn_deterministic": True,
                              "cudnn_allow_tf32": False, "matmul_allow_tf32": False})
    (mdl / "models_export.json").write_text(json.dumps(recipe, indent=1, ensure_ascii=False),
                                            encoding="utf-8")
    runtime_sha = {n: sha256_file(HERE / n) for n in
                   ("delivery_runner.py", "r_verify.py", "r3_component_gate.py",
                    "frozen_mode.py", "r_ledger.py", "r_supervisor.py", "gen_images.py",
                    "make_zip.py", "isolated_precheck.py") if (HERE / n).is_file()}
    bid = "DELIV01-" + hashlib.sha256(json.dumps(
        {"model_sha256": src_sha, "runtime_sha256": runtime_sha, "batch": "DELIV01"},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:32]
    manifest = {"schema_version": "DELIV01-bundle-manifest/1", "bundle_id": bid,
                "created_at": rec["at"],
                "model_reuse": {"from_batch": "EXPORT-01R3", "hardlink": False,
                                "byte_identical": True,
                                "path": str(R3_PT), "sha256": src_sha, "bytes": src_bytes},
                "files": [{"path": "your_model.pt", "role": "jit_model", "sha256": src_sha,
                           "bytes": src_bytes},
                          {"path": "runner.py", "role": "delivery_runner",
                           "sha256": runtime_sha["delivery_runner.py"],
                           "bytes": (mdl / "runner.py").stat().st_size},
                          {"path": "models_export.json", "role": "recipe",
                           "sha256": sha256_file(mdl / "models_export.json")}],
                "runtime_source_sha256": runtime_sha,
                "execution_mode": recipe["runtime"],
                "criteria": {"component_max_abs": 1e-5, "component_mean_abs": 1e-6,
                             "pair_max_abs": 1e-5, "pair_mean_abs": 1e-6,
                             "migration_max_abs": 0.05, "migration_mean_abs": 0.002},
                "not_claimed": ["官方兼容", "目标环境实测", "v7 训练资格"]}
    (ev / "bundle_manifest_deliv01.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    rec.update({"bundle_id": bid, "model_sha256": src_sha, "model_bytes": src_bytes})
    (ev / "prepare_deliv01.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    print(json.dumps({"bundle_id": bid, "model_sha256": src_sha, "bytes": src_bytes},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
