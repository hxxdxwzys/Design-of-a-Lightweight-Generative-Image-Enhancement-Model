"""Strict V3 identity and exact reproduction checks; never accept approximate matches."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
EXPECTED_MODEL = "421be231d504b1e1d20656ff7aa1c45a377434ff4762cc79c05416a2c7b5306d"
EXPECTED_RUNNER = "8e434e0240eed20b924e04f5f7338a037c8ec7821b8067159fe50ac7492ff70b"
EXPECTED_SOURCE = "d3bf0bf602e9d2ec6459ace57753fd0188189aaacf7b1ca36fda5e1d590ca712"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def safe_path(root, rel):
    root = Path(root).resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root) or p == root:
        raise ValueError("Unsafe manifest path")
    return p


def verify_identity(root=HERE):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if m["source_zip"]["sha256"] != EXPECTED_SOURCE:
        raise ValueError("Wrong V3 source identity")
    mandatory = {"model_dir/your_model.pt", "model_dir/runner.py", "infer.py", "selfcheck.py",
                 "samples/case3_in.jpg", "samples/case3_out.jpg", "samples/case17_in.jpg", "samples/case17_out.jpg"}
    if not mandatory.issubset(m["files"]):
        raise ValueError("Missing mandatory manifest entries")
    for rel, entry in m["files"].items():
        p = safe_path(root, rel)
        if not p.is_file() or p.stat().st_size != entry["bytes"] or sha(p) != entry["sha256"]:
            raise ValueError("Missing or changed file: " + rel)
    for rel, expected in [("model_dir/your_model.pt", EXPECTED_MODEL), ("model_dir/runner.py", EXPECTED_RUNNER)]:
        if sha(root / rel) != expected:
            raise ValueError("Not the V3 original: " + rel)
    if m["cases"] != ["case3", "case17"]:
        raise ValueError("Fixed case list changed")
    return m


def compare_jpeg(generated, reference):
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(generated)).convert("RGB"))
    b = np.asarray(Image.open(io.BytesIO(reference)).convert("RGB"))
    same_shape = a.shape == b.shape
    d = np.abs(a.astype(np.int16)-b.astype(np.int16)) if same_shape else None
    return {"jpeg_byte_equal": generated == reference,
            "decoded_rgb_equal": bool(same_shape and np.array_equal(a, b)),
            "decoded_max_abs_diff": int(d.max()) if same_shape else None,
            "decoded_mean_abs_diff": float(d.mean()) if same_shape else None,
            "generated_sha256": hashlib.sha256(generated).hexdigest(),
            "reference_sha256": hashlib.sha256(reference).hexdigest()}


def environment():
    import torch, numpy, PIL
    from PIL import features
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": numpy.__version__, "Pillow": PIL.__version__, "libjpeg": features.version("jpg"),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


class Ledger:
    """Single supervised worker only; reserve before calls; no refunds on failure."""
    def __init__(self, path):
        self.path = Path(path)

    def reserve(self, n):
        d = json.loads(self.path.read_text(encoding="utf-8"))
        if time.time() >= d["deadline_epoch"]:
            raise RuntimeError("Wall-clock budget exceeded")
        if n < 1 or d["reserved_calls"] + n > d["call_cap"]:
            raise RuntimeError("Call budget exceeded")
        d["reserved_calls"] += n
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)


def gpu_check(output, ledger_path):
    import numpy as np
    import torch
    from PIL import Image
    import infer
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "FAIL", "cases": {}, "started_epoch": time.time(),
              "scope": "two fixed samples, not all 100 historical outputs"}
    try:
        manifest = verify_identity()
        report["manifest_sha256"] = sha(HERE/"manifest.json")
        report["environment"] = environment()
        ledger = Ledger(ledger_path)
        ledger.reserve(4)  # Original Runner: 3 constructor warmups + 1 graph capture.
        runner = infer.load_runner(HERE / "model_dir")
        report["cuda_graph_active"] = runner.graph is not None

        class Counted:
            def infer(self, x, prompt=""):
                ledger.reserve(1)
                return runner.infer(x, prompt=prompt)

        counted = Counted()
        x = torch.zeros(1, 3, 512, 512)
        with torch.no_grad():
            y = counted.infer(x)
        report["smoke_512"] = bool(tuple(y.shape) == (1,3,512,512) and torch.isfinite(y).all()
                                  and y.min() >= -1.0001 and y.max() <= 1.0001)
        if not report["smoke_512"]:
            raise RuntimeError("512 smoke check failed")
        for case in manifest["cases"]:
            arr = np.asarray(Image.open(HERE/"samples"/(case+"_in.jpg")).convert("RGB"), dtype=np.uint8)
            rendered = infer.render_4k(counted, arr)
            buf = io.BytesIO()
            Image.fromarray(rendered).save(buf, format="JPEG", quality=98, subsampling=0)
            data = buf.getvalue()
            (output/(case+".jpg")).write_bytes(data)
            reading = compare_jpeg(data, (HERE/"samples"/(case+"_out.jpg")).read_bytes())
            reading["pre_jpeg_array_sha256"] = hashlib.sha256(rendered.tobytes()).hexdigest()
            report["cases"][case] = reading
            print(case, json.dumps(reading), flush=True)
        report["status"] = "PASS" if all(r["jpeg_byte_equal"] for r in report["cases"].values()) else "FAIL"
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        report["ended_epoch"] = time.time()
        report["elapsed_s"] = report["ended_epoch"]-report["started_epoch"]
        (output/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-only", action="store_true")
    ap.add_argument("--output")
    ap.add_argument("--ledger")
    a = ap.parse_args()
    if a.identity_only:
        verify_identity()
        print("IDENTITY_PASS (no inference performed)")
        return 0
    if not a.output or not a.ledger:
        ap.error("GPU check requires --output and --ledger; use verify_reproduction.py for supervised checks")
    return gpu_check(a.output, a.ledger)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        sys.exit(1)
