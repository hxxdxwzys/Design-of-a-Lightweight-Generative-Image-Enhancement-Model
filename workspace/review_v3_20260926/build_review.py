"""Assemble a NEW review directory; preserve original archives. No model forward."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = Path("D:/models/upload/Track2_TinyRefV3.zip")
EXPECTED = "d3bf0bf602e9d2ec6459ace57753fd0188189aaacf7b1ca36fda5e1d590ca712"


def digest_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destination", required=True)
    a = ap.parse_args()
    dst = Path(a.destination).resolve()
    if dst.exists():
        raise FileExistsError("New destination required")
    if digest_file(SOURCE) != EXPECTED:
        raise ValueError("Original ZIP identity changed")
    shutil.copytree(HERE/"package", dst, ignore=shutil.ignore_patterns("__pycache__"))
    test = REPO/"生成式图像增强轻量模型设计挑战"/"赛题二"/"测试集"
    provenance = {}
    members = {}
    selected = {"model_dir/your_model.pt": "model_dir/your_model.pt", "model_dir/runner.py": "model_dir/runner.py",
                "output_dir/case3.jpg": "samples/case3_out.jpg", "output_dir/case17.jpg": "samples/case17_out.jpg"}
    with zipfile.ZipFile(SOURCE) as z:
        entries = [i for i in z.infolist() if not i.is_dir()]
        expected_names = {"model_dir/your_model.pt", "model_dir/runner.py"} | {"output_dir/case%d.jpg" % i for i in range(1,101)}
        if len(entries) != 102 or {i.filename for i in entries} != expected_names:
            raise ValueError("Unexpected original member set")
        for info in entries:
            h = hashlib.sha256()
            with z.open(info) as f:
                for b in iter(lambda: f.read(4*1024*1024), b""):
                    h.update(b)
            members[info.filename] = {"bytes": info.file_size, "sha256": h.hexdigest()}
            if info.filename in selected:
                rel = selected[info.filename]
                p = dst/rel
                p.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as f, p.open("xb") as g:
                    shutil.copyfileobj(f, g)
                provenance[rel] = {"kind": "original_zip_member", "source": info.filename, "source_zip_sha256": EXPECTED}
    for case in ["case3", "case17"]:
        src = test/(case+".jpg")
        rel = "samples/"+case+"_in.jpg"
        shutil.copy2(src, dst/rel)
        provenance[rel] = {"kind": "official_test_input_local_copy", "source": str(src), "sha256": digest_file(src)}
    (dst/"original_members.json").write_text(json.dumps({"source_zip_sha256": EXPECTED, "inventory_time": "2026-09-26", "members": members}, indent=2), encoding="utf-8")
    sys.path.insert(0, str(dst))
    from selfcheck import environment
    (dst/"environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    original_renderer = REPO/"生成式图像增强轻量模型设计挑战"/"workspace"/"export_baseline"/"opt11_render_pkg.py"
    manifest = {"schema": "V3-review-1", "created_date": "2026-09-26", "cases": ["case3", "case17"],
                "source_zip": {"name": SOURCE.name, "sha256": EXPECTED, "bytes": SOURCE.stat().st_size},
                "historical_renderer": {"name": original_renderer.name, "sha256": digest_file(original_renderer)},
                "provenance": provenance, "files": {}}
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            manifest["files"][p.relative_to(dst).as_posix()] = {"bytes": p.stat().st_size, "sha256": digest_file(p)}
    (dst/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"destination": str(dst), "files": len(manifest["files"]), "source_sha256": EXPECTED, "manifest_sha256": digest_file(dst/"manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
