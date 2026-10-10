"""Seal the reviewed engineering DRAFT with evidence. Never mutate original archives."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = Path("D:/models/review_V3_20260926")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    evidence = ROOT/"evidence"
    # Tests produce a real persistent log; do not replace a previous log.
    test = subprocess.run([sys.executable, "-B", str(HERE/"test_review.py")], capture_output=True, text=True, encoding="utf-8")
    with (evidence/"cpu_tests.txt").open("x", encoding="utf-8") as f:
        f.write(test.stdout+test.stderr+"\nEXIT_CODE="+str(test.returncode)+"\n")
    if test.returncode != 0:
        raise RuntimeError("CPU tests failed")
    dst = ROOT/"V3_Review"
    subprocess.run([sys.executable, "-B", str(HERE/"build_review.py"), "--destination", str(dst)], check=True)
    verification = dst/"verification"
    verification.mkdir()
    old_manifest = json.loads((ROOT/"engineering/manifest.json").read_text(encoding="utf-8"))
    for rel in ["infer.py", "selfcheck.py", "verify_reproduction.py", "model_dir/your_model.pt", "model_dir/runner.py",
                "samples/case3_in.jpg", "samples/case3_out.jpg", "samples/case17_in.jpg", "samples/case17_out.jpg"]:
        if sha(dst/rel) != old_manifest["files"][rel]["sha256"]:
            raise ValueError("Delivery differs from GPU-tested source: "+rel)
    shutil.copy2(ROOT/"engineering/manifest.json", verification/"tested_manifest.json")
    run = evidence/"reproduction_01"
    summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "PASS":
        raise ValueError("Failed reproduction cannot be marked accepted")
    for name in ["preregistration.json", "summary.json", "ledger.json", "process_1.log", "process_2.log"]:
        shutil.copy2(run/name, verification/name)
    from PIL import Image
    import numpy as np
    independent = {}
    with zipfile.ZipFile("D:/models/upload/Track2_TinyRefV3.zip") as original:
        for case in ["case3", "case17"]:
            reference = original.read("output_dir/"+case+".jpg")
            reads = []
            for i in [1,2]:
                result = json.loads((run/("process_"+str(i))/"results.json").read_text(encoding="utf-8"))
                actual = (run/("process_"+str(i))/(case+".jpg")).read_bytes()
                if actual != reference:
                    raise ValueError("Independent archive byte comparison failed")
                digest = hashlib.sha256(actual).hexdigest()
                if result["cases"][case]["generated_sha256"] != digest:
                    raise ValueError("Reading hash mismatch")
                a = np.asarray(Image.open(io.BytesIO(actual)).convert("RGB"))
                b = np.asarray(Image.open(io.BytesIO(reference)).convert("RGB"))
                if not np.array_equal(a,b):
                    raise ValueError("Decoded image mismatch")
                reads.append({"process":i,"sha256":digest,"original_zip_byte_equal":True,"decoded_rgb_equal":True})
            independent[case] = reads
    for i in [1,2]:
        shutil.copy2(run/("process_"+str(i))/"results.json", verification/("process_"+str(i)+"_results.json"))
    (verification/"independent_comparison.json").write_text(json.dumps(independent, indent=2), encoding="utf-8")
    shutil.copy2(evidence/"cpu_tests.txt", verification/"cpu_tests.txt")
    shutil.copy2(HERE/"test_review.py", verification/"test_review.py")
    shutil.copy2(HERE/"SLIDE_CONTENT.md", dst/"SLIDE_CONTENT.md")
    # CPU load only: verify actual submitted parameter counts and dtypes.
    import torch
    model = torch.jit.load(str(dst/"model_dir/your_model.pt"), map_location="cpu")
    named = list(model.named_parameters())
    params = {"total":sum(p.numel() for _,p in named),
              "groups":{},"dtypes":sorted({str(p.dtype) for _,p in named}),
              "execution":"CPU load only, no model forward"}
    for name,p in named:
        group=name.split(".")[0]
        params["groups"][group]=params["groups"].get(group,0)+p.numel()
    if params["total"] != 876662:
        raise ValueError("Unexpected V3 parameter count")
    (verification/"model_audit.json").write_text(json.dumps(params,indent=2),encoding="utf-8")
    manifest = json.loads((dst/"manifest.json").read_text(encoding="utf-8"))
    manifest["tested_manifest_sha256"] = sha(verification/"tested_manifest.json")
    manifest["delivery_status"] = "ENGINEERING_ACCEPTED_PPT_MISSING_NOT_READY_FOR_FULL_SUBMISSION"
    manifest["files"] = {p.relative_to(dst).as_posix(): {"bytes":p.stat().st_size,"sha256":sha(p)}
                         for p in sorted(dst.rglob("*")) if p.is_file() and p != dst/"manifest.json"}
    (dst/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run([sys.executable,"-B",str(dst/"selfcheck.py"),"--identity-only"], check=True)
    # Local engineering-only draft ZIP, never overwrite any original or upload it.
    archive = ROOT/"Track2_TinyRefV3_Review_DRAFT_NO_PPT.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED,compresslevel=6,strict_timestamps=False) as z:
        for p in sorted(dst.rglob("*")):
            if p.is_file(): z.write(p,p.relative_to(dst).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError("ZIP CRC failed")
        expected_members={p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()}
        if set(z.namelist()) != expected_members: raise ValueError("ZIP set mismatch")
        for rel in z.namelist():
            if hashlib.sha256(z.read(rel)).hexdigest()!=sha(dst/rel): raise ValueError("ZIP hash mismatch")
    original_hashes = {}
    for name,expected in [("Track2_TinyRefV3.zip","d3bf0bf602e9d2ec6459ace57753fd0188189aaacf7b1ca36fda5e1d590ca712"),
                          ("Track2_TinyRefV4t1.zip","9502384034af5fb4a87d9c89eeedc191eb89008c52dbbef75fa854193cc94bad")]:
        actual = sha(Path("D:/models/upload")/name)
        if actual != expected: raise ValueError("Original archive changed: "+name)
        original_hashes[name] = actual
    final={"status":"ENGINEERING_ACCEPTED_PPT_BLOCKED", "zip":str(archive),"zip_bytes":archive.stat().st_size,
           "zip_sha256":sha(archive),"manifest_sha256":sha(dst/"manifest.json"),
           "originals_unchanged":original_hashes,"zip_member_count":len(expected_members),
           "gpu_reproduction":"PASS_2_CASES_2_PROCESSES","cpu_tests":16,"ppt":"NOT_CREATED_SLIDEMIND_NOT_PAIRED",
           "uploaded":False,"dependencies_changed":False}
    (evidence/"delivery_audit.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
