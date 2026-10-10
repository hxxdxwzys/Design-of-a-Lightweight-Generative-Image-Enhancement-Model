"""Resume archive sealing after legacy input timestamps rejected by ZIP format."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path("D:/models/review_V3_20260926")
DST = ROOT/"V3_Review"
ARCHIVE = ROOT/"Track2_TinyRefV3_Review_DRAFT_NO_PPT.zip"


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    subprocess.run([sys.executable,"-B",str(DST/"selfcheck.py"),"--identity-only"],check=True)
    subprocess.run([sys.executable,"-B",str(DST/"verification/test_review.py")],check=True)
    files={p.relative_to(DST).as_posix():p for p in DST.rglob("*") if p.is_file()}
    with zipfile.ZipFile(ARCHIVE,"x",zipfile.ZIP_DEFLATED,compresslevel=6,strict_timestamps=False) as z:
        for rel,p in sorted(files.items()): z.write(p,rel)
    with zipfile.ZipFile(ARCHIVE) as z:
        if z.testzip() is not None or set(z.namelist()) != set(files):
            raise ValueError("ZIP CRC/set validation failed")
        for rel,p in files.items():
            if hashlib.sha256(z.read(rel)).hexdigest() != sha(p):
                raise ValueError("ZIP byte identity failed: "+rel)
    originals={}
    for name,expected in [("Track2_TinyRefV3.zip","d3bf0bf602e9d2ec6459ace57753fd0188189aaacf7b1ca36fda5e1d590ca712"),
                          ("Track2_TinyRefV4t1.zip","9502384034af5fb4a87d9c89eeedc191eb89008c52dbbef75fa854193cc94bad")]:
        originals[name]=sha(Path("D:/models/upload")/name)
        if originals[name] != expected: raise ValueError("Original archive changed")
    result={"status":"ENGINEERING_ACCEPTED_PPT_BLOCKED","zip":str(ARCHIVE),"zip_bytes":ARCHIVE.stat().st_size,
            "zip_sha256":sha(ARCHIVE),"manifest_sha256":sha(DST/"manifest.json"),"zip_member_count":len(files),
            "originals_unchanged":originals,"gpu_reproduction":"PASS_2_CASES_2_PROCESSES","cpu_tests":16,
            "ppt":"NOT_CREATED_SLIDEMIND_NOT_PAIRED","uploaded":False,"dependencies_changed":False,
            "packaging_recovery":"First ZIP stopped on pre-1980 input mtime; incomplete ZIP retained under evidence. Only ZIP metadata date clamped, source bytes unchanged."}
    with (ROOT/"evidence/delivery_audit.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
