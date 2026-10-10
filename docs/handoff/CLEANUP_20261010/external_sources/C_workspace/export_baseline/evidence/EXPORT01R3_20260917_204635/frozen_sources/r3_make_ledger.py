# -*- coding: utf-8 -*-
"""创建 EXPORT-01R3 新批次总账（关联 R/R2 旧账；旧+新不超硬上限）。

用法: python r3_make_ledger.py <new_ledger_path>
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r_ledger  # noqa: E402

OLD = [Path("C:/models/exports/EXPORT01R_20260917_150506/ledger.json"),
       Path("C:/models/exports/EXPORT01R2_20260917_200903/ledger.json")]


def sha256_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1])
    used = {"top": 0, "top4k": 0, "component": 0}
    refs = []
    wall = 0.0
    for p in OLD:
        d = json.loads(p.read_text(encoding="utf-8"))
        u = d.get("used", {})
        for k in used:
            used[k] += int(u.get(k, 0) or 0)
        refs.append({"ledger": str(p), "sha256": sha256_file(p), "used": u,
                     "supervisor_wall_s": ("71.083" if "150506" in str(p) else "109.669")})
        wall += 71.083 if "150506" in str(p) else 109.669
    prior = {"old_ledgers": refs, "used": used, "wall_s": round(wall, 3),
             "combined_caps": {"component": 48, "wall_s": 1800.0, "top": 17}}
    led = r_ledger.Ledger(out, top=12, top4k=4, component=36, wall_s=1618.0, disk_gib=8.0,
                          prior=prior)
    led._write(led._read())
    print(json.dumps({"ledger": str(out), "limits": led.limits, "prior_used": used,
                      "prior_wall_s": prior["wall_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
