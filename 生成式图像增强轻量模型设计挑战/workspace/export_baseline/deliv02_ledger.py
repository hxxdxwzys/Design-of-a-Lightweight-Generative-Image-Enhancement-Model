# -*- coding: utf-8 -*-
"""DELIV-02 唯一持久总账：本次额度 + 历史消耗（不可退款、不可冲销）。"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r_ledger  # noqa: E402

OLD = [Path("C:/models/exports/EXPORT01R_20260917_150506/ledger.json"),
       Path("C:/models/exports/EXPORT01R2_20260917_200903/ledger.json"),
       Path("C:/models/exports/EXPORT01R3_20260917_204635/ledger.json"),
       Path("C:/models/exports/DELIV01_20260917_211249/ledger.json"),
       Path("C:/models/exports/DELIV01R_20260917_213145/ledger_a3.json"),
       Path("C:/models/exports/DELIV01R_A2_20260917_213518/ledger.json"),
       Path("C:/models/exports/DELIV01R_A2_20260917_213518/ledger_b.json"),
       Path("C:/models/exports/DELIV01R_A2_20260917_213518/ledger_final.json"),
       Path("C:/models/exports/DELIV01R_A2_20260917_213518/ledger_verify.json"),
       Path("C:/models/exports/DELIV01R_A2_20260917_213518/ledger_verify2.json")]


def sha256_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1])
    hist = {"top": 0, "top4k": 0, "component": 0}
    refs = []
    for p in OLD:
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        u = d.get("used", {}) or {}
        for k in hist:
            hist[k] += int(u.get(k, 0) or 0)
        refs.append({"ledger": str(p), "sha256": sha256_file(p), "used": u})
    prior = {"old_ledgers": refs, "note": "历史消耗不可退款、不可冲销",
             "prior_top": hist["top"], "prior_top4k": hist["top4k"],
             "prior_component": hist["component"]}
    led = r_ledger.Ledger(out, top=16, top4k=6, component=48, wall_s=1800.0, disk_gib=20.0,
                          prior=prior)
    led._write(led._read())
    print(json.dumps({"ledger": str(out), "limits": led.limits, "hard_caps": led.hard_caps,
                      "history": hist}, ensure_ascii=False))


if __name__ == "__main__":
    main()
