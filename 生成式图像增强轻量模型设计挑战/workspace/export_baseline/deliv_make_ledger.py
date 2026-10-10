# -*- coding: utf-8 -*-
"""创建 DELIV-01 阶段总账（P1/P2/P3 分阶段上限，先扣后跑，失败不退）。"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r_ledger  # noqa: E402

OLD = [Path("C:/models/exports/EXPORT01R_20260917_150506/ledger.json"),
       Path("C:/models/exports/EXPORT01R2_20260917_200903/ledger.json"),
       Path("C:/models/exports/EXPORT01R3_20260917_204635/ledger.json")]


def sha256_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1])
    stage = sys.argv[2] if len(sys.argv) > 2 else "P1"
    caps = {"P1": {"top": 24, "top4k": 8, "component": 96, "wall_s": 2700.0},
            "P2": {"top": 100, "top4k": 100, "component": 8, "wall_s": 9000.0},
            "P3": {"top": 4, "top4k": 0, "component": 4, "wall_s": 900.0}}
    c = caps[stage]
    prior = {"old_ledgers": [], "note": "旧批次失败与扣款原样保留，不清零、不退还"}
    for p in OLD:
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            prior["old_ledgers"].append({"ledger": str(p), "sha256": sha256_file(p),
                                         "used": d.get("used")})
    led = r_ledger.Ledger(out, top=c["top"], top4k=c["top4k"], component=c["component"],
                          wall_s=c["wall_s"], disk_gib=8.0, prior=prior)
    led._write(led._read())
    print(json.dumps({"stage": stage, "limits": led.limits, "ledger": str(out)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
