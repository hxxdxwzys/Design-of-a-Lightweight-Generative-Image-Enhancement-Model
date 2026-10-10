# -*- coding: utf-8 -*-
"""清理本会话产生的可重建中间产物（逐条落盘记录）。
原则：只删【本会话生成】且【指标已落盘 JSON / 结论已归档】的渲染目录与临时包；
      绝不触碰提交件、归档件、冻结的历史证据、检查点、数据集。
"""
import json, os, shutil, time
from pathlib import Path

OPT11 = Path(r"D:\models\opt11")
MODELS = Path(r"D:\models")

# 本会话生成的 eval_* 渲染目录（对应 test30_*.json 或已归档读数）
EVALS = [
    "eval_b1_iter2", "eval_b1_iter4", "eval_b1_res2", "eval_b1_res3", "eval_b1_res4",
    "eval_ref_c512", "eval_ref_msd", "eval_ref_cap2", "eval_cand3", "eval_bcap64",
    "eval_s1b", "eval_s1", "eval_s1b_smoke", "eval_v4t1_ng", "eval_v4t1_samechain",
    "eval_v3_samechain", "eval_us40_s12_t3", "eval_us60_s08_t3", "eval_us60_s12_t1",
    "eval_us60_s12_t3", "eval_us60_s12_t6", "eval_us60_s20_t3", "eval_us80_s12_t3",
    "eval_d512", "eval_cap2", "eval_lrnd", "eval_thin2",
]
# 本会话生成的临时包目录（模型已导出到 zip/最终包，可重建）
PKGS = ["pkg_s1", "pkg_s1b", "pkg_s1b_smoke", "pkg_s1b_graph", "pkg_cap2ref", "pkg_c512ref",
        "pkg_msdref", "pkg_bcap64", "pkg_cap_test", "pkg_v4t1_ng", "pkg_cap3ref",
        "pkg_final_t1", "pkg_s1_stale"]

# 明确保护清单（绝不删）
PROTECT = [
    "D:\\models\\opt11\\Track2_TinyRefV4t1.zip",
    "D:\\models\\opt11\\Track2_TinyRefV4t1_review.zip",
    "D:\\models\\pkg_submit_final",
    "D:\\models\\finals_20260927",
    "D:\\models\\archive_candidates",
    "D:\\models\\upload",
    "D:\\models\\pkg_TinyRefV3",
    "D:\\models\\pkg_TinyRefV4",
    "D:\\models\\tiny_v1",
    "D:\\models\\refine_v1",
    "D:\\models\\opt11\\v4_render",
    "D:\\models\\opt11\\e8_render",
    "D:\\models\\opt11\\e6_render",
    "D:\\models\\opt11\\R04_render",
    "D:\\models\\opt11\\R08_render",
    "D:\\models\\opt11\\R16_render",
]

rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "deleted": [], "skipped": [], "freed_bytes": 0}


def size_of(p: Path):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def guarded(p: Path) -> bool:
    ps = str(p)
    for g in PROTECT:
        if ps == g or ps.startswith(g + os.sep):
            return False
    return True


for name in EVALS:
    p = OPT11 / name
    if not p.exists():
        rec["skipped"].append({"path": str(p), "why": "not found"}); continue
    if not guarded(p):
        rec["skipped"].append({"path": str(p), "why": "PROTECTED"}); continue
    sz = size_of(p)
    shutil.rmtree(p, ignore_errors=True)
    rec["deleted"].append({"path": str(p), "mb": round(sz / 2**20, 1), "why": "本会话渲染产物，指标已落盘 JSON"})
    rec["freed_bytes"] += sz
    print("  del %-28s %7.1f MB" % (name, sz / 2**20), flush=True)

for name in PKGS:
    p = MODELS / name
    if not p.exists():
        continue
    if not guarded(p):
        rec["skipped"].append({"path": str(p), "why": "PROTECTED"}); continue
    sz = size_of(p)
    shutil.rmtree(p, ignore_errors=True)
    rec["deleted"].append({"path": str(p), "mb": round(sz / 2**20, 1), "why": "本会话临时导出包，可由权重重建"})
    rec["freed_bytes"] += sz
    print("  del %-28s %7.1f MB" % (name, sz / 2**20), flush=True)

rec["freed_gb"] = round(rec["freed_bytes"] / 2**30, 2)
(OPT11 / "cleanup_20260925.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
print("\n释放 %.2f GB，删除 %d 项，跳过 %d 项" % (rec["freed_gb"], len(rec["deleted"]), len(rec["skipped"])), flush=True)
print("CLEANUP_DONE", flush=True)
