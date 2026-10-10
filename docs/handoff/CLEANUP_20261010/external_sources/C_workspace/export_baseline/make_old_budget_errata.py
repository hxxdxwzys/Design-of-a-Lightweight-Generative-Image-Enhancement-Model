# -*- coding: utf-8 -*-
"""旧预算事件账重建（§0A R0 第 4 条）：从既有 EXPORT01* 目录与日志回收可核对的调用事件。

原则：只读旧目录/日志，不改旧报告；能从日志逐条数出的才写数字，其余标 UNKNOWN 或给保守上界。
用法: python make_old_budget_errata.py <out_json>
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path('C:/models/exports')
EB = Path(__file__).resolve().parent


def read_text_any(p):
    """旧日志多为 PowerShell 重定向产物（UTF-16LE/BOM），需按字节嗅探后再解码。"""
    raw = p.read_bytes()
    for enc in ('utf-8-sig', 'utf-16', 'utf-16-le', 'latin-1'):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode('utf-8', errors='replace')


def count_lines(p, pattern):
    if not p.is_file():
        return None
    n = 0
    for line in read_text_any(p).splitlines():
        if re.search(pattern, line):
            n += 1
    return n


def main():
    out = {'schema': 'EXPORT01R-old-budget-errata/1',
           'note': '本轮只重建可核对事件；无法恢复的项标 UNKNOWN，不编造精确次数，不修改旧报告',
           'dirs': [], 'totals': {}}
    top = 0
    top4k = 0
    comp = 0
    unknown = []
    for d in sorted(ROOT.glob('EXPORT01*')):
        if not d.is_dir():
            continue
        rec = {'dir': d.name, 'mtime': d.stat().st_mtime,
               'n_files': sum(1 for _ in d.rglob('*') if _.is_file()),
               'bytes': sum(_.stat().st_size for _ in d.rglob('*') if _.is_file())}
        # 探针阶段日志 → 已完成的 trace/前向
        stages = d / 'probe_stages.log'
        rec['probe_traces'] = count_lines(stages, r'\[stage\] traced ')
        rec['probe_stage_lines'] = count_lines(stages, r'\[stage\] ')
        # 验证日志 → phase1/phase2 行数（每行 = 一次顶层真实 infer，128/4K 混合）
        logs = sorted(d.glob('evidence_verify_c1_stdout*.txt'))
        p1 = p2 = 0
        for lg in logs:
            p1 += count_lines(lg, r'\[phase1\]') or 0
            p2 += count_lines(lg, r'\[phase2\]') or 0
        rec['verify_logs'] = [lg.name for lg in logs]
        rec['phase1_lines'] = p1
        rec['phase2_lines'] = p2
        det = d / 'evidence' / 'determinism_probe.json'
        rec['determinism_component_forwards'] = None
        if det.is_file():
            try:
                rec['determinism_component_forwards'] = json.loads(
                    det.read_text(encoding='utf-8')).get('n_component_forwards')
            except Exception:
                rec['determinism_component_forwards'] = 'UNKNOWN'
        out['dirs'].append(rec)
        if rec['determinism_component_forwards']:
            comp += rec['determinism_component_forwards']
        top += p1 + p2
    # 已知的申报数（本实现报告）与可核对数的对照
    out['implementer_declared'] = {'top_level_infers': 22, 'top4k_infers': 8,
                                   'note': 'EXP-01 交付报告申报数；本重建不对其做等号确认'}
    out['reconstructed'] = {
        'verified_phase_lines_total': top,
        'determinism_component_forwards_total': comp,
        'top4k_breakdown': 'UNKNOWN（旧日志未逐行标注 4K/512，只有 case 名与尺寸）',
        'trace_internal_reruns': 'UNKNOWN（check_trace=False，无法从日志区分）',
        'load_failures_and_cpu_exceptions': 'UNKNOWN（部分日志为空/被截断）',
        'conservative_upper_bound_top': max(top, 22),
        'note': '旧批次额度已用尽且不可复用；本表仅用于勘误与新批次隔离'
    }
    out['post_failure_calls'] = {
        'note': '首次 FAIL（verify_c1 attempt_5 的 fp32 诊断设备错误）之后仍执行了：'
                'attempt_6 的 5 次顶层 infer，以及 1 次 26 前向的 fp16 组件确定性探针'
                '（determinism_probe.json），随后才停止；这些调用均未再支取新额度。'}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'dirs': len(out['dirs']), 'reconstructed_top': top,
                      'component_forwards': comp}, ensure_ascii=False))


if __name__ == '__main__':
    main()
