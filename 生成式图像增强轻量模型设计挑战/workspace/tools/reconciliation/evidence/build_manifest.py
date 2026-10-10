# -*- coding: utf-8 -*-
"""RCON-01 工作包 E — 资产与生成来源证据清单生成器（CPU / 只读输入）

原则（docs/current_task.md §3）:
  * 只做安全读取: 流式 SHA-256、zipfile 结构、safetensors 头部 JSON。
    **不 import torch、不 torch.load、不 unpickle 来源不明的权重。**
  * 只登记本机已核验的文件事实；无法从记录重建的合并/生成关系一律 UNKNOWN，
    不根据目录名或注释填 true。
  * 不复现大资产（不复制），不改动任何被登记的文件。
  * 代码引用带自校验: 若引用的行号/内容对不上，条目状态置为 STALE 而不是沿用旧结论。

用法: D:/Python/python.exe build_manifest.py
输出: manifest_assets.json / source_hash_record.json / gate_workcopy_scope.json / gate_workcopy_diff.txt
      （本目录不写 .md；人读摘要打印到 stdout，正式报告见 docs/handoff/RCON01_20260916_交付报告.md）
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import struct
import sys
import time
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]                     # Git 根 C:/生成式图像增强轻量模型设计挑战
TMP = Path(r'D:/tmp_timing')
MDIR = Path(r'D:/models')
CFIN = Path(r'C:/models/finals')
NOW = time.strftime('%Y-%m-%d %H:%M:%S')
CHUNK = 1 << 23

# review.md 记录的源码哈希（2026-09-16 调查快照）；用于漂移比对
REVIEW_HASHES = {
    'vosr_lora_infer2.py': '11b664d403f3b662067f7e221badb2c8724d8d3c3b0b825398985be0d66fb94b',
    'train_vosr_lora.py': '61409f0ecd785a0a5070d234d7d156e97ffe483e4ff1b650511421800cddee46',
    'preupload_gate.py': 'acff56c784e703a2cb8513519098098cfc08ead8c0047c15840e84b1014c8478',
    'score_pred2.py': '38e27eeda883279b1b674a99071d3035524de6ab4a4c15f6c62385937b949a15',
}


def sha256_file(p: Path, block=CHUNK):
    h = hashlib.sha256()
    n = 0
    with open(p, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            n += len(b)
            h.update(b)
    return h.hexdigest(), n


def asset(aid, path, role, kind='file', scope='', note=''):
    p = Path(path)
    rec = {
        'id': aid, 'path': str(p), 'role': role, 'kind': kind,
        'exists': p.exists(), 'verified_at': NOW, 'scope': scope,
        'evidence': 'build_manifest.py: 直接文件系统读取',
    }
    if not p.exists():
        rec.update(status='MISSING', sha256=None, size_bytes=None)
        if note:
            rec['note'] = note
        return rec
    if p.is_dir():
        files = sorted(x for x in p.rglob('*') if x.is_file())
        rec.update(status='PRESENT', size_bytes=None,
                   sha256=None, file_count=len(files),
                   extensions=sorted({x.suffix.lower() for x in files}))
    else:
        h, n = sha256_file(p)
        rec.update(status='PRESENT', sha256=h, size_bytes=n)
    if note:
        rec['note'] = note
    return rec


def safe_container_peek(p: Path):
    """安全结构读取：不解包、不反序列化对象。"""
    p = Path(p)
    if not p.exists():
        return {'status': 'MISSING'}
    if p.suffix == '.safetensors':
        try:
            with open(p, 'rb') as f:
                raw = f.read(8)
                if len(raw) != 8:
                    return {'status': 'UNKNOWN', 'reason': 'header 长度字段不足 8 字节'}
                n = struct.unpack('<Q', raw)[0]
                if n <= 0 or n > 200 * 1024 * 1024:
                    return {'status': 'UNKNOWN', 'reason': 'header 长度异常: %d' % n}
                hdr = json.loads(f.read(n).decode('utf-8'))
            keys = [k for k in hdr if k != '__metadata__']
            return {'status': 'OK', 'container': 'safetensors',
                    'tensor_count': len(keys), 'has_metadata': '__metadata__' in hdr,
                    'method': '仅解析头部 JSON，未加载张量'}
        except Exception as e:
            return {'status': 'UNKNOWN', 'reason': '%s: %s' % (type(e).__name__, e)}
    try:
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
            return {'status': 'OK', 'container': 'zip(torch.save 新格式)',
                    'member_count': len(names), 'members_sample': names[:6],
                    'method': 'zipfile 目录读取，未 unpickle'}
    except Exception as e:
        return {'status': 'UNKNOWN', 'reason': '%s: %s' % (type(e).__name__, e)}
    return {'status': 'UNKNOWN',
            'reason': '非 zip 容器；按纪律不 torch.load / 不 unpickle 来源不明的权重'}


def code_citations():
    """带自校验的代码引用：行号 + 期望子串。对不上就标 STALE。"""
    specs = [
        ('render.precision.offline', TMP / 'vosr_lora_infer2.py', 131,
         '.to(device).eval()', '离线模型 fp32（无 .half()/autocast）'),
        ('render.seed.offline', TMP / 'vosr_lora_infer2.py', 124,
         'torch.manual_seed', '离线入口设置 torch seed'),
        ('render.z.offline', TMP / 'vosr_lora_infer2.py', 161,
         'z = torch.randn_like(lq_latent)', '离线非分块分支独立随机 z'),
        ('render.align.offline', TMP / 'vosr_lora_infer2.py', 166,
         "args.align_method == 'adain'", '离线颜色对齐分支'),
        ('render.outfmt.offline', TMP / 'vosr_lora_infer2.py', 170,
         "stem + '.png'", '离线输出 PNG'),
        ('vae.posterior.offline', MDIR / 'finals/VOSR/tiled_vae.py', 78,
         'posterior.mode() if posterior_mode else posterior.sample()', 'VAE 后验默认 mode'),
        ('vae.posterior.default', MDIR / 'finals/VOSR/tiled_vae.py', 191,
         "getattr(args, 'posterior_mode', True)", 'encode_dispatch 默认 posterior_mode=True'),
        ('runner.doc.recipe', MDIR / 'final_compliant/model_dir/runner.py', 5,
         'v7_s5000 LoRA(sc0.8', 'runner 自述配方（声明，非证明）'),
        ('runner.doc.samesubject', MDIR / 'final_compliant/model_dir/runner.py', 17,
         '与出图(output_dir)同一模型主体同一套权重', 'runner 自述同主体（声明，非证明）'),
        ('runner.vae.posterior', MDIR / 'final_compliant/model_dir/runner.py', 291,
         'latent_dist.sample()', 'runner 用 sample（与离线 mode 不同）'),
        ('runner.precision', MDIR / 'final_compliant/model_dir/runner.py', 354,
         '.to(self.device).half().eval()', 'runner 半精度'),
        ('runner.autocast', MDIR / 'final_compliant/model_dir/runner.py', 255,
         "torch.autocast('cuda', dtype=torch.float16)", 'runner autocast fp16'),
        ('runner.z', MDIR / 'final_compliant/model_dir/runner.py', 393,
         'z = torch.randn(', 'runner 每次调用独立随机 z'),
        ('runner.clamp', MDIR / 'final_compliant/model_dir/runner.py', 411,
         'return sr.float().clamp(-1.0, 1.0)', 'runner 返回 clamp 后 float32'),
        ('runner.load', MDIR / 'final_compliant/model_dir/runner.py', 343,
         "torch.load(os.path.join(model_dir, 'your_model.pt')", 'runner 用 torch.load 普通权重字典'),
        ('gate.rc_ignored.orig', TMP / 'preupload_gate.py', 212,
         "ok5 = 'FORMAT_PASS' in out and 'FORMAT_FAIL' not in out", '原件忽略子进程退出码（已记录缺陷）'),
        ('gate.harness_marker.orig', TMP / 'preupload_gate.py', 177,
         "ok_h = 'infer OK' in out", '原件接受中间态标记（已记录缺陷）'),
        ('gate.can_upload.orig', TMP / 'preupload_gate.py', 224,
         '可以上传', '原件输出自动上传许可（本副本已移除）'),
        ('harness.terminal.orig', TMP / 'harness_sim_zip.py', 37,
         'HARNESS_ZIP_SIM_PASS', 'harness 终态标记'),
        ('eval.metric_skip', REPO / '生成式图像增强轻量模型设计挑战/workspace/eval.py', 103,
         'except Exception as e:', 'EV 指标创建失败即跳过（TR32）'),
        ('eval.per_image_none', REPO / '生成式图像增强轻量模型设计挑战/workspace/eval.py', 61,
         'except Exception as e:', 'EV 逐图异常返回 None（TR32）'),
        ('eval.mean_valid', REPO / '生成式图像增强轻量模型设计挑战/workspace/eval.py', 60,
         'fn().mean().item()', 'EV 只对有效值求均值（TR32）'),
        ('eval.collect_overwrite', REPO / '生成式图像增强轻量模型设计挑战/workspace/eval.py', 54,
         'd[k] = p', 'EV 同 case 后者覆盖（TR33）'),
        ('eval.silent_resize', REPO / '生成式图像增强轻量模型设计挑战/workspace/eval.py', 113,
         'interpolate(', 'EV 尺寸不一致自动缩放（TR33）'),
        ('qf.gt_glob_bug', TMP / 'q_fit_measure2.py', 47,
         'if not p.exists(): continue', 'GT 通配符未展开 -> 全部跳过（TR37）'),
        ('qf.gt_call', TMP / 'q_fit_measure2.py', 64,
         "'*_gt.*'", 'GT_val 用通配符字符串调用（TR37）'),
        ('qf.identity_pair', TMP / 'q_fit2.py', 17,
         "'w2_m0s1_s090',   34.0989", 'QF 把 s090 目录与 34.0989 配对（TR36）'),
        ('sp.params', TMP / 'score_pred2.py', 18,
         'C = 2.0', 'SP 实参 C=2.0（与文档 v3.1 的 2.515 不同，TR35）'),
        ('q11.sample_ids', TMP / 'q11_align_screen.py', 15,
         "CASES = ['case1', 'case7', 'case12', 'case30', 'case54', 'case77']",
         'Q11 六个样本 ID（返修：原登记行号 14 有误，实为 15；现纳入自校验）'),
        ('train.iqa_loss', TMP / 'train_vosr_lora.py', 323,
         'args.w_clip * loss_cq', '历史可微 CLIPIQA 损失（TR06，须披露不得隐藏）'),
        ('train.iqa_as_loss', TMP / 'train_vosr_lora.py', 226,
         'clipiqa_m.as_loss = True', '历史 CLIPIQA 作为损失（TR06）'),
    ]
    out = []
    for cid, path, line, expect, why in specs:
        rec = {'id': cid, 'file': str(path), 'line': line, 'expect': expect,
               'why': why, 'verified_at': NOW, 'source': 'static-read'}
        try:
            txt = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
        except Exception as e:
            rec.update(status='STALE', detail='读取失败: %s' % e)
            out.append(rec)
            continue
        if line - 1 >= len(txt):
            rec.update(status='STALE', detail='行号超出文件长度 %d' % len(txt))
        elif expect in txt[line - 1]:
            rec.update(status='OK', kind='FACT')
        else:
            rec.update(status='STALE', kind='CONFLICT',
                       detail='实际内容: %s' % txt[line - 1].strip()[:160])
        out.append(rec)
    return out


def lora_availability():
    root = MDIR / 'vosr_lora'
    res = {}
    for v in ('v1', 'v2', 'v3', 'v5', 'v6', 'v7', 'v8', 'v14'):
        d = root / v
        pts = sorted(x.name for x in d.glob('*.pt')) if d.is_dir() else []
        res[v] = {'dir_exists': d.is_dir(), 'pt_count': len(pts),
                  'pt_names': pts if len(pts) <= 20 else pts[:20],
                  'status': 'AVAILABLE' if pts else 'NO_WEIGHTS_IN_THIS_DIR',
                  'verified_at': NOW}
        if v in ('v1', 'v2', 'v3') and not pts:
            res[v]['note'] = ('本目录只有日志/val_gate JSON；其他位置备份 UNKNOWN，'
                              '不得据目录名或指标文件名推定候选存在（TR26）')
    return res


def protocol_artifacts():
    out = {}
    for tag, path in (('Q10', TMP / 'q10_protocol_screen.json'),
                      ('Q11', TMP / 'q11_align_screen.json')):
        p = Path(path)
        rec = {'path': str(p), 'exists': p.exists(), 'verified_at': NOW}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding='utf-8'))
                rec['keys'] = list(data.keys())
                rec['done_true'] = [k for k, v in data.items()
                                    if isinstance(v, dict) and v.get('done') is True]
                rec['incomplete'] = [k for k, v in data.items()
                                     if isinstance(v, dict) and v.get('done') is False]
                rec['metric_values_null'] = [k for k, v in data.items()
                                             if isinstance(v, dict) and v.get('clipiqa') is None]
                rec['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
            except Exception as e:
                rec['status'] = 'UNKNOWN'
                rec['reason'] = str(e)
        out[tag] = rec
    out['Q11_sample_ids_source'] = {
        'see_code_citation': 'q11.sample_ids',
        'verified_at': NOW,
        'note': ('Q11 只用 6 张固定样本；行号与内容已纳入 code_citations 自校验'
                 '（返修：原登记行号 14 有误，实为 15）'),
    }
    out['status_note'] = ('Q10 只有 A/C 成功、B 未完成；Q11 三档均为 6 图对比新 PNG 与旧 JPG，'
                          '不构成完整协议轴闭合（TR28/TR29/TR34）')
    return out


def spot_check_output_vs_archive():
    """n=3 抽样比对本地 output_dir 与归档 zip 内同名成员（有界范围，不等于全量）。"""
    local = MDIR / 'final_compliant/output_dir'
    zp = REPO / '提交历史/Track2_M24_v3——43.6951.zip'
    rec = {'scope': 'n=3 抽样（case1/case50/case100），非全量比对', 'verified_at': NOW,
           'local_dir': str(local), 'archive_zip': str(zp)}
    if not local.is_dir() or not zp.exists():
        rec['status'] = 'UNKNOWN'
        rec['reason'] = 'local 或 archive 缺失'
        return rec
    picks = ['case1.jpg', 'case50.jpg', 'case100.jpg']
    rows = []
    try:
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
            for nm in picks:
                lp = local / nm
                member = 'output_dir/' + nm
                if not lp.exists() or member not in names:
                    rows.append({'name': nm, 'status': 'MISSING'})
                    continue
                lh, _ = sha256_file(lp, 1 << 20)
                zh = hashlib.sha256(z.read(member)).hexdigest()
                rows.append({'name': nm, 'status': 'MATCH' if lh == zh else 'DIFFER',
                             'local_sha256': lh, 'zip_sha256': zh})
    except Exception as e:
        rec['status'] = 'UNKNOWN'
        rec['reason'] = '%s: %s' % (type(e).__name__, e)
        return rec
    rec['rows'] = rows
    rec['status'] = 'ALL_MATCH' if rows and all(r.get('status') == 'MATCH' for r in rows) else 'CHECK'
    rec['note'] = ('归档一致性抽样，只说明本地目录与归档包在这些样本上一致；'
                   '不证明这批图由某条命令生成（生成 manifest 缺失）')
    return rec


def name_agreement():
    test = REPO / '生成式图像增强轻量模型设计挑战/赛题二/测试集'
    out = MDIR / 'final_compliant/output_dir'
    rec = {'verified_at': NOW, 'test_dir': str(test), 'output_dir': str(out)}
    if not test.is_dir() or not out.is_dir():
        rec.update(status='UNKNOWN', reason='目录缺失')
        return rec
    tn = {p.name for p in test.glob('*.jpg')}
    on = {p.name for p in out.glob('*')}
    rec.update(test_count=len(tn), output_count=len(on),
               same_set=(tn == on),
               only_in_output=sorted(on - tn)[:5],
               status='OK' if tn == on else 'DIFFER')
    return rec


def asset_path_drift():
    """登记 asset_profiles.json 中记录与实际归档不一致的路径（只登记，不按名称相似改指向）。"""
    p = TMP / 'asset_profiles.json'
    rec = {'file': str(p), 'verified_at': NOW, 'kind': 'DRIFT'}
    if not p.exists():
        rec.update(status='MISSING')
        return rec
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        z = data.get('assets', {}).get('M24', {}).get('zip')
    except Exception as e:
        rec.update(status='UNKNOWN', reason='%s: %s' % (type(e).__name__, e))
        return rec
    hist = REPO / '提交历史'
    actual = sorted(x.name for x in hist.glob('*.zip')) if hist.is_dir() else []
    rec.update(recorded_field='assets.M24.zip', recorded_value=z,
               recorded_path_exists=bool(z) and Path(z).exists(),
               archive_dir=str(hist), archive_zip_count=len(actual),
               archive_names_sample=actual[:3],
               status='DRIFT' if (z and not Path(z).exists()) else 'OK',
               action='仅登记为资产路径漂移；不按名称相似改指向（名称相似不等于同一对象）')
    return rec


def generator_identity():
    """记录生成器自身的哈希，避免报告与当前生成器版本不一致（返修 4c）。"""
    me = Path(__file__).resolve()
    return {'path': str(me), 'sha256': hash_of(me), 'computed_at': NOW}


def hash_of(p: Path):
    p = Path(p)
    if not p.exists():
        return None
    return sha256_file(p)[0]


def _func_sources(text):
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.get_source_segment(text, node) or ''
    return out


def scope_check(src: Path, dst: Path):
    """证明“业务检查函数未被改动”的可审查证据（最小差异主张）。"""
    a_text = src.read_text(encoding='utf-8', errors='replace')
    b_text = dst.read_text(encoding='utf-8', errors='replace')
    fa, fb = _func_sources(a_text), _func_sources(b_text)
    rows = []
    for name in sorted(set(fa) | set(fb)):
        if name in fa and name in fb:
            same = fa[name] == fb[name]
            rows.append({'function': name, 'in_both': True, 'byte_identical': same,
                         'status': 'UNCHANGED' if same else 'CHANGED'})
        elif name in fa:
            rows.append({'function': name, 'in_both': False,
                         'status': 'REMOVED_IN_WORKCOPY'})
        else:
            rows.append({'function': name, 'in_both': False,
                         'status': 'ADDED_IN_WORKCOPY'})
    kept = [r['function'] for r in rows if r['status'] == 'UNCHANGED']
    changed = [r['function'] for r in rows if r['status'] == 'CHANGED']
    rec = {'generated_at': NOW, 'original': str(src), 'workcopy': str(dst),
           'functions': rows, 'unchanged': kept, 'changed': changed,
           'business_checks_unchanged': all(
               f in kept for f in ('check_ascii_name', 'check_inner_names',
                                   'check_structure', 'check_size')),
           'note': ('check_jpg_dims 是唯一改动的本地检查：缺 PIL 时由 PASS 改为 NOT_RUN；'
                    '尺寸阈值 min(w,h)>=2000 未改。run_tool 被可注入执行器取代。')}
    (HERE / 'gate_workcopy_scope.json').write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    return rec


def write_diff():
    src = TMP / 'preupload_gate.py'
    dst = HERE.parent / 'preupload_gate_rcon.py'
    scope = scope_check(src, dst)
    a = src.read_text(encoding='utf-8', errors='replace').splitlines(keepends=True)
    b = dst.read_text(encoding='utf-8', errors='replace').splitlines(keepends=True)
    diff = list(difflib.unified_diff(a, b, fromfile='ORIGINAL D:/tmp_timing/preupload_gate.py',
                                     tofile='WORKCOPY workspace/tools/reconciliation/preupload_gate_rcon.py',
                                     n=3))
    hdr = ('# RCON-01 工作包G：原件 -> 工作副本 统一差异（生成于 %s）\n'
           '# 原件 sha256(未改)  %s\n# 副本 sha256(本次)  %s\n'
           '# 说明：只做失败传播/拒绝副作用/可注入性的窄修复；本地检查判据保持不变\n\n'
           % (NOW, hash_of(src), hash_of(dst)))
    p = HERE / 'gate_workcopy_diff.txt'
    p.write_text(hdr + ''.join(diff), encoding='utf-8')
    return {'path': str(p), 'lines': len(diff),
            'added': sum(1 for l in diff if l.startswith('+') and not l.startswith('+++')),
            'removed': sum(1 for l in diff if l.startswith('-') and not l.startswith('---')),
            'scope': {'unchanged_functions': scope['unchanged'],
                      'changed_functions': scope['changed'],
                      'business_checks_unchanged': scope['business_checks_unchanged']}}


def main():
    assets = [
        asset('A01.infer2', TMP / 'vosr_lora_infer2.py', '离线4K渲染入口（主线）', scope='源码'),
        asset('A02.m24_runner', MDIR / 'final_compliant/model_dir/runner.py', 'M24 交付 runner', scope='源码'),
        asset('A03.merged_weights', MDIR / 'final_compliant/model_dir/your_model.pt',
              'M24 合并权重（runner 加载对象）', scope='权重容器'),
        asset('A04.base_ema', CFIN / 'preset/ckpts/VOSR_0.5B_os/checkpoints/ema_model.safetensors',
              'VOSR 0.5B 基座权重', scope='权重'),
        asset('A05.args_json', CFIN / 'preset/ckpts/VOSR_0.5B_os/args.json', '基座配置', scope='配置'),
        asset('A06.lora_v7_s5000', MDIR / 'vosr_lora/v7/lora_s5000.pt',
              '文档所述 M24 配方 LoRA（v7_s5000, scale 0.8）', scope='LoRA 权重'),
        asset('A07.lwdecoder', CFIN / 'preset/ckpts/sd21_lwdecoder.pth', 'LightDecoder 权重', scope='权重'),
        asset('A08.output_dir', MDIR / 'final_compliant/output_dir', 'M24 100 张出图目录', kind='dir', scope='图像'),
        asset('A09.archive_m24_v3', REPO / '提交历史/Track2_M24_v3——43.6951.zip',
              '归档成绩包（43.6951）', scope='归档 ZIP'),
        asset('A10.gate_orig', TMP / 'preupload_gate.py', '门禁原件（只读）', scope='源码'),
        asset('A11.harness_orig', TMP / 'harness_sim_zip.py', 'harness 模拟原件', scope='源码'),
        asset('A12.audit_py_compat', TMP / 'audit_py_compat.py', '子工具②', scope='源码'),
        asset('A13.audit_shadow', TMP / 'audit_iqa_shadow.py', '子工具③', scope='源码'),
        asset('A14.audit_shadow_hard', TMP / 'audit_iqa_shadow_hard.py', '子工具④', scope='源码'),
        asset('A15.audit_format', TMP / 'audit_official_format.py', '子工具⑤', scope='源码'),
        asset('A16.score_pred2', TMP / 'score_pred2.py', '分数工具（v3 参数）', scope='源码'),
        asset('A17.gate_workcopy', HERE.parent / 'preupload_gate_rcon.py',
              'RCON-01 门禁工作副本（待审查）', scope='源码'),
        asset('A18.gate_tests', HERE.parent / 'tests/test_gate_failure_propagation.py',
              'RCON-01 CPU 测试', scope='测试'),
    ]

    drift = []
    for name, want in REVIEW_HASHES.items():
        got = hash_of(TMP / name)
        drift.append({'file': str(TMP / name), 'review_recorded': want, 'current': got,
                      'drift': (got != want), 'verified_at': NOW})
    drifted = [d for d in drift if d['drift']]

    containers = {
        'merged_weights': safe_container_peek(MDIR / 'final_compliant/model_dir/your_model.pt'),
        'base_ema': safe_container_peek(
            CFIN / 'preset/ckpts/VOSR_0.5B_os/checkpoints/ema_model.safetensors'),
        'lora_v7_s5000': safe_container_peek(MDIR / 'vosr_lora/v7/lora_s5000.pt'),
    }

    relations = [
        {'id': 'R01', 'subject': 'M24 output_dir 的 100 张 jpg',
         'known_recipe_recorded': 'v7_s5000 + LoRA scale 0.8 + align=wavelet + seed 8 + tile512/ov32 + vae1024（README §7 / Q11 反推）',
         'proven_generated_by': 'UNKNOWN',
         'why': '没有生成 manifest（命令、权重哈希、输入清单、输出哈希）落盘可查；Q11 只是 6 图相似度，不是身份认证（TR29）',
         'status': 'UNKNOWN'},
        {'id': 'R02', 'subject': 'your_model.pt（合并权重）',
         'known_recipe_recorded': 'runner 文档字符串自述 VOSR 0.5B os + v7_s5000 LoRA(sc0.8, 已合并进权重)',
         'proven_generated_by': 'UNKNOWN',
         'why': '没有“基座+LoRA+scale -> 合并权重”的转换记录或等价性验证；D18 §3 要求转换记录，现缺',
         'status': 'UNKNOWN'},
        {'id': 'R03', 'subject': 'runner 512 前向 与 离线 4K 出图 的“同一模型主体”声明',
         'known_recipe_recorded': 'runner 第 17 行声明“与出图(output_dir)同一模型主体同一套权重”',
         'proven_generated_by': 'NOT_PROVEN',
         'why': '两条入口在精度/VAE 后验/分块/颜色后处理/随机数上均不同（见 code_citations），'
                '且没有成对复现验收；声明是文本，不是证据（TR05/TR27/D18）',
         'status': 'CONFLICT'},
        {'id': 'R04', 'subject': '本地 final_compliant/output_dir 与归档 Track2_M24_v3.zip 的关系',
         'known_recipe_recorded': ('无直接记录。progress §4「同图核验」的 100/100 一致指的是'
                                   '【归档包之间】（M24img_V4grun / M24img_V7ganrun / ProbeSlow8 vs M24_v3），'
                                   '并不覆盖本地目录；返修前把两者错接，现更正'),
         'proven_generated_by': 'SAMPLED_n3（本轮，仅 case1/case50/case100）',
         'why': ('仅 n=3 抽样（见 archive_spot_check）；不得外推为 100/100，也不证明任何生成来源'),
         'status': 'FACT(仅 n=3，局限范围)'},
        {'id': 'R05', 'subject': 'v1–v3 LoRA 候选',
         'known_recipe_recorded': '旧任务单曾把 v1/v2/v3 列为候选',
         'proven_generated_by': 'NOT_AVAILABLE',
         'why': '目录内无 .pt；其他位置备份 UNKNOWN；不得据日志/指标文件名推定候选存在（TR26）',
         'status': 'MISSING'},
    ]

    tr = [
        {'tr': 'TR06', 'statement': '主力 v7 家族含历史可微 CLIPIQA 训练（w_clip=2.0）',
         'evidence': 'train_vosr_lora.py:226,323 + D:/models/vosr_lora/v7/train.log:1',
         'status': 'FACT / 已披露，不隐藏；官方是否接受 UNKNOWN'},
        {'tr': 'TR26', 'statement': 'v1–v3 本目录无权重；v5 命名不同',
         'evidence': 'lora_availability 块（本清单）', 'status': 'FACT'},
        {'tr': 'TR27', 'statement': '旧任务单用固定 M24 runner 配新候选图，无法保证同权重',
         'evidence': 'relations R03 + decisions D17/D18', 'status': 'CONFLICT（策略已由 D17 取代，未执行）'},
        {'tr': 'TR28/TR29', 'statement': 'Q10 六配置只 A/C 完成、B 未完成；Q11 是 6 图近似',
         'evidence': 'protocol_artifacts 块', 'status': 'FACT / ASSUMPTION（不得称协议轴闭合）'},
        {'tr': 'TR32', 'statement': 'eval.py 指标创建失败即省略、逐图异常记 None、均值只算有效值',
         'evidence': 'eval.py:60,61,101,103（自校验引用）', 'status': 'FACT，未修'},
        {'tr': 'TR33', 'statement': 'eval.py 同 case 后者覆盖、尺寸不一致自动缩放',
         'evidence': 'eval.py:54,113（自校验引用）', 'status': 'FACT，未修'},
        {'tr': 'TR34', 'statement': 'PNG/JPG、样本 ID、排序、裁切口径需进协议；test30 非 case1–30',
         'evidence': 'q11_align_screen.py 直测 PNG；q_fit_measure2.py 取字典序前 N',
         'status': 'FACT，未修'},
        {'tr': 'TR35', 'statement': 'score_pred2.py 实参 C=2.0/p=0.19，与文档 v3.1(2.515/0.198) 不同',
         'evidence': 'score_pred2.py:17-19（自校验引用）', 'status': 'FACT；本轮不改 SP'},
        {'tr': 'TR36', 'statement': 'Q 拟合存在资产-成绩身份错配（s090 目录配 34.0989）',
         'evidence': 'q_fit2.py:17 + asset_profiles.json', 'status': 'FACT，未修；该回归不得用作选优依据'},
        {'tr': 'TR37', 'statement': 'GT 通配符未展开 -> GT_val n=0',
         'evidence': 'q_fit_measure2.py:46,47,64（自校验引用）', 'status': 'FACT，未修'},
        {'tr': 'TR41', 'statement': '门禁实为 10 项（5 基础 + 兼容 + harness + 2 遮蔽 + 格式）',
         'evidence': 'preupload_gate.py 与工作副本的 TOOL_ORDER/local_checks', 'status': 'FACT'},
        {'tr': 'TR42', 'statement': '门禁假阳性：忽略子退出码、接受 infer OK 中间态、复用 SIM_DIR、SKIP 计 PASS',
         'evidence': '工作副本 + tests/test_gate_failure_propagation.py（18 项）',
         'status': 'FIXED_IN_WORKCOPY（仅在受跟踪工作副本内；原调用入口未切换，待 Architect 审查）'},
        {'tr': 'TR52', 'statement': 'score_ledger.json 内新旧快照并存，消费者读不同键得到不同现状',
         'evidence': 'score_ledger.json（本轮未改）', 'status': 'FACT，未修；本清单不指定其为真源'},
    ]

    manifest = {
        'manifest_id': 'RCON-01-E-20260916-R2',
        'rework_r2': [
            'A 只读复跑副作用: 默认改为计划模式（不加 --execute-tools 不调用任何子工具）',
            'A --stage-upload: 拒绝后立即停止，不再调用任何子工具',
            'B --sim-dir 绑定: 要求目录本次被重建 + harness 自报解压数量与包内 output_dir 成员数一致',
            'C 成功标记: 改为整行相等（NOT_X / X_FAILED 不再判 PASS）；失败标记按整词',
            'D 证据更正: R04 错接更正；Q11 行号 15 并纳入自校验；记录生成器自哈希；资产路径漂移登记',
        ],
        'kind': 'FACT清单 + UNKNOWN登记',
        'generated_at': NOW,
        'generator': 'workspace/tools/reconciliation/evidence/build_manifest.py',
        'scope': 'CPU / 只读输入；无 GPU 前向、无质量比较、无真实测速、无官方核验',
        'disclaimer': ('本清单是新增审计产物，不迁移旧 JSON schema，不指定既有台账哪一个自动胜出；'
                       'manifest 只登记可核验事实与显式 UNKNOWN，不证明“同主体通过/合规通过/候选已生成”。'),
        'git': {'head': 'e50000c',
                'note': ('本轮未提交/未推送；门禁工作副本与测试是【仓库内未跟踪工作副本】'
                         '（Architect 裁决：不需要 git add，索引保持不变）')},
        'assets': assets,
        'safe_container_peek': containers,
        'source_hash_drift': {'review_recorded': REVIEW_HASHES, 'checked': drift,
                              'any_drift': bool(drifted)},
        'lora_availability': lora_availability(),
        'protocol_artifacts': protocol_artifacts(),
        'name_agreement': name_agreement(),
        'archive_spot_check': spot_check_output_vs_archive(),
        'code_citations': code_citations(),
        'identity_relations': relations,
        'asset_path_drift': asset_path_drift(),
        'generator_identity': generator_identity(),
        'tr_registration': tr,
        'unknowns': [
            'M24 100 张图的完整生成 manifest（命令/权重哈希/输入清单/输出哈希）',
            'your_model.pt 由基座+LoRA(s5000,sc0.8) 合并而来的转换记录或等价性验证',
            'runner 与离线两条入口的成对逐像素复现与数值误差上限',
            'v1–v3 权重是否存在于本机其他位置或外部备份',
            '官方对 JIT 形式、同主体粒度、IQA 训练历史的实际口径',
            'Q10 未完成配置（B/D/E/F/REF）的产物状态与是否放弃',
        ],
    }

    (HERE / 'manifest_assets.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    (HERE / 'source_hash_record.json').write_text(
        json.dumps({'generated_at': NOW, 'review_recorded': REVIEW_HASHES,
                    'checked': drift, 'any_drift': bool(drifted),
                    'workcopy': {'path': str(HERE.parent / 'preupload_gate_rcon.py'),
                                 'sha256': hash_of(HERE.parent / 'preupload_gate_rcon.py')},
                    'tests': {'path': str(HERE.parent / 'tests/test_gate_failure_propagation.py'),
                              'sha256': hash_of(HERE.parent / 'tests/test_gate_failure_propagation.py')},
                    'generator': generator_identity(),
                    'asset_path_drift': asset_path_drift()},
                   ensure_ascii=False, indent=1), encoding='utf-8')
    diff_rec = write_diff()

    stale = [c['id'] for c in manifest['code_citations'] if c['status'] != 'OK']
    lines = ['# RCON-01 证据清单摘要（工作包E）', '',
             '- manifest: evidence/manifest_assets.json（生成于 %s）' % NOW,
             '- 资产条目: %d（存在 %d / 缺失 %d）' % (
                 len(assets), sum(1 for a in assets if a['exists']),
                 sum(1 for a in assets if not a['exists'])),
             '- 源码哈希漂移: %s' % ('无（4/4 与 review 记录一致）' if not drifted
                                     else '有 -> %s' % [d['file'] for d in drifted]),
             '- 工作副本差异: +%d/-%d 行，记录见 evidence/gate_workcopy_diff.txt'
             % (diff_rec['added'], diff_rec['removed']),
             '- 代码引用自校验: %d 条，STALE %d 条 %s'
             % (len(manifest['code_citations']), len(stale), stale or ''),
             '- 生成/合并关系: %d 条，其中 UNKNOWN/CONFLICT/MISSING %d 条'
             % (len(relations), sum(1 for r in relations if r['status'] != 'FACT(局限范围)')),
             '- 归档抽样: %s（n=3）' % manifest['archive_spot_check'].get('status'),
             '- 出图名与官方测试集一致: %s' % manifest['name_agreement'].get('same_set'),
             '', '## 安全读取边界', '',
             '- 全程未 import torch、未 torch.load、未 unpickle；.pt 只做 zipfile 目录读取，',
             '  .safetensors 只解析头部 JSON；大文件用流式 SHA-256，未复制任何资产。',
             '', '## UNKNOWN（不得由目录名或注释填补）', '']
    lines += ['- %s' % u for u in manifest['unknowns']]
    # 用户 2026-09-16 指令：md 文档统一放 docs/；本目录只留机器可读产物与日志。
    print('\n'.join(lines))

    print('manifest 写入: %s' % (HERE / 'manifest_assets.json'))
    print('资产 %d 条; 漂移 %s; 引用STALE %d; 差异 +%d/-%d'
          % (len(assets), bool(drifted), len(stale), diff_rec['added'], diff_rec['removed']))
    print('容器安全读取: %s' % {k: v.get('status') for k, v in containers.items()})
    print('归档抽样: %s' % manifest['archive_spot_check'].get('status'))
    print('名称一致: %s' % manifest['name_agreement'].get('same_set'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
