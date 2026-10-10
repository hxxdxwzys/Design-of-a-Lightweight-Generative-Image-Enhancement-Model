import json
from pathlib import Path
P = Path(r'D:/tmp_timing/score_ledger.json')
d = json.loads(P.read_text(encoding='utf-8'))
d['entries'] += [
 {"acct":"C","date":"2026-09-14","file":"Track2_TinyV2g.zip","model":"TinyV2 图(逐像素同) + CUDA Graph runner, 53.9x",
  "score":33.2872,"status":"完成","ratio":53.9,
  "note":"速度单变量: 19.5x->53.9x = +3.86 分 / 1.47 翻倍 = 2.63/翻倍 (1x 以上汇率)"},
 {"acct":"C","date":"2026-09-14","file":"Track2_TinyV5d.zip","model":"蒸馏 M24 观感(256域+G)A Graph), 53.9x",
  "score":33.0382,"status":"完成","ratio":53.9,
  "test30":{"clipiqa":0.3897,"niqe":5.996,"lapvar":91.7,"LFdiff":3.461},
  "note":"与 TinyV2g 同速度档、图像差异大(LFdiff 1.10->3.46)却同分 => LFdiff/结构轴证伪"}]
d['derived']['falsified_axes'] = d['derived'].get('falsified_axes', {})
d['derived']['falsified_axes'].update({
 "detail_lapvar": "ProbeDetail lapvar x2.27 -> -0.63",
 "fidelity_toward_LQ": "ProbeBlend alpha0.85(有参三指标全改善) -> -3.16",
 "structure_LFdiff": "TinyV5d LFdiff 1.10->3.46 同速度 -> -0.25 (噪声级)",
 "sharpening": "本地对照: v5 图 lapvar 92->296, clipiqa 0.392->0.379 (不动)",
 "survivors": {"speed": "1x以下 4.98-6.85/翻倍; 1x以上 2.63/翻倍",
               "M24_texture_cell": "lapvar 508-762 + clipiqa 0.90 -> 43.6951 (唯一未复现的档)"}})
P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('entries =', len(d['entries']))
