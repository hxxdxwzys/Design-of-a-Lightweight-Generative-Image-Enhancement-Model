# -*- coding: utf-8 -*-
import json, io
from pathlib import Path
P = Path(r'D:/tmp_timing/score_ledger.json')
d = json.loads(P.read_text(encoding='utf-8'))
new = [
 {"acct": "C", "date": "2026-09-13", "file": "Track2_ProbeSlow8.zip",
  "model": "M24 图(逐字节同) + runner 故意 8x 算力",
  "score": 28.7538, "status": "完成", "ratio": 0.098, "latency_ms": 2930,
  "note": "决定性: 图像不变, 仅时延 x8 -> -14.94 分 / 3.00 翻倍 = 4.98 分/翻倍"},
 {"acct": "C", "date": "2026-09-13", "file": "Track2_ProbeDetail.zip",
  "model": "M24 图 + 确定性细节增强(lapvar x2.27), runner 同算子",
  "score": 43.0699, "status": "完成", "ratio": 0.780,
  "note": "细节轴证伪: lapvar +1.18 翻倍 -> 实测 -0.63(噪声级); lapvar 只是伴随相关"},
 {"acct": "C", "date": "2026-09-13", "file": "Track2_ProbeBlend.zip",
  "model": "M24 图向 LQ 回退 alpha=0.85(lapvar x0.77), runner 同算子",
  "score": None, "status": "已提交待评测", "ratio": 0.780,
  "note": "有参保真轴: val 上 PSNR/SSIM/LPIPS 三指标 Pareto 点"},
]
d['entries'].extend(new)
d['derived']['speed_axis_new_world'] = {
 "same_image_points": [
   {"pkg": "M24", "latency_ms": 366.2, "ratio": 0.780, "score": 43.6951},
   {"pkg": "avg2", "latency_ms": 748.5, "ratio": 0.382, "score": 36.6414},
   {"pkg": "ProbeSlow8", "latency_ms": 2930, "ratio": 0.098, "score": 28.7538}],
 "slope_per_doubling": {"M24->avg2": 6.84, "avg2->Slow8": 4.00, "M24->Slow8": 4.98},
 "shape": "凹(越靠近 1x 越陡) -> 更快方向斜率 >= 5-7 分/翻倍",
 "falsified": ["lapvar/细节轴 (ProbeDetail -0.63)", "niqe/clipiqa 质量轴 (前 43.4-43.7 平台)"],
 "caveat": "加速比必须由同一模型主体真实取得(规则三); 外推不可当承诺"
}
d['baseline_best'] = 43.6951
P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('entries =', len(d['entries']))
print('ok, best =', d['baseline_best'])
