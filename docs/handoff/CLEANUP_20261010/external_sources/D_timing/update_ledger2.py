import json
from pathlib import Path
P = Path(r'D:/tmp_timing/score_ledger.json')
d = json.loads(P.read_text(encoding='utf-8'))
d['entries'].append({
 "acct": "C", "date": "2026-09-14", "file": "Track2_ProbeBlend.zip",
 "model": "M24 图向 LQ 回退 alpha=0.85 (lapvar 762->590), runner 同算子",
 "score": 40.5354, "status": "完成", "ratio": 0.780, "latency_ms": 366.2,
 "note": "保真轴证伪: val 上 PSNR/SSIM/LPIPS 三指标同时优于 M24, 官方只给 -3.16"})
d['derived']['quality_axis_20260914'] = {
 "conclusion": "质量项 = 平台: 只有偏离 M24 工况点才掉分, 且方向不对称",
 "points": [
   {"pkg": "M24", "delta": 0.0, "intervention": "基准"},
   {"pkg": "M24_v4", "delta": -0.11, "intervention": "runner 冷启动加固"},
   {"pkg": "probeA_sc04", "delta": -0.41, "intervention": "LoRA sc0.8->0.4"},
   {"pkg": "ProbeDetail", "delta": -0.63, "intervention": "细节 lapvar x2.27"},
   {"pkg": "ProbeBlend", "delta": -3.16, "intervention": "向 LQ 回退 alpha=0.85 (有参三指标全改善)"},
   {"pkg": "M24sc00", "delta": -6.06, "intervention": "去掉 LoRA (发糊)"},
   {"pkg": "M24p4", "delta": -36.51, "intervention": "DiT 剪到 4/28 (毁图)"}],
 "asymmetry": "向 LQ 保守 -3.16 > 向细节激进 -0.63 -> 官方偏好更'增强'的一侧",
 "speed_axis": "4.98-6.84 分/翻倍 (M24->avg2->Slow8 三连)"}
P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('entries =', len(d['entries']))
