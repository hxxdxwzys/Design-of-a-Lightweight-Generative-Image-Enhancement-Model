import json
from pathlib import Path
P = Path(r'D:/tmp_timing/score_ledger.json')
d = json.loads(P.read_text(encoding='utf-8'))
d['entries'] += [
 {"acct": "C", "date": "2026-09-14", "file": "Track2_TinyV1.zip",
  "model": "微扩散 1 步 0.87M (512 域), 6.1x", "score": 27.3521, "status": "完成", "ratio": 6.12,
  "test30": {"clipiqa": 0.3367, "niqe": 5.7899, "lapvar": 139.4},
  "note": "质量项崩: clipiqa 0.34 (M24 0.90, LQ 0.275) -> 学的是温和去噪"},
 {"acct": "C", "date": "2026-09-14", "file": "Track2_TinyV2.zip",
  "model": "微扩散 1 步 0.87M (256 域), 19.5x", "score": 29.4225, "status": "完成", "ratio": 19.5,
  "test30": {"clipiqa": 0.3532, "niqe": 5.8442, "lapvar": 111.0},
  "note": "同上; 速度更快(+2.07)但质量同样崩"}]
d['derived']['tiny_model_diagnosis'] = {
 "finding": "官方质量项 ~ '看被增强过多少' (clipiqa-like); 微模型只做温和去噪, clipiqa 掉到 LQ 档",
 "test_clipiqa": {"M24": 0.9023, "TinyV1": 0.3367, "TinyV2": 0.3532, "LQ": 0.2752},
 "fix": "蒸馏 M24 的输出 (而不是复原 GT), 让微模型继承 M24 的观感档位",
 "speed_axis_still_real": "Slow8 (图逐字节同 M24, 时延x8) = -14.94"}
P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('entries =', len(d['entries']))
