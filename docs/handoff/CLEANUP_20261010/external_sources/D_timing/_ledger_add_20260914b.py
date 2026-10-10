# -*- coding: utf-8 -*-
import json, shutil
P = r'D:\tmp_timing\score_ledger.json'
d = json.load(open(P, encoding='utf-8'))
shutil.copy(P, P + '.bak_20260914b')

def has(f):
    return any(e.get('file') == f for e in d['entries'])

new = []
if not has('Track2_PixB.zip'):
    new.append({
        "acct": "C", "date": "2026-09-14", "file": "Track2_PixB.zip",
        "model": "PixRestore-b 单步 (像素空间条件流匹配 DiT + DINOv2-B 条件, 403MB, 家族外资产)",
        "score": None, "status": "待评测", "ratio": 1.8,
        "val": {"psnr": 23.709, "ssim": 0.9199, "lpips": 0.4107, "ms512": 155.9},
        "test30": {"clipiqa": 0.3143, "niqe": 5.4254},
        "ms512_verified": 155.91,
        "note": "家族外第一个参照系: 有参三项全面优于 M24(22.615/0.9016/0.4408) 且快 2.4 倍. 若官方分>43.7 => 有参保真轴成立, 我们整周方向错了. 风险: 随机张量输入输出 std 0.027 (官方 repo 同行为, 非 bug; 真图 std 0.192), 若官方测速+sanity check 可能'测速失败'(不计次). 判据见 新世界模型分数对应表.md 一·C"
    })
if not has('Track2_TinyV7gan.zip'):
    new.append({
        "acct": "C", "date": "2026-09-14", "file": "Track2_TinyV7gan.zip",
        "model": "v7gan 0.87M (GAN 蒸馏 M24 观感, 256 域 + CUDA Graph)",
        "score": None, "status": "待评测", "ratio": 60.3,
        "test30": {"clipiqa": 0.5059, "niqe": 5.3249},
        "ms512_verified": 4.62,
        "note": "GAN 观感路线: clipiqa 0.3897(v5d)->0.5059(v7gan). 若官方分明显>33 (v5d 档) => GAN 观感真有官方价值, v8 宽模型值得做; 若仍 ~33 => 观感代理(clipiqa)也是假的"
    })
d['entries'].extend(new)

d['derived']['pixrestore_axis_20260914'] = {
    "what": "PixRestore (PolyU+OPPO 2026.8 开源) = 像素空间条件流匹配扩散, 支持 1 步 (DINO-GAN 蒸馏), 1x 同分辨率复原. 本项目第一个家族外权重.",
    "val_512_center_crop": {
        "LQ":              {"psnr": 23.953, "ssim": 0.9249, "lpips": 0.4983},
        "M24":             {"psnr": 22.615, "ssim": 0.9016, "lpips": 0.4408, "ms": 366.2, "ratio": 0.78},
        "PixRestore_b":    {"psnr": 23.709, "ssim": 0.9199, "lpips": 0.4107, "ms": 155.9, "ratio": 1.8},
        "PixRestore_l":    {"psnr": 23.659, "ssim": 0.9187, "lpips": 0.3876, "ms": 2521,  "ratio": 0.11}
    },
    "reading": "PixRestore-b 在有参三项上全面优于我方最佳 M24, 且快 2.4 倍; 唯一劣势是'平' (test30 clipiqa 0.3143 vs M24 0.9023)",
    "decision_tree": {
        "gt_43.7": "有参保真轴成立 => PixRestore 成新基线 (试 l/xl, 10 步; HYPIR 合规 runner)",
        "30_43":   "有参轴有效但不足 => 保真+冲击力组合",
        "~30":     "感知/冲击力轴主导 => 回 GAN 路线 (v7gan 0.5059 -> v8 宽模型)"
    },
    "risk": "IFACE 自查: 随机张量输出 std 0.027 < 0.05 阈值. 官方 repo 路径同样行为 (OOD 输入), 真图 std 0.192 => 不是 bug. 若官方'测速失败', 结论应是'runner 需要非随机输入兜底', 不是'有参轴不成立'"
}

d['derived']['latency_remeasure_20260914'] = {
    "why": "发现链子内时延读数受 GPU 占用影响 (v7gan 链内 8.44ms/33.8x, 空闲复测 4.62-4.80ms/59.5-61.9x)",
    "verified_idle": {
        "package_pkg_TinyV7gan": {"ms": 4.62, "ratio": 61.9},
        "package_pkg_TinyV5d":   {"ms": 4.65, "ratio": 61.4},
        "package_pkg_TinyV2g":   {"ms": 4.69, "ratio": 60.9},
        "package_pkg_PixB":      {"ms": 155.91, "ratio": 1.8}
    },
    "lesson": "台账里的加速比多为链内读数, 系统性偏保守 ~14% (TinyV2g 记 53.9x, 空闲 60.9x); 关键判决前必须空闲复测"
}

d['_最后更新'] = '2026-09-14 夜'
json.dump(d, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('entries', len(d['entries']))
print('derived', list(d['derived'].keys()))
