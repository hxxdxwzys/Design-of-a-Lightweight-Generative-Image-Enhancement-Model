# -*- coding: utf-8 -*-
import json, shutil
P = r'D:\tmp_timing\score_ledger.json'
d = json.load(open(P, encoding='utf-8'))
shutil.copy(P, P + '.bak_20260915b')

V = {
 'lq':            {'ms': None,   'ratio': None,  'psnr': 23.897, 'ssim': 0.9246, 'lpips': 0.4977, 'dists': 0.3139},
 'm24':           {'ms': 352.9,  'ratio': 0.81,  'psnr': 22.480, 'ssim': 0.8966, 'lpips': 0.4476, 'dists': 0.3642, 'official': 43.6951, 'test30_clipiqa': 0.9023},
 'pkg_v2g':       {'ms': 4.3,    'ratio': 66.4,  'psnr': 23.575, 'ssim': 0.9193, 'lpips': 0.4570, 'dists': 0.3062, 'official': 33.2872, 'test30_clipiqa': 0.3532},
 'pkg_v5d':       {'ms': 3.6,    'ratio': 79.3,  'psnr': 22.639, 'ssim': 0.9078, 'lpips': 0.4575, 'dists': 0.3282, 'official': 33.0382, 'test30_clipiqa': 0.3897},
 'pkg_v7gan':     {'ms': 3.6,    'ratio': 79.3,  'psnr': 22.548, 'ssim': 0.9058, 'lpips': 0.4680, 'dists': 0.3409, 'official': 41.7346, 'test30_clipiqa': 0.5059},
 'pix_s_1step':   {'ms': 88.4,   'ratio': 3.2,   'psnr': 23.400, 'ssim': 0.9152, 'lpips': 0.4059, 'dists': 0.2935},
 'pix_b_1step':   {'ms': 199.4,  'ratio': 1.4,   'psnr': 23.707, 'ssim': 0.9199, 'lpips': 0.4109, 'dists': 0.2907},
 'pix_b_10step':  {'ms': 1559.6, 'ratio': 0.18,  'psnr': 23.265, 'ssim': 0.9193, 'lpips': 0.4412, 'dists': 0.2972},
 'pix_l_1step':   {'ms': 539.8,  'ratio': 0.53,  'psnr': 23.662, 'ssim': 0.9187, 'lpips': 0.3874, 'dists': 0.2904},
 'pix_l_10step':  {'ms': 4229.1, 'ratio': 0.07,  'psnr': 23.378, 'ssim': 0.9199, 'lpips': 0.3951, 'dists': 0.2887},
 'hypir_ct150':   {'ms': None,   'ratio': None,  'psnr': 20.786, 'ssim': 0.8730, 'lpips': 0.3959, 'dists': 0.2633, 'test30_clipiqa': 0.4478},
}

d['derived']['val_fullref_20260915'] = {
 'harness': 'val 3 对 512 中心裁切; PSNR/SSIM 自算, LPIPS=lpips(alex), DISTS=pyiqa; 每配置一个进程内同一指标器',
 'table': V,
 'three_counterexamples': [
   'M24 四项有参全部劣于 LQ (23.897/0.9246/0.4977/0.3139), 官方却 43.6951',
   '同一速度档 3.6 ms: v5d->v7gan 只加对抗损失, 四项有参全部变差, 官方 +8.70',
   'TinyV2g 四项有参全优于 v5d/v7gan, 官方只有 33.2872 (比 v7gan 低 8.45)',
 ],
 'conclusion': '官方质量项主战场 = 无参观感(clipiqa/musiq/maniqa/niqe), 不是对私有 GT 的保真; 本地有参四项只用于排除崩坏',
 'falsifies': '交接提示词 §2.2 "有参指标必然是主战场" 的推断',
 'falsified_also': [
   '多步扩散兼顾保真与纹理: PixRestore-b 10 步 PSNR/LPIPS/DISTS 三项变差, 时延 7.8x',
   'HYPIR 观感值得蒸馏: HYPIR test30 clipiqa 仅 0.4478 (v7gan 0.5059, M24 0.9023)',
   'HYPIR val LPIPS 0.3364: 同口径复测 0.3959 => 旧数字口径不同, 不得跨口径比较',
 ],
}

d['derived']['v7gan_trajectory_20260915'] = {
 'harness': 'render_subset(字典序前 30 张 = 与 measure_test30 同口径) + measure_dir',
 'points': {'8k': {'clipiqa': 0.4446, 'niqe': 5.5726, 'lapvar': 128.7},
            '16k': {'clipiqa': 0.4817, 'niqe': 5.2264, 'lapvar': 135.2},
            '25k': {'clipiqa': 0.5059, 'niqe': 5.3249, 'lapvar': 127.6}},
 'reading': '观感代理随步数单调上升, 25k 未饱和 => 训练更久是免费杠杆 (但增量递减: +0.037 8k->16k, +0.024 16k->25k)',
 'consistency_check': '25k 的 0.5059 与已提交包当时的测量逐位一致 => 我们的渲染/测量链路可复现',
}

d['derived']['perceptual_axis_20260915'] = {
 'target': '在 >=50x 速度下最大化"观感代理 clipiqa" (唯一被两个家族同时验证与官方分同向的量)',
 'clipiqa_vs_official': {'M24': [0.9023, 43.6951], 'probeA': [0.7700, 43.2837], 'sc00': [0.6448, 37.6316],
                         'v2g': [0.3532, 33.2872], 'v5d': [0.3897, 33.0382], 'v7gan': [0.5059, 41.7346]},
 'hypothesis': 'v7gan 走 down=2 (512->256->512) 输出承载不了高频纹理 => clipiqa 卡 0.51; 512 域 (down=1) 应显著抬高',
 'running': 'v9b = 512 域 ch 32,32,32 25k 步 (近同速); 完成后按 prereg_20260915.txt 判据自动分支',
 'prereg': 'D:/tmp_timing/prereg_20260915.txt',
}
d['_最后更新'] = '2026-09-15 上午'
json.dump(d, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('derived blocks:', len(d['derived']))
print(list(d['derived'].keys()))
