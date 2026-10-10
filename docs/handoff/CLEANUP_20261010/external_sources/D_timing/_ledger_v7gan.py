# -*- coding: utf-8 -*-
import json, shutil
P = r'D:\tmp_timing\score_ledger.json'
d = json.load(open(P, encoding='utf-8'))
shutil.copy(P, P + '.bak_20260915a')

hit = None
for e in d['entries']:
    if e.get('file') == 'Track2_TinyV7gan.zip':
        hit = e
if hit is None:
    raise SystemExit('entry not found')
hit['score'] = 41.7346
hit['status'] = '已评分'
hit['ratio_note'] = '链内 33.8x / 空闲 61.9x (4.62 ms)'
hit['note'] = ('决定性: 与 TinyV5d(33.0382) 同架构(0.87M)/同256域/同数据/同速度档, '
               '唯一主线差异是对抗损失(PatchGAN) => +8.70 全来自图像侧. '
               '=> 在 60x 档, 图像质量轴是活的且斜率极陡 (clipiqa 0.39->0.51 换 +8.70); '
               '§一·B 的"质量轴无空间"只在 M24 的 0.78x 工况点成立')

d['derived']['v7gan_reading_20260915'] = {
    'what': '速度固定、只换图的单变量读数 (本日最重要)',
    'points': {
        'TinyV5d':   {'arch': '0.87M/256域', 'img': 'L1+0.2LPIPS 蒸馏 M24 观感 (无GAN)', 'ms': 4.65, 'ratio': 61.4, 'score': 33.0382, 'test30_clipiqa': 0.3897, 'test30_niqe': 5.996},
        'TinyV7gan': {'arch': '0.87M/256域', 'img': '同上 + 0.1 LSGAN PatchGAN', 'ms': 4.62, 'ratio': 61.9, 'score': 41.7346, 'test30_clipiqa': 0.5059, 'test30_niqe': 5.3249}
    },
    'delta': 8.70,
    'reading': [
        '速度逐毫秒相同 => +8.70 完全来自图像侧',
        '推翻"质量轴没有可挖空间"的普适说法: 那只在 M24 的 0.78x 工况点成立',
        '合成图景: M24(0.78x, 图A)=43.70; v7gan(60x, 图B)=41.73 => 60倍加速+更差的图只掉 1.96 分',
        '高倍速区"图像质量"与"速度"是两条可同时吃的轴'
    ],
    'prereg_next': {
        'lever': '对抗纹理 (唯一被证明活的图像侧杠杆)',
        'candidates': ['更宽生成器 0.87M->3.5M (同256域同~60x)',
                       '更强/更久的对抗训练',
                       '换教师 (PixRestore / HYPIR 的更保真观感)'],
        'criterion': '本地测试集 clipiqa >= 0.55 (v7gan 0.5059) 且加速比 >= 50x => 值得占一发提交'
    }
}

d['derived']['val_fullref_20260915'] = {
    'harness': 'val 3 对 512 中心裁切, PSNR/SSIM/LPIPS(alex)/DISTS(pyiqa), 同一脚本同一进程外指标器',
    'note': 'DISTS 是官方四项里的第四项, 本项目此前从未对家族内资产测过',
    'pending': '结果由 D:/tmp_timing/sweep_val_fullref.json 汇总 (链 C 完成后回填)'
}
d['_最后更新'] = '2026-09-15 上午'
json.dump(d, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ok entries', len(d['entries']), '| v7gan score', hit['score'])
print('derived', list(d['derived'].keys()))
