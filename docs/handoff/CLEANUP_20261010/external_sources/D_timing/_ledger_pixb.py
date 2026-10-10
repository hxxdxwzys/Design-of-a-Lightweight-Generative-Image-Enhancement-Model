# -*- coding: utf-8 -*-
import json, shutil
P = r'D:\tmp_timing\score_ledger.json'
d = json.load(open(P, encoding='utf-8'))
shutil.copy(P, P + '.bak_20260915c')

for e in d['entries']:
    if e.get('file') == 'Track2_PixB.zip':
        e['score'] = 18.9710
        e['status'] = '已评分'
        e['note'] = ('决定性否定: 有参四项为全家族最好(DISTS 0.2907 全场最佳)且比 M24 快(1.4~1.8x vs 0.81x), '
                     '官方仅 18.9710 => "更接近真值"的路线一律不加分甚至重罚. 观感(clipiqa 0.3143≈原图档)才是主项. '
                     'PixRestore 家族据此封盘')
        break

d['derived']['score_model_v2_20260915'] = {
    'form': 'score ~= L(clipiqa_4K) + S(speed_ratio)  (加法; 两族数据共用同一条 L 曲线)',
    'L_points': {'0.31': 0.0, '0.35': 0.0, '0.39': -0.25, '0.51': 8.44, '0.64': 20.6, '0.77': 26.3, '0.90': 26.7},
    'L_sources': {'0.31': 'PixB@1.5x (18.9710)', '0.35': 'TinyV2g@66x (33.2872)', '0.39': 'TinyV5d@79x (33.0382)',
                  '0.51': 'TinyV7gan@79x (41.7346)', '0.64': 'sc00@0.81x (37.6316)', '0.77': 'probeA@0.81x (43.2837)',
                  '0.90': 'M24@0.81x (43.6951)'},
    'L_shape': '0.35->0.64 段极陡 (~71 分/单位); 0.64->0.90 段近饱和 (~23 分/单位, 仅 +6 分)',
    'S_shape': {'below_1x': '5~6.9 分/翻倍', 'above_1x': '2.63 分/翻倍'},
    'self_consistency': 'L(0.31)~=L(0.35) => PixB 预测 18.97+S(1.5) 与实测 18.9710 吻合; 速度模型把"又准又快"的 PixB 与"糊但 66x"的 TinyV2g 摆平 => 观感是主项',
    'action': [
        'PixRestore 家族封盘 (l/xl/多步不再试)',
        '唯一值钱方向: >=50x 下把观感 0.51->0.64+ (每 +0.01 clipiqa ~ +0.7 分; +0.13 ~ +12 分 => ~54 分档)',
        '杠杆: 学生自身生成能力 = 容量 / 输入分辨率 / 对抗强度 / 训练时长',
        '在跑: v9b (512 域 down=1, 0.38M, 同 GAN 配方, ~36x) 分辨率单变量',
    ],
    'caveat': 'L 由 0.51->0.64 段外推了一小段 (该段两端都有真实锚点, 但 0.55~0.63 之间没有直接读数); 不得再往 0.64 以上外推收益',
}
d['_最后更新'] = '2026-09-15 PixB 出分后'
json.dump(d, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ok; entries', len(d['entries']), '| derived', len(d['derived']))
print('PixB score =', [e['score'] for e in d['entries'] if e['file'] == 'Track2_PixB.zip'])
