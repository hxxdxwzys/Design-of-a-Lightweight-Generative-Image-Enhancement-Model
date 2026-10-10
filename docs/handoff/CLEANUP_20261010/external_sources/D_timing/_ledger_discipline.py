# -*- coding: utf-8 -*-
import json, shutil
P = r'D:\tmp_timing\score_ledger.json'
d = json.load(open(P, encoding='utf-8'))
shutil.copy(P, P + '.bak_20260915d')

d['_纪律_20260915'] = {
    '触发': '2026-09-15 用户指出"求保真又被否定掉了, 不要左右脑互搏, 幻觉少一点"',
    '错误实例': '我在 09-15 上午提出"缺失杠杆可能是有参保真(PSNR/SSIM 提升)" —— 这直接违反本台账 falsified_axes.fidelity_toward_LQ '
              '(ProbeBlend α0.85: 有参三项同时改善 -> 官方 -3.16) 与 PixB 实测(有参全家族最好却只有 18.9710)',
    '规则': [
        '1) 提出任何"新方向/新假设"之前, 必须先对照 derived.falsified_axes; 若与任一直接读数冲突, 禁止作为假设提出',
        '2) 重提已证伪的轴 = 幻觉; 只能"用新的直接读数"去推翻它, 不能靠推理复活',
        '3) 60 分以上的分数模型从未被验证 => 对 124~217 的差距只能说"未知", 不得编造解释',
        '4) 每次发言区分三档: 【实测】/【有区间的推断】/【未知】; 未知就说未知',
    ],
}
d['derived']['falsified_axes']['fidelity_restated_20260915'] = \
    '再次确认(两次直接读数): 保真方向不加分。ProbeBlend -3.16; PixB(有参全域最好) 18.9710。禁止再把它当"缺失杠杆"提出。'
json.dump(d, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ledger: 纪律块已写入')
