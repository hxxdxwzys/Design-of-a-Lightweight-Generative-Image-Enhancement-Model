import json, numpy as np
j = json.load(open(r'D:/tmp_timing/degradation_analysis.json', encoding='utf-8'))
print('== summary ==')
print(json.dumps(j['summary'], ensure_ascii=False, indent=1)[:1500])
print('\n== blur_calibration ==')
print(json.dumps(j['blur_calibration'], ensure_ascii=False)[:600])
print('\n== noise_calibration ==')
print(json.dumps(j['noise_calibration'], ensure_ascii=False)[:400])
pi = j['per_image']
print('\nper_image n =', len(pi))
print('sample:', json.dumps(pi[:3], ensure_ascii=False))
test = [r for r in pi if r.get('tag') == 'test']
for k in ('est_sigma', 'est_n', 'jpeg_q'):
    v = np.array([r[k] for r in test if r.get(k) is not None], dtype=float)
    print(f'{k}: n={len(v)} mean={v.mean():.2f} p10={np.percentile(v,10):.2f} median={np.median(v):.2f} p90={np.percentile(v,90):.2f}')
