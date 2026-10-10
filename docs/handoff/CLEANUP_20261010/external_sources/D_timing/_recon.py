import json,sys
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
d=json.load(open(r'D:/tmp_timing/score_ledger.json',encoding='utf-8'))
print('KEYS:',list(d.keys()))
print('=== accounts ===')
for k,v in d['accounts'].items():
    print(k,'::',json.dumps(v,ensure_ascii=False)[:700])
print('=== derived ===')
for k,v in d['derived'].items():
    print(k,'=',json.dumps(v,ensure_ascii=False)[:320])
print('=== baseline_best ===',d['baseline_best'])
print('=== score_model_v3 ===')
print(json.dumps(d['score_model_v3_20260915pm'],ensure_ascii=False,indent=1)[:2000])
print('=== entries (compact) ===')
for e in d['entries']:
    print(json.dumps(e,ensure_ascii=False)[:260])