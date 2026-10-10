import json,sys
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
d=json.load(open(r'D:/tmp_timing/score_ledger.json',encoding='utf-8'))
print(json.dumps(d['derived'].get('val_fullref_20260915'),ensure_ascii=False,indent=1))
print('---- pixrestore_axis ----')
print(json.dumps(d['derived'].get('pixrestore_axis_20260914'),ensure_ascii=False,indent=1)[:1500])