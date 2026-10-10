import json,hashlib,sys
from pathlib import Path
from PIL import Image
from common import ROOT,LQ,TGT,BASE,PARENT,digest

ROOT.mkdir(exist_ok=False)
groups=[{p.name for p in d.glob('*.jpg')} for d in [LQ,TGT,BASE]]
assert groups[0]==groups[1]==groups[2] and len(groups[0])==2839
names=sorted(groups[0],key=lambda s:hashlib.sha256(s.encode()).hexdigest())
records=[]
for name in names:
    r={'name':name,'files':{}}
    for kind,d in [('lq',LQ),('target',TGT),('student_cache',BASE)]:
        p=d/name
        with Image.open(p) as im:
            if im.size!=(512,512):raise ValueError(str(p))
        r['files'][kind]={'sha256':digest(p),'bytes':p.stat().st_size}
    records.append(r)
data={'parent':str(PARENT),'parent_sha256':digest(PARENT),'validation':names[:128],'train':names[128:],
      'records':records,'note':'Holdout only for this fine-tune. Parent may have seen these samples.'}
(ROOT/'data_manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print('DATA_READY',len(data['train']),len(data['validation']),flush=True)
