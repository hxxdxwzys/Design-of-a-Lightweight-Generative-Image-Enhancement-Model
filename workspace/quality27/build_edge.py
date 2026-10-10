"""Copy only known assets into an independent review package; never modify source assets."""
import json,shutil
from pathlib import Path
from common import ROOT,PARENT,digest
PKG=ROOT/'V4t1_EdgeFix_Review'
src=Path(__file__).parent


def seal():
    m={'version':'V4t1_EdgeFix_20260927','status':'ENGINEERING_REVISION_NOT_ORIGINAL_SCORED_SUBMISSION',
       'source_model_sha256':digest(PARENT),'official_score':None,
       'organizer_acceptance_of_post_deadline_changes':'UNKNOWN',
       'files':{p.relative_to(PKG).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size}
                for p in sorted(PKG.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
    (PKG/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    (PKG/'model_dir').mkdir(parents=True,exist_ok=False)
    (PKG/'samples').mkdir();(PKG/'verification').mkdir()
    for name in ['your_model.pt','runner.py']:shutil.copy2(PARENT.parent/name,PKG/'model_dir'/name)
    for name in ['infer.py','selfcheck.py','test_edge.py']:shutil.copy2(src/name,PKG/name)
    old=Path('D:/models/review_V3_20260926/V3_Review_Final')
    for name in ['requirements.txt','environment.json']:shutil.copy2(old/name,PKG/name)
    for case in ['case3','case17']:shutil.copy2(old/'samples'/(case+'_in.jpg'),PKG/'samples'/(case+'_in.jpg'))
    seal();print(str(PKG))
