"""Seal the final review materials after adding the validated one-page PPT."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path('D:/models/review_V3_20260926')
DST = ROOT/'V3_Review_Final'
ARCHIVE = ROOT/'Track2_TinyRefV3_Review_Final.zip'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    old = json.loads((ROOT/'V3_Review/manifest.json').read_text(encoding='utf-8'))
    protected = ['model_dir/your_model.pt','model_dir/runner.py','infer.py','selfcheck.py','verify_reproduction.py',
                 'requirements.txt','samples/case3_in.jpg','samples/case3_out.jpg','samples/case17_in.jpg','samples/case17_out.jpg']
    for rel in protected:
        if sha(DST/rel) != old['files'][rel]['sha256']:
            raise ValueError('GPU-tested protected asset changed: '+rel)
    ppt = DST/'Proposal_OnePage_V3.pptx'
    with zipfile.ZipFile(ppt) as z:
        slides = [n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')]
        if len(slides) != 1 or z.testzip() is not None:
            raise ValueError('PPT page count or CRC failed')
        xml = ET.fromstring(z.read(slides[0]))
        runs = [e.text or '' for e in xml.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t')]
        text = '\n'.join(runs)
        for required in ['TinyRef V3','876,662','871,139','5,523','63','case3','case17','逐字节一致']:
            if required not in text: raise ValueError('Missing required slide fact: '+required)
        for stale in ['11K','6×8','零时延','0.8042','NOT_READY']:
            if stale in text: raise ValueError('Stale claim on slide: '+stale)
    receipt = json.loads((ROOT/'ppt_work/private/validation.json').read_text(encoding='utf-8'))
    if sha(ppt) != receipt['finalSha256']:
        raise ValueError('PPT differs from validated artifact')
    final_notes = {
        'status':'READY_FOR_USER_SUBMISSION',
        'scope':'local completeness and stated two-sample reproduction only; not official certification',
        'ppt_sha256':sha(ppt),'ppt_slides':1,'native_text_runs':len(runs),
        'rendered_preview_reviewed':True,'native_powerpoint_opened':False,
        'protected_assets_unchanged':{rel:sha(DST/rel) for rel in protected},
        'new_GPU_calls':0,'user_override':'User explicitly requested no slidemind for this task',
        'original_draft_manifest_sha256':sha(ROOT/'V3_Review/manifest.json'),
        'official_submission_id':'not independently obtained; user to match platform record',
        'uploaded':False,
    }
    (DST/'verification/final_materials.json').write_text(json.dumps(final_notes,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest = old.copy()
    manifest['delivery_status'] = 'READY_FOR_USER_SUBMISSION_NOT_OFFICIAL_CERTIFICATION'
    manifest['previous_draft_manifest_sha256'] = sha(ROOT/'V3_Review/manifest.json')
    manifest['files'] = {p.relative_to(DST).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
                         for p in sorted(DST.rglob('*')) if p.is_file() and p != DST/'manifest.json'}
    (DST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,'-B',str(DST/'selfcheck.py'),'--identity-only'],check=True)
    with zipfile.ZipFile(ARCHIVE,'x',zipfile.ZIP_DEFLATED,compresslevel=6,strict_timestamps=False) as z:
        for p in sorted(DST.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(DST).as_posix())
    isolated = Path(tempfile.mkdtemp(prefix='final_isolated_',dir=ROOT)).resolve()
    with zipfile.ZipFile(ARCHIVE) as z:
        names = z.namelist()
        expected = set(manifest['files']) | {'manifest.json'}
        if len(names) != len(expected) or set(names) != expected or z.testzip() is not None:
            raise ValueError('Archive CRC/set failed')
        for name in names:
            if not (isolated/name).resolve().is_relative_to(isolated): raise ValueError('Unsafe ZIP path')
            if hashlib.sha256(z.read(name)).hexdigest() != sha(DST/name): raise ValueError('ZIP byte mismatch')
        z.extractall(isolated)
    results = []
    for command in [[sys.executable,'-B',str(isolated/'selfcheck.py'),'--identity-only'],
                    [sys.executable,'-B',str(isolated/'verification/test_review.py')]]:
        result = subprocess.run(command,cwd=isolated,capture_output=True,text=True,encoding='utf-8')
        results.append({'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    report = {'status':'PASS' if all(r['exit_code']==0 for r in results) else 'FAIL',
              'archive':str(ARCHIVE),'zip_sha256':sha(ARCHIVE),'zip_bytes':ARCHIVE.stat().st_size,
              'zip_members':len(names),'ppt_sha256':sha(ppt),'manifest_sha256':sha(DST/'manifest.json'),
              'isolated_directory':str(isolated),'checks':results,'GPU_calls':0,
              'visual_review':'Rendered final PPTX reviewed, no clipping/overlap observed; no native PowerPoint verification',
              'uploaded':False}
    for name, expected in [('Track2_TinyRefV3.zip','d3bf0bf602e9d2ec6459ace57753fd0188189aaacf7b1ca36fda5e1d590ca712'),
                           ('Track2_TinyRefV4t1.zip','9502384034af5fb4a87d9c89eeedc191eb89008c52dbbef75fa854193cc94bad')]:
        actual=sha(Path('D:/models/upload')/name)
        if actual!=expected: raise ValueError('Original changed: '+name)
    report['original_archives_unchanged']=True
    (ROOT/'evidence/final_materials_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))
    if report['status']!='PASS': raise RuntimeError('Isolated validation failed')


if __name__ == '__main__': main()
