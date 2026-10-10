"""Final packaging and independent read-back, with no model execution."""
import io,json,shutil,subprocess,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
from common import ROOT,PARENT,digest
from build_edge import PKG,seal
from selfcheck import verify_identity


def main():
    a=json.loads((ROOT/'edge_phase1.json').read_text());b=json.loads((ROOT/'edge_phase2.json').read_text())
    assert a['status']==b['status']=='PASS' and a['pid']!=b['pid']
    for k in ['model_sha256','runner_sha256','infer_sha256']:
        assert a[k]==b[k]
    for c in ['case3','case17']:
        assert a['cases'][c]['output_sha256']==b['cases'][c]['output_sha256']
        assert a['cases'][c]['pre_jpeg_array_sha256']==b['cases'][c]['pre_jpeg_array_sha256']
    assert digest(PKG/'model_dir/your_model.pt')==digest(PARENT)==a['model_sha256']
    assert digest(PKG/'infer.py')==a['infer_sha256']
    for name in ['edge_phase1.json','edge_phase2.json','edge_preregistration.json','edge_ledger.json']:
        shutil.copy2(ROOT/name,PKG/'verification'/name)
    shutil.copy2(Path(__file__).parent/'test_edge.py',PKG/'test_edge.py')
    shutil.copy2(ROOT/'ppt_work/output/Proposal_OnePage_V4t1_EdgeFix.pptx',PKG/'Proposal_OnePage_V4t1_EdgeFix.pptx')
    (PKG/'environment.json').write_text(json.dumps(b['environment'],indent=2),encoding='utf-8')
    rows=[]
    for c in ['case3','case17']:
        rows.append({'input':'samples/'+c+'_in.jpg','output':'samples/'+c+'_out.jpg',
                     'input_sha256':a['cases'][c]['input_sha256'],'output_sha256':a['cases'][c]['output_sha256'],
                     'pre_jpeg_array_sha256':a['cases'][c]['pre_jpeg_array_sha256'],
                     'model_sha256':a['model_sha256'],'runner_sha256':a['runner_sha256'],'infer_sha256':a['infer_sha256'],
                     'generation_execution':'verification/edge_phase1.json','reproduction_execution':'verification/edge_phase2.json',
                     'recipe':{'tile':512,'overlap':32,'exterior_weight':1,'uint8':'nearest-even','jpeg_quality':98,'subsampling':0}})
    manifest={'schema':'image-model-binding.v1','bundle_id':'V4t1_EdgeFix_20260927-'+a['model_sha256'][:12],
              'scope':'two regenerated samples only, not 100','files':rows}
    (PKG/'generation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    cp=subprocess.run([sys.executable,'-B',str(PKG/'test_edge.py')],capture_output=True,text=True)
    (PKG/'verification/cpu_tests.txt').write_text(cp.stdout+cp.stderr,encoding='utf-8')
    if cp.returncode:raise RuntimeError('Package CPU tests failed')
    seal();verify_identity(PKG)
    target=ROOT/'Track2_V4t1_EdgeFix_Review_20260927.zip'
    # Some source members retain Unix epoch mtimes. Clamp ZIP metadata only;
    # numerical bytes and archived failure evidence are never rewritten.
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,strict_timestamps=False) as z:
        for p in sorted(PKG.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PKG).as_posix())
    iso=ROOT/'isolated_zip_check';iso.mkdir(exist_ok=False)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for n in z.namelist():
            p=(iso/n).resolve()
            if not p.is_relative_to(iso.resolve()):raise ValueError('Unsafe archive path')
        z.extractall(iso)
    # Actual entry script from the extracted package, not just imported build sources.
    identity=subprocess.run([sys.executable,'-B',str(iso/'selfcheck.py')],capture_output=True,text=True)
    tests=subprocess.run([sys.executable,'-B',str(iso/'test_edge.py')],capture_output=True,text=True)
    receipt={'zip':str(target),'zip_sha256':digest(target),'bytes':target.stat().st_size,
             'manifest_sha256':digest(PKG/'manifest.json'),'ppt_sha256':digest(PKG/'Proposal_OnePage_V4t1_EdgeFix.pptx'),
             'isolated_identity_exit':identity.returncode,'isolated_cpu_exit':tests.returncode,
             'isolated_identity_output':identity.stdout+identity.stderr,'isolated_cpu_output':tests.stdout+tests.stderr,
             'created_utc':datetime.now(timezone.utc).isoformat(),
             'status':'ENGINEERING_REVISION_READY_NOT_OFFICIAL_SCORE_CERTIFICATION'}
    (ROOT/'edge_delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    if identity.returncode or tests.returncode:raise RuntimeError('Isolated package test failed')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
