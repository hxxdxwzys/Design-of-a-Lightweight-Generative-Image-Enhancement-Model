"""Read-back audit and aggregate existing evidence; no GPU inference."""
import json,subprocess,sys
from pathlib import Path
from common import ROOT,digest


def main():
    load=lambda n:json.loads((ROOT/n).read_text(encoding='utf-8'))
    v=load('full_val.json');e1=load('edge_phase1.json');e2=load('edge_phase2.json')
    train=load('validate_2000.json');long=load('validate_6000.json')
    assert not v['gate'] and not load('export_check.json')['pass'] and not load('execution_recheck.json')['pass']
    assert e1['status']==e2['status']=='PASS' and e1['pid']!=e2['pid']
    assert not (ROOT/'.gpu.lock').exists()
    old=Path('D:/models/review_V3_20260926/Track2_TinyRefV3_Review_Final.zip')
    assert digest(old)=='b7d71e71bbf08439c2a66707286b1fb489c2817697bc990de5cc1fddac3b027b'
    cp=subprocess.run([sys.executable,'-B',str(Path(__file__).parent/'test_cpu.py')],capture_output=True,text=True)
    (ROOT/'training_cpu_final.txt').write_text(cp.stdout+cp.stderr,encoding='utf-8')
    if cp.returncode:raise RuntimeError('CPU regression failed')
    result={'trained_candidate_status':'REJECTED_FOR_DELIVERY','training_steps':6000,
            'training_wall_s':265.6421712,'selected_steps':2000,'selection_holdout':train['mean'],
            'step6000_holdout':long['mean'],'full_validation':v['mean'],'full_quality_gate':False,
            'full_val_count':3,'metric_calls':105,'render_and_verify_reserved':v['render_calls']+load('edge_ledger.json')['used'],
            'export_conservative_charges':22,'export_cap_after_preregistered_execution_repair':32,
            'new_training_weights_delivered':False,'test30_performed':False,'production100_performed':False,
            'edge_revision_status':'ACCEPTED_ENGINEERING_SCOPE_ONLY','edge_gpu_calls':262,
            'edge_cpu_tests':10,'conditional_cpu_tests':4,'original_v3_zip_unchanged':True,
            'new_tree_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
            'authority_boundary':'No upload, no Git add/commit, no change to original V3/V4t1 archives; official score and replacement permission UNKNOWN',
            'evidence':{p.name:digest(p) for p in ROOT.glob('*.json') if p.name!='closeout.json'}}
    (ROOT/'closeout.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
