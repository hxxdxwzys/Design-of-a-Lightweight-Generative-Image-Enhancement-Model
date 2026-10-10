import json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from common import ROOT,PARENT,EXPECTED,sha


def main():
    read=lambda name:json.loads((ROOT/name).read_text())
    v=read('full_val.json');selection=read('selection.json');s=selection['steps']
    md=ROOT/'candidates'/('s%d'%s)/'model_dir'
    assert not v['gate'] and read('reload_gate.json')['gate'] and read('holdout_3000.json')['gate']
    assert sha(PARENT)==EXPECTED and not (ROOT/'.gpu.lock').exists()
    cp=subprocess.run([sys.executable,'-B',str(Path(__file__).parent/'test_cpu.py')],capture_output=True,text=True)
    (ROOT/'cpu_final.txt').write_text(cp.stdout+cp.stderr,encoding='utf-8')
    if cp.returncode:raise RuntimeError('CPU regression failed')
    b,c=v['mean']['parent'],v['mean']['candidate']
    result={'status':'RESEARCH_CANDIDATE_PARTIAL_FIDELITY_GAIN_NOT_APPROVED_FOR_DELIVERY',
            'created_utc':datetime.now(timezone.utc).isoformat(),'selected_steps':s,
            'model_sha256':sha(md/'your_model.pt'),'runner_sha256':sha(md/'runner.py'),
            'full_val':v['mean'],'psnr_mean_delta_db':c['psnr']-b['psnr'],
            'lpips_relative_improvement':1-c['lpips']/b['lpips'],'delivery_gate':False,
            'reason':'LPIPS full-image mean improvement below preregistered 2%; no retuning.',
            'dev':read('dev_3000.json')['mean'],'holdout':read('holdout_3000.json')['mean'],
            'cpu_tests':5,'reload_final_file_gate':True,'shadow_reference_exact':False,
            'resources':read('ledger.json'),'test30_performed':False,'bench_performed':False,
            'production100_performed':False,'new_review_package_created':False,
            'old_v3_unchanged':True,'tree_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
            'files':{p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.glob('*.json') if p.name!='final_status.json'}}
    with (ROOT/'final_status.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','resources']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
