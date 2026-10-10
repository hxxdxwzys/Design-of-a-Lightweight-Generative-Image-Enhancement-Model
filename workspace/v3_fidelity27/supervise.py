import argparse,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,DEADLINE,CAPS,sha,guard


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tag');ap.add_argument('script');ap.add_argument('args',nargs=argparse.REMAINDER);a=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);guard();src=Path(__file__).parent
    lock=ROOT/'.gpu.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if not (ROOT/'ledger.json').exists():
            (ROOT/'ledger.json').write_text(json.dumps({'caps':CAPS,'used':{k:0 for k in CAPS},'deadline':DEADLINE},indent=2))
            (ROOT/'preregistration.json').write_text(json.dumps({'epoch':time.time(),'protocol':(src/'PROTOCOL.md').read_text(encoding='utf-8'),
                'protocol_sha256':sha(src/'PROTOCOL.md'),'sources':{p.name:sha(p) for p in src.glob('*.py')}},ensure_ascii=False,indent=2),encoding='utf-8')
        meta=ROOT/(a.tag+'_execution.json');log=ROOT/(a.tag+'.log')
        if meta.exists() or log.exists():raise FileExistsError('No overwrite/retry of execution tag')
        d={'status':'RUNNING','start':time.time(),'sources':{p.name:sha(p) for p in src.glob('*.py')}}
        meta.write_text(json.dumps(d,indent=2))
        with log.open('x',encoding='utf-8') as f:
            p=subprocess.Popen([sys.executable,'-B',str(src/a.script)]+a.args,stdout=f,stderr=subprocess.STDOUT,
                               env=dict(os.environ,PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'))
            try:code=p.wait(timeout=min(DEADLINE-time.time(),3600))
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait();code=124
        d.update(status='PASS' if code==0 else 'FAIL',exit_code=code,wall_s=time.time()-d['start']);meta.write_text(json.dumps(d,indent=2))
        print(json.dumps({'tag':a.tag,'status':d['status'],'wall_s':d['wall_s'],'exit_code':code}),flush=True);return code
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':sys.exit(main())
