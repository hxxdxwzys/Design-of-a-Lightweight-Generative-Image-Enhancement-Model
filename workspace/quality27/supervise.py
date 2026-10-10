import argparse,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
from common import ROOT,DEADLINE,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tag');ap.add_argument('script');ap.add_argument('args',nargs=argparse.REMAINDER)
    a=ap.parse_args();source=Path(__file__).parent
    lock=ROOT/'.gpu.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        meta=ROOT/(a.tag+'_execution.json');log=ROOT/(a.tag+'_stdout.log')
        if meta.exists() or log.exists():raise FileExistsError('No automatic retry of existing tag')
        if not (ROOT/'preregistration.json').exists():
            record={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(source/'PROTOCOL.md'),
                    'deadline_epoch':DEADLINE,'sources':{p.name:digest(p) for p in source.glob('*.py')},
                    'protocol':(source/'PROTOCOL.md').read_text(encoding='utf-8')}
            (ROOT/'preregistration.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        start=time.time()
        if start>=DEADLINE:raise RuntimeError('Deadline reached')
        record={'status':'RUNNING','start_epoch':start,'command':[sys.executable,'-B',str(source/a.script)]+a.args,
                'sources':{p.name:digest(p) for p in source.glob('*.py')}}
        meta.write_text(json.dumps(record,indent=2),encoding='utf-8')
        env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1')
        with log.open('x',encoding='utf-8') as f:
            p=subprocess.Popen(record['command'],stdout=f,stderr=subprocess.STDOUT,env=env)
            try:code=p.wait(timeout=min(DEADLINE-start,5400 if a.script=='train.py' else 3600))
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait();code=124
        record.update(status='PASS' if code==0 else 'FAIL',exit_code=code,wall_s=time.time()-start)
        meta.write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record),flush=True)
        return code
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':sys.exit(main())
