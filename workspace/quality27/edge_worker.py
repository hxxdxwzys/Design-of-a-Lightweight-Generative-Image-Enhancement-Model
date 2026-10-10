import argparse,hashlib,importlib.util,io,json,os,platform,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import PIL
import torch
from common import ROOT,digest,guard
from build_edge import PKG,seal


def module(name,file):
    s=importlib.util.spec_from_file_location(name,str(file));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def reserve(n):
    p=ROOT/'edge_ledger.json';d=json.loads(p.read_text())
    if time.time()>d['deadline_epoch'] or d['used']+n>d['cap']:raise RuntimeError('Edge verification budget exhausted')
    guard();d['used']+=n;p.write_text(json.dumps(d,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',type=int,choices=[1,2],required=True);a=ap.parse_args()
    module('edge_check',PKG/'selfcheck.py').verify_identity(PKG)
    entry=module('edge_infer',PKG/'infer.py')
    if a.phase==1:
        with (ROOT/'edge_ledger.json').open('x') as f:json.dump({'used':0,'cap':300,'deadline_epoch':time.time()+900},f,indent=2)
    reserve(4);r=entry.load_runner(PKG/'model_dir')
    reserve(1)
    with torch.no_grad(): y=r.infer(torch.zeros(1,3,512,512),prompt='')
    assert tuple(y.shape)==(1,3,512,512) and torch.isfinite(y).all() and y.min()>=-1 and y.max()<=1
    result={'phase':a.phase,'status':'RUNNING','cases':{},'pid':os.getpid(),'model_sha256':digest(PKG/'model_dir/your_model.pt'),
            'runner_sha256':digest(PKG/'model_dir/runner.py'),'infer_sha256':digest(PKG/'infer.py'),
            'environment':{'python':sys.version,'torch':torch.__version__,'Pillow':PIL.__version__,'numpy':np.__version__,
                           'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'platform':platform.platform()},
            'cuda_graph_active':r.graph is not None}
    for case in ['case3','case17']:
        p=PKG/'samples'/(case+'_in.jpg');arr=np.array(Image.open(p).convert('RGB'));H,W=arr.shape[:2]
        coords=iter((y,x) for y in entry.positions(H) for x in entry.positions(W))
        acc=np.zeros((H,W,3),np.float32);ws=np.zeros((H,W,1),np.float32)
        mask=np.ones((512,512,1),np.float32);v=np.linspace(0,1,16,dtype=np.float32)
        mask[:16,:,0]*=v[:,None];mask[-16:,:,0]*=v[::-1,None]
        mask[:,:16,0]*=v[None,:];mask[:,-16:,0]*=v[None,::-1]
        class Audited:
            def infer(self,x,prompt=''):
                reserve(1);out=r.infer(x,prompt=prompt)
                y0,x0=next(coords);z=out[0].permute(1,2,0).float().cpu().numpy()
                acc[y0:y0+512,x0:x0+512]+=np.clip((z+1)*127.5,0,255)*mask
                ws[y0:y0+512,x0:x0+512]+=mask
                return out
        start=time.perf_counter();out=entry.render(Audited(),arr);wall=time.perf_counter()-start
        old=np.clip(acc/np.maximum(ws,1e-6),0,255).astype(np.uint8)
        b=io.BytesIO();Image.fromarray(out).save(b,format='JPEG',quality=98,subsampling=0);data=b.getvalue()
        edge=np.concatenate([out[0],out[-1],out[1:-1,0],out[1:-1,-1]])
        oldedge=np.concatenate([old[0],old[-1],old[1:-1,0],old[1:-1,-1]])
        diff=np.abs(out[16:-16,16:-16].astype(np.int16)-old[16:-16,16:-16].astype(np.int16))
        dest=PKG/'samples'/(case+'_out.jpg')
        if a.phase==1:
            with dest.open('xb') as f:f.write(data)
        same=data==dest.read_bytes()
        row={'input_sha256':digest(p),'output_sha256':hashlib.sha256(data).hexdigest(),
             'pre_jpeg_array_sha256':hashlib.sha256(out.tobytes()).hexdigest(),'shape':list(out.shape),
             'wall_s':wall,'old_outer_all_black':bool(np.all(oldedge==0)),
             'new_outer_all_black':bool(np.all(edge==0)),'new_outer_mean':float(edge.mean()),
             'interior_u8max':int(diff.max()),'jpeg_identical':same}
        assert row['old_outer_all_black'] and not row['new_outer_all_black'] and row['interior_u8max']<=1 and same
        result['cases'][case]=row;print(case,json.dumps(row),flush=True)
    result['status']='PASS'
    (ROOT/('edge_phase%d.json'%a.phase)).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if a.phase==1:seal()


if __name__=='__main__':main()
