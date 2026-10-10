import os, subprocess, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY=r"D:\Python\python.exe"; T=Path(r"D:\tmp_timing")
EB=Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
ENV=dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
IN=r"D:\models\refine_v1\in"; TGT=r"D:\datasets\distill256\tgt"; REF="ref_c512"
def run(name, cmd, timeout=10800):
    print("\n[chain] >>> %s" % name, flush=True); t0=time.time()
    r=subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=ENV)
    print("[chain] %s rc=%d  %.1f min" % (name, r.returncode, (time.time()-t0)/60), flush=True)
    for l in [x for x in (r.stdout or "").strip().splitlines() if x.strip()][-3:]:
        print("        "+l.encode("ascii","replace").decode("ascii"), flush=True)
    if r.returncode!=0: print("[chain] STDERR:", (r.stderr or "")[-900:].encode("ascii","replace").decode("ascii"), flush=True)
    return r.returncode
print("[chain] C2: refiner TRAINING DOMAIN 256 -> 512 crops (params unchanged 16ch/2bl)", flush=True)
print("[chain] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
run("refiner512_a", [PY,"-u",str(T/"train_refiner_gan2.py"),"--in_dir",IN,"--tgt_dir",TGT,
                     "--ch","16","--blocks","2","--crop","512","--bs","4","--steps","8000",
                     "--w_gan","0.1","--tag",REF+"a"])
run("refiner512_b", [PY,"-u",str(T/"train_refiner_gan2.py"),"--in_dir",IN,"--tgt_dir",TGT,
                     "--ch","16","--blocks","2","--crop","512","--bs","4","--steps","6000",
                     "--w_gan","0.3","--lr_d","1e-4","--tag",REF+"b",
                     "--init", r"D:\models\refine_v1\refiner_%sa_s8000.pt"%REF])
ref_ck=Path(r"D:\models\refine_v1")/("refiner_%sb_s6000.pt"%REF)
if not ref_ck.exists(): print("[chain] refiner 未产出 -> stop", flush=True); sys.exit(1)
md=Path(r"D:\models\pkg_c512ref")
run("export",[PY,"-u",str(EB/"opt11_export_us.py"),"--ckpt",r"D:\models\tiny_v1\v21_s21000.pt",
              "--refiner",str(ref_ck),"--modeldir",str(md),"--unsharp","60","--sigma","1.2","--thr","1.0"],timeout=3600)
outd=Path(r"D:\models\opt11")/("eval_"+REF)
run("render30",[PY,"-u",str(EB/"opt11_render_pkg.py"),"--modeldir",str(md),"--out",str(outd),"--n","30"],timeout=7200)
run("measure",[PY,"-u",str(T/"measure_test30.py"),"--dir",str(outd),"--n","30"],timeout=3600)
mp=Path(r"D:\models\opt11\test30_metrics.json")
print("\n[chain] ===== RESULT =====", flush=True)
if mp.exists(): print("[chain] refiner-512domain full chain: %s" % mp.read_text(encoding="utf-8").replace("\n"," "), flush=True)
print("[chain] refs: SUBMITTED V4 = clipiqa 0.8042/niqe 4.9967 ; same refiner @256crop = +0.1736", flush=True)
print("[chain] criteria: >=0.83 wall broken ; <=0.8042 clear negative", flush=True)
print("C2_CHAIN_DONE", flush=True)
