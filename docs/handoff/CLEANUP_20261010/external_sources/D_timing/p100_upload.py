# -*- coding: utf-8 -*-
"""P100 服务器上传器 (2026-09-02): paramiko SFTP, 跳过同名同大小, 支持 tar 打包

用法:
  1) 先打 tar:  D:/Python/python.exe p100_upload.py --make-tars
  2) 上传:      D:/Python/python.exe p100_upload.py --upload [--only small|weights|data]
  3) 服务器解压: 由 --upload 尾部自动执行 (untar)
"""
import os
import sys
import tarfile
import time
from pathlib import Path

import paramiko

HOST = os.environ.get('P100_HOST', '10.96.138.171')
PORT = int(os.environ.get('P100_PORT', '2222'))
USER, PWD = 'tempuser', 'temp76'
BASE = 'p100'  # SFTP 相对家目录 (chroot 环境, 不能用绝对路径)
BASE_ABS = '$HOME/p100'  # exec 命令里用

LOCAL_ROOT = Path(r'D:/tmp_timing')
TARS = {
    'div2k.tar': (Path(r'D:/datasets/div2k/pngtxt_ori'), 'data/'),
    'torch_cache.tar': (Path(r'C:/models/finals/preset/ckpts/torch_cache'), 'weights/'),
}

# (本地路径, 远端相对 BASE 路径, 是否目录)
ITEMS_SMALL = [
    ('train_vosr_lora.py', 'tools/'),
    ('train_vosr_lora_14b.py', 'tools/'),
    ('train_v2.py', 'tools/'),
    ('vosr_lora_infer.py', 'tools/'),
    ('val_gate.py', 'tools/'),
    ('b_14b_fp16_align.py', 'tools/'),
    ('degradation_analysis.json', 'tools/'),
    ('jpeg_blockiness.json', 'tools/'),
    ('dl_chunked.py', 'tools/'),
    ('extract_flickr2k.py', 'tools/'),
    ('p100_setup.sh', 'tools/'),
    ('p100_fix.sh', 'tools/'),
    ('p100_dl.py', 'tools/'),
    ('p100_dl2.py', 'tools/'),
    ('p100_dl3.sh', 'tools/'),
    ('p100_check_repo.py', 'tools/'),
    ('p100_ms_search.py', 'tools/'),
    ('p100_dl4.py', 'tools/'),
    ('p100_ready.sh', 'tools/'),
    ('p100_fill_pyiqa.py', 'tools/'),
    ('p100_fix2.sh', 'tools/'),
    ('p100_export_m24.sh', 'tools/'),
    ('p100_merge_m24.sh', 'tools/'),
    ('p100_resume.sh', 'tools/'),
    ('p100_dl_14b.py', 'tools/'),
    ('p100_guard.sh', 'tools/'),
    ('p100_launch_14b.sh', 'tools/'),
    ('p100_smoke14b.sh', 'tools/'),
    ('p100_resume_14b.sh', 'tools/'),
    ('p100_smoke768.sh', 'tools/'),
    ('p100_launch.sh', 'tools/'),
    ('p100_step2.sh', 'tools/'),
    ('p100_step3.sh', 'tools/'),
    ('p100_auto.sh', 'tools/'),
]
ITEMS_WEIGHTS = [
    (r'C:/models/finals/preset/ckpts/VOSR_0.5B_os', 'weights/VOSR_0.5B_os/', True),
    (r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base/vae', 'weights/sd21base/vae/', True),
    (r'C:/models/finals/preset/ckpts/sd21_lwdecoder.pth', 'weights/sd21_lwdecoder.pth', False),
    (r'D:/models/finals/VOSR', 'VOSR/', True),
    (r'C:\Users\Yuki Nagato\.cache\torch\hub\pyiqa', '.cache/torch/hub/pyiqa/', True),
    (r'C:\Users\Yuki Nagato\.cache\torch\hub\checkpoints', '.cache/torch/hub/checkpoints/', True),
    (r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集', 'val/', True),
]
ITEMS_DATA = [
    ('div2k.tar', 'data/div2k.tar'),
    ('torch_cache.tar', 'weights/torch_cache.tar'),
]


def make_tars():
    for name, (src, _) in TARS.items():
        dst = LOCAL_ROOT / name
        if dst.exists() and dst.stat().st_size > 100_000_000:
            print(f'[skip] {name} 已存在 ({dst.stat().st_size / 1e9:.2f} GB)')
            continue
        print(f'[tar] {src} -> {dst} ...', flush=True)
        t0 = time.time()
        with tarfile.open(dst, 'w') as tf:
            for p in sorted(src.rglob('*')):
                if p.is_file():
                    tf.add(p, arcname=p.relative_to(src.parent))
        print(f'[tar] {name} done ({dst.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)', flush=True)


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for i in range(3):
        try:
            cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
            return cli
        except Exception as e:
            print(f'[warn] 连接失败 {i + 1}: {e}', flush=True)
            time.sleep(5)
    raise RuntimeError('连不上服务器')


def sftp_put(sftp, local, remote, label):
    """跳过同名同大小; 返回是否上传"""
    try:
        if sftp.stat(remote).st_size == local.stat().st_size:
            print(f'[skip] {label} (已存在)', flush=True)
            return False
    except Exception:
        pass
    t0 = time.time()
    sftp.put(str(local), remote)
    print(f'[put] {label} ({local.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)', flush=True)
    return True


def _dir_exists(sftp, d):
    """容错 stat: 任何异常都视为不存在 (SFTP chroot 下 stat 目录偶发 Failure)"""
    try:
        sftp.stat(d)
        return True
    except Exception:
        return False


def _mkdir(sftp, d):
    """容错 mkdir: 已存在/并发创建都静默通过"""
    try:
        sftp.mkdir(d)
    except Exception:
        pass


def upload_dir(sftp, local_dir, remote_dir):
    for p in sorted(Path(local_dir).rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir)
        if any(seg.startswith('.') for seg in rel.parts):
            continue  # 跳过隐藏文件/目录 (如 .git) — 服务器只需要工作文件
        rp = os.path.join(remote_dir, rel.as_posix())
        try:
            if sftp.stat(rp).st_size == p.stat().st_size:
                continue
        except Exception:
            pass
        rparent = os.path.dirname(rp)
        # 逐级建父目录 (容错)
        parts = []
        cur = remote_dir
        parts.append(cur)
        for seg in rel.parts[:-1]:
            cur = os.path.join(cur, seg)
            parts.append(cur)
        for d in parts:
            _mkdir(sftp, d)
        sftp.put(str(p), rp)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--upload'
    if mode == '--make-tars':
        make_tars()
        return
    cli = connect()
    sftp = cli.open_sftp()
    # 建目录 (全容错)
    _mkdir(sftp, BASE)
    for d in ['tools', 'weights', 'VOSR', 'data', 'val', 'runs', 'logs']:
        _mkdir(sftp, f'{BASE}/{d}')
    only = sys.argv[2] if len(sys.argv) > 2 else None
    if only in (None, 'small'):
        print('=== 小文件 ===', flush=True)
        for f, d in ITEMS_SMALL:
            sftp_put(sftp, LOCAL_ROOT / f, f'{BASE}/{d}{f}', f)
    if only in (None, 'weights'):
        print('=== 权重/仓库/缓存/验证集 ===', flush=True)
        for lp, rp, is_dir in ITEMS_WEIGHTS:
            if is_dir:
                upload_dir(sftp, Path(lp), f'{BASE}/{rp}')
                print(f'[dir] {rp} 完成', flush=True)
            else:
                sftp_put(sftp, Path(lp), f'{BASE}/{rp}', rp)
    if only in (None, 'data'):
        print('=== 大 tar ===', flush=True)
        for f, rp in ITEMS_DATA:
            sftp_put(sftp, LOCAL_ROOT / f, f'{BASE}/{rp}', f)
    # 服务器侧: 解压 tar + pyiqa 缓存归位
    print('=== 服务器后处理 ===', flush=True)
    post = (
        'set -e; '
        'mkdir -p ~/.cache/torch/hub; '
        'cp -rn ~/p100/.cache/torch/hub/pyiqa ~/.cache/torch/hub/ 2>/dev/null || true; '
        'cp -rn ~/p100/.cache/torch/hub/checkpoints ~/.cache/torch/hub/ 2>/dev/null || true; '
        'tar -xf ~/p100/data/div2k.tar -C ~/p100/data/ && echo UNTAR_DIV2K_OK; '
        'tar -xf ~/p100/weights/torch_cache.tar -C ~/p100/weights/ && echo UNTAR_CACHE_OK'
    )
    _in, _out, _err = cli.exec_command(post, timeout=600)
    print(_out.read().decode('utf-8', errors='replace').strip(), flush=True)
    e = _err.read().decode('utf-8', errors='replace').strip()
    if e:
        print(f'[stderr] {e[:500]}', flush=True)
    if '--run-setup' in sys.argv:
        print('=== 启动环境安装 (nohup) ===', flush=True)
        _in, _out, _err = cli.exec_command(
            f'mkdir -p ~/p100/logs && nohup bash ~/p100/tools/p100_setup.sh > ~/p100/logs/setup.log 2>&1 & echo SETUP_STARTED',
            timeout=30)
        print(_out.read().decode('utf-8', errors='replace').strip(), flush=True)
    sftp.close()
    cli.close()
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
