# -*- coding: utf-8 -*-
"""全流程自动编排: 下载(续传) -> 解压 -> 校验 -> 训练 -> sweep -> 渲染 -> 打包
每一步调用已有脚本, 失败即停(下载会重试3次)。日志写 D:/tmp_timing/pipeline.log。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:\Python\python.exe'
TMP = Path(r'D:\tmp_timing')
LOG = Path(r'D:\tmp_timing\pipeline.log')
DATA = r'D:/datasets/div2k/pngtxt_ori,D:/datasets/flickr2k'
OUT = r'D:/models/metric_net_f2k'


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(args, slog):
    log('>>> ' + ' '.join(args))
    with open(slog, 'a', encoding='utf-8') as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    log(f'<<< exit {r.returncode}')
    return r.returncode


def step_download():
    for attempt in range(1, 4):
        rc = run([PY, str(TMP / 'dl_chunked.py')], r'E:\flickr2k\dl_chunked.log')
        if rc == 0:
            return 0
        log(f'下载第 {attempt} 次失败, 重试')
        time.sleep(30)
    return 1


def main():
    log('===== pipeline start =====')

    if step_download() != 0:
        log('下载最终失败'); sys.exit(1)

    if run([PY, str(TMP / 'extract_flickr2k.py'),
            '--zip', 'E:/flickr2k/Flickr2K.zip',
            '--out', 'D:/datasets/flickr2k'], r'E:\flickr2k\extract.log') != 0:
        log('解压失败'); sys.exit(1)

    if run([PY, str(TMP / 'validate_imgs.py'),
            '--dir', 'D:/datasets/flickr2k',
            '--dir', 'D:/datasets/div2k/pngtxt_ori'], r'D:\tmp_timing\validate.log') != 0:
        log('校验失败'); sys.exit(1)

    if run([PY, str(TMP / 'train_df2k.py'),
            '--data', DATA, '--steps', '16000', '--out', OUT],
            r'D:\tmp_timing\train_f2k.log') != 0:
        log('训练失败'); sys.exit(1)

    if run([PY, str(TMP / 'sweep_df2k.py'),
            '--ckpt-dir', OUT, '--width', '32',
            '--out', r'D:\tmp_timing\sweep_f2k.json'], r'D:\tmp_timing\sweep_f2k.log') != 0:
        log('sweep 失败'); sys.exit(1)

    with open(r'D:\tmp_timing\sweep_f2k.json') as f:
        results = json.load(f)
    results.sort(key=lambda r: -r['q'])
    best = results[0]
    log(f'[best] {best["ckpt"]} Q={best["q"]:.2f} lpips={best["lpips"]:.4f} clipiqa={best["clipiqa"]:.4f}')

    if run([PY, str(TMP / 'net_infer.py'),
            '--ckpt', f'{OUT}/{best["ckpt"]}', '--mode', 'test',
            '--out', 'D:/models/cnn_out_f2k'], r'D:\tmp_timing\render_f2k.log') != 0:
        log('渲染失败'); sys.exit(1)

    if run([PY, str(TMP / 'package_submit.py'),
            '--out-dir', 'D:/models/cnn_out_f2k',
            '--model-dir', 'D:/models/final_graph7/model_dir',
            '--zip', 'D:/models/Track2_MetricF2K_Remay_13000000000.zip'],
            r'D:\tmp_timing\package_f2k.log') != 0:
        log('打包失败'); sys.exit(1)

    log('===== pipeline ALL_DONE =====')


if __name__ == '__main__':
    main()
