# -*- coding: utf-8 -*-
"""检查仓库文件清单 v2: huggingface_hub list_repo_files + ModelScope 双路"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import urllib.request  # noqa: E402
import json  # noqa: E402


def main():
    # 路1: huggingface_hub
    try:
        from huggingface_hub import list_repo_files
        tok = open(os.path.expanduser('~/p100/hf_token')).read().strip()
        files = list_repo_files('CSWRY/VOSR', token=tok)
        print(f'HF 路: 总 {len(files)} 文件', flush=True)
        for kw in ('RN50', 'train/', 'clip', 'lwdecoder'):
            hits = [f for f in files if kw in f]
            print(f'--- "{kw}" ({len(hits)}) ---', flush=True)
            for h in hits[:6]:
                print(' ', h, flush=True)
        return
    except Exception as e:
        print(f'HF 路失败: {type(e).__name__}: {str(e)[:150]}', flush=True)

    # 路2: ModelScope files API
    try:
        req = urllib.request.Request(
            'https://www.modelscope.cn/api/v1/models/LULALULALU/VOSR_CKPT/repo/files?Revision=master',
            headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode('utf-8'))
        files = [f['Path'] for f in d.get('Data', {}).get('Files', [])]
        print(f'MS 路: 总 {len(files)} 文件', flush=True)
        for kw in ('RN50', 'train/', 'clip', 'lwdecoder', '0.5B'):
            hits = [f for f in files if kw in f]
            print(f'--- "{kw}" ({len(hits)}) ---', flush=True)
            for h in hits[:6]:
                print(' ', h, flush=True)
    except Exception as e:
        print(f'MS 路失败: {type(e).__name__}: {str(e)[:150]}', flush=True)


if __name__ == '__main__':
    main()
