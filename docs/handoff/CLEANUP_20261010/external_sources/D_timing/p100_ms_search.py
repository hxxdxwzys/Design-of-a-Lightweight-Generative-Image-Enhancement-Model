# -*- coding: utf-8 -*-
"""ModelScope 搜索 SD2.1 仓库并试下 vae (2026-09-02)"""
import json
import urllib.request

UA = {'User-Agent': 'Mozilla/5.0'}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))


def main():
    # 搜索
    try:
        d = get_json('https://www.modelscope.cn/api/v1/models?PageSize=20&PageNumber=1&Name=stable-diffusion-2-1')
        models = d.get('Data', {}).get('Models', [])
        print(f'搜索到 {len(models)} 个:', flush=True)
        for m in models:
            print(' ', m.get('Path') or m.get('Name'), flush=True)
    except Exception as e:
        print(f'搜索失败: {type(e).__name__}: {str(e)[:120]}', flush=True)
        return
    # 候选仓库逐个试 vae
    for cand in ['AI-ModelScope/stable-diffusion-2-1', 'AI-ModelScope/stable-diffusion-2-1-base',
                 'iic/stable-diffusion-2-1', 'stable-diffusion-v1-5/stable-diffusion-v1-5']:
        url = f'https://www.modelscope.cn/api/v1/models/{cand}/repo?FilePath=vae/diffusion_pytorch_model.safetensors&Revision=master'
        try:
            req = urllib.request.Request(url, headers=UA, method='HEAD')
            with urllib.request.urlopen(req, timeout=20) as r:
                print(f'{cand}: HEAD OK, content-length={r.headers.get("Content-Length")}', flush=True)
        except Exception as e:
            print(f'{cand}: {type(e).__name__} {str(e)[:80]}', flush=True)


if __name__ == '__main__':
    main()
