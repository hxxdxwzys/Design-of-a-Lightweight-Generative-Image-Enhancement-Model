# -*- coding: utf-8 -*-
"""列出候选 HF 仓库的文件与体积 (用 API, 不下载)"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.pop('HF_HUB_OFFLINE', None)
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
from huggingface_hub import HfApi
api = HfApi()
REPOS = ['EvanEternal/VARestorer', 'VCLab-PolyU/PixRestore', 'zty557/TADSR',
         'FoundationVision/Infinity', 'double8fun/ODTSR']
for rid in REPOS:
    try:
        info = api.model_info(rid, files_metadata=True)
        files = [(s.rfilename, (s.size or 0)) for s in info.siblings]
        tot = sum(f[1] for f in files)
        print(f'=== {rid}: {len(files)} 个文件, 合计 {tot/1e9:.2f} GB')
        for n, sz in sorted(files, key=lambda x: -x[1])[:10]:
            print(f'   {sz/1e6:9.1f} MB  {n}')
    except Exception as e:
        print(f'=== {rid}: 失败 {type(e).__name__}: {str(e)[:100]}')
print('LIST_DONE')
