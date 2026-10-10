# -*- coding: utf-8 -*-
import os, sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
for tag, p in [('v9b', r'D:/models/pkg_TinyV9b/model_dir/your_model.pt'),
               ('v4g', r'D:/models/pkg_TinyV4g/model_dir/your_model.pt'),
               ('v7gan', r'D:/models/pkg_TinyV7gan/model_dir/your_model.pt'),
               ('v1', r'D:/models/pkg_TinyV1/model_dir/your_model.pt'),
               ('v11b', r'D:/models/pkg_TinyV11b/model_dir/your_model.pt')]:
    try:
        m = torch.jit.load(p, map_location='cpu')
        tot = 0
        lines = []
        for k, v in m.state_dict().items():
            n = v.numel(); tot += n
            lines.append('   %-46s %s' % (k, tuple(v.shape)))
        print('===== %s  params=%.3fM  sd_items=%d' % (tag, tot/1e6, len(lines)))
        print('\n'.join(lines[:60]))
    except Exception as e:
        print('=====', tag, 'ERR', repr(e)[:200])
