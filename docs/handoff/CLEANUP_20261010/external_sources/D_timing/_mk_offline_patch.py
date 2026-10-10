# -*- coding: utf-8 -*-
import io, os, sys
src = io.open(r'D:/tmp_timing/vosr_lora_infer.py', encoding='utf-8').read()
patch = '''
# ---- 离线 dinov2 补丁 (2026-09-15 夜) ----
# 原 _hub_load_offline_first 用相对路径 'preset/ckpts/torch_cache', 只有 CWD=C:/models/finals
# 才命中本地仓库, 否则回落 torch.hub.load('facebookresearch/dinov2') 走网络 -> 断网 60s 超时崩。
import inference_vosr_onestep as _iv
os.environ.setdefault('TORCH_HOME', r'C:/models/finals/preset/ckpts/torch_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
_HUB_DIR = r'C:/models/finals/preset/ckpts/torch_cache'


def _offline_first_abs(name):
    local = os.path.join(_HUB_DIR, 'facebookresearch_dinov2_main')
    if os.path.isdir(local):
        try:
            return torch.hub.load(local, name, source='local')
        except Exception as e:
            print('[warn] local dinov2 load failed (%s); fallback remote' % e, flush=True)
    return torch.hub.load('facebookresearch/dinov2', name)


_iv._hub_load_offline_first = _offline_first_abs
# ---- 补丁结束 ----

'''
i = src.index('TORCH_CACHE = ')
io.open(r'D:/tmp_timing/vosr_lora_infer2.py', 'w', encoding='utf-8').write(src[:i] + patch + src[i:])
print('patched ->', r'D:/tmp_timing/vosr_lora_infer2.py')
