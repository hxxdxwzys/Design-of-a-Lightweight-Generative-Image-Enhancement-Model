# -*- coding: utf-8 -*-
"""PIPE-01 共享推理核心（同主体最小可复现推理闭环）。

身份与出处（只读外部参考；本文件为受控重写，不复制整套实验历史）：
  REF-A D:/models/finals/VOSR/inference_vosr_onestep.py
        参考: preprocess_raw_image(131-135)、get_venc_features(181-189)、
        tiled_latent_inference(207-309) 的 DiT 分块与一步流匹配更新。
  REF-B D:/models/finals/VOSR/tiled_vae.py
        直接复用: encode_dispatch / decode_dispatch / _gaussian_weights / _make_tile_grid
        （经显式 sys.path + 源码哈希校验导入，原件只读）。
  REF-C D:/tmp_timing/vosr_lora_infer2.py
        参考其资产组装顺序（基座 -> peft 包装 -> scale 只作用于 lora_B）。

相对参考实现的**有意差异**（均在 PIPE-01 任务单 §3 要求内）：
  1. 显式冻结配置：全部参数来自 config.json，不接受 CLI 默认值静默覆盖。
  2. 每次 infer 使用局部随机协议：新建 torch.Generator(seed)，不依赖进程级 seed、
     加载顺序或上一张图。
  3. 完全离线：DINO 只用本地 hub 目录 + 本地 checkpoint，缺失即报错，无远程回退。
  4. wavelet 颜色处理属于共享核心（张量进、张量出），tensor 入口同样执行。
  5. 权重安全加载：LoRA 用 weights_only=True，失败即停止，不静默降级。
  6. 记录资产指纹与有效参数指纹，供两入口一致性核对。

本模块顶层不导入 torch/diffusers/peft —— 保证普通导入与 --help 不加载模型、不启动 CUDA。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / 'config.json'
CHUNK = 1 << 23

REQUIRED_KEYS = (
    'base_dir', 'base_weights', 'lora_path', 'lora_scale', 'ae_dir', 'light_decoder',
    'torch_cache', 'dinov2_repo', 'dinov2_ckpt', 'vosr_dir', 'dtype', 'autocast',
    'torch_compile', 'vae_posterior_mode', 'infer_steps', 't_start', 't_end',
    'upscale', 'align_method', 'align_sigma', 'seed', 'dit_tile_size',
    'dit_tile_overlap_cli', 'vae_tile_size', 'vae_tile_overlap_cli',
    'expected_patch_size', 'expected_ae_type', 'expected_dinov2_size',
    'expected_layer_dinov2b_list', 'expected_enc_type', 'expected_ae_factor',
    'base_key_allowlist_missing', 'base_key_allowlist_unexpected', 'lora_keys_must_load',
    'pad_mode', 'crop_back_to_input',
)


class PipelineError(RuntimeError):
    """本管线所有可预期失败的基类（配置/资产/离线/数值）。"""


class ConfigError(PipelineError):
    pass


class AssetError(PipelineError):
    pass


class OfflineError(PipelineError):
    pass


class OutputValidationError(PipelineError):
    pass


# ───────────────────────── 配置 ─────────────────────────

def load_config(path=None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    try:
        raw = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        raise ConfigError('无法读取配置 %s: %s: %s' % (p, type(e).__name__, e))
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ConfigError('配置缺少必需键: %s' % missing)
    unknown = [k for k in raw if not k.startswith('_') and k not in REQUIRED_KEYS]
    if unknown:
        raise ConfigError('配置含未登记键（禁止静默扩展）: %s' % unknown)
    cfg = {k: raw[k] for k in REQUIRED_KEYS}
    if str(cfg['dtype']) != 'float32':
        raise ConfigError('PIPE-01 固定 eager/fp32，收到 dtype=%r' % cfg['dtype'])
    if cfg['autocast'] or cfg['torch_compile']:
        raise ConfigError('PIPE-01 固定 无autocast/无compile')
    if not cfg['vae_posterior_mode']:
        raise ConfigError('PIPE-01 固定 VAE posterior mode')
    if int(cfg['infer_steps']) != 1 or float(cfg['t_start']) != 1.0 or float(cfg['t_end']) != 0.0:
        raise ConfigError('PIPE-01 固定 一步 t=1->0')
    if str(cfg['align_method']) != 'wavelet' or int(cfg['upscale']) != 1:
        raise ConfigError('PIPE-01 固定 wavelet + upscale=1')
    if str(cfg['pad_mode']) != 'reflect' or not cfg['crop_back_to_input']:
        raise ConfigError('PIPE-01 固定 reflect 补齐 + decode 后裁回输入尺寸')
    cfg['_config_path'] = str(p)
    cfg['_config_sha256'] = sha256_file(p)
    return cfg


def dit_tile_params(cfg, patch_size, lh, lw):
    """DiT latent 分块参数，逐字对齐 REF-A:235-238。"""
    af = int(cfg['expected_ae_factor'])
    lt_size = max((int(cfg['dit_tile_size']) // af // patch_size) * patch_size, patch_size)
    lt_overlap = max(int(cfg['dit_tile_overlap_cli']) // af, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)
    return lt_size, lt_overlap


def vae_tile_params(cfg, lh, lw):
    """VAE latent 分块参数，逐字对齐 REF-B:_tile_params(96-102)。"""
    af = int(cfg['expected_ae_factor'])
    lt_size = max(int(cfg['vae_tile_size']) // af, 1)
    lt_overlap = max(int(cfg['vae_tile_overlap_cli']) // af, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)
    return lt_size, lt_overlap


def tile_config_report(cfg, patch_size=2, lh=384, lw=512):
    """把任务单 §3 的声明与代码推导值对齐；不一致直接报错（不自行换配方）。"""
    af = int(cfg['expected_ae_factor'])
    ds, dov = dit_tile_params(cfg, patch_size, lh, lw)
    vs, vov = vae_tile_params(cfg, lh, lw)
    rep = {
        'dit_tile_size_px': int(cfg['dit_tile_size']),
        'dit_tile_latent': ds, 'dit_overlap_latent': dov,
        'dit_effective_overlap_px': dov * af, 'dit_stride_px': (ds - dov) * af,
        'vae_tile_size_px': int(cfg['vae_tile_size']),
        'vae_tile_latent': vs, 'vae_overlap_latent': vov,
        'vae_effective_overlap_px': vov * af, 'vae_stride_px': (vs - vov) * af,
    }
    if rep['dit_stride_px'] != 448 or rep['dit_effective_overlap_px'] != 64:
        raise ConfigError('DiT 分块与任务单声明不符: %s' % rep)
    if rep['vae_stride_px'] != 896 or rep['vae_effective_overlap_px'] != 128:
        raise ConfigError('VAE 分块与任务单声明不符: %s' % rep)
    return rep


# ───────────────────────── 哈希与指纹 ─────────────────────────

def sha256_file(p, chunk=CHUNK) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def asset_fingerprint(cfg) -> dict:
    items = {
        'base_weights': cfg['base_weights'],
        'lora': cfg['lora_path'],
        'light_decoder': cfg['light_decoder'],
        'dinov2_ckpt': cfg['dinov2_ckpt'],
        'args_json': str(Path(cfg['base_dir']) / 'args.json'),
        'config_json': cfg['_config_path'],
    }
    out = {}
    for k, v in items.items():
        p = Path(v)
        if not p.is_file():
            raise AssetError('资产缺失 %s: %s' % (k, p))
        out[k] = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
    hub = Path(cfg['dinov2_repo']) / 'hubconf.py'
    if not hub.is_file():
        raise AssetError('DINO hubconf 缺失: %s' % hub)
    out['dinov2_repo_hubconf'] = {'path': str(hub), 'sha256': sha256_file(hub)}
    dino_src = Path(cfg['dinov2_repo']) / 'dinov2' / 'models' / 'vision_transformer.py'
    if not dino_src.is_file():
        raise AssetError('本地 dinov2 源码缺失: %s' % dino_src)
    out['dinov2_vision_transformer'] = {'path': str(dino_src),
                                        'sha256': sha256_file(dino_src)}
    vae_dir = Path(cfg['ae_dir']) / 'vae'
    if not vae_dir.is_dir():
        raise AssetError('VAE 目录缺失: %s' % vae_dir)
    vae_files = {}
    for p in sorted(x for x in vae_dir.rglob('*') if x.is_file()):
        vae_files[str(p.relative_to(vae_dir)).replace('\\', '/')] = {
            'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
    if not vae_files:
        raise AssetError('VAE 目录下没有文件: %s' % vae_dir)
    out['ae_vae_dir'] = {'dir': str(vae_dir), 'n_files': len(vae_files), 'files': vae_files}
    return out


def implementation_fingerprint() -> dict:
    """PIPE-01 自身实现的源码身份（R03）：相对路径 + 绝对 __file__ + 字节数 + SHA-256。"""
    files = ('core.py', 'runner.py', 'offline.py', 'pair_verify.py', 'make_evidence.py',
             'config.json', 'tests/test_pipe01_cpu.py')
    out = {}
    for rel in files:
        p = HERE / rel
        if not p.is_file():
            raise AssetError('实现文件缺失: %s' % p)
        out[rel] = {'path': str(p.resolve()), 'bytes': p.stat().st_size,
                    'sha256': sha256_file(p)}
    return out


def env_fingerprint() -> dict:
    import platform
    out = {'python': sys.version.split()[0], 'platform': platform.platform()}
    for m in ('torch', 'torchvision', 'diffusers', 'peft', 'safetensors', 'numpy', 'cv2'):
        try:
            mod = __import__(m)
            out[m] = str(getattr(mod, '__version__', 'unknown'))
        except Exception as e:
            out[m] = 'UNKNOWN(%s)' % type(e).__name__
    return out


def source_fingerprint(cfg) -> dict:
    vosr = Path(cfg['vosr_dir'])
    files = [
        vosr / 'inference_vosr_onestep.py', vosr / 'tiled_vae.py',
        vosr / 'models' / 'lightningdit.py', vosr / 'models' / 'light_decoder.py',
        Path('D:/tmp_timing/vosr_lora_infer2.py'),
    ]
    out = {}
    for p in files:
        if not p.is_file():
            raise AssetError('参考源码缺失: %s' % p)
        out[str(p)] = {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
    return out


def fingerprint_state_dict(sd, with_bytes=True) -> dict:
    keys = sorted(sd.keys())
    shapes, dtypes = {}, {}
    h = hashlib.sha256()
    total = 0
    for k in keys:
        t = sd[k]
        if not hasattr(t, 'detach'):
            raise AssetError('state_dict 含非张量项: %s (%s)' % (k, type(t).__name__))
        t = t.detach()
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
        shapes[str(tuple(t.shape))] = shapes.get(str(tuple(t.shape)), 0) + 1
        total += t.numel()
        if with_bytes:
            h.update(k.encode('utf-8'))
            h.update(str(t.dtype).encode('utf-8'))
            h.update(str(tuple(t.shape)).encode('utf-8'))
            h.update(t.contiguous().to('cpu').float().numpy().tobytes())
    return {
        'n_keys': len(keys), 'n_params': int(total),
        'dtypes': dtypes, 'shape_histogram': shapes,
        'first_keys': keys[:5], 'last_keys': keys[-5:],
        'digest_sha256': h.hexdigest() if with_bytes else None,
        'byte_protocol': 'key|dtype|shape|float32-raw' if with_bytes else None,
    }


# ───────────────────────── 离线环境与权重加载 ─────────────────────────

def ensure_offline_env(cfg):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ.pop(k, None)
    # 不再使用 torch.hub（PIPE-R02）：只通过环境变量声明缓存位置，本管线不调用任何 hub 入口
    os.environ['TORCH_HOME'] = str(cfg['torch_cache'])


def ensure_vosr_import_path(cfg):
    """显式插入 VOSR 源码路径（只读）后再导入；路径无效即报错。"""
    d = str(cfg['vosr_dir'])
    if not (Path(d) / 'tiled_vae.py').is_file():
        raise AssetError('VOSR 源码路径无效: %s' % d)
    if d not in sys.path:
        sys.path.insert(0, d)
    import tiled_vae  # noqa: F401
    return d


def load_lora_payload(path):
    """LoRA 安全加载：weights_only=True；失败即停止，不降级为 False。"""
    import torch
    try:
        payload = torch.load(str(path), map_location='cpu', weights_only=True)
    except Exception as e:
        raise AssetError('LoRA 无法以 weights_only=True 安全加载（不降级、不静默继续）: %s: %s'
                         % (type(e).__name__, e))
    if not isinstance(payload, dict) or 'state_dict' not in payload:
        raise AssetError('LoRA payload 结构不符: keys=%s' % list(payload)[:8])
    return payload


def build_dit(cfg, args_ns, base_state=None):
    from models.lightningdit import LightningDiT
    base_ch = 4 if args_ns.ae_type == 'sd2' else 16
    model = LightningDiT(
        input_size=args_ns.resolution // 8, patch_size=args_ns.patch_size,
        in_channels=2 * base_ch, out_channels=base_ch,
        hidden_size=args_ns.dim, depth=args_ns.depth, num_heads=args_ns.num_heads,
        mlp_ratio=args_ns.mlp_ratio, z_dims=args_ns.enc_dim,
        encdim_ratio=args_ns.encdim_ratio,
        auxiliary_time_cond=bool(getattr(args_ns, 'auxiliary_time_cond', True)),
        use_qknorm=args_ns.use_qknorm, use_swiglu=args_ns.use_swiglu,
        use_rope=args_ns.use_rope, use_rmsnorm=args_ns.use_rmsnorm,
        wo_shift=args_ns.wo_shift,
        num_fused_layers=len(args_ns.layer_dinov2b_list),
    )
    if base_state is None:
        return model, {}
    res = model.load_state_dict(base_state, strict=False)
    missing, unexpected = sorted(res.missing_keys), sorted(res.unexpected_keys)
    if missing != sorted(cfg['base_key_allowlist_missing']) or \
       unexpected != sorted(cfg['base_key_allowlist_unexpected']):
        raise AssetError('基座键集与白名单不符（不静默继续）。missing=%d 例:%s ; unexpected=%d 例:%s'
                         % (len(missing), missing[:5], len(unexpected), unexpected[:5]))
    return model, {'missing': missing, 'unexpected': unexpected}


def apply_lora(model, payload, scale):
    from peft import LoraConfig, get_peft_model
    if 'lora_config' not in payload:
        raise AssetError('LoRA payload 缺 lora_config')
    lc = payload['lora_config']
    lora_cfg = json.loads(lc) if isinstance(lc, str) else lc
    model = get_peft_model(model, LoraConfig(**lora_cfg))
    lora_sd = {k: v for k, v in payload['state_dict'].items() if 'lora' in k}
    if not lora_sd:
        raise AssetError('LoRA payload 里没有 lora_* 权重键')
    if float(scale) != 1.0:
        lora_sd = {k: (v * float(scale) if 'lora_B' in k else v) for k, v in lora_sd.items()}
    res = model.load_state_dict(lora_sd, strict=False)
    missing_lora = sorted(k for k in res.missing_keys if 'lora' in k)
    if missing_lora:
        raise AssetError('LoRA 键未被加载（禁止用 strict=False 掩盖）: %d 个 例:%s'
                         % (len(missing_lora), missing_lora[:5]))
    if res.unexpected_keys:
        raise AssetError('LoRA 注入出现多余键: %d 个 例:%s'
                         % (len(res.unexpected_keys), sorted(res.unexpected_keys)[:5]))
    return model, {'lora_keys_loaded': len(lora_sd), 'lora_scale': float(scale)}


# 与本地 dinov2 hubconf 的 vitb14 构造参数一致（dinov2/hub/backbones.py:_make_dinov2_model 默认值）
DINOV2_VITB14_KWARGS = dict(img_size=518, patch_size=14, init_values=1.0, ffn_layer='mlp',
                            block_chunks=0, num_register_tokens=0,
                            interpolate_antialias=False, interpolate_offset=0.1)


def load_dinov2_local(cfg, device):
    """严格离线 DINO（PIPE-R02 修正）：直接用本地 dinov2 源码构造 + 指定的本地 checkpoint 安全加载。

    不再调用 torch.hub（其 source='local' 仍会走 load_state_dict_from_url 的取权重路径）。
    构造不触发任何预训练下载；checkpoint 就是预登记所指文件；weights_only=True 失败即停。
    """
    import types as _types
    import numpy as _np  # noqa: F401  (dinov2 vision_transformer 顶层需要)
    import torch
    repo, ckpt = Path(cfg['dinov2_repo']), Path(cfg['dinov2_ckpt'])
    if not repo.is_dir():
        raise OfflineError('本地 DINO 源码目录缺失（不远程回退）: %s' % repo)
    if not ckpt.is_file():
        raise OfflineError('本地 DINO 权重缺失（不远程回退）: %s' % ckpt)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from dinov2.models import vision_transformer as vits
    except Exception as e:
        raise OfflineError('本地 dinov2 源码不可导入（不远程回退）: %s: %s'
                           % (type(e).__name__, e))
    try:
        enc = vits.vit_base(**DINOV2_VITB14_KWARGS)
    except Exception as e:
        raise OfflineError('本地 dinov2 构造失败（不远程回退）: %s: %s' % (type(e).__name__, e))
    try:
        sd = torch.load(str(ckpt), map_location='cpu', weights_only=True)
    except Exception as e:
        raise AssetError('DINO checkpoint 无法以 weights_only=True 安全加载（不降级）: %s: %s'
                         % (type(e).__name__, e))
    try:
        enc.load_state_dict(sd, strict=True)
    except Exception as e:
        raise AssetError('DINO 键集与 checkpoint 不符（strict，不静默忽略）: %s'
                         % str(e)[:300])
    n_dino_keys = len(sd)
    del sd
    dino_info = {'loader': 'local source vit_base + torch.load(weights_only=True) + strict=True',
                 'uses_torch_hub': False, 'weights_only': True, 'strict': True,
                 'n_keys': n_dino_keys, 'ckpt': str(ckpt)}
    if hasattr(enc, 'head'):
        enc.head = torch.nn.Identity()

    def forward_with_features(self, x, masks=None):
        features = {}
        x = self.prepare_tokens_with_masks(x, masks)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            features['layer_%d' % i] = x[:, 1:]
        return features, self.norm(x)[:, 1:]

    enc.forward_with_features = _types.MethodType(forward_with_features, enc)
    enc = enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, dino_info


def build_args_ns(cfg):
    """由冻结配置 + 基座 args.json 构造 args（无 CLI 覆盖）。"""
    import argparse
    raw = json.loads((Path(cfg['base_dir']) / 'args.json').read_text(encoding='utf-8'))
    for k, expect in (('patch_size', cfg['expected_patch_size']),
                      ('ae_type', cfg['expected_ae_type']),
                      ('dinov2_size', cfg['expected_dinov2_size']),
                      ('layer_dinov2b_list', cfg['expected_layer_dinov2b_list']),
                      ('enc_type', cfg['expected_enc_type'])):
        if k in raw and raw[k] != expect:
            raise ConfigError('基座 args.json 的 %s=%r 与冻结配置 %r 不符' % (k, raw[k], expect))
    ns = argparse.Namespace(**raw)
    ns.ae_path = cfg['ae_dir']
    ns.tile_size = int(cfg['dit_tile_size'])
    ns.tile_overlap = int(cfg['dit_tile_overlap_cli'])
    ns.vae_tile_size = int(cfg['vae_tile_size'])
    ns.vae_tile_overlap = int(cfg['vae_tile_overlap_cli'])
    ns.posterior_mode = bool(cfg['vae_posterior_mode'])
    ns.infer_steps = int(cfg['infer_steps'])
    ns.auxiliary_time_cond = bool(getattr(ns, 'auxiliary_time_cond', True))
    return ns


# ───────────────────────── 数值边界与 DINO 预处理（R01 / R05） ─────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_dinov2_input(lq_tensor, dinov2_size):
    """DINOv2 条件输入预处理 —— 逐字对齐 REF-A:131-135 preprocess_raw_image。

    参考顺序（不可调换）：x(0..255) -> /255 -> bicubic 缩放到 dinov2_size -> clip(0,1)
    -> ImageNet 归一化。PIPE-R01 修正点：此前漏掉 /255，导致灰色输入被 clip 成近白。
    """
    import torch
    x = (0.5 * lq_tensor + 0.5) * 255.0          # 对齐 REF-A:183 raw_image 的量纲
    x = x / 255.0                                 # ← 参考实现第一步
    x = torch.nn.functional.interpolate(x, size=int(dinov2_size), mode='bicubic')
    x = x.clip(0., 1.)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def require_finite(t, where):
    """在量化/后处理之前拦截 NaN/Inf（PIPE-R05）。"""
    import torch
    bad = ~torch.isfinite(t)
    if bool(bad.any()):
        raise OutputValidationError('%s 出现 NaN/Inf（%d 个元素）——不得进入量化或写图'
                                    % (where, int(bad.sum())))


# ───────────────────────── 资产组装 ─────────────────────────

class Assets:
    """一次加载的全部推理资产 + 身份/参数指纹。两入口共享同一实例。"""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        ensure_offline_env(cfg)
        ensure_vosr_import_path(cfg)
        self.asset_fp = asset_fingerprint(cfg)
        self.source_fp = source_fingerprint(cfg)
        self.args = build_args_ns(cfg)
        self.tile_report = tile_config_report(cfg, self.args.patch_size)
        self._tile_feat_cache = {}
        self._full_feat = None

        import torch
        from safetensors.torch import load_file
        from diffusers import AutoencoderKL
        from models.light_decoder import LightDecoder

        base_sd = load_file(str(cfg['base_weights']))
        model, base_report = build_dit(cfg, self.args, base_state=base_sd)
        del base_sd
        payload = load_lora_payload(cfg['lora_path'])
        model, lora_report = apply_lora(model, payload, cfg['lora_scale'])
        del payload
        model.forward = model.forward_flexible
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(
            str(cfg['ae_dir']), subfolder='vae', local_files_only=True).to(device).eval()
        lw = torch.load(str(cfg['light_decoder']), map_location='cpu', weights_only=True)
        keys = ('in_channels', 'out_channels', 'block_out_channels', 'layers_per_block')
        self.dec = LightDecoder(**{k: lw['config'][k] for k in keys})
        self.dec.load_state_dict(lw['model_state_dict'])
        self.dec = self.dec.to(device).eval()
        del lw
        self.venc, dino_info = load_dinov2_local(cfg, device)

        self.load_report = {
            'base_keys': base_report, 'lora': lora_report,
            'ae_local_files_only': True,
            'dino': dino_info,
            'light_decoder_weights_only': True,
        }
        self.effective_fp = {
            'model': fingerprint_state_dict(self.model.state_dict()),
            'vae': fingerprint_state_dict(self.vae.state_dict()),
            'dino': fingerprint_state_dict(self.venc.state_dict()),
            'light_decoder': fingerprint_state_dict(self.dec.state_dict()),
        }
        self.implementation_fp = implementation_fingerprint()
        self.env_fp = env_fingerprint()
        import importlib

        def _modfile(name):
            """稳健地取模块文件路径：不在 sys.modules 就显式导入，失败即报错（不静默）。"""
            mod = sys.modules.get(name)
            if mod is None:
                try:
                    mod = importlib.import_module(name)
                except Exception as e:
                    raise AssetError('无法定位模块文件 %s（%s: %s）'
                                     % (name, type(e).__name__, e))
            f = getattr(mod, '__file__', None)
            if not f:
                raise AssetError('模块 %s 没有 __file__，无法登记身份' % name)
            return str(Path(f).resolve())

        self.module_files = {
            'core': str(Path(__file__).resolve()),
            'tiled_vae': _modfile('tiled_vae'),
            'lightningdit': _modfile('models.lightningdit'),
            'light_decoder': _modfile('models.light_decoder'),
        }

    def release(self):
        """显式释放全部学习模块（RA03）：清引用 + gc + empty_cache。

        注意：只有在【没有其他强引用】时才会真正释放；驱动必须同时丢弃自己的引用。
        """
        import gc
        for attr in ('model', 'vae', 'venc', 'dec'):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def identity(self, entry=None, model_dir=None, device=None):
        """完整运行身份（PIPE-R03）：配置/资产/源码/实现/有效参数/模块文件/环境/加载报告。"""
        return {
            'entry': entry,
            'model_dir': model_dir,
            'device': device if device is not None else self.device,
            'config_sha256': self.cfg['_config_sha256'],
            'config': {k: v for k, v in self.cfg.items() if not k.startswith('_')},
            'asset_fingerprint': self.asset_fp,
            'source_fingerprint': self.source_fp,
            'implementation_fingerprint': self.implementation_fp,
            'effective_fingerprint': self.effective_fp,
            'module_files': self.module_files,
            'load_report': self.load_report,
            'tile_report': self.tile_report,
            'env': self.env_fp,
        }

    def _venc_features(self, x):
        import torch
        with torch.no_grad():
            raw = preprocess_dinov2_input(x, self.args.dinov2_size)
            feats, x_norm = self.venc.forward_with_features(raw)
            z = [v for k, v in feats.items() if k.startswith('layer_')]
            z[-1] = x_norm
            return [z[i] for i in self.args.layer_dinov2b_list]

    def prepare_features(self, x):
        """VAE 编码 + DINO 条件预计算（对齐 REF-A:228-275）。"""
        import torch
        import tiled_vae as TV
        with torch.no_grad():
            lq_latent, mean, std = TV.encode_dispatch(self.vae, x, self.args, self.device)
        _, _, lh, lw = lq_latent.shape
        lt_size, lt_overlap = dit_tile_params(self.cfg, self.args.patch_size, lh, lw)
        self._tile_feat_cache = {}
        self._full_feat = None
        if lh <= lt_size and lw <= lt_size:
            self._full_feat = self._venc_features(x)
        else:
            h_pos = TV._make_tile_grid(lh, lt_size, lt_overlap)
            w_pos = TV._make_tile_grid(lw, lt_size, lt_overlap)
            for hi in h_pos:
                for wi in w_pos:
                    ph_s, pw_s = hi * 8, wi * 8
                    ph_e = min((hi + lt_size) * 8, int(x.shape[2]))
                    pw_e = min((wi + lt_size) * 8, int(x.shape[3]))
                    self._tile_feat_cache[(hi, wi)] = self._venc_features(
                        x[:, :, ph_s:ph_e, pw_s:pw_e])
        return lq_latent, mean, std

    def one_step_tiled(self, lq_latent, generator):
        """潜空间 DiT 一步更新（对齐 REF-A:255-305）；z 由局部 generator 生成。"""
        import torch
        import tiled_vae as TV
        _, _, lh, lw = lq_latent.shape
        lt_size, lt_overlap = dit_tile_params(self.cfg, self.args.patch_size, lh, lw)
        b = int(lq_latent.shape[0])
        t_cur = torch.full((b,), float(self.cfg['t_start']), device=lq_latent.device)
        t_nxt = torch.full((b,), float(self.cfg['t_end']), device=lq_latent.device)
        dt = t_cur[0] - t_nxt[0]
        z = torch.randn(lq_latent.shape, generator=generator,
                        device=lq_latent.device, dtype=lq_latent.dtype)
        if lh <= lt_size and lw <= lt_size:
            u = self.model(torch.cat([lq_latent, z], 1), t_cur, t_nxt, self._full_feat)
            return z - dt * u
        h_pos = TV._make_tile_grid(lh, lt_size, lt_overlap)
        w_pos = TV._make_tile_grid(lw, lt_size, lt_overlap)
        g = TV._gaussian_weights(lt_size, lt_size, int(lq_latent.shape[1]), lq_latent.device)
        u_acc = torch.zeros_like(lq_latent)
        w_acc = torch.zeros_like(lq_latent)
        for hi in h_pos:
            for wi in w_pos:
                he, we = hi + lt_size, wi + lt_size
                inp = torch.cat([lq_latent[:, :, hi:he, wi:we], z[:, :, hi:he, wi:we]], dim=1)
                u_tile = self.model(inp, t_cur, t_nxt, self._tile_feat_cache[(hi, wi)])
                u_acc[:, :, hi:he, wi:we] += u_tile * g
                w_acc[:, :, hi:he, wi:we] += g
        return z - dt * (u_acc / w_acc)


# ───────────────────────── 共享后处理与入口契约 ─────────────────────────

def tensor_to_pil(t):
    """唯一写图编码器：[1,3,H,W] 写图前张量 -> uint8 PIL。两入口必须共用。"""
    from torchvision.transforms.functional import to_pil_image
    tt = t.detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5
    if tt.dim() == 4:
        if int(tt.shape[0]) != 1:
            raise PipelineError('写图张量 B 必须为 1，实为 %d' % int(tt.shape[0]))
        tt = tt[0]
    return to_pil_image(tt)


def pil_to_tensor(pil):
    from torchvision.transforms.functional import to_tensor
    return (to_tensor(pil) * 2.0 - 1.0).unsqueeze(0)


def wavelet_color_fix_tensor(target, source, sigma=5):
    """对齐 REF-A:78-87 的 wavelet 颜色处理，核心内以张量进/张量出。"""
    import numpy as np
    import cv2
    from PIL import Image
    tgt = tensor_to_pil(target)
    src = tensor_to_pil(source)
    target_np = np.array(tgt).astype(np.float32) / 255.0
    source_np = np.array(src.resize(tgt.size, Image.LANCZOS)).astype(np.float32) / 255.0
    source_low = cv2.GaussianBlur(source_np, (0, 0), sigma)
    target_low = cv2.GaussianBlur(target_np, (0, 0), sigma)
    target_high = target_np - target_low
    result = np.clip(source_low + target_high, 0, 1) * 255.0
    return pil_to_tensor(Image.fromarray(result.astype(np.uint8)))


IDENTITY_ENTRY_KEYS = ('entry',)


def identity_comparable(idy):
    """两份身份用于严格相等比较时，去掉仅描述“哪个入口”的字段。"""
    return {k: v for k, v in idy.items() if k not in IDENTITY_ENTRY_KEYS}


def make_generator(seed, device):
    """局部随机协议：每次调用重建同一初始状态。"""
    import torch
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def validate_output(t, h, w):
    import torch
    if not isinstance(t, torch.Tensor):
        raise OutputValidationError('输出不是张量: %s' % type(t).__name__)
    if t.dim() != 4 or int(t.shape[0]) != 1 or int(t.shape[1]) != 3:
        raise OutputValidationError('输出形状应为 [1,3,H,W]，实为 %s' % tuple(t.shape))
    if (int(t.shape[2]), int(t.shape[3])) != (h, w):
        raise OutputValidationError('输出尺寸 %s 与输入 (%d,%d) 不符'
                                    % (tuple(t.shape[2:]), h, w))
    if t.dtype != torch.float32:
        raise OutputValidationError('输出 dtype 应为 float32，实为 %s' % t.dtype)
    if not bool(torch.isfinite(t).all()):
        raise OutputValidationError('输出含 NaN/Inf')
    if float(t.min()) < -1.0000001 or float(t.max()) > 1.0000001:
        raise OutputValidationError('输出超出 [-1,1]: [%.6f, %.6f]'
                                    % (float(t.min()), float(t.max())))
    return True


def _device_sync(device):
    """统一同步边界：CUDA 上先同步再计时/收尾，使墙钟反映设备完成。"""
    try:
        import torch
        if str(device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def infer_core(assets, image_tensor, seed=None, timing=None):
    """共享核心：完整前向 + 共享后处理，返回写图前张量（[-1,1] fp32）。

    timing 非空时写入 'infer_s' 与事件序列 'events'：**两个入口共用同一计时边界**
    —— 起点在本函数入口（同步后）、终点在返回前（同步后），因此**包含设备搬运、
    随机数生成、共享后处理与数值校验**；读图/写图不在其中（由入口另行记录）。PIPE-RA06。
    """
    import torch
    import tiled_vae as TV
    if timing is not None:
        timing.setdefault('events', []).append('enter')
    _device_sync(assets.device)
    t_start = time.perf_counter()
    cfg = assets.cfg
    if image_tensor.dim() != 4 or int(image_tensor.shape[1]) != 3 \
            or int(image_tensor.shape[0]) != 1:
        raise PipelineError('输入须为 [1,3,H,W]，实为 %s' % tuple(image_tensor.shape))
    x = image_tensor.to(assets.device, dtype=torch.float32)   # 搬运在计时区间内（两入口一致）
    if timing is not None:
        timing['events'].append('to_device')
    h, w = int(x.shape[2]), int(x.shape[3])
    lq_latent, mean, std = assets.prepare_features(x)
    if timing is not None:
        timing['events'].append('prepare_features')
    require_finite(lq_latent, 'VAE 潜码')
    gen = make_generator(cfg['seed'] if seed is None else seed, x.device)
    with torch.no_grad():
        z1 = assets.one_step_tiled(lq_latent, gen)
        require_finite(z1, 'DiT 一步更新后的潜码')
        if timing is not None:
            timing['events'].append('one_step')
        sr = TV.decode_dispatch(assets.vae, z1, assets.args, mean, std,
                                light_decoder=assets.dec)
        if timing is not None:
            timing['events'].append('decode')
    # 关键：在 wavelet / PIL 量化之前拦截 NaN/Inf，避免被 uint8 舍入“洗白”（PIPE-R05）
    require_finite(sr, 'LightDecoder 解码输出（量化前）')
    if (int(sr.shape[2]), int(sr.shape[3])) != (h, w):
        # VAE 只保证 8 的倍数：非整倍数输入（如 511x513）必须裁回原尺寸
        if int(sr.shape[2]) < h or int(sr.shape[3]) < w:
            raise OutputValidationError('decode 输出 %s 小于输入 (%d,%d)，无法裁回'
                                        % (tuple(sr.shape[2:]), h, w))
        sr = sr[:, :, :h, :w]
    out = wavelet_color_fix_tensor(sr, x, sigma=int(cfg['align_sigma']))   # 共享后处理，计入
    if timing is not None:
        timing['events'].append('postprocess')
    validate_output(out, h, w)
    _device_sync(assets.device)
    if timing is not None:
        timing['infer_s'] = time.perf_counter() - t_start
        timing['events'].append('exit')
    return out
