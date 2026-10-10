# -*- coding: utf-8 -*-
"""SCORE-01 性能候选 runner（v2）：与交付基线 v1 **数值等价**，仅去除确定的非数值开销。

相对 v1 的全部改动（由 _score01_gen_v2.py 机械替换生成，其余行逐字节相同）：
  A. 6 处仅为阶段计时服务的 torch.cuda.synchronize()（self._dev_sync）改为仅在 timing 请求时执行；
  B. 缓存与输入内容无关的常量：Gaussian 融合权重（键 h/w/通道/设备，<=4 条）、tile 网格
     （键 length/tile/overlap，<=8 条，返回副本）、t_cur/t_nxt 常量张量（键 batch/值/设备/dtype，<=8 条）、
     复用同一 Generator（每次仍 manual_seed）；
  C. 所有 finite 检查、clamp/量化顺序、分块/融合数学与接口保持 v1 不变。

模型文件 your_model.pt 逐字节复用（不改权重/精度）。
"""
from __future__ import annotations

import json
import os
import time

import torch

AE_FACTOR = 8


_GAUSS_CACHE = {}
_GAUSS_CACHE_MAX = 4
_GRID_CACHE = {}
_GRID_CACHE_MAX = 8
_T_CACHE = {}
_T_CACHE_MAX = 8


def _gaussian_weights_uncached(tile_h, tile_w, channels, device):
    var = 0.01
    mid_h, mid_w = (tile_h - 1) / 2, (tile_w - 1) / 2
    y = torch.arange(tile_h, dtype=torch.float32)
    x = torch.arange(tile_w, dtype=torch.float32)
    wy = torch.exp(-((y - mid_h) / tile_h) ** 2 / (2 * var))
    wx = torch.exp(-((x - mid_w) / tile_w) ** 2 / (2 * var))
    w = wy[:, None] * wx[None, :]
    return w.to(device).unsqueeze(0).unsqueeze(0).expand(1, channels, -1, -1)


def _gaussian_weights(tile_h, tile_w, channels, device):
    """v1 的同一常量；仅按 (h,w,c,device) 缓存（与输入内容无关）。"""
    key = (int(tile_h), int(tile_w), int(channels), str(device))
    hit = _GAUSS_CACHE.get(key)
    if hit is not None and hit.device == torch.device(device):
        return hit
    w = _gaussian_weights_uncached(tile_h, tile_w, channels, device)
    if len(_GAUSS_CACHE) >= _GAUSS_CACHE_MAX:
        _GAUSS_CACHE.pop(next(iter(_GAUSS_CACHE)))
    _GAUSS_CACHE[key] = w
    return w


def _make_tile_grid(length, tile, overlap):
    """v1 的同一网格；仅按 (length,tile,overlap) 缓存并返回副本。"""
    key = (int(length), int(tile), int(overlap))
    hit = _GRID_CACHE.get(key)
    if hit is not None:
        return list(hit)
    stride = max(tile - overlap, 1)
    if length <= tile:
        positions = [0]
    else:
        positions = list(range(0, length - tile + 1, stride))
        if positions[-1] + tile < length:
            positions.append(length - tile)
        positions = sorted(set(positions))
    if len(_GRID_CACHE) >= _GRID_CACHE_MAX:
        _GRID_CACHE.pop(next(iter(_GRID_CACHE)))
    _GRID_CACHE[key] = tuple(positions)
    return list(positions)


class RunnerError(RuntimeError):
    pass


def _finite_ok(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


class Runner:
    def __init__(self, model_dir: str, device: str = None):
        self.model_dir = str(model_dir)
        cfg_path = os.path.join(self.model_dir, 'models_export.json')
        if not os.path.isfile(cfg_path):
            raise RunnerError('缺少 models_export.json: %s' % cfg_path)
        with open(cfg_path, 'r', encoding='utf-8') as f:
            self.recipe = json.load(f)
        ckpt = os.path.join(self.model_dir, self.recipe['model_file'])
        if not os.path.isfile(ckpt):
            raise RunnerError('缺少模型文件: %s' % ckpt)
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.model = torch.jit.load(ckpt, map_location=self.device).eval()
        r = self.recipe
        self.dtype = torch.float16 if r['weight_dtype'] == 'float16' else torch.float32
        self.seed = int(r['seed'])
        self.t_cur_val = float(r['t_start'])
        self.t_nxt_val = float(r['t_end'])
        self.dit_tile = int(r['dit_tile_size'])
        self.dit_overlap_cli = int(r['dit_tile_overlap_cli'])
        self.vae_tile = int(r['vae_tile_size'])
        self.vae_overlap_cli = int(r['vae_tile_overlap_cli'])
        self.trace_tile_px = [int(v) for v in r['vae_enc_trace_px']]
        self.trace_dec_latent = [int(v) for v in r['dec_trace_latent']]
        # 冻结执行模式：交付 runner 自身也要兑现，不能只在验证脚本里套上下文（EXP-R09/R11）
        self.jit_optimized_execution = bool(r.get('jit_optimized_execution', False))

    # ── 分块参数（与参考实现同式）──
    def _dit_params(self, lh, lw):
        patch = int(self.recipe['patch_size'])
        lt = max((self.dit_tile // AE_FACTOR // patch) * patch, patch)
        lov = max(self.dit_overlap_cli // AE_FACTOR, lt // 8)
        lt = min(lt, min(lh, lw))
        lov = min(lov, lt - 1)
        return lt, lov

    def _vae_params(self, lh, lw):
        lt = max(self.vae_tile // AE_FACTOR, 1)
        lov = max(self.vae_overlap_cli // AE_FACTOR, lt // 8)
        lt = min(lt, min(lh, lw))
        lov = min(lov, lt - 1)
        return lt, lov

    @torch.no_grad()
    def _encode_latent(self, x):
        h, w = int(x.shape[2]), int(x.shape[3])
        ph = (AE_FACTOR - h % AE_FACTOR) % AE_FACTOR
        pw = (AE_FACTOR - w % AE_FACTOR) % AE_FACTOR
        pad = torch.nn.functional.pad(x, (0, pw, 0, ph), mode='reflect') if (ph or pw) else x
        lh, lw = int(pad.shape[2]) // AE_FACTOR, int(pad.shape[3]) // AE_FACTOR
        lt, lov = self._vae_params(lh, lw)
        tile_px = lt * AE_FACTOR
        if tile_px not in self.trace_tile_px:
            raise RunnerError('VAE 编码块像素 %d 不在已导出 trace 尺寸 %s 内'
                              % (tile_px, self.trace_tile_px))
        out_lh, out_lw = -(-h // AE_FACTOR), -(-w // AE_FACTOR)
        if lh <= lt and lw <= lt:
            z = self.model.encode(pad.to(self.dtype).contiguous(), tile_px)
            self._require_finite(z, 'VAE 编码潜码(单块)')
            return z[:, :, :out_lh, :out_lw].contiguous()
        h_pos, w_pos = _make_tile_grid(lh, lt, lov), _make_tile_grid(lw, lt, lov)
        acc = wacc = g = None
        for hi in h_pos:
            for wi in w_pos:
                crop = pad[:, :, hi * AE_FACTOR:(hi + lt) * AE_FACTOR,
                           wi * AE_FACTOR:(wi + lt) * AE_FACTOR].contiguous()
                zt = self.model.encode(crop.to(self.dtype), tile_px)
                self._require_finite(zt, 'VAE 编码潜码(分块)')
                if acc is None:
                    acc = torch.zeros(1, int(zt.shape[1]), lh, lw, device=zt.device,
                                      dtype=torch.float32)
                    wacc = torch.zeros_like(acc)
                    g = _gaussian_weights(lt, lt, int(zt.shape[1]), zt.device)
                acc[:, :, hi:hi + lt, wi:wi + lt] += zt.float() * g
                wacc[:, :, hi:hi + lt, wi:wi + lt] += g
        blended = acc / wacc
        self._require_finite(blended, 'VAE 编码潜码(分块融合)')
        return blended[:, :, :out_lh, :out_lw].to(self.dtype).contiguous()

    @torch.no_grad()
    def _features(self, x, lh, lw, lt, h_pos, w_pos):
        if lh <= lt and lw <= lt:
            f = self.model.features(x.to(self.dtype).contiguous())
            self._require_finite(f, 'DINO 条件特征(单块)')
            return {(0, 0): f}
        feats = {}
        for hi in h_pos:
            for wi in w_pos:
                ph_s, pw_s = hi * AE_FACTOR, wi * AE_FACTOR
                ph_e = min((hi + lt) * AE_FACTOR, int(x.shape[2]))
                pw_e = min((wi + lt) * AE_FACTOR, int(x.shape[3]))
                f = self.model.features(x[:, :, ph_s:ph_e, pw_s:pw_e].to(self.dtype).contiguous())
                self._require_finite(f, 'DINO 条件特征(分块)')
                feats[(hi, wi)] = f
        return feats

    @torch.no_grad()
    def _one_step(self, lq_latent, gen, feats, lh, lw, lt, h_pos, w_pos):
        b = int(lq_latent.shape[0])
        t_cur = self._cached_full(b, self.t_cur_val, lq_latent)
        t_nxt = self._cached_full(b, self.t_nxt_val, lq_latent)
        z = torch.randn(lq_latent.shape, generator=gen, device=lq_latent.device,
                        dtype=torch.float32).to(self.dtype)
        if lh <= lt and lw <= lt:
            z1 = self.model.step(lq_latent, z, t_cur, t_nxt, feats[(0, 0)])
            self._require_finite(z1, 'DiT 一步更新后的潜码(单块)')
            return z1
        # EXP-R01：step 返回的**已经是更新后的潜码 z1**（z - dt·u）。分块只做加权融合，
        # 不能再减一次 dt·velocity。数学上 Σg(z-dt·u)/Σg = z - dt·Σg·u/Σg，与参考实现等价。
        g = _gaussian_weights(lt, lt, int(lq_latent.shape[1]), lq_latent.device)
        z1_acc = torch.zeros(1, int(lq_latent.shape[1]), lh, lw, device=lq_latent.device,
                             dtype=torch.float32)
        w_acc = torch.zeros_like(z1_acc)
        for hi in h_pos:
            for wi in w_pos:
                he, we = hi + lt, wi + lt
                z1 = self.model.step(lq_latent[:, :, hi:he, wi:we].contiguous(),
                                     z[:, :, hi:he, wi:we].contiguous(),
                                     t_cur, t_nxt, feats[(hi, wi)])
                z1_acc[:, :, hi:he, wi:we] += z1.float() * g
                w_acc[:, :, hi:he, wi:we] += g
        out = (z1_acc / w_acc).to(self.dtype)
        self._require_finite(out, 'DiT 一步更新后的潜码(分块融合)')
        return out

    @torch.no_grad()
    def _decode(self, sr_latent):
        lh, lw = int(sr_latent.shape[2]), int(sr_latent.shape[3])
        lt, lov = self._vae_params(lh, lw)
        if lt not in self.trace_dec_latent:
            raise RunnerError('VAE 解码块 %d 不在已导出 trace 尺寸 %s 内'
                              % (lt, self.trace_dec_latent))
        if lh <= lt and lw <= lt:
            pix = self.model.decode(sr_latent, lt)
            self._require_finite(pix, '解码原始输出(clamp 前, 单块)')
            return pix.clamp(-1.0, 1.0)
        h_pos, w_pos = _make_tile_grid(lh, lt, lov), _make_tile_grid(lw, lt, lov)
        g = _gaussian_weights(lt * AE_FACTOR, lt * AE_FACTOR, 3, sr_latent.device)
        acc = torch.zeros(1, 3, lh * AE_FACTOR, lw * AE_FACTOR, device=sr_latent.device,
                          dtype=torch.float32)
        wacc = torch.zeros_like(acc)
        for hi in h_pos:
            for wi in w_pos:
                he, we = hi + lt, wi + lt
                pix = self.model.decode(sr_latent[:, :, hi:he, wi:we].contiguous(), lt)
                self._require_finite(pix, '解码原始输出(clamp 前, 分块)')
                pix = pix.clamp(-1.0, 1.0)      # EXP-R08：逐块 clamp 后再融合（恢复原计算顺序）
                acc[:, :, hi * AE_FACTOR:he * AE_FACTOR, wi * AE_FACTOR:we * AE_FACTOR] += \
                    pix.float() * g
                wacc[:, :, hi * AE_FACTOR:he * AE_FACTOR, wi * AE_FACTOR:we * AE_FACTOR] += g
        blended = acc / wacc
        self._require_finite(blended, '解码融合输出(最终 clamp 前)')
        return blended.clamp(-1.0, 1.0)

    @torch.no_grad()
    def infer(self, image_tensor: torch.Tensor, prompt: str = "", timing: dict = None):
        with torch.jit.optimized_execution(self.jit_optimized_execution):
            return self._infer_body(image_tensor, prompt=prompt, timing=timing)

    @torch.no_grad()
    def _infer_body(self, image_tensor: torch.Tensor, prompt: str = "", timing: dict = None):
        want_timing = timing is not None      # SCORE-01：无 profiling 时不做仅为计时的同步
        t0 = time.perf_counter()
        if not isinstance(image_tensor, torch.Tensor) or image_tensor.dim() != 4 or \
                int(image_tensor.shape[0]) != 1 or int(image_tensor.shape[1]) != 3:
            raise RunnerError('输入须为 [1,3,H,W]')
        h, w = int(image_tensor.shape[2]), int(image_tensor.shape[3])
        x = image_tensor.detach().to(self.device, dtype=torch.float32).contiguous()
        if want_timing:
            self._dev_sync()
        t_in = time.perf_counter()
        lq = self._encode_latent(x)
        lh, lw = int(lq.shape[2]), int(lq.shape[3])
        lt, lov = self._dit_params(lh, lw)
        h_pos = _make_tile_grid(lh, lt, lov)
        w_pos = _make_tile_grid(lw, lt, lov)
        if want_timing:
            self._dev_sync()
        t_enc = time.perf_counter()
        feats = self._features(x, lh, lw, lt, h_pos, w_pos)
        if want_timing:
            self._dev_sync()
        t_feat = time.perf_counter()
        gen = self._generator()
        z1 = self._one_step(lq, gen, feats, lh, lw, lt, h_pos, w_pos)
        if want_timing:
            self._dev_sync()
        t_dit = time.perf_counter()
        sr = self._decode(z1)
        if want_timing:
            self._dev_sync()
        t_dec = time.perf_counter()
        if int(sr.shape[2]) != h or int(sr.shape[3]) != w:
            if int(sr.shape[2]) < h or int(sr.shape[3]) < w:
                raise RunnerError('decode 输出 %s 小于输入 (%d,%d)'
                                  % (tuple(sr.shape[2:]), h, w))
            sr = sr[:, :, :h, :w].contiguous()
        self._require_finite(sr, '解码输出(clamp 后)')
        self._require_finite(x, '输入图像')
        out = self.model.post(sr.float().contiguous(), x.float().contiguous())
        if want_timing:
            self._dev_sync()
        t_post = time.perf_counter()
        out = out.float().clamp(-1.0, 1.0)
        if not bool(torch.isfinite(out).all()):
            raise RunnerError('输出含 NaN/Inf')
        if timing is not None:
            timing.update({'encode_s': round(t_enc - t_in, 4),
                           'features_s': round(t_feat - t_enc, 4),
                           'dit_s': round(t_dit - t_feat, 4),
                           'decode_s': round(t_dec - t_dit, 4),
                           'postprocess_s': round(t_post - t_dec, 4),
                           'infer_s': round(t_post - t0, 4)})
        return out

    def _cached_full(self, b, val, like):
        key = (int(b), float(val), str(like.device), str(self.dtype))
        hit = _T_CACHE.get(key)
        if hit is None:
            hit = torch.full((int(b),), float(val), device=like.device, dtype=self.dtype)
            if len(_T_CACHE) >= _T_CACHE_MAX:
                _T_CACHE.pop(next(iter(_T_CACHE)))
            _T_CACHE[key] = hit
        return hit

    def _generator(self):
        """复用同一个 Generator：每次仍 manual_seed(self.seed)，噪声逐位不变。"""
        if getattr(self, '_gen', None) is None:
            self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(self.seed)
        return self._gen

    def _require_finite(self, t, where):
        """EXP-R03：在 clamp / uint8 量化之前拦截 NaN/Inf（否则会被掩成有限像素）。"""
        if not _finite_ok(t):
            raise RunnerError('%s 出现 NaN/Inf（%d 个元素）' % (where, int((~torch.isfinite(t)).sum())))

    def _dev_sync(self):
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
