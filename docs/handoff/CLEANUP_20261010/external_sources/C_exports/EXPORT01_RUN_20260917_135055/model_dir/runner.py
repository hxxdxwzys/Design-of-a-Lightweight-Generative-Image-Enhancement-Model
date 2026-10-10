# -*- coding: utf-8 -*-
"""EXPORT-01 交付入口（薄 runner）：Runner(model_dir).infer(image_tensor, prompt="")。

职责边界：
  - 只做编排：分块网格、Gaussian 融合、局部 RNG、尺寸补齐/裁回、接口校验与计时；
  - 数值主体全部在 model_dir/your_model.pt（单一 ScriptModule，含 VAE 编码 / DINO 条件 /
    DiT 一步 / LightDecoder / 忠实张量化 wavelet）；
  - 噪声由本 runner 用局部 Generator(seed) 生成并**显式传入**，模型内无随机算子。

依赖：仅 torch 与标准库（不依赖 PIL/cv2/numpy/diffusers/peft/dinov2 源码）。
"""
from __future__ import annotations

import json
import os
import time

import torch

AE_FACTOR = 8


def _gaussian_weights(tile_h, tile_w, channels, device):
    var = 0.01
    mid_h, mid_w = (tile_h - 1) / 2, (tile_w - 1) / 2
    y = torch.arange(tile_h, dtype=torch.float32)
    x = torch.arange(tile_w, dtype=torch.float32)
    wy = torch.exp(-((y - mid_h) / tile_h) ** 2 / (2 * var))
    wx = torch.exp(-((x - mid_w) / tile_w) ** 2 / (2 * var))
    w = wy[:, None] * wx[None, :]
    return w.to(device).unsqueeze(0).unsqueeze(0).expand(1, channels, -1, -1)


def _make_tile_grid(length, tile, overlap):
    stride = max(tile - overlap, 1)
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return sorted(set(positions))


class RunnerError(RuntimeError):
    pass


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
            return z[:, :, :out_lh, :out_lw].contiguous()
        h_pos, w_pos = _make_tile_grid(lh, lt, lov), _make_tile_grid(lw, lt, lov)
        acc = wacc = g = None
        for hi in h_pos:
            for wi in w_pos:
                crop = pad[:, :, hi * AE_FACTOR:(hi + lt) * AE_FACTOR,
                           wi * AE_FACTOR:(wi + lt) * AE_FACTOR].contiguous()
                zt = self.model.encode(crop.to(self.dtype), tile_px)
                if acc is None:
                    acc = torch.zeros(1, int(zt.shape[1]), lh, lw, device=zt.device,
                                      dtype=torch.float32)
                    wacc = torch.zeros_like(acc)
                    g = _gaussian_weights(lt, lt, int(zt.shape[1]), zt.device)
                acc[:, :, hi:hi + lt, wi:wi + lt] += zt.float() * g
                wacc[:, :, hi:hi + lt, wi:wi + lt] += g
        blended = acc / wacc
        return blended[:, :, :out_lh, :out_lw].to(self.dtype).contiguous()

    @torch.no_grad()
    def _features(self, x, lh, lw, lt, h_pos, w_pos):
        if lh <= lt and lw <= lt:
            return {(0, 0): self.model.features(x.to(self.dtype).contiguous())}
        feats = {}
        for hi in h_pos:
            for wi in w_pos:
                ph_s, pw_s = hi * AE_FACTOR, wi * AE_FACTOR
                ph_e = min((hi + lt) * AE_FACTOR, int(x.shape[2]))
                pw_e = min((wi + lt) * AE_FACTOR, int(x.shape[3]))
                feats[(hi, wi)] = self.model.features(
                    x[:, :, ph_s:ph_e, pw_s:pw_e].to(self.dtype).contiguous())
        return feats

    @torch.no_grad()
    def _one_step(self, lq_latent, gen, feats, lh, lw, lt, h_pos, w_pos):
        b = int(lq_latent.shape[0])
        t_cur = torch.full((b,), self.t_cur_val, device=lq_latent.device, dtype=self.dtype)
        t_nxt = torch.full((b,), self.t_nxt_val, device=lq_latent.device, dtype=self.dtype)
        z = torch.randn(lq_latent.shape, generator=gen, device=lq_latent.device,
                        dtype=torch.float32).to(self.dtype)
        if lh <= lt and lw <= lt:
            return self.model.step(lq_latent, z, t_cur, t_nxt, feats[(0, 0)])
        g = _gaussian_weights(lt, lt, int(lq_latent.shape[1]), lq_latent.device)
        u_acc = torch.zeros(1, int(lq_latent.shape[1]), lh, lw, device=lq_latent.device,
                            dtype=torch.float32)
        w_acc = torch.zeros_like(u_acc)
        for hi in h_pos:
            for wi in w_pos:
                he, we = hi + lt, wi + lt
                u = self.model.step(lq_latent[:, :, hi:he, wi:we].contiguous(),
                                    z[:, :, hi:he, wi:we].contiguous(),
                                    t_cur, t_nxt, feats[(hi, wi)])
                u_acc[:, :, hi:he, wi:we] += u.float() * g
                w_acc[:, :, hi:he, wi:we] += g
        dt = self.t_cur_val - self.t_nxt_val
        return (z.float() - dt * (u_acc / w_acc)).to(self.dtype)

    @torch.no_grad()
    def _decode(self, sr_latent):
        lh, lw = int(sr_latent.shape[2]), int(sr_latent.shape[3])
        lt, lov = self._vae_params(lh, lw)
        if lt not in self.trace_dec_latent:
            raise RunnerError('VAE 解码块 %d 不在已导出 trace 尺寸 %s 内'
                              % (lt, self.trace_dec_latent))
        if lh <= lt and lw <= lt:
            return self.model.decode(sr_latent, lt)
        h_pos, w_pos = _make_tile_grid(lh, lt, lov), _make_tile_grid(lw, lt, lov)
        g = _gaussian_weights(lt * AE_FACTOR, lt * AE_FACTOR, 3, sr_latent.device)
        acc = torch.zeros(1, 3, lh * AE_FACTOR, lw * AE_FACTOR, device=sr_latent.device,
                          dtype=torch.float32)
        wacc = torch.zeros_like(acc)
        for hi in h_pos:
            for wi in w_pos:
                he, we = hi + lt, wi + lt
                pix = self.model.decode(sr_latent[:, :, hi:he, wi:we].contiguous(), lt)
                acc[:, :, hi * AE_FACTOR:he * AE_FACTOR, wi * AE_FACTOR:we * AE_FACTOR] += \
                    pix.float() * g
                wacc[:, :, hi * AE_FACTOR:he * AE_FACTOR, wi * AE_FACTOR:we * AE_FACTOR] += g
        return (acc / wacc).clamp(-1, 1)

    @torch.no_grad()
    def infer(self, image_tensor: torch.Tensor, prompt: str = "", timing: dict = None):
        t0 = time.perf_counter()
        if not isinstance(image_tensor, torch.Tensor) or image_tensor.dim() != 4 or \
                int(image_tensor.shape[0]) != 1 or int(image_tensor.shape[1]) != 3:
            raise RunnerError('输入须为 [1,3,H,W]')
        h, w = int(image_tensor.shape[2]), int(image_tensor.shape[3])
        x = image_tensor.detach().to(self.device, dtype=torch.float32).contiguous()
        self._dev_sync()
        t_in = time.perf_counter()
        lq = self._encode_latent(x)
        lh, lw = int(lq.shape[2]), int(lq.shape[3])
        lt, lov = self._dit_params(lh, lw)
        h_pos = _make_tile_grid(lh, lt, lov)
        w_pos = _make_tile_grid(lw, lt, lov)
        self._dev_sync()
        t_enc = time.perf_counter()
        feats = self._features(x, lh, lw, lt, h_pos, w_pos)
        self._dev_sync()
        t_feat = time.perf_counter()
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)
        z1 = self._one_step(lq, gen, feats, lh, lw, lt, h_pos, w_pos)
        self._dev_sync()
        t_dit = time.perf_counter()
        sr = self._decode(z1)
        self._dev_sync()
        t_dec = time.perf_counter()
        if int(sr.shape[2]) != h or int(sr.shape[3]) != w:
            if int(sr.shape[2]) < h or int(sr.shape[3]) < w:
                raise RunnerError('decode 输出 %s 小于输入 (%d,%d)'
                                  % (tuple(sr.shape[2:]), h, w))
            sr = sr[:, :, :h, :w].contiguous()
        out = self.model.post(sr.float().contiguous(), x.float().contiguous())
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

    def _dev_sync(self):
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
