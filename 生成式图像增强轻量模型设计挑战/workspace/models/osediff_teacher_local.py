"""OSEDiff 教师推理封装 (本地 diffusers 0.39 适配版)。

官方仓库 (D:/models/OSEDiff-main/) 的 osediff.py 依赖旧版 diffusers 0.25
(FromOriginalVAEMixin/PositionNet 等已删 API)。本文件用本地 diffusers 0.39
官方 AutoencoderKL/UNet2DConditionModel 重建 OSEDiff_test 的加载与推理，
LoRA 注入逻辑 (initialize + load_ckpt) 从 osediff.py 复制。

差异点:
  - 类: diffusers 官方 AutoencoderKL / UNet2DConditionModel (非仓库自定义)
  - merge: fuse_lora() 替代 merge_and_unload()
  - 推理: 保留 tiled VAE + tiled latent (4K 分块, 8GB 可跑)

用法:
    from models.osediff_teacher_local import OSEDiffTeacherLocal
    teacher = OSEDiffTeacherLocal(sd_path='D:/models/sd21base',
                                  osediff_dir='D:/models/OSEDiff-main/OSEDiff-main')
    out_pil = teacher.infer(lq_pil, prompt='a photo of ...')  # 单张
    teacher.infer_batch(lq_pil_list, prompts, out_dir)         # 批量 4K 分块
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

# ---- transformers 5.14 兼容: 旧 API 已移除, 注入等价实现 (RAM bert.py 依赖) ----
import transformers.modeling_utils as _mu
if not hasattr(_mu, 'apply_chunking_to_forward'):
    def apply_chunking_to_forward(chunk_size, chunk_dim, forward_fn, *args, **kwargs):
        """transformers 4.x 的 chunked forward; 5.14 移除, 此处提供等价的简单实现。"""
        if chunk_size == 0:
            return forward_fn(*args, **kwargs)
        import torch as _t
        x = args[0]
        if x.size(chunk_dim) <= chunk_size:
            return forward_fn(*args, **kwargs)
        outputs = [_t.cat([forward_fn(x[i:i + chunk_size], *args[1:], **kwargs)
                           for i in range(0, x.size(chunk_dim), chunk_size)], dim=chunk_dim)]
        return outputs[0]
    _mu.apply_chunking_to_forward = apply_chunking_to_forward

if not hasattr(_mu, 'find_pruneable_heads_and_indices'):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        """transformers 4.x 的 head 剪枝辅助; 5.14 移除。RAM 未真正调用, 提供兼容实现。"""
        # 简化实现: 直接报出未实现 (RAM 推理不用剪枝)
        raise NotImplementedError('find_pruneable_heads_and_indices is not available in transformers 5.x')
    _mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

if not hasattr(_mu, 'prune_linear_layer'):
    def prune_linear_layer(layer, index, dim=0):
        """transformers 4.x 的线性层剪枝; 5.14 移除。简化: 按 index 裁剪。"""
        import torch as _t
        if index is None or len(index) == layer.weight.shape[dim]:
            return layer
        if dim == 0:
            new_w = _t.nn.Parameter(layer.weight.data.index_select(0, index))
            new_b = None if layer.bias is None else _t.nn.Parameter(layer.bias.data.index_select(0, index))
        elif dim == 1:
            new_w = _t.nn.Parameter(layer.weight.data.index_select(1, index))
            new_b = layer.bias
        else:
            raise ValueError(f'dim {dim} not supported')
        new_layer = _t.nn.Linear(new_w.shape[1], new_w.shape[0])
        new_layer.weight = new_w
        if new_b is not None:
            new_layer.bias = new_b
        return new_layer
    _mu.prune_linear_layer = prune_linear_layer

# file_utils.ModelOutput -> modeling_outputs.ModelOutput (5.x 里 ModelOutput 在 modeling_outputs)
import transformers.file_utils as _fu
if not hasattr(_fu, 'ModelOutput'):
    from transformers.modeling_outputs import ModelOutput
    _fu.ModelOutput = ModelOutput

# 5.14 移除 PreTrainedModel.all_tied_weights_keys 类属性 (BertModel 初始化需要)
# 且 5.14 的 tie_weights 有 bug: get_expanded_tied_weights_keys 返回 tuple 却调 .items()
# RAM 的 BertModel 触发此路径 → tie_weights 为 no-op (RAM 权重已完整, 不需要绑定)
import transformers.modeling_utils as _mu2
if not hasattr(_mu2.PreTrainedModel, 'all_tied_weights_keys'):
    _mu2.PreTrainedModel.all_tied_weights_keys = ()
if hasattr(_mu2.PreTrainedModel, 'tie_weights'):
    def _noop_tie_weights(self, *args, **kwargs):
        return None
    _mu2.PreTrainedModel.tie_weights = _noop_tie_weights

# RAM 的 init_tokenizer 用 transformers 4.x API (additional_special_tokens_ids),
# 5.14 已移除。给 BertTokenizer 类加属性补丁 (RAM 内部任意处调用都生效)。
try:
    import transformers as _tf
    _BertTokenizer = _tf.BertTokenizer
    if not hasattr(_BertTokenizer, 'additional_special_tokens_ids'):
        def _get_additional_special_tokens_ids(self):
            try:
                # 4.x: additional_special_tokens = 通过 add_special_tokens 新加的自定义 token
                # 5.14 里这些在 added_tokens_encoder (排除 base vocab 内置特殊 token)
                base_ids = set(self.convert_tokens_to_ids(
                    ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']))
                return [i for i in self.added_tokens_encoder.values()
                        if i not in base_ids]
            except Exception:
                return []
        _BertTokenizer.additional_special_tokens_ids = property(
            _get_additional_special_tokens_ids)
except Exception:
    pass


class OSEDiffTeacherLocal:
    """本地适配 OSEDiff 教师: LoRA 注入 SD21 UNet+VAE, 单步 t=999, 支持 4K tiled。"""

    def __init__(self, sd_path='D:/models/sd21base',
                 osediff_dir='D:/models/OSEDiff-main/OSEDiff-main',
                 dape_path=None, device='cuda', fp16=True):
        self.device = torch.device(device)
        self.fp16 = fp16
        self.weight_dtype = torch.float16 if fp16 else torch.float32
        self.osediff_dir = osediff_dir
        sys.path.insert(0, osediff_dir)

        # ---- SD21 基座组件 (本地官方类) ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            sd_path, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path, subfolder='text_encoder').to(self.device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            sd_path, subfolder='scheduler')
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder='vae').to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet').to(self.device)

        # ---- 加载 osediff.pkl (LoRA) ----
        ck = torch.load(os.path.join(osediff_dir, 'preset', 'models', 'osediff.pkl'),
                        map_location='cpu', weights_only=False)
        self._load_ckpt(ck)

        # ---- LoRA 动态应用 (不 fuse! 0.39 的 fuse_lora 与 OSEDiff 训练缩放语义不同, 融合后输出压平) ----
        # set_adapter 已由 load_ckpt 设置, peft backend 在 forward 时自动应用

        # ---- 推理精度 ----
        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        for m in (self.vae, self.unet, self.text_encoder):
            m.eval()
        self.timesteps = torch.tensor([999], device=self.device).long()
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        # ---- 4K tiled VAE (复用官方 my_utils.vaehook) ----
        # 注意: VAEHook 与 vae LoRA 可能冲突 (输出压平), 可用 SKIP_TILED=1 隔离测试
        if os.environ.get('SKIP_TILED') != '1':
            from my_utils.vaehook import VAEHook
            enc, dec = self.vae.encoder, self.vae.decoder
            if not hasattr(enc, 'original_forward'):
                setattr(enc, 'original_forward', enc.forward)
            if not hasattr(dec, 'original_forward'):
                setattr(dec, 'original_forward', dec.forward)
            self.vae.encoder.forward = VAEHook(
                enc, 1024, is_decoder=False, fast_decoder=False,
                fast_encoder=False, color_fix=False, to_gpu=True)
            self.vae.decoder.forward = VAEHook(
                dec, 224, is_decoder=True, fast_decoder=False,
                fast_encoder=False, color_fix=False, to_gpu=True)

        # RAM 标签模型 (可选; 缺权重则用固定 prompt)
        self.ram = None
        ram_path = os.path.join(osediff_dir, 'preset', 'models', 'ram_swin_large_14m.pth')
        if not os.path.exists(ram_path):
            ram_path = os.path.join(os.path.dirname(osediff_dir), '..', 'ram_swin_large_14m.pth')
            if not os.path.exists(ram_path):
                ram_path = 'D:/models/ram_swin_large_14m.pth'
        if os.path.exists(ram_path):
            try:
                from ram.models.ram_lora import ram as ram_cls
                from ram import inference_ram as inference
                from ram import get_transform
                self.ram = ram_cls(pretrained=ram_path, pretrained_condition=dape_path,
                                   image_size=384, vit='swin_l').to(self.device).eval()
                if fp16:
                    self.ram = self.ram.half()
                self.ram_transform = get_transform(image_size=384)
                self.ram_infer = inference
                print(f'[teacher] RAM loaded from {ram_path}', flush=True)
            except Exception as e:
                print(f'[teacher] RAM load skipped: {e}', flush=True)
                self.ram = None

    # ---------- LoRA 注入 (从官方 osediff.py 复制) ----------
    def _initialize_unet(self):
        l_enc, l_dec, l_others = [], [], []
        l_grep = ['to_k', 'to_q', 'to_v', 'to_out.0', 'conv', 'conv1', 'conv2',
                  'conv_in', 'conv_shortcut', 'conv_out', 'proj_out', 'proj_in',
                  'ff.net.2', 'ff.net.0.proj']
        for n, _p in self.unet.named_parameters():
            if 'bias' in n or 'norm' in n:
                continue
            for pat in l_grep:
                if pat in n and ('down_blocks' in n or 'conv_in' in n):
                    l_enc.append(n.replace('.weight', '')); break
                elif pat in n and ('up_blocks' in n or 'conv_out' in n):
                    l_dec.append(n.replace('.weight', '')); break
                elif pat in n:
                    l_others.append(n.replace('.weight', '')); break
        return l_enc, l_dec, l_others

    def _initialize_vae(self, part=None):
        """收集 vae 的 LoRA 目标模块。part: 'encoder'/'decoder'/None(全部)。
        osediff.pkl 的 state_dict_vae 同时含 encoder+decoder 键 (官方 save_model 保存全部 lora),"""
        l_mods = []
        l_grep = ['conv1', 'conv2', 'conv_in', 'conv_shortcut', 'conv', 'conv_out',
                  'to_k', 'to_q', 'to_v', 'to_out.0']
        for n, _p in self.vae.named_parameters():
            if 'bias' in n or 'norm' in n:
                continue
            if part == 'decoder' and 'post_quant_conv' in n:
                l_mods.append(n.replace('.weight', '')); continue
            if part == 'encoder' and 'quant_conv' in n and 'post_quant_conv' not in n:
                l_mods.append(n.replace('.weight', '')); continue
            for pat in l_grep:
                if part == 'encoder' and 'encoder' in n and pat in n:
                    l_mods.append(n.replace('.weight', '')); break
                elif part == 'decoder' and 'decoder' in n and pat in n:
                    l_mods.append(n.replace('.weight', '')); break
                elif part is None and pat in n:
                    l_mods.append(n.replace('.weight', '')); break
                elif part is None and 'quant_conv' in n and 'post_quant_conv' not in n:
                    l_mods.append(n.replace('.weight', '')); break
        return l_mods

    def _load_ckpt(self, model):
        # unet lora (3 adapter: encoder/decoder/others)
        l_enc, l_dec, l_others = self._initialize_unet()
        for name, tgt in [('default_encoder', l_enc), ('default_decoder', l_dec), ('default_others', l_others)]:
            conf = LoraConfig(r=model['rank_unet'], init_lora_weights='gaussian', target_modules=tgt)
            self.unet.add_adapter(conf, adapter_name=name)
        for n, p in self.unet.named_parameters():
            if 'lora' in n or 'conv_in' in n:
                p.data.copy_(model['state_dict_unet'][n])
        self.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

        # vae lora: 官方 initialize_vae 只给 encoder 加 LoRA (decoder 键是保存冗余, 官方不加载)
        # 实测: 给 decoder 加 adapter 会导致 decode 输出压平 (std 112->12), 与官方行为一致只做 encoder
        l_enc_v = self._initialize_vae(part='encoder')
        if l_enc_v:
            conf = LoraConfig(r=model['rank_vae'], init_lora_weights='gaussian', target_modules=l_enc_v)
            self.vae.add_adapter(conf, adapter_name='default_encoder')
            for n, p in self.vae.named_parameters():
                if 'lora' in n and n in model['state_dict_vae']:
                    p.data.copy_(model['state_dict_vae'][n])
            self.vae.set_adapter(['default_encoder'])

    # ---------- 推理 ----------
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tok = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                             padding='max_length', truncation=True, return_tensors='pt')
        emb = self.text_encoder(tok.input_ids.to(self.device))[0]
        return emb.to(self.weight_dtype)

    def get_prompt(self, lq_pil: Image.Image, user_prompt: str = '') -> str:
        """RAM 标签 -> prompt (无 RAM 则用固定通用 prompt)。"""
        if self.ram is not None:
            try:
                img_t = self.ram_transform(lq_pil.convert('RGB')).unsqueeze(0).to(self.device)
                if self.fp16:
                    img_t = img_t.half()
                captions = self.ram_infer(img_t, self.ram)
                if captions and captions[0]:
                    return f'{captions[0]}, {user_prompt},'
            except Exception as e:
                print(f'[teacher] RAM infer failed, fallback: {e}', flush=True)
        return f'{user_prompt},' if user_prompt else 'a photo of a high-quality, sharp, clean scene,'

    @torch.no_grad()
    def infer_tensor(self, lq: torch.Tensor, prompt: str = '') -> torch.Tensor:
        """lq[-1,1] tensor [1,3,H,W] -> out[-1,1]。4K 需 tiled。"""
        prompt_embeds = self._encode_prompt(prompt)
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        out = self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample
        return out.clamp(-1, 1)

    @torch.no_grad()
    def infer_tiled(self, lq_pil: Image.Image, prompt: str = '',
                    tile=512, overlap=64) -> Image.Image:
        """4K 分块推理 (高斯融合), 返回 PIL。"""
        from infer import gaussian_weights  # 复用 workspace infer.py 的融合
        img_pil = lq_pil.convert('RGB')
        W, H = img_pil.size
        arr = np.asarray(img_pil, dtype=np.float32) / 127.5 - 1.0
        pw = (tile - W % tile) % tile
        ph = (tile - H % tile) % tile
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode='reflect')
        H2, W2 = arr.shape[:2]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = torch.zeros(1, 3, H2, W2, device=self.device)
        wsum = torch.zeros(1, 1, H2, W2, device=self.device)
        g = gaussian_weights(tile, overlap).to(self.device)
        step = tile - overlap
        for y in range(0, H2, step):
            for x in range(0, W2, step):
                y1 = min(y, H2 - tile)
                x1 = min(x, W2 - tile)
                patch = t[:, :, y1:y1 + tile, x1:x1 + tile]
                enh = self.infer_tensor(patch, prompt)
                out[:, :, y1:y1 + tile, x1:x1 + tile] += enh * g
                wsum[:, :, y1:y1 + tile, x1:x1 + tile] += g
        out = out / wsum.clamp(min=1e-8)
        out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
        out_np = np.clip(out_np[:H, :W], 0, 255).astype(np.uint8)
        return Image.fromarray(out_np)

    def infer(self, lq_pil: Image.Image, prompt: str = '') -> Image.Image:
        """自动选择: 小图直接, 大图(4K)分块。"""
        W, H = lq_pil.size
        if W * H <= 512 * 512:
            arr = np.asarray(lq_pil.convert('RGB'), dtype=np.float32) / 127.5 - 1.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            out = self.infer_tensor(x, prompt)
            out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
            return Image.fromarray(np.clip(out_np, 0, 255).astype(np.uint8))
        return self.infer_tiled(lq_pil, prompt)
