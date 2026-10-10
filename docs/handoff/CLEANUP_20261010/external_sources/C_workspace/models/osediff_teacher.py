"""OSEDiff 教师推理封装（v5 蒸馏用）

依赖 OSEDiff 官方仓库 (/data/osediff/OSEDiff/)：
  - osediff.py: OSEDiff 类 (VAE+UNet+scheduler 一步推理)
  - osediff.pkl / DAPE.pth / ram_swin_large_14m.pth
  - RAM 模型用于提取图像标签 -> CLIP 文本条件

⚠️ 本文件按官方仓库 API 编写，服务器上需按 /data/osediff/OSEDiff 实际结构核对后微调。
   所有路径通过 --osediff_dir 等参数传入，不硬编码。

用法（推理函数）:
    from models.osediff_teacher import OSEDiffTeacher
    teacher = OSEDiffTeacher(osediff_dir, sd_path, ram_path, dape_path, device)
    hq = teacher.infer(lq_pil)   # lq_pil: PIL RGB 任意尺寸 -> hq_pil
"""
import os
import sys
import numpy as np
import torch
from PIL import Image


class OSEDiffTeacher:
    def __init__(self, osediff_dir, sd_path, ram_path, dape_path=None,
                 device='cuda', fp16=True):
        """初始化教师模型（RAM + CLIP text encoder + OSEDiff 单步核心）。

        Args:
            osediff_dir: OSEDiff 官方仓库根目录（含 osediff.py, ram/, configs/）
            sd_path:     SD 基座目录（含 vae/ unet/ text_encoder/ tokenizer/）
            ram_path:    ram_swin_large_14m.pth
            dape_path:   DAPE.pth（可选）
        """
        self.device = torch.device(device)
        self.fp16 = fp16
        self.osediff_dir = osediff_dir
        sys.path.insert(0, osediff_dir)

        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer

        # --- SD 基座组件 ---
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder='vae').to(device)
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet').to(device)
        self.scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder='scheduler')
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path, subfolder='text_encoder').to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder='tokenizer')
        self.scaling = self.vae.config.scaling_factor

        # --- 加载 OSEDiff 蒸馏权重 (osediff.pkl) ---
        pkl_path = os.path.join(osediff_dir, 'osediff.pkl')
        ckpt = torch.load(pkl_path, map_location='cpu', weights_only=False)
        if 'model' in ckpt:
            ckpt = ckpt['model']  # 官方蒸馏保存格式: {"model": state_dict}
        # osediff.pkl 可能同时含 unet 与 vae 权重，分开加载（strict 各自允许缺）
        unet_sd = {k: v for k, v in ckpt.items() if k.startswith('unet.')}
        if unet_sd:
            self.unet.load_state_dict(
                {k[len('unet.'):]: v for k, v in unet_sd.items()}, strict=False)
        vae_sd = {k: v for k, v in ckpt.items() if k.startswith('vae.')}
        if vae_sd:
            self.vae.load_state_dict(
                {k[len('vae.'):]: v for k, v in vae_sd.items()}, strict=False)
        # 直接以 unet/vae 为前缀的 key 也试一次（不同保存格式）
        direct = {k: v for k, v in ckpt.items()
                  if not k.startswith('unet.') and not k.startswith('vae.')}
        if direct and not unet_sd and not vae_sd:
            self.unet.load_state_dict(direct, strict=False)

        # --- DAPE 适配器（若存在） ---
        if dape_path and os.path.exists(dape_path):
            dape = torch.load(dape_path, map_location='cpu', weights_only=False)
            dape = dape.get('model', dape)
            # DAPE 注入文本条件，加载到 text_encoder 对应层或 unet；strict=False 容错
            try:
                self.text_encoder.load_state_dict(dape, strict=False)
            except Exception:
                try:
                    self.unet.load_state_dict(dape, strict=False)
                except Exception as e:
                    print(f'[teacher] DAPE load skipped: {e}', flush=True)

        # --- RAM 标签模型 ---
        from ram.models.ram_swin_large import RAM_Swin_Large
        from ram import get_transform
        self.ram = RAM_Swin_Large(pretrained=ram_path, image_size=384, vit='swin_l')
        self.ram = self.ram.to(device).eval()
        self.ram_transform = get_transform(image_size=384)

        for m in (self.vae, self.unet, self.text_encoder, self.ram):
            m.eval()
            if fp16:
                m.half()

        self.t = torch.tensor([self.scheduler.config.num_train_timesteps - 1],
                              device=device)  # t = 999（官方单步）
        self.alpha_t = self.scheduler.alphas_cumprod[self.t[0].item()].to(device)

    @torch.no_grad()
    def _get_tags(self, lq_pil: Image.Image) -> str:
        """RAM 整图标签 -> prompt 文本。"""
        from ram import inference_ram as inference
        img_t = self.ram_transform(lq_pil.convert('RGB')).unsqueeze(0).to(self.device)
        if self.fp16:
            img_t = img_t.half()
        tags, _ = inference(self.ram, img_t, self.ram.normalize, None, None, False)
        return f'a photo of {tags[0]}'

    @torch.no_grad()
    def _encode_prompt(self, text: str) -> torch.Tensor:
        """文本 -> CLIP embeddings [1,77,1024]（官方 prompt 模板）。"""
        text_inputs = self.tokenizer(
            text, max_length=self.tokenizer.model_max_length,
            padding='max_length', truncation=True, return_tensors='pt')
        emb = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        return emb.half() if self.fp16 else emb

    @torch.no_grad()
    def infer(self, lq_pil: Image.Image) -> Image.Image:
        """单步 OSEDiff 推理：LQ PIL -> HQ PIL（同分辨率）。"""
        tag_text = self._get_tags(lq_pil)
        prompt_embeds = self._encode_prompt(tag_text)

        arr = np.asarray(lq_pil.convert('RGB'), dtype=np.float32) / 127.5 - 1.0
        t_in = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.fp16:
            t_in = t_in.half()

        latents = self.vae.encode(t_in).latent_dist.sample() * self.scaling
        noise_pred = self.unet(
            latents, self.t, encoder_hidden_states=prompt_embeds).sample
        # 显式 x0（与 v1-v4 学生相同的公式，可导且确定）
        x0 = (latents - (1 - self.alpha_t).sqrt() * noise_pred) / self.alpha_t.sqrt()
        out = self.vae.decode(x0 / self.scaling).sample.clamp(-1, 1)

        out_np = (out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                  * 127.5 + 127.5)
        return Image.fromarray(np.clip(out_np, 0, 255).astype(np.uint8))
