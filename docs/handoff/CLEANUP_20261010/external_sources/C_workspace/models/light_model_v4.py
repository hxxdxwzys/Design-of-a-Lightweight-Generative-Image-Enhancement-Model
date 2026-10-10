"""v4 模型: LoRA 微调 SD2.1 UNet (预训练扩散先验) + 冻结 VAE + CLIP 固定 prompt, 去 RAM。
1-step 显式 x0 预测 (t=500)。比 v3(从零56M) 多了 SD 先验 → 感知质量上限大幅提升。
训练只更新 LoRA + conv_in；VAE/CLIP/UNet基座冻结。
"""
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, AutoTokenizer
from peft import LoraConfig


class LightLatentDiffV4(nn.Module):
    def __init__(self, sd_path, lora_rank=16, t=500,
                 prompt="high quality, detailed, sharp, clear, high resolution"):
        super().__init__()
        # VAE (冻结)
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.scaling = self.vae.config.scaling_factor

        # UNet (预训练, 冻结基座 + 加 LoRA + 梯度检查点省显存)
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        self.unet.requires_grad_(False)
        self.unet.enable_gradient_checkpointing()
        targets = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj"]
        self.unet.add_adapter(LoraConfig(r=lora_rank, init_lora_weights="gaussian",
                                         target_modules=targets))
        # conv_in 可训(输入是LQ潜码,与原SD噪声分布不同)
        for p in self.unet.conv_in.parameters():
            p.requires_grad_(True)
        self.unet.train()

        # CLIP 文本编码 (冻结, 固定 prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        with torch.no_grad():
            ids = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                 padding="max_length", truncation=True, return_tensors="pt").input_ids
            self.register_buffer("prompt_embeds", self.text_encoder(ids)[0])  # [1,77,1024]

        # 调度器 (仅用 alphas_cumprod)
        self.scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.t = t
        self.register_buffer("alpha_t", self.scheduler.alphas_cumprod[t].clone())

    @torch.no_grad()
    def encode(self, img):
        return self.vae.encode(img).latent_dist.sample() * self.scaling

    def decode(self, lat):
        return self.vae.decode(lat / self.scaling).sample.clamp(-1, 1)

    def forward(self, lq, return_x0=False):
        lq_lat = self.encode(lq)
        bs = lq.size(0)
        t = torch.full((bs,), self.t, device=lq.device, dtype=torch.long)
        pe = self.prompt_embeds.expand(bs, -1, -1)
        pred = self.unet(lq_lat, t, encoder_hidden_states=pe).sample
        a = self.alpha_t
        x0 = (lq_lat - (1.0 - a).sqrt() * pred) / a.sqrt()
        out = self.decode(x0)
        if return_x0:
            return out, x0, lq_lat
        return out

    @torch.no_grad()
    def infer(self, lq):
        self.eval()
        return self.forward(lq)

    def trainable_params(self):
        return [p for p in self.unet.parameters() if p.requires_grad]

    def param_count(self):
        return sum(p.numel() for p in self.trainable_params())
