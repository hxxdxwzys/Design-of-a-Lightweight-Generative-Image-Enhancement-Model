"""提交用推理代码 (用于推理时延测速)。
加载 torch.jit 导出的 your_model.pt，暴露 Runner(model_dir).infer(image_tensor, prompt)。
- 模型: fp16 一步潜扩散 (SD2.1 VAE + 小型 UNet，显式 x0 预测，无 RAM/CLIP)
- 输入/输出: [1,3,H,W] ∈ [-1,1]
"""
import os
import torch


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.float16
        path = os.path.join(model_dir, "your_model.pt")
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, image_tensor: torch.Tensor, prompt: str = "") -> torch.Tensor:
        x = image_tensor.to(self.device, dtype=self.weight_dtype)
        out = self.model(x)
        return out
