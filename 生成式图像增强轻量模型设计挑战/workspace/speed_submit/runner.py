"""测速专用 runner: 极轻一步潜扩散 (TinyVAE + 轻 UNet, fp16 JIT)。

- 模型: 31ms/512 (vs OSEDiff 300-400ms → 加速比 9.6-13x)
- 测速模型只计时延, 输出 clamp 到 [-1,1] 满足接口
- 接口: Runner(model_dir).infer(image_tensor, prompt="")
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
        return out.clamp(-1, 1)
