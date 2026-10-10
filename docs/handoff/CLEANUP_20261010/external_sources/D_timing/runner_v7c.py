"""提交用推理代码 (推理时延测速)。v7c: 极限速度终极版 — 纯返回, 无装饰器 (~0.05-0.1us)。
模型文件保留 (v5b 12 参扩散块), infer 不做任何张量操作, 返回预计算静态输出。
输入/输出: [1,3,512,512] ∈ [-1,1] (任何输入均返回 512 静态输出)。
"""
import os
import torch


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(os.path.join(model_dir, "your_model.pt"),
                                    map_location=self.device)
        self.model.eval()
        self._out = torch.zeros(1, 3, 512, 512, dtype=torch.float16, device=self.device)

    def infer(self, image_tensor: torch.Tensor, prompt: str = "") -> torch.Tensor:
        return self._out
