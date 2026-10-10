"""提交用推理代码 (推理时延测速)。
轻量一步扩散: 熔合 1-block UNet (slice 下采样 -> conv/gn/silu x2 -> 显式 x0 去噪 -> nearest 上采样)。
CUDA Graph 捕获 + 静态输入缓冲重放, 砍掉 kernel 启动开销; 任何异常自动回退 eager。
输入/输出: [1,3,512,512] ∈ [-1,1] (动态尺寸亦支持)。
"""
import os
import torch


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(os.path.join(model_dir, "your_model.pt"),
                                    map_location=self.device)
        self.model.eval()
        self.graph = None
        self.static_in = None
        self.static_out = None
        if self.device.type == "cuda":
            try:
                self.static_in = torch.zeros(1, 3, 512, 512, dtype=torch.float16,
                                             device=self.device)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    with torch.no_grad():
                        for _ in range(3):
                            self.model(self.static_in)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.no_grad():
                    with torch.cuda.graph(self.graph):
                        self.static_out = self.model(self.static_in)
                torch.cuda.synchronize()
            except Exception:
                self.graph = None
                self.static_in = None

    @torch.no_grad()
    def infer(self, image_tensor: torch.Tensor, prompt: str = "") -> torch.Tensor:
        t = image_tensor.to(self.device, dtype=torch.float16)
        if self.graph is not None and t.shape == self.static_in.shape:
            self.static_in.copy_(t)
            self.graph.replay()
            return self.static_out
        return self.model(t)
