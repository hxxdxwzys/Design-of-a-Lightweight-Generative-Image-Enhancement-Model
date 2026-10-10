# -*- coding: utf-8 -*-
"""速度实验: CUDA Graph 重放 vs eager 1.40ms 熔合模型。
模拟官方测速方法论: 10 次 warmup + 200 次计时 (与本地 204x 校准一致)。
输入: CPU fp32 tensor (模拟 harness 传 CPU 张量), runner 内部 .to(device, fp16)。
"""
import os
import time
import torch

MODEL_PATH = r"D:\models\final_204x\model_dir\your_model.pt"
BASELINE_MS = 287.0  # 官方 OSEDiff 基准


class GraphRunner:
    def __init__(self, model_dir):
        self.device = torch.device("cuda:0")
        self.model = torch.jit.load(model_dir, map_location=self.device)
        self.model.eval()
        self.static_in = torch.zeros(1, 3, 512, 512, dtype=torch.float16, device=self.device)
        self.static_out = None
        self.graph = None
        self.graph_ok = False
        self.capture_wall_ms = None
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.model(self.static_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = self.model(self.static_in)
            torch.cuda.synchronize()
            self.capture_wall_ms = (time.perf_counter() - t0) * 1000
            self.graph_ok = True
            print(f"[graph] capture ok, wall={self.capture_wall_ms:.1f}ms", flush=True)
        except Exception as e:
            print(f"[graph] capture failed: {e}", flush=True)
            self.graph_ok = False

    @torch.no_grad()
    def infer(self, image_tensor, prompt=""):
        if self.graph_ok:
            t = image_tensor.to(self.device, dtype=torch.float16)
            self.static_in.copy_(t)
            self.graph.replay()
            return self.static_out
        return self.model(image_tensor.to(self.device, dtype=torch.float16))


class EagerRunner:
    def __init__(self, model_dir):
        self.device = torch.device("cuda:0")
        self.model = torch.jit.load(model_dir, map_location=self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, image_tensor, prompt=""):
        return self.model(image_tensor.to(self.device, dtype=torch.float16))


def bench(runner, name, cpu_input=True):
    # 模拟官方: 输入 CPU fp32 (含显式 to(device,fp16))
    if cpu_input:
        x = torch.randn(1, 3, 512, 512, dtype=torch.float32).clamp(-1, 1)
    else:
        x = torch.randn(1, 3, 512, 512, dtype=torch.float16, device="cuda").clamp(-1, 1)
    with torch.no_grad():
        for _ in range(10):
            out = runner.infer(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 200
        for _ in range(N):
            out = runner.infer(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N * 1000
    ratio = BASELINE_MS / dt
    print(f"[{name}] {dt:.3f}ms -> {ratio:.0f}x   out={tuple(out.shape)} nan={torch.isnan(out.float()).sum().item() if torch.is_tensor(out) else '?'}", flush=True)
    return dt, ratio


if __name__ == "__main__":
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
    er = EagerRunner(MODEL_PATH)
    bench(er, "eager(CPU in)")
    bench(er, "eager(GPU in)")

    gr = GraphRunner(MODEL_PATH)
    bench(gr, "graph(CPU in)")
    bench(gr, "graph(GPU in)")
    # 首次调用 (无 warmup) 计时 — 若官方不 warmup
    x = torch.randn(1, 3, 512, 512, dtype=torch.float32).clamp(-1, 1)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gr.infer(x)
        torch.cuda.synchronize()
        dt1 = (time.perf_counter() - t0) * 1000
    print(f"[graph] first-call (no warmup): {dt1:.3f}ms -> {BASELINE_MS/dt1:.0f}x", flush=True)
