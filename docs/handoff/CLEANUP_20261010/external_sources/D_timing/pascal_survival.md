# Pascal-Era GPU Survival Guide

**Keep your GTX 9xx / 10xx (Maxwell & Pascal) and Tesla M/P GPUs doing real AI and video work in 2026 — after CUDA 13, cuDNN 9.11 and TensorRT 10 dropped support for them.**

Your card is not dead. The toolchain moved and left Pascal (compute capability 6.1) behind. This repo is the frozen, tested version matrix that still works — so one `pip install` or driver update does not break a working machine.

## The problem you already have

You own a GTX 1060 / 1070 / 1080 Ti / 970 / Tesla P100. It still works fine for plenty of AI and video work. Then you ran one `pip install` or one driver update and hit:

- `RuntimeError: CUDA error: no kernel image is available for execution`
- `CUDNN_STATUS_ARCH_MISMATCH`
- `nvcc` no longer sees your card
- TensorRT refusing your SM 6.1 device

CUDA 13, cuDNN 9.11 and TensorRT 10 all dropped Pascal. If you blindly take "the latest", you break a working setup.

## The frozen version matrix that still works (2026)

| Component | Use this | Do NOT upgrade past | Why |
|---|---|---|---|
| Driver | 566.36 (Production Branch) | — | Last branch keeping Pascal current on Win11 |
| CUDA Toolkit | 12.4 - 12.9 | 13.0 | 13.x removes Pascal offline compile |
| cuDNN | 9.10.x (or 8.x legacy) | 9.11 | 9.11 dropped Maxwell / Pascal / Volta |
| Inference | ONNX Runtime CUDA / PyTorch native | TensorRT 10.x | TRT 10 requires SM 7.5+; Pascal is 6.1 |
| ffmpeg / NVENC | ffmpeg 7.0.2 portable | ffmpeg 8.x | ffnvcodec 13 in ffmpeg 8 needs driver >= 570 |
| PyTorch | 2.x cu124 wheel | cu128 / cu129 | cu128+ wheels dropped Pascal kernels |
| CuPy | cupy-cuda12x | — | Matches the CUDA 12.x runtime |

Pin these and a used Pascal card keeps doing real work.

## 3 traps that cost people an evening each

**Trap 1 — "Just upgrade CUDA to the latest."**
You install CUDA 13.x. Pascal offline compile is gone; `nvcc` builds fail or produce no Pascal kernel. Fix: stay on the 12.4-12.9 line; keep one 12.x toolkit and set `CUDA_PATH` explicitly.

**Trap 2 — "Use TensorRT, it's the fast path."**
TensorRT 10.x rejects SM 6.1. Fix: use ONNX Runtime CUDA — it runs on Pascal today. Export your model to ONNX and run it with the CUDA execution provider.

**Trap 3 — ffmpeg NVENC silently stops working after an ffmpeg update.**
ffmpeg 8's ffnvcodec 13 needs driver >= 570; Pascal tops out below that. Fix: use the ffmpeg 7.0.2 portable build (NVENC SDK 12.x), and add `-bf 0` — Pascal has no B-frames, and without it HEVC NVENC can silently misbehave.

## One benchmark — measured and reproducible

Measured on a GTX 1060 6GB (driver 566.36 / CUDA 12.4 / torch 2.6.0+cu124):

| Workload | GPU | CPU | Speedup |
|---|---|---|---|
| PyTorch 4096x4096 fp32 matmul | 38.8 ms | 305.5 ms | ~7.9x |
| CuPy 100M-element sqrt*2+1 | 15.8 ms | 330.6 ms | ~21x |

Don't trust the numbers — reproduce them on your own card:

```bash
python benchmark/bench.py
```

## Quick start

```bash
# Linux
bash setup/pin-pascal-stack.sh

# Windows — PowerShell, inside an activated Python 3.11 venv
.\setup\pin-pascal-stack.ps1
```

Both scripts pin the Pascal-safe stack and then hard-verify that real GPU kernels execute (they fail loudly if a cu128+ wheel slipped in). They never touch your driver.

## Who this is for

- You run a Pascal / Maxwell / Tesla M-P card and would rather not buy a new GPU to do what your current card still does
- You maintain a fleet of older boxes
- You just want the version-hell solved once

**Not for you** if you are on RTX 30/40/50 — you do not need any of this, go use the latest.

## Full guide

This repo is the free core: the matrix, the traps, one benchmark, the setup scripts.

The complete guide goes further — exact Windows + Linux install commands, the cuDNN archive recipe, a full TensorRT-to-ONNX migration walkthrough, the ffmpeg portable setup, 5 real-world use cases (video encode, image batch, small VLM, GPU text scan, FAISS-GPU), full benchmark tables, and a troubleshooting appendix:

- **Free 8-page sample (PDF):** https://1219763523552.gumroad.com/l/vdqymr
- **Full guide:** https://1219763523552.gumroad.com/l/xvvrp

## License

The scripts in `setup/` and `benchmark/` are released under the Apache License 2.0 — see [LICENSE](LICENSE). The version-matrix research is provided as-is; verify each version against current vendor release notes before relying on it in production.

