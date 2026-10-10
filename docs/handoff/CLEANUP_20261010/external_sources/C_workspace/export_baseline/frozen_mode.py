# -*- coding: utf-8 -*-
"""DELIV-01 冻结执行模式：所有实际 GPU 子进程必须自己设置并记录（不假设继承父进程 torch 开关）。"""
import os


ENV_KEY = "CUBLAS_WORKSPACE_CONFIG"
ENV_VALUE = ":4096:8"


def prepare_env():
    """必须在 CUDA 初始化（首次使用 CUDA）之前调用。"""
    os.environ[ENV_KEY] = ENV_VALUE
    return {ENV_KEY: os.environ.get(ENV_KEY)}


def apply_frozen_mode():
    """设置进程内 torch 开关；返回可记录的开关字典。"""
    import torch
    prepare_env()
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return read_flags()


def read_flags():
    import torch
    return {"CUBLAS_WORKSPACE_CONFIG": os.environ.get(ENV_KEY),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch": torch.__version__}


def flags_match(a, b, keys=None):
    """比较两侧开关是否一致（默认比较全部开关，torch 版本只作信息）。"""
    keys = keys or ["CUBLAS_WORKSPACE_CONFIG", "deterministic_algorithms", "cudnn_benchmark",
                    "cudnn_deterministic", "cudnn_allow_tf32", "matmul_allow_tf32"]
    diff = {k: {"ref": a.get(k), "cur": b.get(k)} for k in keys if a.get(k) != b.get(k)}
    return (not diff), diff


def assert_invalid_protocol(cond, detail):
    """协议条件不满足时抛 INVALID_PROTOCOL（不得据此判模型失败或成功）。"""
    if not cond:
        raise RuntimeError("INVALID_PROTOCOL: %s" % detail)
