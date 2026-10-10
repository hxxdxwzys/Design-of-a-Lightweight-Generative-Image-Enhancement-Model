from __future__ import annotations
"""fairscale shim: 单卡模式 (推理), rmsnorm.py 原样可导入"""
import torch
import torch.nn as nn


class _FSInit:
    @staticmethod
    def get_model_parallel_world_size():
        return 1

    @staticmethod
    def get_model_parallel_rank():
        return 0

    @staticmethod
    def get_data_parallel_world_size():
        return 1


fs_init = _FSInit()


class ColumnParallelLinear(nn.Linear):
    pass


class RowParallelLinear(nn.Linear):
    pass


class ParallelEmbedding(nn.Embedding):
    pass
