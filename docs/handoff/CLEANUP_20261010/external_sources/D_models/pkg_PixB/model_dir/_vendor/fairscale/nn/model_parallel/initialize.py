from __future__ import annotations
"""fairscale.nn.model_parallel.initialize shim (单卡推理)"""


def get_model_parallel_world_size():
    return 1


def get_model_parallel_rank():
    return 0


def get_data_parallel_world_size():
    return 1


def get_data_parallel_rank():
    return 0
