from __future__ import annotations
"""fairscale.nn.model_parallel.layers shim (单卡, 别名 nn.Linear/nn.Embedding)"""
import torch.nn as nn

ColumnParallelLinear = nn.Linear
RowParallelLinear = nn.Linear
ParallelEmbedding = nn.Embedding
