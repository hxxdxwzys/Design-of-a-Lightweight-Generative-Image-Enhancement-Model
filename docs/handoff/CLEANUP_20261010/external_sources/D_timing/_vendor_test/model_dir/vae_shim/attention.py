from __future__ import annotations
# vae_shim.attention
# Minimal self-attention extracted from diffusers.models.attention_processor (diffusers 0.39.0).
#
# Kept from the official `Attention` class:
#   * the parameter layout used by the VAE mid-block attention checkpoint keys:
#     group_norm.*, to_q.*, to_k.*, to_v.*, to_out.0.*,
#   * the exact default forward math (equivalent to the official default
#     AttnProcessor2_0 path):  GroupNorm(query channels) -> linear q/k/v -> reshape heads
#     -> F.scaled_dot_product_attention -> to_out[0] (+dropout) -> residual add
#     -> / rescale_output_factor.
#
# Trimmed away: cross-attention (norm_cross / encoder_hidden_states), spatial norm,
# qk norms, added-KV projections, fused projections, upcast_softmax re-implementations
# (the official SDPA path never used it), all other processors (xformers / sliced / ...).
#
# Constructor accepts the same keyword arguments the official VAE blocks pass, so the
# build sites of Encoder/Decoder/Mid are unchanged in spirit.
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_dim: int | None = None,
        cross_attention_norm: str | None = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: str | None = None,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        norm_num_groups: int | None = None,
        spatial_norm_dim: int | None = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor=None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()

        if cross_attention_dim is not None and cross_attention_dim != query_dim:
            raise NotImplementedError(
                "vae_shim.Attention only implements self-attention (cross_attention_dim "
                "must be None or equal to query_dim)."
            )
        if cross_attention_norm is not None:
            raise NotImplementedError("vae_shim.Attention does not implement cross_attention_norm.")
        if qk_norm is not None:
            raise NotImplementedError("vae_shim.Attention does not implement qk_norm.")
        if added_kv_proj_dim is not None or only_cross_attention:
            raise NotImplementedError("vae_shim.Attention does not implement added-KV projections.")
        if spatial_norm_dim is not None:
            raise NotImplementedError("vae_shim.Attention does not implement spatial norm.")

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = False
        self.cross_attention_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = None
        self.only_cross_attention = False

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None
        self.spatial_norm = None

        self.norm_q = None
        self.norm_k = None
        self.norm_cross = None

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        self.add_q_proj = None
        self.add_k_proj = None
        self.add_v_proj = None

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))
        else:
            self.to_out = None
        self.to_add_out = None
        self.norm_added_q = None
        self.norm_added_k = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """Self-attention with the exact math of the official default (AttnProcessor2_0) path."""

        if encoder_hidden_states is not None:
            raise NotImplementedError("vae_shim.Attention only implements self-attention.")
        if attention_mask is not None:
            raise NotImplementedError("vae_shim.Attention does not implement attention masks.")

        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states
