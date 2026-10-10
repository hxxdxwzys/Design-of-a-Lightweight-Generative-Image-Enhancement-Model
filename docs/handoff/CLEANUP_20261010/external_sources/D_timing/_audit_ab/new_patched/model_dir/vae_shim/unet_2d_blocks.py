from __future__ import annotations
# vae_shim.unet_2d_blocks
# Standalone 2D blocks extracted from diffusers.models.unets.unet_2d_blocks (diffusers 0.39.0)
# restricted to the three blocks the VAE family needs:
#   * DownEncoderBlock2D  - VAE Encoder down blocks,
#   * UpDecoderBlock2D   - VAE / LightDecoder up blocks,
#   * UNetMidBlock2D     - VAE / LightDecoder mid block (resnets + optional attention).
#
# Trimmed away: every cross-attention block variant, spatial (ResnetBlockCondNorm2D) paths,
# temb/time-embedding conditioning (VAE blocks run with temb_channels=None), gradient
# checkpointing branches and the deprecated `scale` argument handling.
#
# Parameter/module registration is identical to the diffusers originals, so the produced
# state_dict keys (resnets.<i>.norm1/norm2/conv1/conv2/conv_shortcut, downsamplers.<i>.conv,
# upsamplers.<i>.conv, attentions.<i>.group_norm/to_q/to_k/to_v/to_out.0) exactly match the
# official checkpoints.
import torch
import torch.nn as nn

from .attention import Attention
from .downsampling import Downsample2D
from .resnet import ResnetBlock2D
from .upsampling import Upsample2D


def _make_resnet(
    *,
    in_channels: int,
    out_channels: int,
    temb_channels,
    resnet_eps: float,
    resnet_groups: int,
    dropout: float,
    resnet_time_scale_shift: str,
    resnet_act_fn: str,
    output_scale_factor: float,
    resnet_pre_norm: bool,
) -> ResnetBlock2D:
    if resnet_time_scale_shift == "spatial":
        raise NotImplementedError("vae_shim does not implement ResnetBlockCondNorm2D ('spatial') paths.")
    return ResnetBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        eps=resnet_eps,
        groups=resnet_groups,
        dropout=dropout,
        time_embedding_norm=resnet_time_scale_shift,
        non_linearity=resnet_act_fn,
        output_scale_factor=output_scale_factor,
        pre_norm=resnet_pre_norm,
    )


class DownEncoderBlock2D(nn.Module):
    """VAE encoder down block: resnets (+ 2x2 stride-2 conv downsample unless final)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                _make_resnet(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=None,
                    resnet_eps=resnet_eps,
                    resnet_groups=resnet_groups,
                    dropout=dropout,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    resnet_pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class UpDecoderBlock2D(nn.Module):
    """VAE / LightDecoder up block: resnets (+ nearest-2x upsample conv unless final)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: int | None = None,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                _make_resnet(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    resnet_eps=resnet_eps,
                    resnet_groups=resnet_groups,
                    dropout=dropout,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    resnet_pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class UNetMidBlock2D(nn.Module):
    """VAE / LightDecoder mid block: num_layers+1 resnets with optional attention after
    every resnet but the last (mirrors the diffusers ordering resnet -> attn -> resnet)."""

    def __init__(
        self,
        in_channels: int,
        temb_channels: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: int | None = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        if resnet_time_scale_shift == "spatial":
            raise NotImplementedError("vae_shim does not implement ResnetBlockCondNorm2D ('spatial') paths.")
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # there is always at least one resnet
        resnets = [
            _make_resnet(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                resnet_eps=resnet_eps,
                resnet_groups=resnet_groups,
                dropout=dropout,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_act_fn=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                resnet_pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                _make_resnet(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    resnet_eps=resnet_eps,
                    resnet_groups=resnet_groups,
                    dropout=dropout,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    resnet_pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states
