from __future__ import annotations
# vae_shim.resnet
# Standalone ResnetBlock2D extracted from diffusers.models.resnet (diffusers 0.39.0),
# trimmed to the subsets used by the VAE Encoder/Decoder/Mid blocks and the LightDecoder.
#
# Trimmed away (the VAE path never uses them):
#   * FIR / SDE-VP in-block up/downsampling (up/down=True),
#   * ResnetBlockCondNorm2D / "ada_group" / "spatial" time-embedding norms,
#   * conv_shortcut=True flag (shortcuts are driven by use_in_shortcut, as in diffusers).
#
# Math kept bit-for-bit identical to diffusers 0.39 forward():
#   norm1 -> silu -> conv1 -> (+temb, unused here) -> norm2 -> silu -> dropout -> conv2
#   -> (1x1 conv shortcut when in_channels != conv_2d_out_channels) -> (x + h) / output_scale_factor
# with GroupNorm eps=1e-6, nn.SiLU activation.
import torch
import torch.nn as nn

from .activations import get_activation


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: int | None = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # "default", "scale_shift" (VAE uses default/group only)
        kernel: torch.Tensor | None = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: bool | None = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: int | None = None,
    ):
        super().__init__()
        if time_embedding_norm == "ada_group":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==ada_group`, please use "
                "ResnetBlockCondNorm2D instead (not part of the vae_shim subset)."
            )
        if time_embedding_norm == "spatial":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==spatial`, please use "
                "ResnetBlockCondNorm2D instead (not part of the vae_shim subset)."
            )
        if up or down:
            raise NotImplementedError(
                "vae_shim.ResnetBlock2D does not support in-block up/downsampling (VAE blocks "
                "build with up=False, down=False and use dedicated Upsample2D/Downsample2D modules)."
            )

        # diffusers 0.39 hard-codes this to True (forward no longer branches on it).
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError("`temb` should not be None when `time_embedding_norm` is 'scale_shift'")
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            # time_embedding_norm == "group" (used by the decoder up-blocks with norm_type="group"):
            # identical to "default" with temb=None.
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor
