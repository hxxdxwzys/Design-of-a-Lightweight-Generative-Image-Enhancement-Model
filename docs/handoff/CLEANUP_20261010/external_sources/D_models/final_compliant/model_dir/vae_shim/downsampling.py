from __future__ import annotations
# vae_shim.downsampling
# Standalone Downsample2D extracted from diffusers.models.downsampling (diffusers 0.39.0),
# trimmed to the code paths used by the VAE Encoder (conv downsampling, optional right/bottom
# zero padding when padding==0). LayerNorm / RMSNorm variants and FIR downsampling are dropped.
import torch.nn as nn
import torch.nn.functional as F


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution (stride 2).

    Matches diffusers 0.39 ``Downsample2D`` module/parameter layout so that
    ``state_dict`` keys (``downsamplers.<i>.conv.weight`` for ``name="op"``) are
    identical to the official SD2.1 VAE checkpoint.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int | None = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        if norm_type is not None:
            raise NotImplementedError(
                "vae_shim.Downsample2D only supports the plain (norm_type=None) VAE path."
            )
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        self.name = name
        self.norm = None
        stride = 2

        if use_conv:
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # Mirror diffusers' naming logic exactly (VAE encoder passes name="op",
        # which registers the conv under the plain name "conv").
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        # diffusers 0.39 always routes the forward through self.conv (every
        # registration path above also stores the conv under the name "conv").
        hidden_states = self.conv(hidden_states)

        return hidden_states
