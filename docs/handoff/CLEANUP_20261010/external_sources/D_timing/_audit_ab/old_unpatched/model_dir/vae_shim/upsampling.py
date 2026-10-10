from __future__ import annotations
# vae_shim.upsampling
# Standalone Upsample2D extracted from diffusers.models.upsampling (diffusers 0.39.0),
# trimmed to the code paths used by the VAE Decoder / LightDecoder (nearest-2x
# interpolation followed by an optional 3x3 conv). ConvTranspose, norm and FIR variants dropped.
import torch.nn as nn
import torch.nn.functional as F


class Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Matches diffusers 0.39 ``Upsample2D`` module/parameter layout so that
    ``state_dict`` keys (``upsamplers.<i>.conv.weight``) are identical to the
    official checkpoints.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: int | None = None,
        name: str = "conv",
        kernel_size: int | None = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        if norm_type is not None:
            raise NotImplementedError("vae_shim.Upsample2D only supports the plain VAE path.")
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        self.norm = None

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        # Mirror diffusers' naming logic (VAE decoder upsamplers use name="conv").
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size: int | None = None):
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Nearest-neighbour 2x upsampling (float32/bfloat16-safe on every torch >= 2.1).
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states
