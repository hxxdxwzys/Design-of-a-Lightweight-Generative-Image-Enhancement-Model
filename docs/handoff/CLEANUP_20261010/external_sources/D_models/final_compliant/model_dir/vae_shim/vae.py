from __future__ import annotations
# vae_shim.vae
# Standalone Encoder / Decoder / DiagonalGaussianDistribution extracted from
# diffusers.models.autoencoders.vae (diffusers 0.39.0).
#
# Encoder  : conv_in -> 4x DownEncoderBlock2D -> UNetMidBlock2D(+attention)
#            -> GroupNorm(1e-6) -> SiLU -> conv_out(double_z)
# Decoder  : conv_in -> UNetMidBlock2D(+attention) -> 4x UpDecoderBlock2D
#            -> GroupNorm(1e-6) -> SiLU -> conv_out
#
# The block *factories* (get_down_block / get_up_block) are replaced by direct
# construction of the (shim) block classes with the exact arguments the factories
# would have forwarded for the "DownEncoderBlock2D"/"UpDecoderBlock2D" types, so the
# module trees - and therefore the state_dict keys - are identical to the diffusers
# originals (e.g. encoder.down_blocks.<i>.downsamplers.<j>.conv.weight).
#
# Trimmed: gradient checkpointing, tiny/asymmetric/temporal variants, spatial norm.
import torch
import torch.nn as nn

from .unet_2d_blocks import DownEncoderBlock2D, UNetMidBlock2D, UpDecoderBlock2D


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type != "DownEncoderBlock2D":
                raise NotImplementedError(f"vae_shim Encoder only supports DownEncoderBlock2D, got {down_block_type}")

            down_block = DownEncoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None
        if norm_type == "spatial":
            raise NotImplementedError("vae_shim Decoder only supports norm_type='group'.")

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            if up_block_type != "UpDecoderBlock2D":
                raise NotImplementedError(f"vae_shim Decoder only supports UpDecoderBlock2D, got {up_block_type}")

            up_block = UpDecoderBlock2D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temb_channels=None,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor, latent_embeds: torch.Tensor | None = None) -> torch.Tensor:
        sample = self.conv_in(sample)

        sample = self.mid_block(sample, latent_embeds)

        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class DiagonalGaussianDistribution:
    """Copied from diffusers.models.autoencoders.vae.DiagonalGaussianDistribution
    (mean/std/logvar split with logvar clamped to [-30, 20]); `sample` uses the same
    formula as diffusers' randn_tensor-backed sample(): mean + std * randn_like."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator=None) -> torch.Tensor:
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )

    def nll(self, sample: torch.Tensor, dims=(1, 2, 3)) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        import math

        logtwopi = math.log(2.0 * math.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class IdentityDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.parameters = parameters

    def sample(self, generator=None) -> torch.Tensor:
        return self.parameters

    def mode(self) -> torch.Tensor:
        return self.parameters
