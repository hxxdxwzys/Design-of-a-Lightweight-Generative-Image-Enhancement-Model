from __future__ import annotations
# vae_shim.autoencoder_kl
# Standalone AutoencoderKL extracted from diffusers.models.autoencoders.autoencoder_kl
# (diffusers 0.39.0).
#
# Purpose: forward/encode/decode-compatible replacement for the official AutoencoderKL
# that can be built with `AutoencoderKL(**config_kwargs)` and filled with
# `load_state_dict(merged_state_dict, strict=True)`. No ModelMixin / ConfigMixin /
# from_pretrained machinery: the runner builds the class from the same hyperparameters
# (block_out_channels etc.) and loads the merged weights directly.
#
# Module tree mirrors the official class:
#   encoder (Encoder) / decoder (Decoder) / quant_conv / post_quant_conv
# => state_dict keys encoder.*, decoder.*, quant_conv.*, post_quant_conv.* identical
#    to the official diffusers SD2.1 VAE.
#
# Trimmed away: tiling/slicing, gradient-checkpointing hooks, attention-processor
# plumbing (set_processor/fuse_qkv...), apply_forward_hook decorators.
import torch
import torch.nn as nn

from .vae import Decoder, DiagonalGaussianDistribution, Encoder


class AutoencoderKLOutput:
    """Lightweight stand-in for diffusers.models.modeling_outputs.AutoencoderKLOutput."""

    def __init__(self, latent_dist: DiagonalGaussianDistribution):
        self.latent_dist = latent_dist

    def __getitem__(self, index):
        if index == 0:
            return self.latent_dist
        raise IndexError(index)


class DecoderOutput:
    """Lightweight stand-in for diffusers.models.vae.DecoderOutput."""

    def __init__(self, sample: torch.Tensor = None, commit_loss: torch.Tensor | None = None):
        self.sample = sample
        self.commit_loss = commit_loss

    def __getitem__(self, index):
        if index == 0:
            return self.sample
        if index == 1:
            return self.commit_loss
        raise IndexError(index)


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: float | None = None,
        latents_mean=None,
        latents_std=None,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()

        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

    # ------------------------------------------------------------------ encode
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)
        return enc

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    # ------------------------------------------------------------------ decode
    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def decode(self, z: torch.Tensor, return_dict: bool = True, generator=None):
        decoded = self._decode(z).sample
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator=None,
    ):
        """Full VAE round trip: encode -> (sample posterior or mode) -> decode."""
        posterior = self.encode(sample).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
