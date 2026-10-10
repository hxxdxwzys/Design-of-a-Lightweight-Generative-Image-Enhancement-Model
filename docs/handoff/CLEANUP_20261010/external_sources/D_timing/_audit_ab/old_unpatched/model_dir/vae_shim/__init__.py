from __future__ import annotations
# vae_shim
# Minimal, dependency-free (torch only) re-implementation of the parts of diffusers 0.39
# needed to run the SD2.1 AutoencoderKL VAE and the LightDecoder forward passes on the
# evaluation environment (which has no diffusers installed).
#
# Usage from the runner:
#     import vae_shim
#     vae = vae_shim.AutoencoderKL(
#         in_channels=3, out_channels=3,
#         down_block_types=("DownEncoderBlock2D",) * 4,
#         up_block_types=("UpDecoderBlock2D",) * 4,
#         block_out_channels=(128, 256, 512, 512),   # SD2.1 VAE
#         layers_per_block=2, latent_channels=4, norm_num_groups=32,
#     )
#     vae.load_state_dict(merged_sd21_vae_state_dict)   # keys: encoder.* decoder.* quant_conv.* post_quant_conv.*
#
#     decoder = vae_shim.Decoder(          # LightDecoder (halved decoder, decoder-only)
#         in_channels=4, out_channels=3,
#         up_block_types=("UpDecoderBlock2D",) * 4,
#         block_out_channels=(128, 128, 256, 256), layers_per_block=2,
#     )
#     decoder.load_state_dict(ld_state_dict)
#
# Note: the actual sd21_lwdecoder.pth config is {in_channels:4, out_channels:3,
# block_out_channels:[128,128,256,256], layers_per_block:2} (read from the file itself),
# i.e. the two top-resolution blocks are 128/128 (not 64 as sometimes quoted). The shim
# Decoder/UpDecoderBlock2D/UNetMidBlock2D are generic; pass whatever the weights use.

from .activations import get_activation  # noqa: F401
from .attention import Attention  # noqa: F401
from .autoencoder_kl import AutoencoderKL, AutoencoderKLOutput, DecoderOutput  # noqa: F401
from .downsampling import Downsample2D  # noqa: F401
from .resnet import ResnetBlock2D  # noqa: F401
from .unet_2d_blocks import DownEncoderBlock2D, UNetMidBlock2D, UpDecoderBlock2D  # noqa: F401
from .upsampling import Upsample2D  # noqa: F401
from .vae import Decoder, DiagonalGaussianDistribution, Encoder, IdentityDistribution  # noqa: F401

# Alias so runner code can read `vae_shim.LightDecoder` for the decoder-only model.
LightDecoder = Decoder

__all__ = [
    "AutoencoderKL",
    "AutoencoderKLOutput",
    "DecoderOutput",
    "Encoder",
    "Decoder",
    "LightDecoder",
    "DiagonalGaussianDistribution",
    "IdentityDistribution",
    "DownEncoderBlock2D",
    "UpDecoderBlock2D",
    "UNetMidBlock2D",
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    "Attention",
    "get_activation",
]
