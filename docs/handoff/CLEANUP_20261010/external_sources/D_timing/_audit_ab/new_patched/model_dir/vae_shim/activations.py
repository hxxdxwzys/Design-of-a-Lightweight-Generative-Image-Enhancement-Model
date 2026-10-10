from __future__ import annotations
# vae_shim.activations
# Standalone activation helper extracted from diffusers.models.activations.
# The VAE / LightDecoder code path only ever uses "silu" (alias "swish"), both map to nn.SiLU.
# No third-party dependency: only torch.nn.
import torch.nn as nn

_ACTIVATIONS = {"silu": nn.SiLU, "swish": nn.SiLU}


def get_activation(act_fn: str):
    """Return an nn.Module implementing the requested activation.

    diffusers' VAE encoder/decoder/mid blocks are built with ``act_fn="silu"`` /
    ``resnet_act_fn="silu"``. ``"swish"`` is accepted as an alias (diffusers maps it
    to SiLU as well) so that checkpoints configured with either string load.
    """
    if act_fn not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation {act_fn!r}. The standalone VAE subset only supports "
            f"'silu'/'swish' (got from diffusers get_activation which also supports many more)."
        )
    return _ACTIVATIONS[act_fn]()
