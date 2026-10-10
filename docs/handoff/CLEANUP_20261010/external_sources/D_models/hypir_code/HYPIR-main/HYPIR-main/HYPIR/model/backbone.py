import os
import torch
from torch import nn
import open_clip
from open_clip.factory import CLIP


def _visual_forward(
    model: CLIP,
    image: torch.Tensor,
    return_feats: bool = False,
    return_pooled_feats: bool = False,
):
    # stem, stages, norm_pre
    x, intermediates = model.visual.trunk.forward_intermediates(
        image,
        indices=None,
        norm=False,  # useless
        stop_early=False,
        intermediates_only=False,
    )
    # timm 版本兼容: 有些版本 intermediates[0] 是输入 x, 有些直接是 stage1
    # (若 [0] 是 x 则丢它; 否则保留全部 stage 特征)
    if len(intermediates) and intermediates[0].shape[1] == 3:
        feats = intermediates[1:]
    else:
        feats = list(intermediates)
    if return_feats:
        return feats
    # trunk.head
    x = model.visual.trunk.forward_head(x)
    # visual.head
    x = model.visual.head(x)
    if return_pooled_feats:
        feats[-1] = x
        return feats
    return x


class ImageOpenCLIPConvNext(nn.Module):

    def __init__(self, precision="fp32"):
        super().__init__()
        # 本地权重 (hf-mirror 下载), 避免联网; 也可用环境变量覆盖
        ckpt = os.environ.get(
            "HYPIR_OPENCLIP_CKPT",
            r"D:/models/open_clip/convnext_xxlarge-laion2b_s34b_b82k_augreg_soup.bin",
        )
        self.model, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge",
            pretrained=ckpt if os.path.isfile(ckpt) else "laion2b_s34b_b82k_augreg_soup",
            precision=precision,
        )

    def encode_image(self, image, return_feats=False, return_pooled_feats=False):
        return _visual_forward(
            self.model,
            image,
            return_feats,
            return_pooled_feats,
        )
