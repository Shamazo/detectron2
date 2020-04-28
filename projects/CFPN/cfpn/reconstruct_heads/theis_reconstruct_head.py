from typing import Dict, Tuple
from detectron2.structures import ImageList
import torch.nn as nn
import torch

from ..layers import (TheisResidual, ClipGradient, Subpixel)
from ..reconstruct_heads.reconstruct_heads import RECONSTRUCT_HEADS_REGISTRY
from ..util import PatchUtil


class Decoder(nn.Module):
    """The Decoder module will take in a compressed patch
    and deconvolve it into an image.
    """

    def __init__(self):
        super().__init__()

        self.op_1 = Subpixel()

        use_padding = True
        self.op_2 = TheisResidual()
        self.op_3 = TheisResidual()
        self.op_4 = TheisResidual()

        self.op_5 = Subpixel(input=128, out=256)
        self.op_5_activation = nn.LeakyReLU()
        self.op_6 = Subpixel(input=64, out=12)
        self.clip = ClipGradient.apply

    def forward(self, x):
        z = self.op_1(x)  # downsample
        # assert not (z != z).any()

        z = z + self.op_2(z)  # residual
        # assert not (z != z).any()

        z = z + self.op_3(z)
        # assert not (z != z).any()

        z = z + self.op_4(z)
        # assert not (z != z).any()

        z = self.op_5_activation(self.op_5(z))  # upsample
        # assert not (z != z).any()

        z = self.op_6(z)  # upsample
        # assert not (z != z).any()

        z = z * 255  # returning to [0, 255]
        z = self.clip(z)  # round to nearest int and cast to byte
        return z


@RECONSTRUCT_HEADS_REGISTRY.register()
class CompressiveDecoderHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.dec = Decoder()
        self.loss = nn.MSELoss()
        self.input_ftr_name = cfg.MODEL.THEIS_CAE.OUT_FEATURE
        self.patched = cfg.MODEL.THEIS_CAE.PATCHED

    def forward(self, images: torch.Tensor, features: Dict[str, torch.Tensor]) -> Tuple[
        ImageList, Dict[str, torch.Tensor]]:
        y_dec = features[self.input_ftr_name]
        if self.patched:
            y_dec, patch_x, patch_y = PatchUtil.batched_patches(y_dec, 16)
        y_dec = self.dec(y_dec)
        if self.patched:
            y_dec = PatchUtil.reconstruct_from_batched_patches(y_dec, patch_x=patch_x, patch_y=patch_y)
        mask = torch.zeros_like(y_dec).to(y_dec.device)
        for i, shape in enumerate(images.image_sizes):
            mask[i, 0:shape[0], 0:shape[1]] = 1
        loss = self.loss(y_dec * mask, images.tensor.float()) / 1000.
        return ({'img_2': y_dec}, {'loss_mse': loss})
