import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
RECONSTRUCT_HEADS_REGISTRY = Registry("RECONSTRUCT_HEADS")
RECONSTRUCT_HEADS_REGISTRY.__doc__ = """
Registry for reconstruction heads for use with FPN backbones.
ReconstructHeads take in a feature map and reconstruct the image input
to the original network.

The registered object will be called with `obj(cfg, input_shape, output_shape)`.
The call is expected to return an :class:`ReconstructHeads`.
"""

logger = logging.getLogger(__name__)


def build_reconstruct_heads(cfg, input_shape):
    """
    Build ReconstructHeads defined by `cfg.MODEL.RECONSTRUCT_HEADS.NAME`.
    """
    name = cfg.MODEL.RECONSTRUCT_HEADS.NAME
    return RECONSTRUCT_HEADS_REGISTRY.get(name)(cfg, input_shape)


class ReconstructHeads(torch.nn.Module):
    """
    ReconstructHeads reconstruct the input image of the backbone network.
    """
    def __init__(self, cfg, input_shape):
        super(ReconstructHeads, self).__init__()
        self.in_channels = cfg.MODEL.RECONSTRUCT_HEADS.IN_CHANNELS
        self.in_features = cfg.MODEL.RECONSTRUCT_HEADS.IN_FEATURES
        self.device = torch.device(cfg.MODEL.DEVICE)


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[ImageList, Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).

        Returns:
            ImageList: batch size image list of reconstructed images
            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()

def Subpixel(in_channels=256, out_channels=512, scale=2) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 3, stride=1, padding=1),
        nn.modules.PixelShuffle(scale)
    )

class ClipGradient(torch.autograd.Function):
    """
    Clips the output to [0, 255] and casts it to an integer
    """

    @staticmethod
    def forward(ctx, input):
        return torch.clamp(input, 0.0, 255.0).round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

@RECONSTRUCT_HEADS_REGISTRY.register()
class SPHead(ReconstructHeads):
    """
    Basic reconstruction head using two Subpixel convolutions
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.op1 = Subpixel(in_channels=self.in_channels, out_channels=256, scale=2)
        self.op2 = Subpixel(in_channels=64, out_channels=12, scale=2)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.loss = nn.MSELoss()
        self.clip = ClipGradient()
    def forward(
        self,
        images: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[ImageList, Dict[str, torch.Tensor]]:
        """
        Args:
            images (Tensor): (n, c, h, w)
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).

        Returns:
            Image (Tensor): (n, c, h, w)
            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        x = self.op1(features["p2"])
        x = self.op2(x)
        # x = (x * self.pixel_std) + self.pixel_mean  # returning to [0, 255]
        x = self.clip.apply(x)  # round to nearest int
        mask = torch.zeros_like(x).to(x.device)
        npx = 0
        for i, shape in enumerate(images.image_sizes):
            mask[i, 0:shape[0], 0:shape[1]] = 1
            npx += shape[0] * shape[1]
        #flattening is probably bas
        loss = self.loss(x * mask, images.tensor.float())
        # diff2 = ((x) - (images.tensor.to("cuda:0"))) ** 2.0 * mask
        # print(diff2)
        # loss = torch.sum(diff2) / npx
        # print(loss)
        return(x, {"MSE": loss})


