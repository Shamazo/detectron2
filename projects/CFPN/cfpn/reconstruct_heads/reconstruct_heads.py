import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.layers import ShapeSpec, Conv2d
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
            Image (Tensor): (n, c, h, w)
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
    This uses the lowest level input features
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
            dict[str->Tensor] Image (Tensor): (n, c, h, w)
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
        return({'img_2': x}, {"MSE": loss})


class SubPixelReconstruct(nn.Module):
    def __init__(self, in_channels):
        super(SubPixelReconstruct, self).__init__()
        self.op1 = Subpixel(in_channels=in_channels, out_channels=256, scale=2)
        self.op2 = Subpixel(in_channels=64, out_channels=12, scale=2)
        self.clip = ClipGradient()

    def forward(self, x: torch.tensor):
        """

        Args:
            x: tensor, nchw

        Returns: tensor, NOT clipped to 0-255

        """
        x = self.op1(x)
        x = self.op2(x)
        return x

@RECONSTRUCT_HEADS_REGISTRY.register()
class MLSPHead(ReconstructHeads):
    """
    Multi level reconstruction head using the features from multiple levels of the FPN backbone
    """
    def __init__(self, cfg, input_shape, fuse_type='sum'):
        super().__init__(cfg, input_shape)
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type
        self.in_features = cfg.MODEL.RECONSTRUCT_HEADS.IN_FEATURES

        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.loss = nn.MSELoss()
        self.clip = ClipGradient()

        reconstruct_convs = []
        output_convs = []
        stages = []
        for idx, key in enumerate(self.in_features):
            # output_conv = Conv2d(
            #     3,
            #     3,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1
            # )
            reconstruct_conv = SubPixelReconstruct(self.in_channels)

            # weight_init.c2_xavier_fill(reconstruct_conv)
            # weight_init.c2_xavier_fill(output_conv)

            reconstruct_convs.append(reconstruct_conv)
            # output_convs.append(output_conv)
            stage = int(key[1])
            stages.append(stage)
            self.add_module("MLSP_recon{}".format(stage), reconstruct_conv)
            # self.add_module("MLSP_output{}".format(stage), output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.stages = stages[::-1]
        self.reconstruct_convs = reconstruct_convs[::-1]
        # self.output_convs = output_convs[::-1]

    def forward(
        self,
        images: ImageList,
        top_down_features: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, ImageList], Dict[str, torch.Tensor]]:
        """
        Args:
            images (Tensor): (n, c, h, w)
            top_down_features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).

        Returns:
            dict[str->tensor] Image (Tensor): mapping output images to tensor(n, c, h, w)
            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        x = [top_down_features[f] for f in self.in_features[::-1]]
        results = {}
        losses = {}
        mask = torch.zeros_like(images.tensor, device=images.device)
        for i, shape in enumerate(images.image_sizes):
            mask[i, 0:shape[0], 0:shape[1]] = 1
        # loss = self.loss(x * mask, images.tensor.float())

        prev_features = self.reconstruct_convs[0](x[0])
        results["img_{}".format(self.stages[0])] = self.clip.apply(prev_features)

        for features, reconstruct_conv, stage in zip(
                x[1:], self.reconstruct_convs[1:], self.stages[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = reconstruct_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            reconstructed_images = self.clip.apply(prev_features)
            with torch.no_grad():
                _, _, height, width = reconstructed_images.shape
                ds_images = F.interpolate(images.tensor.float(), size=(height, width))
                # not sure if treating mask as a float is problematic, but interpolate only operates on floats
                ds_mask = F.interpolate(mask.float(), size=(height, width))

            losses["img_{}_MSE".format(stage)] = self.loss(reconstructed_images * ds_mask, ds_images)
            results["img_{}".format(stage)] = reconstructed_images

        return results, losses


