# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone, build_resnet_backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from ..reconstruct_heads import build_reconstruct_heads
from ..quantization import build_quantizer
from detectron2.modeling.backbone import FPN


__all__ = ["CFPN"]

@BACKBONE_REGISTRY.register()
class QFPN(FPN):
    """
    A quantized feature pyramid network, the lateral inputs are quantized
    """
    def __init__(self, cfg, input_shape):
        bottom_up = build_resnet_backbone(cfg, input_shape)
        in_features = cfg.MODEL.FPN.IN_FEATURES
        out_channels = cfg.MODEL.FPN.OUT_CHANNELS
        super().__init__(
            bottom_up=bottom_up,
            in_features=in_features,
            out_channels=out_channels,
            norm=cfg.MODEL.FPN.NORM,
            top_block=LastLevelMaxPool(),
            fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        )
        input_shapes = bottom_up.output_shape()
        self.quantizer = build_quantizer(cfg, input_shapes)

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        # quantize the bottom up features
        bottom_up_features, quantization_losses = self.quantizer(bottom_up_features)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results)), quantization_losses

#qfpn which subclasses fpn

#build resnet backbone but make it qfpn





@BACKBONE_REGISTRY.register()
class QFPN(FPN):
    """
    A quantized feature pyramid network, the lateral inputs are quantized
    """
    def __init__(self, cfg, input_shape):
        bottom_up = build_resnet_backbone(cfg, input_shape)
        in_features = cfg.MODEL.FPN.IN_FEATURES
        out_channels = cfg.MODEL.FPN.OUT_CHANNELS
        super().__init__(
            bottom_up=bottom_up,
            in_features=in_features,
            out_channels=out_channels,
            norm=cfg.MODEL.FPN.NORM,
            top_block=LastLevelMaxPool(),
            fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        )
        input_shapes = bottom_up.output_shape()
        self.quantizer = build_quantizer(cfg, input_shapes)

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        # quantize the bottom up features
        bottom_up_features, quantization_losses = self.quantizer(bottom_up_features)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results)), quantization_losses

#qfpn which subclasses fpn

#build resnet backbone but make it qfpn


@META_ARCH_REGISTRY.register()
class CFPN(nn.Module):
    """
    Compressive Feature Pyramid Network
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg).to(self.device)
        self.reconstruct_heads = build_reconstruct_heads(cfg, self.backbone.output_shape()).to(self.device)
        self.input_format = cfg.INPUT.FORMAT
        self.vis_period = cfg.VIS_PERIOD
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            dict:
                Contains the losses
        """
        if not self.training:
            return self.inference(batched_inputs)

        normed_images = self.preprocess_image(batched_inputs)
        images = self.preprocess_image(batched_inputs, norm=False)

        features = self.backbone(normed_images.tensor)
<<<<<<< HEAD:projects/CFPN/cfpn/meta_arch/cfpn.py
        if isinstance(features, tuple): # if using quantization the backbone returns features, losses
=======
        if isinstance(features, tuple): #if using quantization the backbone returns features, losses
>>>>>>> 35d962dcd22060d119a3a5b2381ceb6873e6ffca:projects/CFPN/cfpn/cfpn.py
            features, quantization_losses = features
        else:
            quantization_losses = {}

        reconstructed_images, reconstruction_losses = self.reconstruct_heads(images, features)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                 self.visualize_training(reconstructed_images, images)

        comb_losses = {**reconstruction_losses, **quantization_losses}
        return comb_losses

    def inference(self, batched_inputs):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
        Returns:
            Dict[str->tensor] mapping output image names to the tensor in n,c,h,w format
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor.float())
<<<<<<< HEAD:projects/CFPN/cfpn/meta_arch/cfpn.py
        if isinstance(features, tuple):  # if using quantization the backbone returns features, losses
=======
        if isinstance(features, tuple): #if using quantization the backbone returns features, losses
>>>>>>> 35d962dcd22060d119a3a5b2381ceb6873e6ffca:projects/CFPN/cfpn/cfpn.py
            features, _ = features

        reconstructed_images, loss_dict = self.reconstruct_heads(images, features)
        return reconstructed_images

    def visualize_training(self, reconstructed_images, images):
        storage = get_event_storage()
        orig_img = images.tensor[0].detach().cpu().numpy()
        if self.input_format == "BGR":
            orig_img = orig_img[::-1, :, :]
        for key in reconstructed_images:
            recon_img = reconstructed_images[key][0].detach().cpu().numpy()
            assert orig_img.shape[0] == 3, "Images should have 3 channels."
            assert recon_img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                recon_img = recon_img[::-1, :, :]
            _, height, width = recon_img.shape


            t_orig_img = orig_img.transpose(1, 2, 0).astype("uint8")
            ds_orig_img = cv2.resize(t_orig_img, dsize=(height, width), interpolation=cv2.INTER_NEAREST)
            recon_img = recon_img.transpose(1, 2, 0).astype("uint8")
            # print(ds_orig_img.shape, recon_img.shape)
            vis_img = np.concatenate((ds_orig_img, recon_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "{} ;Left: GT image;  Right: reconstructed image".format(key)
            storage.put_image(vis_name, vis_img) #takes in c,h,w images
        return

    def preprocess_image(self, batched_inputs, norm=True):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if norm:
            images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, 512)
        images = images.to(self.device)
        return images



