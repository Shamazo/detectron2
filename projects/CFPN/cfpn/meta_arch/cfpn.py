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

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY

from ..reconstruct_heads import build_reconstruct_heads

__all__ = ["CFPN"]


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

        reconstructed_images, reconstruction_losses = self.reconstruct_heads(images, features)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                 self.visualize_training(reconstructed_images, images)

        return reconstruction_losses

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



