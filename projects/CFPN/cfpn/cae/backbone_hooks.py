import logging
from typing import Dict, Tuple
from detectron2.structures import ImageList
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

import torch.nn as nn
import torch

from .models import Encoder, Decoder
from ..reconstruct_heads.reconstruct_heads import RECONSTRUCT_HEADS_REGISTRY, build_reconstruct_heads


class PatchUtil:
    """Class functioning as a namespace for utils related
    to patching.
    """

    @staticmethod
    def batched_patches(images, patch_size):
        patches = images.unfold(-2, patch_size, patch_size)
        patches = patches.unfold(-2, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, -2, -1)
        _, patch_x, patch_y, _, _, _ = patches.shape
        return patches.permute(0, 2, 3, 1, -2, -1).flatten(0, 2), patch_x, patch_y

    @staticmethod
    def reconstruct_from_batched_patches(batched_out, patch_x=4, patch_y=4):
        batches, channels, x, y = batched_out.shape
        assert x == y
        bsz = batches / (patch_x * patch_y)
        out = batched_out.view(bsz, patch_x, patch_y, channels, x, y).permute(0, 3, 1, 2, -2, -1)
        return out.transpose(-3, -2).flatten(-2, -1).flatten(-3, -2)


@META_ARCH_REGISTRY.register()
class RCNNwithReconstruction(GeneralizedRCNN):
    def __init__(self, cfg):
        super(RCNNwithReconstruction, self).__init__(cfg)
        self.reconstruct_heads = build_reconstruct_heads(cfg, self.backbone.output_shape()).to(self.device)

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        _, reconstruction_losses = self.reconstruct_heads(images, features)

        losses = {}
        # losses.update(detector_losses)
        # losses.update(proposal_losses)
        losses.update(reconstruction_losses)
        return losses

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


@BACKBONE_REGISTRY.register()
class CompressiveEncoderBackbone(Backbone):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(CompressiveEncoderBackbone, self).__init__()
        assert input_shape.height == input_shape.width and "Width must be equal to height for Theis CAE."
        assert not input_shape.width or input_shape.width == 128 and "Either no width or the width is 128"
        self.name = cfg.MODEL.THEIS_CAE.OUT_FEATURE
        self.patched = cfg.MODEL.THEIS_CAE.PATCHED
        self.enc = Encoder()
        self._size_divisibility = 128  # this is needed to fix some image errors

    @property
    def size_divisibility(self):
        return self._size_divisibility


    def forward(self, image: torch.Tensor):
        if self.patched:
            image, patch_x, patch_y = PatchUtil.batched_patches(image, self._size_divisibility)
        out = self.enc(image)
        if self.patched:
            out = PatchUtil.reconstruct_from_batched_patches(out, patch_x=patch_x, patch_y=patch_y)
        return {self.name: out}

    def output_shape(self):
        return {self.name: ShapeSpec(stride=8, channels=96, height=16, width=16)}


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
        # print("features.shape {} {}".format(self.input_ftr_name, features[self.input_ftr_name].shape))
        # print("y_dec.shape {}".format(y_dec.shape))
        # print("mask.shape {}".format(mask.shape))
        # print("images.shape {}".format(images.tensor.shape))
        loss = self.loss(y_dec * mask, images.tensor.float())
        return ({'img_2': y_dec}, {'mse': loss})
