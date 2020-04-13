import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from torch.distributions import Normal, Uniform
from detectron2.layers import ShapeSpec, Conv2d
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
QUANTIZER_REGISTRY = Registry("QUANTIZERS")
QUANTIZER_REGISTRY.__doc__ = """
Registry for quantizers. These are for use in compressive networks.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`Quantizer`. The output shape
is assumed to be the same as the input shape, but values are quantized.
"""

logger = logging.getLogger(__name__)


def build_quantizer(cfg, input_shape):
    """
    Build ReconstructHeads defined by `cfg.MODEL.RECONSTRUCT_HEADS.NAME`.
    """
    name = cfg.MODEL.QUANTIZER.NAME
    return QUANTIZER_REGISTRY.get(name)(cfg, input_shape)


class Quantizer(torch.nn.Module):
    """
    Quantizers 'round' floats to the nearest int, which is necessary for compression
    of codes.
    """
    def __init__(self, cfg, input_shape):
        super(Quantizer, self).__init__()
        self.input_shape = input_shape
        self.in_features = cfg.MODEL.QUANTIZER.IN_FEATURES
        self.feat_weights = cfg.MODEL.QUANTIZER.FEAT_WEIGHTS
        self.device = torch.device(cfg.MODEL.DEVICE)


    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).

        Returns:
            dict[str->Tensor]: output data as a mapping from the same feature names as the input
                to tensor. Same as input dimensions.
            dict[str->Tensor]: mapping from a named loss to a tensor storing the loss.
                Used during training only. Optional
        """
        raise NotImplementedError()



class TheisRounding(torch.autograd.Function):
    """
    You compute r(x) = [x]. This a transformation from
    """

    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            z = input.round()
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

@QUANTIZER_REGISTRY.register()
class GSM(Quantizer):
    """The GSM provides an estimate of the entropy
    of the quantized distribution.
    """

    def __init__(self, cfg, input_shape, s=6, in_channels=96, patch=128, bsz=None):
        self.s = s
        super(GSM, self).__init__(cfg, input_shape)
        self.params = dict()
        for feat in self.in_features:
            print("GSM FEAT", feat)
            variance = torch.randn([1, input_shape[feat].channels, 1, 1, self.s])
            pi = torch.randn(1, input_shape[feat].channels, 1, 1, self.s)
            self.params[feat] = dict()
            self.params[feat]['variance'] = torch.nn.Parameter(variance)
            self.params[feat]['pi'] = torch.nn.Parameter(pi)


        self.eps = 1e-7
        self.quantize = TheisRounding.apply
        self.uni: Uniform = Uniform(-0.5, 0.5)
        self.eps = 0.0001


    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.training:
            return self.inference(features)
        outputs = dict()
        losses = dict()
        for feat, feat_weight in zip(self.in_features, self.feat_weights):
            assert feat in features, "Feature name: {} not in input dict".format(feat)
            x = features[feat]
            shape: Tuple[int, int, int, int] = x.shape  # type: ignore
            batch, k, i, j = shape

            pi = F.softmax(self.params[feat]['pi'], dim=-1).repeat(batch, 1, i, j, 1)
            u = (
                self.uni.rsample(sample_shape=(batch, k, i, j, 1))
                    .repeat(1, 1, 1, 1, self.s)
                    .to(self.device)
            )
            y = x.unsqueeze(4).repeat(1, 1, 1, 1, self.s)
            variance = self.params[feat]['variance'].repeat(batch, 1, i, j, 1)

            exp_terms = (-0.5 * ((y + u) ** 2) / variance.exp()).exp()
            leading_terms = 1 / (2 * 3.14159 * (self.eps + variance).exp()).sqrt()
            normal_pdfs = leading_terms * exp_terms
            total_pdfs = (pi * normal_pdfs).sum(axis=-1) + self.eps
            outputs[feat] = self.quantize(x)
            loss = feat_weight * -1 * total_pdfs.log2().mean(dim=(0, 1, 2, 3))
            losses["{}_GSM_loss".format(feat)] = loss
        return outputs, losses

    def inference(self, features: Dict[str, torch.Tensor]):
        assert not self.training
        outputs = dict()
        for feat, feat_weight in zip(self.in_features, self.feat_weights):
            assert feat in features, "Feature name: {} not in input dict".format(feat)
            x = features[feat]
            outputs[feat] = self.quantize(x)

        return outputs



