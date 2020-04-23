import torch
import torch.nn as nn
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from ..util import PatchUtil
from ..layers import (TheisRounding, TheisConv, TheisResidual)

class Encoder(nn.Module):
    """The Encoder module will take in 128x128x3 ('width'x'height'x'channel') patches from the
    original image and compress it into a vector.
    """

    def __init__(self, quantize=True):
        super().__init__()

        self.op_1: nn.Sequential = TheisConv(stride=2)
        self.op_2: nn.Sequential = TheisConv(stride=2, input=64, out=128)
        self.op_3: nn.Sequential = TheisResidual()
        self.op_4: nn.Sequential = TheisResidual()
        self.op_5: nn.Sequential = TheisResidual()
        self.op_6: nn.Sequential = TheisConv(stride=2, input=128, out=64)

        self.quantize = TheisRounding.apply if quantize else (lambda x: x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.op_1(x)  # downsample
        z = self.op_2(z)

        z = z + self.op_3(z)  # residual
        z = z + self.op_4(z)
        z = z + self.op_5(z)

        z = self.op_6(z)  # upsample

        z = self.quantize(z)  # quantization trick
        return z

@BACKBONE_REGISTRY.register()
class CompressiveInferenceBackbone(Backbone):
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
