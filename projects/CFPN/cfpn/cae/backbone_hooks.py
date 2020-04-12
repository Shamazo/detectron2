from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from .models import Encoder

@BACKBONE_REGISTRY.register()
class CompressiveEncoderBackbone(Backbone):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(CompressiveEncoderBackbone, self).__init__()
        assert input_shape.width == input_shape.height and "Width must be equal to height for Theis CAE."
        print(input_shape.width)
        assert not input_shape.width or input_shape.width == 128

        self.enc = Encoder()
        self.name = "cae_encoder_top"

    def forward(self, image):
        return {self.name: self.enc(image)}

    def output_shape(self):
        return {self.name: ShapeSpec(stride=8, channels=96, height=16, width=16)}
