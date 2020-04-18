import logging
from detectron2.config import CfgNode
from detectron2.structures import ImageList
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
import torch

from ..reconstruct_heads.reconstruct_heads import build_reconstruct_heads
from ..backbone import CompressiveEncoderBackbone # import so they are included in the registry!
from ..reconstruct_heads import CompressiveDecoderHead


@META_ARCH_REGISTRY.register()
class RCNNwithReconstruction(GeneralizedRCNN):
    def __init__(self, cfg: CfgNode):
        super(RCNNwithReconstruction, self).__init__(cfg)
        self.reconstruct_heads = build_reconstruct_heads(cfg, self.backbone.output_shape()).to(self.device)

    def forward(self, batched_inputs: torch.Tensor):
        if not self.training:
            return self.inference(batched_inputs, )
        images_to_compare = self.preprocess_image(batched_inputs, norm=False)
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

        _, reconstruction_losses = self.reconstruct_heads(images_to_compare, features)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(reconstruction_losses)
        return losses

    def inference(self, batched_inputs, **kwargs):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
        Returns:
            Dict[str->tensor] mapping output image names to the tensor in n,c,h,w format
            :param batched_inputs:
            :param **kwargs:
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor.float())

        reconstructed_images, loss_dict = self.reconstruct_heads(images, features)
        return reconstructed_images

    def preprocess_image(self, batched_inputs: torch.Tensor, norm=True):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if norm:
            images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
